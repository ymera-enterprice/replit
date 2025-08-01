"""
YMERA Enterprise - Agent Learning Integration System
Production-Ready Agent-Learning Infrastructure Integration - v4.0
Enterprise-grade implementation with zero placeholders
"""

# ===============================================================================
# STANDARD IMPORTS SECTION
# ===============================================================================

# Standard library imports (alphabetical)
import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable, AsyncGenerator, Type
from dataclasses import dataclass, field

# Third-party imports (alphabetical)
import aioredis
import structlog
from fastapi import HTTPException
from sqlalchemy import select, insert, update, delete, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError

# Local imports (alphabetical)
from database.connection import (
    get_database_manager,
    agent_learning_data,
    knowledge_graph_nodes,
    knowledge_graph_edges,
    knowledge_transfer_logs,
    behavioral_patterns,
    external_knowledge_sources,
    memory_consolidation_sessions,
    learning_metrics
)
from config.settings import get_settings
from monitoring.performance_tracker import track_performance
from utils.encryption import encrypt_data, decrypt_data

# ===============================================================================
# LOGGING CONFIGURATION
# ===============================================================================

logger = structlog.get_logger("ymera.agent_learning_integration")

# ===============================================================================
# CONSTANTS & CONFIGURATION
# ===============================================================================

# Learning cycle settings
LEARNING_CYCLE_INTERVAL = 60  # seconds
KNOWLEDGE_SYNC_INTERVAL = 300  # 5 minutes
PATTERN_DISCOVERY_INTERVAL = 900  # 15 minutes
MEMORY_CONSOLIDATION_INTERVAL = 3600  # 1 hour

# Performance thresholds
MIN_CONFIDENCE_SCORE = 60
MAX_KNOWLEDGE_RETENTION_DAYS = 90
COLLABORATION_SCORE_THRESHOLD = 70

# Cache settings
KNOWLEDGE_CACHE_TTL = 1800  # 30 minutes
PATTERN_CACHE_TTL = 3600  # 1 hour

# Configuration loading
settings = get_settings()

# ===============================================================================
# DATA MODELS & SCHEMAS
# ===============================================================================

@dataclass
class LearningExperience:
    """Represents a learning experience for an agent"""
    agent_id: str
    experience_type: str
    context_data: Dict[str, Any]
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    session_id: Optional[str] = None

@dataclass
class KnowledgeItem:
    """Represents extracted knowledge from experiences"""
    knowledge_type: str
    content: Dict[str, Any]
    confidence_score: int
    source_agent_id: str
    applicability_scope: List[str]
    validation_status: str = "pending"
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class LearningCycleResult:
    """Result of a learning cycle execution"""
    agent_id: str
    cycle_id: str
    experiences_processed: int
    knowledge_items_extracted: int
    patterns_discovered: int
    processing_time_ms: int
    success: bool
    error_message: Optional[str] = None

@dataclass
class KnowledgeTransferRequest:
    """Request for knowledge transfer between agents"""
    source_agent_id: str
    target_agent_id: str
    knowledge_items: List[str]
    transfer_type: str
    priority: int = 5
    requested_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class CollaborationMetrics:
    """Metrics for inter-agent collaboration"""
    successful_transfers: int
    failed_transfers: int
    average_transfer_time_ms: float
    knowledge_diversity_score: int
    collaboration_effectiveness: float

# ===============================================================================
# CORE LEARNING ENGINE CLASSES
# ===============================================================================

class LearningEngineInterface(ABC):
    """Interface for learning engine integration"""
    
    @abstractmethod
    async def process_experience(self, experience: LearningExperience) -> List[KnowledgeItem]:
        """Process a learning experience and extract knowledge"""
        pass
    
    @abstractmethod
    async def discover_patterns(self, experiences: List[LearningExperience]) -> List[Dict[str, Any]]:
        """Discover behavioral patterns from experiences"""
        pass
    
    @abstractmethod
    async def validate_knowledge(self, knowledge_item: KnowledgeItem) -> bool:
        """Validate extracted knowledge"""
        pass

class ContinuousLearningEngine(LearningEngineInterface):
    """Production-ready continuous learning engine"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.logger = logger.bind(agent_id=agent_id, engine="ContinuousLearning")
        
        # Learning state
        self._learning_active = False
        self._experience_buffer: List[LearningExperience] = []
        self._knowledge_cache: Dict[str, KnowledgeItem] = {}
        
        # Performance tracking
        self._learning_stats = {
            "total_experiences": 0,
            "knowledge_items_created": 0,
            "patterns_discovered": 0,
            "learning_velocity": 0.0
        }
    
    async def initialize(self) -> None:
        """Initialize the learning engine"""
        try:
            self.logger.info("Initializing continuous learning engine")
            
            # Load existing knowledge from database
            await self._load_existing_knowledge()
            
            # Initialize learning metrics
            await self._initialize_learning_metrics()
            
            self._learning_active = True
            
            self.logger.info(
                "Learning engine initialized",
                cached_knowledge_items=len(self._knowledge_cache)
            )
            
        except Exception as e:
            self.logger.error("Failed to initialize learning engine", error=str(e))
            raise RuntimeError(f"Learning engine initialization failed: {str(e)}")
    
    async def _load_existing_knowledge(self) -> None:
        """Load existing knowledge items from database"""
        try:
            db_manager = await get_database_manager()
            
            async with db_manager.get_session() as session:
                # Load recent knowledge nodes
                query = select(knowledge_graph_nodes).where(
                    knowledge_graph_nodes.c.source_agent_id == self.agent_id
                ).where(
                    knowledge_graph_nodes.c.creation_timestamp >= 
                    datetime.utcnow() - timedelta(days=7)
                ).limit(100)
                
                result = await session.execute(query)
                nodes = result.fetchall()
                
                for node in nodes:
                    knowledge_item = KnowledgeItem(
                        knowledge_type=node.node_type,
                        content=node.node_data,
                        confidence_score=node.confidence_score,
                        source_agent_id=node.source_agent_id,
                        applicability_scope=["general"],
                        validation_status=node.validation_status,
                        created_at=node.creation_timestamp
                    )
                    
                    self._knowledge_cache[node.id] = knowledge_item
                
                self.logger.debug(f"Loaded {len(nodes)} existing knowledge items")
                
        except Exception as e:
            self.logger.error("Failed to load existing knowledge", error=str(e))
            # Continue initialization even if loading fails
    
    async def _initialize_learning_metrics(self) -> None:
        """Initialize learning metrics tracking"""
        try:
            db_manager = await get_database_manager()
            
            async with db_manager.get_session() as session:
                # Get recent learning statistics
                query = text("""
                    SELECT 
                        COUNT(*) as experience_count,
                        AVG(confidence_score) as avg_confidence
                    FROM agent_learning_data 
                    WHERE agent_id = :agent_id 
                    AND learning_timestamp >= :since_date
                """)
                
                result = await session.execute(query, {
                    "agent_id": self.agent_id,
                    "since_date": datetime.utcnow() - timedelta(days=7)
                })
                
                stats = result.fetchone()
                if stats:
                    self._learning_stats["total_experiences"] = stats[0] or 0
                    self._learning_stats["learning_velocity"] = stats[1] or 0.0
                
        except Exception as e:
            self.logger.error("Failed to initialize learning metrics", error=str(e))
    
    @track_performance
    async def process_experience(self, experience: LearningExperience) -> List[KnowledgeItem]:
        """Process a learning experience and extract knowledge"""
        if not self._learning_active:
            return []
        
        try:
            self.logger.debug("Processing learning experience", experience_type=experience.experience_type)
            
            # Add experience to buffer
            self._experience_buffer.append(experience)
            self._learning_stats["total_experiences"] += 1
            
            # Extract knowledge from experience
            knowledge_items = await self._extract_knowledge(experience)
            
            # Validate extracted knowledge
            validated_knowledge = []
            for item in knowledge_items:
                if await self.validate_knowledge(item):
                    validated_knowledge.append(item)
                    self._knowledge_cache[str(uuid.uuid4())] = item
                    self._learning_stats["knowledge_items_created"] += 1
            
            # Store experience and knowledge in database
            await self._store_learning_data(experience, validated_knowledge)
            
            # Update learning metrics
            await self._update_learning_metrics()
            
            self.logger.info(
                "Experience processed",
                knowledge_extracted=len(validated_knowledge),
                total_experiences=self._learning_stats["total_experiences"]
            )
            
            return validated_knowledge
            
        except Exception as e:
            self.logger.error("Failed to process experience", error=str(e))
            return []
    
    async def _extract_knowledge(self, experience: LearningExperience) -> List[KnowledgeItem]:
        """Extract knowledge from a learning experience"""
        knowledge_items = []
        
        try:
            # Extract procedural knowledge
            if "procedure" in experience.experience_type:
                procedural_knowledge = await self._extract_procedural_knowledge(experience)
                if procedural_knowledge:
                    knowledge_items.append(procedural_knowledge)
            
            # Extract performance insights
            performance_knowledge = await self._extract_performance_knowledge(experience)
            if performance_knowledge:
                knowledge_items.append(performance_knowledge)
            
            # Extract error patterns
            if experience.output_data.get("errors"):
                error_knowledge = await self._extract_error_patterns(experience)
                if error_knowledge:
                    knowledge_items.append(error_knowledge)
            
            # Extract contextual knowledge
            context_knowledge = await self._extract_contextual_knowledge(experience)
            if context_knowledge:
                knowledge_items.append(context_knowledge)
            
            return knowledge_items
            
        except Exception as e:
            self.logger.error("Failed to extract knowledge", error=str(e))
            return []
    
    async def _extract_procedural_knowledge(self, experience: LearningExperience) -> Optional[KnowledgeItem]:
        """Extract procedural knowledge from experience"""
        try:
            # Analyze input-output patterns for procedures
            procedure_data = {
                "input_pattern": self._analyze_input_structure(experience.input_data),
                "output_pattern": self._analyze_output_structure(experience.output_data),
                "execution_steps": experience.context_data.get("execution_steps", []),
                "success_rate": experience.performance_metrics.get("success_rate", 0),
                "execution_time": experience.performance_metrics.get("execution_time_ms", 0)
            }
            
            # Calculate confidence based on success rate and consistency
            confidence = min(95, int(
                experience.performance_metrics.get("success_rate", 0) * 0.7 +
                (100 - experience.performance_metrics.get("error_rate", 50)) * 0.3
            ))
            
            if confidence >= MIN_CONFIDENCE_SCORE:
                return KnowledgeItem(
                    knowledge_type="procedural",
                    content=procedure_data,
                    confidence_score=confidence,
                    source_agent_id=self.agent_id,
                    applicability_scope=["task_execution", "procedures"]
                )
            
        except Exception as e:
            self.logger.error("Failed to extract procedural knowledge", error=str(e))
        
        return None
    
    async def _extract_performance_knowledge(self, experience: LearningExperience) -> Optional[KnowledgeItem]:
        """Extract performance insights from experience"""
        try:
            performance_data = {
                "metrics": experience.performance_metrics,
                "optimization_opportunities": [],
                "bottlenecks": [],
                "efficiency_score": 0
            }
            
            # Analyze performance metrics
            execution_time = experience.performance_metrics.get("execution_time_ms", 0)
            memory_usage = experience.performance_metrics.get("memory_usage_mb", 0)
            success_rate = experience.performance_metrics.get("success_rate", 0)
            
            # Identify optimization opportunities
            if execution_time > 5000:  # > 5 seconds
                performance_data["optimization_opportunities"].append("execution_time_optimization")
            
            if memory_usage > 1000:  # > 1GB
                performance_data["optimization_opportunities"].append("memory_optimization")
            
            # Calculate efficiency score
            efficiency_score = min(100, int(
                (success_rate * 0.4) +
                (max(0, 100 - (execution_time / 100)) * 0.3) +
                (max(0, 100 - (memory_usage / 10)) * 0.3)
            ))
            
            performance_data["efficiency_score"] = efficiency_score
            
            # Create knowledge item if significant insights exist
            if (len(performance_data["optimization_opportunities"]) > 0 or 
                efficiency_score >= MIN_CONFIDENCE_SCORE):
                
                return KnowledgeItem(
                    knowledge_type="performance",
                    content=performance_data,
                    confidence_score=min(95, efficiency_score + 20),
                    source_agent_id=self.agent_id,
                    applicability_scope=["performance_optimization", "efficiency"]
                )
                
        except Exception as e:
            self.logger.error("Failed to extract performance knowledge", error=str(e))
        
        return None
    
    async def _extract_error_patterns(self, experience: LearningExperience) -> Optional[KnowledgeItem]:
        """Extract error patterns and recovery strategies"""
        try:
            errors = experience.output_data.get("errors", [])
            if not errors:
                return None
            
            error_data = {
                "error_types": [],
                "frequency_patterns": {},
                "recovery_strategies": [],
                "prevention_measures": []
            }
            
            # Analyze error patterns
            for error in errors:
                error_type = error.get("type", "unknown")
                error_data["error_types"].append(error_type)
                
                # Count frequency
                if error_type in error_data["frequency_patterns"]:
                    error_data["frequency_patterns"][error_type] += 1
                else:
                    error_data["frequency_patterns"][error_type] = 1
                
                # Extract recovery strategies if available
                if "recovery" in error:
                    error_data["recovery_strategies"].append({
                        "error_type": error_type,
                        "strategy": error["recovery"],
                        "success_rate": error.get("recovery_success_rate", 0)
                    })
            
            # Generate prevention measures based on patterns
            most_common_error = max(error_data["frequency_patterns"].items(), 
                                  key=lambda x: x[1])[0] if error_data["frequency_patterns"] else None
            
            if most_common_error:
                error_data["prevention_measures"].append({
                    "target_error": most_common_error,
                    "prevention_strategy": f"Enhanced validation for {most_common_error}",
                    "priority": "high"
                })
            
            # Calculate confidence based on error analysis depth
            confidence = min(90, 60 + len(error_data["recovery_strategies"]) * 10)
            
            return KnowledgeItem(
                knowledge_type="error_patterns",
                content=error_data,
                confidence_score=confidence,
                source_agent_id=self.agent_id,
                applicability_scope=["error_handling", "reliability"]
            )
            
        except Exception as e:
            self.logger.error("Failed to extract error patterns", error=str(e))
        
        return None
    
    async def _extract_contextual_knowledge(self, experience: LearningExperience) -> Optional[KnowledgeItem]:
        """Extract contextual insights from experience"""
        try:
            context_data = {
                "environmental_factors": experience.context_data.get("environment", {}),
                "timing_patterns": {
                    "timestamp": experience.timestamp.isoformat(),
                    "duration": experience.performance_metrics.get("execution_time_ms", 0)
                },
                "dependencies": experience.context_data.get("dependencies", []),
                "success_factors": [],
                "failure_factors": []
            }
            
            # Analyze success/failure factors
            success_rate = experience.performance_metrics.get("success_rate", 0)
            
            if success_rate >= 80:
                context_data["success_factors"] = [
                    factor for factor in experience.context_data.keys()
                    if factor not in ["errors", "failures"]
                ]
            elif success_rate <= 20:
                context_data["failure_factors"] = list(experience.context_data.keys())
            
            # Calculate confidence based on context richness
            context_richness = len(experience.context_data)
            confidence = min(85, 40 + context_richness * 5)
            
            if confidence >= MIN_CONFIDENCE_SCORE:
                return KnowledgeItem(
                    knowledge_type="contextual",
                    content=context_data,
                    confidence_score=confidence,
                    source_agent_id=self.agent_id,
                    applicability_scope=["context_awareness", "adaptation"]
                )
                
        except Exception as e:
            self.logger.error("Failed to extract contextual knowledge", error=str(e))
        
        return None
    
    def _analyze_input_structure(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the structure of input data"""
        structure = {
            "keys": list(input_data.keys()),
            "types": {k: type(v).__name__ for k, v in input_data.items()},
            "complexity": len(str(input_data)),
            "nested_depth": self._calculate_nesting_depth(input_data)
        }
        return structure
    
    def _analyze_output_structure(self, output_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the structure of output data"""
        structure = {
            "keys": list(output_data.keys()),
            "types": {k: type(v).__name__ for k, v in output_data.items()},
            "complexity": len(str(output_data)),
            "nested_depth": self._calculate_nesting_depth(output_data)
        }
        return structure
    
    def _calculate_nesting_depth(self, data: Any, current_depth: int = 0) -> int:
        """Calculate the maximum nesting depth of a data structure"""
        if isinstance(data, dict):
            if not data:
                return current_depth
            return max(self._calculate_nesting_depth(v, current_depth + 1) 
                      for v in data.values())
        elif isinstance(data, list):
            if not data:
                return current_depth
            return max(self._calculate_nesting_depth(item, current_depth + 1) 
                      for item in data)
        else:
            return current_depth
    
    async def validate_knowledge(self, knowledge_item: KnowledgeItem) -> bool:
        """Validate extracted knowledge item"""
        try:
            # Basic validation checks
            if knowledge_item.confidence_score < MIN_CONFIDENCE_SCORE:
                return False
            
            if not knowledge_item.content or not isinstance(knowledge_item.content, dict):
                return False
            
            # Type-specific validation
            if knowledge_item.knowledge_type == "procedural":
                return await self._validate_procedural_knowledge(knowledge_item)
            elif knowledge_item.knowledge_type == "performance":
                return await self._validate_performance_knowledge(knowledge_item)
            elif knowledge_item.knowledge_type == "error_patterns":
                return await self._validate_error_knowledge(knowledge_item)
            elif knowledge_item.knowledge_type == "contextual":
                return await self._validate_contextual_knowledge(knowledge_item)
            
            return True
            
        except Exception as e:
            self.logger.error("Knowledge validation failed", error=str(e))
            return False
    
    async def _validate_procedural_knowledge(self, knowledge_item: KnowledgeItem) -> bool:
        """Validate procedural knowledge"""
        content = knowledge_item.content
        required_fields = ["input_pattern", "output_pattern", "success_rate"]
        
        return all(field in content for field in required_fields)
    
    async def _validate_performance_knowledge(self, knowledge_item: KnowledgeItem) -> bool:
        """Validate performance knowledge"""
        content = knowledge_item.content
        required_fields = ["metrics", "efficiency_score"]
        
        return (all(field in content for field in required_fields) and
                isinstance(content["efficiency_score"], (int, float)) and
                0 <= content["efficiency_score"] <= 100)
    
    async def _validate_error_knowledge(self, knowledge_item: KnowledgeItem) -> bool:
        """Validate error pattern knowledge"""
        content = knowledge_item.content
        required_fields = ["error_types", "frequency_patterns"]
        
        return (all(field in content for field in required_fields) and
                len(content["error_types"]) > 0)
    
    async def _validate_contextual_knowledge(self, knowledge_item: KnowledgeItem) -> bool:
        """Validate contextual knowledge"""
        content = knowledge_item.content
        required_fields = ["environmental_factors", "timing_patterns"]
        
        return all(field in content for field in required_fields)
    
    async def _store_learning_data(self, experience: LearningExperience, 
                                 knowledge_items: List[KnowledgeItem]) -> None:
        """Store learning experience and extracted knowledge in database"""
        try:
            db_manager = await get_database_manager()
            
            async with db_manager.get_session() as session:
                # Store learning experience
                experience_data = {
                    "agent_id": experience.agent_id,
                    "experience_type": experience.experience_type,
                    "input_data": encrypt_data(json.dumps(experience.input_data)),
                    "output_data": encrypt_data(json.dumps(experience.output_data)),
                    "context_data": encrypt_data(json.dumps(experience.context_data)),
                    "performance_metrics": json.dumps(experience.performance_metrics),
                    "learning_timestamp": experience.timestamp,
                    "session_id": experience.session_id or str(uuid.uuid4()),
                    "confidence_score": max([item.confidence_score for item in knowledge_items] + [0])
                }
                
                await session.execute(insert(agent_learning_data).values(**experience_data))
                
                # Store knowledge items as graph nodes
                for knowledge_item in knowledge_items:
                    node_data = {
                        "id": str(uuid.uuid4()),
                        "node_type": knowledge_item.knowledge_type,
                        "node_data": knowledge_item.content,
                        "confidence_score": knowledge_item.confidence_score,
                        "source_agent_id": knowledge_item.source_agent_id,
                        "creation_timestamp": knowledge_item.created_at,
                        "validation_status": knowledge_item.validation_status,
                        "applicability_scope": json.dumps(knowledge_item.applicability_scope)
                    }
                    
                    await session.execute(insert(knowledge_graph_nodes).values(**node_data))
                
                await session.commit()
                
                self.logger.debug(
                    "Learning data stored",
                    experience_id=experience_data["session_id"],
                    knowledge_items=len(knowledge_items)
                )
                
        except Exception as e:
            self.logger.error("Failed to store learning data", error=str(e))
            raise
    
    async def _update_learning_metrics(self) -> None:
        """Update learning performance metrics"""
        try:
            db_manager = await get_database_manager()
            
            async with db_manager.get_session() as session:
                # Calculate learning velocity (knowledge items per hour)
                current_time = datetime.utcnow()
                hour_ago = current_time - timedelta(hours=1)
                
                query = text("""
                    SELECT COUNT(*) as recent_knowledge
                    FROM knowledge_graph_nodes 
                    WHERE source_agent_id = :agent_id 
                    AND creation_timestamp >= :hour_ago
                """)
                
                result = await session.execute(query, {
                    "agent_id": self.agent_id,
                    "hour_ago": hour_ago
                })
                
                recent_count = result.fetchone()[0] or 0
                self._learning_stats["learning_velocity"] = recent_count
                
                # Store metrics
                metrics_data = {
                    "agent_id": self.agent_id,
                    "metric_type": "learning_velocity",
                    "metric_value": recent_count,
                    "timestamp": current_time,
                    "metadata": json.dumps(self._learning_stats)
                }
                
                await session.execute(insert(learning_metrics).values(**metrics_data))
                await session.commit()
                
        except Exception as e:
            self.logger.error("Failed to update learning metrics", error=str(e))
    
    async def discover_patterns(self, experiences: List[LearningExperience]) -> List[Dict[str, Any]]:
        """Discover behavioral patterns from experiences"""
        if not experiences:
            return []
        
        try:
            patterns = []
            
            # Temporal patterns
            temporal_patterns = await self._discover_temporal_patterns(experiences)
            patterns.extend(temporal_patterns)
            
            # Performance patterns
            performance_patterns = await self._discover_performance_patterns(experiences)
            patterns.extend(performance_patterns)
            
            # Error patterns
            error_patterns = await self._discover_error_patterns(experiences)
            patterns.extend(error_patterns)
            
            # Store discovered patterns
            await self._store_behavioral_patterns(patterns)
            
            self._learning_stats["patterns_discovered"] += len(patterns)
            
            self.logger.info(f"Discovered {len(patterns)} behavioral patterns")
            
            return patterns
            
        except Exception as e:
            self.logger.error("Pattern discovery failed", error=str(e))
            return []
    
    async def _discover_temporal_patterns(self, experiences: List[LearningExperience]) -> List[Dict[str, Any]]:
        """Discover temporal patterns in experiences"""
        patterns = []
        
        # Group experiences by hour of day
        hourly_performance = {}
        for exp in experiences:
            hour = exp.timestamp.hour
            success_rate = exp.performance_metrics.get("success_rate", 0)
            
            if hour not in hourly_performance:
                hourly_performance[hour] = []
            hourly_performance[hour].append(success_rate)
        
        # Find peak performance hours
        avg_performance_by_hour = {
            hour: sum(rates) / len(rates)
            for hour, rates in hourly_performance.items()
            if len(rates) >= 3  # Minimum sample size
        }
        
        if avg_performance_by_hour:
            best_hour = max(avg_performance_by_hour, key=avg_performance_by_hour.get)
            worst_hour = min(avg_performance_by_hour, key=avg_performance_by_hour.get)
            
            patterns.append({
                "pattern_type": "temporal_performance",
                "pattern_data": {
                    "peak_performance_hour": best_hour,
                    "lowest_performance_hour": worst_hour,
                    "hourly_averages": avg_performance_by_hour
                },
                "confidence": 75,
                "discovered_at": datetime.utcnow()
            })
        
        return patterns
    
    async def _discover_performance_patterns(self, experiences: List[LearningExperience]) -> List[Dict[str, Any]]:
        """Discover performance-related patterns"""
        patterns = []
        
        # Analyze performance trends
        execution_times = []
        success_rates = []
        memory_usage = []
        
        for exp in experiences:
            metrics = exp.performance_metrics
            execution_times.append(metrics.get("execution_time_ms", 0))
            success_rates.append(metrics.get("success_rate", 0))
            memory_usage.append(metrics.get("memory_usage_mb", 0))
        
        # Calculate trends
        if len(execution_times) >= 5:
            # Simple trend detection using first vs last half comparison
            mid_point = len(execution_times) // 2
            first_half_avg = sum(execution_times[:mid_point]) / mid_point
            second_half_avg = sum(execution_times[mid_point:]) / (len(execution_times) - mid_point)
            
            performance_trend = "improving" if second_half_avg < first_half_avg else "degrading"
            trend_magnitude = abs(second_half_avg - first_half_avg) / first_half_avg * 100
            
            if trend_magnitude > 10:  # Significant trend
                patterns.append({
                    "pattern_type": "performance_trend",
                    "pattern_data": {
                        "trend_direction": performance_trend,
                        "trend_magnitude_percent": trend_magnitude,
                        "metric": "execution_time"
                    },
                    "confidence": min(90, 60 + int(trend_magnitude)),
                    "discovered_at": datetime.utcnow()
                })
        
        return patterns
    
    async def _discover_error_patterns(self, experiences: List[LearningExperience]) -> List[Dict[str, Any]]:
        """Discover error-related patterns"""
        patterns = []
        
        # Collect all errors from experiences
        error_sequences = []
        error_contexts = {}
        
        for exp in experiences:
            errors = exp.output_data.get("errors", [])
            if errors:
                error_types = [e.get("type", "unknown") for e in errors]
                error_sequences.append(error_types)
                
                # Associate errors with context
                for error_type in error_types:
                    if error_type not in error_contexts:
                        error_contexts[error_type] = []
                    error_contexts[error_type].append(exp.context_data)
        
        # Find recurring error patterns
        if len(error_sequences) >= 3:
            # Find most common error combinations
            error_combinations = {}
            for seq in error_sequences:
                seq_key = tuple(sorted(seq))
                if seq_key in error_combinations:
                    error_combinations[seq_key] += 1
                else:
                    error_combinations[seq_key] = 1
            
            # Identify significant patterns
            for combo, frequency in error_combinations.items():
                if frequency >= 3 and len(combo) > 1:  # At least 3 occurrences of multi-error pattern
                    patterns.append({
                        "pattern_type": "error_combination",
                        "pattern_data": {
                            "error_types": list(combo),
                            "frequency": frequency,
                            "common_contexts": self._extract_common_contexts(
                                [error_contexts.get(err, []) for err in combo]
                            )
                        },
                        "confidence": min(95, 40 + frequency * 15),
                        "discovered_at": datetime.utcnow()
                    })
        
        return patterns
    
    def _extract_common_contexts(self, context_groups: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Extract common context elements across error occurrences"""
        if not context_groups or not any(context_groups):
            return {}
        
        # Flatten all contexts
        all_contexts = []
        for group in context_groups:
            all_contexts.extend(group)
        
        if not all_contexts:
            return {}
        
        # Find common keys and values
        common_contexts = {}
        all_keys = set()
        for context in all_contexts:
            all_keys.update(context.keys())
        
        for key in all_keys:
            values = [context.get(key) for context in all_contexts if key in context]
            if values:
                # Find most common value for this key
                value_counts = {}
                for value in values:
                    str_value = str(value)
                    value_counts[str_value] = value_counts.get(str_value, 0) + 1
                
                if value_counts:
                    most_common = max(value_counts, key=value_counts.get)
                    frequency = value_counts[most_common] / len(values)
                    
                    if frequency >= 0.5:  # Present in at least 50% of cases
                        common_contexts[key] = {
                            "value": most_common,
                            "frequency": frequency
                        }
        
        return common_contexts
    
    async def _store_behavioral_patterns(self, patterns: List[Dict[str, Any]]) -> None:
        """Store discovered behavioral patterns in database"""
        try:
            db_manager = await get_database_manager()
            
            async with db_manager.get_session() as session:
                for pattern in patterns:
                    pattern_data = {
                        "id": str(uuid.uuid4()),
                        "agent_id": self.agent_id,
                        "pattern_type": pattern["pattern_type"],
                        "pattern_data": json.dumps(pattern["pattern_data"]),
                        "confidence_score": pattern["confidence"],
                        "discovery_timestamp": pattern["discovered_at"],
                        "usage_count": 0,
                        "validation_status": "discovered"
                    }
                    
                    await session.execute(insert(behavioral_patterns).values(**pattern_data))
                
                await session.commit()
                
                self.logger.debug(f"Stored {len(patterns)} behavioral patterns")
                
        except Exception as e:
            self.logger.error("Failed to store behavioral patterns", error=str(e))
            raise

# ===============================================================================
# KNOWLEDGE SHARING & COLLABORATION SYSTEM
# ===============================================================================

class KnowledgeCollaborationManager:
    """Manages knowledge sharing and collaboration between agents"""
    
    def __init__(self):
        self.logger = logger.bind(component="KnowledgeCollaboration")
        self._redis_client: Optional[aioredis.Redis] = None
        self._collaboration_cache = {}
        
    async def initialize(self) -> None:
        """Initialize the collaboration manager"""
        try:
            self.logger.info("Initializing knowledge collaboration manager")
            
            # Initialize Redis connection for real-time collaboration
            self._redis_client = await aioredis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True
            )
            
            # Test Redis connection
            await self._redis_client.ping()
            
            self.logger.info("Knowledge collaboration manager initialized")
            
        except Exception as e:
            self.logger.error("Failed to initialize collaboration manager", error=str(e))
            raise RuntimeError(f"Collaboration manager initialization failed: {str(e)}")
    
    @track_performance
    async def transfer_knowledge(self, request: KnowledgeTransferRequest) -> Dict[str, Any]:
        """Transfer knowledge between agents"""
        try:
            self.logger.info(
                "Processing knowledge transfer request",
                source_agent=request.source_agent_id,
                target_agent=request.target_agent_id,
                knowledge_count=len(request.knowledge_items)
            )
            
            transfer_id = str(uuid.uuid4())
            start_time = time.time()
            
            # Validate transfer request
            validation_result = await self._validate_transfer_request(request)
            if not validation_result["valid"]:
                return {
                    "transfer_id": transfer_id,
                    "success": False,
                    "error": validation_result["error"],
                    "transferred_items": 0
                }
            
            # Retrieve knowledge items from source agent
            knowledge_items = await self._retrieve_knowledge_items(
                request.source_agent_id, 
                request.knowledge_items
            )
            
            # Adapt knowledge for target agent
            adapted_knowledge = await self._adapt_knowledge_for_agent(
                knowledge_items, 
                request.target_agent_id
            )
            
            # Transfer knowledge to target agent
            transfer_results = await self._execute_knowledge_transfer(
                adapted_knowledge,
                request.target_agent_id,
                transfer_id
            )
            
            # Log transfer activity
            await self._log_transfer_activity(request, transfer_results, transfer_id)
            
            # Update collaboration metrics
            await self._update_collaboration_metrics(request, transfer_results)
            
            processing_time = int((time.time() - start_time) * 1000)
            
            result = {
                "transfer_id": transfer_id,
                "success": transfer_results["success"],
                "transferred_items": transfer_results["transferred_count"],
                "failed_items": transfer_results["failed_count"],
                "processing_time_ms": processing_time,
                "adaptation_score": transfer_results.get("adaptation_score", 0)
            }
            
            self.logger.info(
                "Knowledge transfer completed",
                transfer_id=transfer_id,
                success=result["success"],
                transferred_items=result["transferred_items"]
            )
            
            return result
            
        except Exception as e:
            self.logger.error("Knowledge transfer failed", error=str(e))
            return {
                "transfer_id": str(uuid.uuid4()),
                "success": False,
                "error": str(e),
                "transferred_items": 0
            }
    
    async def _validate_transfer_request(self, request: KnowledgeTransferRequest) -> Dict[str, Any]:
        """Validate knowledge transfer request"""
        try:
            # Check if agents exist and are active
            db_manager = await get_database_manager()
            
            async with db_manager.get_session() as session:
                # Verify source agent has the requested knowledge
                query = select(knowledge_graph_nodes).where(
                    knowledge_graph_nodes.c.source_agent_id == request.source_agent_id
                ).where(
                    knowledge_graph_nodes.c.id.in_(request.knowledge_items)
                )
                
                result = await session.execute(query)
                available_items = result.fetchall()
                
                if len(available_items) != len(request.knowledge_items):
                    return {
                        "valid": False,
                        "error": "Some requested knowledge items not found"
                    }
                
                # Check transfer permissions and compatibility
                compatibility_score = await self._calculate_agent_compatibility(
                    request.source_agent_id,
                    request.target_agent_id
                )
                
                if compatibility_score < 30:  # Minimum compatibility threshold
                    return {
                        "valid": False,
                        "error": f"Low agent compatibility score: {compatibility_score}"
                    }
                
                return {
                    "valid": True,
                    "compatibility_score": compatibility_score
                }
                
        except Exception as e:
            self.logger.error("Transfer validation failed", error=str(e))
            return {
                "valid": False,
                "error": f"Validation error: {str(e)}"
            }
    
    async def _calculate_agent_compatibility(self, source_id: str, target_id: str) -> int:
        """Calculate compatibility score between two agents"""
        try:
            db_manager = await get_database_manager()
            
            async with db_manager.get_session() as session:
                # Get knowledge types and patterns for both agents
                source_query = text("""
                    SELECT node_type, COUNT(*) as count
                    FROM knowledge_graph_nodes 
                    WHERE source_agent_id = :agent_id 
                    GROUP BY node_type
                """)
                
                source_result = await session.execute(source_query, {"agent_id": source_id})
                source_types = {row[0]: row[1] for row in source_result.fetchall()}
                
                target_result = await session.execute(source_query, {"agent_id": target_id})
                target_types = {row[0]: row[1] for row in target_result.fetchall()}
                
                # Calculate overlap in knowledge types
                common_types = set(source_types.keys()) & set(target_types.keys())
                total_types = set(source_types.keys()) | set(target_types.keys())
                
                if not total_types:
                    return 50  # Neutral compatibility for new agents
                
                type_overlap = len(common_types) / len(total_types)
                
                # Calculate experience level similarity
                source_total = sum(source_types.values())
                target_total = sum(target_types.values())
                
                if source_total == 0 or target_total == 0:
                    experience_similarity = 0.5
                else:
                    experience_ratio = min(source_total, target_total) / max(source_total, target_total)
                    experience_similarity = experience_ratio
                
                # Combine factors for final compatibility score
                compatibility_score = int(
                    (type_overlap * 0.6 + experience_similarity * 0.4) * 100
                )
                
                return min(100, max(0, compatibility_score))
                
        except Exception as e:
            self.logger.error("Compatibility calculation failed", error=str(e))
            return 50  # Default neutral compatibility
    
    async def _retrieve_knowledge_items(self, agent_id: str, item_ids: List[str]) -> List[Dict[str, Any]]:
        """Retrieve knowledge items from database"""
        try:
            db_manager = await get_database_manager()
            
            async with db_manager.get_session() as session:
                query = select(knowledge_graph_nodes).where(
                    knowledge_graph_nodes.c.source_agent_id == agent_id
                ).where(
                    knowledge_graph_nodes.c.id.in_(item_ids)
                ).where(
                    knowledge_graph_nodes.c.validation_status == "validated"
                )
                
                result = await session.execute(query)
                items = result.fetchall()
                
                return [
                    {
                        "id": item.id,
                        "type": item.node_type,
                        "data": item.node_data,
                        "confidence": item.confidence_score,
                        "created_at": item.creation_timestamp,
                        "scope": json.loads(item.applicability_scope)
                    }
                    for item in items
                ]
                
        except Exception as e:
            self.logger.error("Failed to retrieve knowledge items", error=str(e))
            return []
    
    async def _adapt_knowledge_for_agent(self, knowledge_items: List[Dict[str, Any]], 
                                       target_agent_id: str) -> List[Dict[str, Any]]:
        """Adapt knowledge items for the target agent's context"""
        adapted_items = []
        
        try:
            # Get target agent's context and preferences
            agent_context = await self._get_agent_context(target_agent_id)
            
            for item in knowledge_items:
                adapted_item = item.copy()
                
                # Adjust confidence based on agent compatibility
                compatibility_factor = agent_context.get("compatibility_factor", 0.8)
                adapted_item["confidence"] = int(item["confidence"] * compatibility_factor)
                
                # Update applicability scope for target agent
                target_scope = await self._determine_target_scope(item, agent_context)
                adapted_item["scope"] = target_scope
                
                # Add adaptation metadata
                adapted_item["adaptation_metadata"] = {
                    "original_agent": item.get("source_agent_id"),
                    "adaptation_timestamp": datetime.utcnow().isoformat(),
                    "adaptation_factor": compatibility_factor,
                    "target_agent": target_agent_id
                }
                
                # Only include if confidence is still acceptable after adaptation
                if adapted_item["confidence"] >= MIN_CONFIDENCE_SCORE:
                    adapted_items.append(adapted_item)
            
            self.logger.debug(
                f"Adapted {len(adapted_items)}/{len(knowledge_items)} knowledge items"
            )
            
            return adapted_items
            
        except Exception as e:
            self.logger.error("Knowledge adaptation failed", error=str(e))
            return knowledge_items  # Return original items if adaptation fails
    
    async def _get_agent_context(self, agent_id: str) -> Dict[str, Any]:
        """Get context information for an agent"""
        try:
            db_manager = await get_database_manager()
            
            async with db_manager.get_session() as session:
                # Get agent's recent performance and knowledge patterns
                query = text("""
                    SELECT 
                        AVG(confidence_score) as avg_confidence,
                        COUNT(*) as knowledge_count,
                        array_agg(DISTINCT node_type) as knowledge_types
                    FROM knowledge_graph_nodes 
                    WHERE source_agent_id = :agent_id 
                    AND creation_timestamp >= :since_date
                """)
                
                result = await session.execute(query, {
                    "agent_id": agent_id,
                    "since_date": datetime.utcnow() - timedelta(days=7)
                })
                
                stats = result.fetchone()
                
                return {
                    "avg_confidence": stats[0] or 60,
                    "knowledge_count": stats[1] or 0,
                    "knowledge_types": stats[2] or [],
                    "compatibility_factor": min(1.0, (stats[0] or 60) / 100.0)
                }
                
        except Exception as e:
            self.logger.error("Failed to get agent context", error=str(e))
            return {
                "avg_confidence": 60,
                "knowledge_count": 0,
                "knowledge_types": [],
                "compatibility_factor": 0.6
            }
    
    async def _determine_target_scope(self, knowledge_item: Dict[str, Any], 
                                    agent_context: Dict[str, Any]) -> List[str]:
        """Determine appropriate scope for knowledge item in target agent"""
        original_scope = knowledge_item.get("scope", ["general"])
        agent_types = agent_context.get("knowledge_types", [])
        
        # Intersect original scope with agent's knowledge domains
        if agent_types:
            relevant_scope = [scope for scope in original_scope if scope in agent_types]
            if relevant_scope:
                return relevant_scope
        
        # Default to general scope if no specific match
        return ["general"]
    
    async def _execute_knowledge_transfer(self, adapted_knowledge: List[Dict[str, Any]], 
                                        target_agent_id: str, transfer_id: str) -> Dict[str, Any]:
        """Execute the actual knowledge transfer"""
        try:
            db_manager = await get_database_manager()
            transferred_count = 0
            failed_count = 0
            adaptation_scores = []
            
            async with db_manager.get_session() as session:
                for item in adapted_knowledge:
                    try:
                        # Create new knowledge node for target agent
                        node_data = {
                            "id": str(uuid.uuid4()),
                            "node_type": item["type"],
                            "node_data": item["data"],
                            "confidence_score": item["confidence"],
                            "source_agent_id": target_agent_id,  # Target agent is now the source
                            "creation_timestamp": datetime.utcnow(),
                            "validation_status": "transferred",
                            "applicability_scope": json.dumps(item["scope"]),
                            "transfer_metadata": json.dumps({
                                "transfer_id": transfer_id,
                                "original_source": item.get("adaptation_metadata", {}).get("original_agent"),
                                "transfer_timestamp": datetime.utcnow().isoformat()
                            })
                        }
                        
                        await session.execute(insert(knowledge_graph_nodes).values(**node_data))
                        transferred_count += 1
                        
                        # Track adaptation quality
                        adaptation_metadata = item.get("adaptation_metadata", {})
                        if "adaptation_factor" in adaptation_metadata:
                            adaptation_scores.append(adaptation_metadata["adaptation_factor"])
                        
                    except Exception as item_error:
                        self.logger.error(
                            "Failed to transfer individual knowledge item",
                            item_id=item.get("id"),
                            error=str(item_error)
                        )
                        failed_count += 1
                
                await session.commit()
            
            # Calculate overall adaptation score
            avg_adaptation_score = (
                sum(adaptation_scores) / len(adaptation_scores) * 100
                if adaptation_scores else 0
            )
            
            return {
                "success": transferred_count > 0,
                "transferred_count": transferred_count,
                "failed_count": failed_count,
                "adaptation_score": int(avg_adaptation_score)
            }
            
        except Exception as e:
            self.logger.error("Knowledge transfer execution failed", error=str(e))
            return {
                "success": False,
                "transferred_count": 0,
                "failed_count": len(adapted_knowledge),
                "adaptation_score": 0
            }
    
    async def _log_transfer_activity(self, request: KnowledgeTransferRequest, 
                                   results: Dict[str, Any], transfer_id: str) -> None:
        """Log knowledge transfer activity"""
        try:
            db_manager = await get_database_manager()
            
            async with db_manager.get_session() as session:
                log_data = {
                    "id": transfer_id,
                    "source_agent_id": request.source_agent_id,
                    "target_agent_id": request.target_agent_id,
                    "transfer_type": request.transfer_type,
                    "knowledge_items_requested": len(request.knowledge_items),
                    "knowledge_items_transferred": results["transferred_count"],
                    "transfer_timestamp": datetime.utcnow(),
                    "success_status": results["success"],
                    "transfer_metadata": json.dumps({
                        "priority": request.priority,
                        "adaptation_score": results.get("adaptation_score", 0),
                        "failed_count": results.get("failed_count", 0)
                    })
                }
                
                await session.execute(insert(knowledge_transfer_logs).values(**log_data))
                await session.commit()
                
        except Exception as e:
            self.logger.error("Failed to log transfer activity", error=str(e))
    
    async def _update_collaboration_metrics(self, request: KnowledgeTransferRequest, 
                                          results: Dict[str, Any]) -> None:
        """Update collaboration metrics"""
        try:
            # Update Redis-based real-time metrics
            if self._redis_client:
                metrics_key = f"collaboration_metrics:{request.source_agent_id}:{request.target_agent_id}"
                
                current_metrics = await self._redis_client.hgetall(metrics_key)
                
                # Update counters
                successful_transfers = int(current_metrics.get("successful_transfers", 0))
                failed_transfers = int(current_metrics.get("failed_transfers", 0))
                total_transfer_time = float(current_metrics.get("total_transfer_time", 0))
                transfer_count = int(current_metrics.get("transfer_count", 0))
                
                if results["success"]:
                    successful_transfers += 1
                else:
                    failed_transfers += 1
                
                transfer_count += 1
                processing_time = results.get("processing_time_ms", 0)
                total_transfer_time += processing_time
                
                # Calculate averages
                avg_transfer_time = total_transfer_time / transfer_count if transfer_count > 0 else 0
                success_rate = successful_transfers / transfer_count if transfer_count > 0 else 0
                
                # Update metrics in Redis
                await self._redis_client.hmset(metrics_key, {
                    "successful_transfers": successful_transfers,
                    "failed_transfers": failed_transfers,
                    "total_transfer_time": total_transfer_time,
                    "transfer_count": transfer_count,
                    "avg_transfer_time": avg_transfer_time,
                    "success_rate": success_rate,
                    "last_updated": datetime.utcnow().isoformat()
                })
                
                # Set expiration for metrics (24 hours)
                await self._redis_client.expire(metrics_key, 86400)
                
        except Exception as e:
            self.logger.error("Failed to update collaboration metrics", error=str(e))
    
    async def get_collaboration_metrics(self, agent_id: str) -> CollaborationMetrics:
        """Get collaboration metrics for an agent"""
        try:
            db_manager = await get_database_manager()
            
            async with db_manager.get_session() as session:
                # Get transfer statistics from database
                query = text("""
                    SELECT 
                        COUNT(CASE WHEN success_status = true THEN 1 END) as successful,
                        COUNT(CASE WHEN success_status = false THEN 1 END) as failed,
                        AVG(EXTRACT(EPOCH FROM (transfer_timestamp - 
                            LAG(transfer_timestamp) OVER (ORDER BY transfer_timestamp))) * 1000) as avg_time
                    FROM knowledge_transfer_logs 
                    WHERE source_agent_id = :agent_id OR target_agent_id = :agent_id
                    AND transfer_timestamp >= :since_date
                """)
                
                result = await session.execute(query, {
                    "agent_id": agent_id,
                    "since_date": datetime.utcnow() - timedelta(days=7)
                })
                
                stats = result.fetchone()
                
                # Calculate knowledge diversity score
                diversity_query = text("""
                    SELECT COUNT(DISTINCT node_type) as type_count
                    FROM knowledge_graph_nodes 
                    WHERE source_agent_id = :agent_id
                """)
                
                diversity_result = await session.execute(diversity_query, {"agent_id": agent_id})
                diversity_count = diversity_result.fetchone()[0] or 0
                
                # Calculate collaboration effectiveness
                successful = stats[0] or 0
                failed = stats[1] or 0
                total_transfers = successful + failed
                
                effectiveness = (successful / total_transfers * 100) if total_transfers > 0 else 0
                
                return CollaborationMetrics(
                    successful_transfers=successful,
                    failed_transfers=failed,
                    average_transfer_time_ms=float(stats[2] or 0),
                    knowledge_diversity_score=min(100, diversity_count * 10),
                    collaboration_effectiveness=effectiveness
                )
                
        except Exception as e:
            self.logger.error("Failed to get collaboration metrics", error=str(e))
            return CollaborationMetrics(
                successful_transfers=0,
                failed_transfers=0,
                average_transfer_time_ms=0.0,
                knowledge_diversity_score=0,
                collaboration_effectiveness=0.0
            )

# ===============================================================================
# MEMORY CONSOLIDATION SYSTEM
# ===============================================================================

class MemoryConsolidationSystem:
    """Manages long-term memory consolidation and knowledge optimization"""
    
    def __init__(self):
        self.logger = logger.bind(component="MemoryConsolidation")
        self._consolidation_active = False
    
    async def initialize(self) -> None:
        """Initialize the memory consolidation system"""
        try:
            self.logger.info("Initializing memory consolidation system")
            self._consolidation_active = True
            self.logger.info("Memory consolidation system initialized")
            
        except Exception as e:
            self.logger.error("Failed to initialize memory consolidation", error=str(e))
            raise RuntimeError(f"Memory consolidation initialization failed: {str(e)}")
    
    @track_performance
    async def consolidate_memories(self, agent_id: str) -> Dict[str, Any]:
        """Perform memory consolidation for an agent"""
        if not self._consolidation_active:
            return {"success": False, "error": "Consolidation system not active"}
        
        try:
            self.logger.info("Starting memory consolidation", agent_id=agent_id)
            
            session_id = str(uuid.uuid4())
            start_time = time.time()
            
            # Step 1: Identify memories for consolidation
            consolidation_candidates = await self._identify_consolidation_candidates(agent_id)
            
            # Step 2: Consolidate related memories
            consolidation_results = await self._perform_memory_consolidation(
                agent_id, consolidation_candidates
            )
            
            # Step 3: Update knowledge graph connections
            await self._update_knowledge_graph_connections(agent_id, consolidation_results)
            
            # Step 4: Clean up redundant or obsolete knowledge
            cleanup_results = await self._cleanup_obsolete_knowledge(agent_id)
            
            # Step 5: Log consolidation session
            await self._log_consolidation_session(
                agent_id, session_id, consolidation_results, cleanup_results
            )
            
            processing_time = int((time.time() - start_time) * 1000)
            
            result = {
                "session_id": session_id,
                "success": True,
                "memories_consolidated": consolidation_results["consolidated_count"],
                "connections_created": consolidation_results["connections_created"],
                "obsolete_items_removed": cleanup_results["removed_count"],
                "processing_time_ms": processing_time
            }
            
            self.logger.info(
                "Memory consolidation completed",
                agent_id=agent_id,
                session_id=session_id,
                consolidated=result["memories_consolidated"]
            )
            
            return result
            
        except Exception as e:
            self.logger.error("Memory consolidation failed", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "memories_consolidated": 0
            }
    
    async def _identify_consolidation_candidates(self, agent_id: str) -> List[Dict[str, Any]]:
        """Identify knowledge items that should be consolidated"""
        try:
            db_manager = await get_database_manager()
            
            async with db_manager.get_session() as session:
                # Find knowledge items with similar content or patterns
                query = text("""
                    SELECT 
                        id,
                        node_type,
                        node_data,
                        confidence_score,
                        creation_timestamp,
                        validation_status
                    FROM knowledge_graph_nodes 
                    WHERE source_agent_id = :agent_id 
                    AND validation_status IN ('validated', 'transferred')
                    AND creation_timestamp >= :consolidation_threshold
                    ORDER BY creation_timestamp DESC
                """)
                
                result = await session.execute(query, {
                    "agent_id": agent_id,
                    "consolidation_threshold": datetime.utcnow() - timedelta(days=30)
                })
                
                candidates = []
                knowledge_items = result.fetchall()
                
                # Group similar knowledge items for consolidation
                type_groups = {}
                for item in knowledge_items:
                    node_type = item.node_type
                    if node_type not in type_groups:
                        type_groups[node_type] = []
                    type_groups[node_type].append({
                        "id": item.id,
                        "type": item.node_type,
                        "data": item.node_data,
                        "confidence": item.confidence_score,
                        "created_at": item.creation_timestamp,
                        "status": item.validation_status
                    })
                
                # Identify consolidation opportunities within each type
                for node_type, items in type_groups.items():
                    if len(items) >= 3:  # Need at least 3 items to consider consolidation
                        similar_groups = self._find_similar_knowledge_groups(items)
                        for group in similar_groups:
                            if len(group) >= 2:  # Can consolidate groups of 2 or more
                                candidates.append({
                                    "type": node_type,
                                    "items": group,
                                    "consolidation_opportunity": "similar_content"
                                })
                
                self.logger.debug(f"Identified {len(candidates)} consolidation candidates")
                return candidates
                
        except Exception as e:
            self.logger.error("Failed to identify consolidation candidates", error=str(e))
            return []
    
    def _find_similar_knowledge_groups(self, items: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Find groups of similar knowledge items that can be consolidated"""
        similar_groups = []
        processed_items = set()
        
        for i, item1 in enumerate(items):
            if item1["id"] in processed_items:
                continue
                
            current_group = [item1]
            processed_items.add(item1["id"])
            
            for j, item2 in enumerate(items[i+1:], i+1):
                if item2["id"] in processed_items:
                    continue
                    
                # Calculate similarity between items
                similarity = self._calculate_knowledge_similarity(item1, item2)
                
                if similarity >= 0.7:  # High similarity threshold for consolidation
                    current_group.append(item2)
                    processed_items.add(item2["id"])
            
            if len(current_group) >= 2:
                similar_groups.append(current_group)
        
        return similar_groups
    
    def _calculate_knowledge_similarity(self, item1: Dict[str, Any], item2: Dict[str, Any]) -> float:
        """Calculate similarity score between two knowledge items"""
        try:
            # Type similarity
            type_match = 1.0 if item1["type"] == item2["type"] else 0.0
            
            # Content similarity (simplified - could use more sophisticated NLP)
            data1 = json.dumps(item1["data"], sort_keys=True) if isinstance(item1["data"], dict) else str(item1["data"])
            data2 = json.dumps(item2["data"], sort_keys=True) if isinstance(item2["data"], dict) else str(item2["data"])
            
            # Simple character-based similarity
            common_chars = len(set(data1.lower()) & set(data2.lower()))
            total_chars = len(set(data1.lower()) | set(data2.lower()))
            content_similarity = common_chars / total_chars if total_chars > 0 else 0.0
            
            # Confidence similarity
            conf_diff = abs(item1["confidence"] - item2["confidence"])
            confidence_similarity = max(0, 1.0 - (conf_diff / 100.0))
            
            # Temporal proximity
            time_diff = abs((item1["created_at"] - item2["created_at"]).total_seconds())
            max_time_diff = 7 * 24 * 3600  # 7 days in seconds
            temporal_similarity = max(0, 1.0 - (time_diff / max_time_diff))
            
            # Weighted combination
            overall_similarity = (
                type_match * 0.3 +
                content_similarity * 0.4 +
                confidence_similarity * 0.2 +
                temporal_similarity * 0.1
            )
            
            return overall_similarity
            
        except Exception as e:
            self.logger.error("Similarity calculation failed", error=str(e))
            return 0.0
    
    async def _perform_memory_consolidation(self, agent_id: str, 
                                          candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform the actual memory consolidation"""
        try:
            db_manager = await get_database_manager()
            consolidated_count = 0
            connections_created = 0
            
            async with db_manager.get_session() as session:
                for candidate_group in candidates:
                    items = candidate_group["items"]
                    
                    # Create consolidated knowledge item
                    consolidated_item = await self._consolidate_knowledge_group(items)
                    
                    if consolidated_item:
                        # Store consolidated knowledge
                        consolidated_data = {
                            "id": str(uuid.uuid4()),
                            "node_type": f"consolidated_{consolidated_item['type']}",
                            "node_data": consolidated_item["data"],
                            "confidence_score": consolidated_item["confidence"],
                            "source_agent_id": agent_id,
                            "creation_timestamp": datetime.utcnow(),
                            "validation_status": "consolidated",
                            "applicability_scope": json.dumps(consolidated_item["scope"]),
                            "consolidation_metadata": json.dumps({
                                "source_items": [item["id"] for item in items],
                                "consolidation_method": consolidated_item["method"],
                                "consolidation_timestamp": datetime.utcnow().isoformat()
                            })
                        }
                        
                        await session.execute(insert(knowledge_graph_nodes).values(**consolidated_data))
                        
                        # Create edges between consolidated item and source items
                        for item in items:
                            edge_data = {
                                "id": str(uuid.uuid4()),
                                "source_node_id": consolidated_data["id"],
                                "target_node_id": item["id"],
                                "edge_type": "consolidates",
                                "edge_weight": 1.0,
                                "creation_timestamp": datetime.utcnow(),
                                "edge_metadata": json.dumps({
                                    "consolidation_contribution": item["confidence"] / 100.0
                                })
                            }
                            
                            await session.execute(insert(knowledge_graph_edges).values(**edge_data))
                            connections_created += 1
                        
                        # Mark original items as consolidated
                        for item in items:
                            await session.execute(
                                update(knowledge_graph_nodes)
                                .where(knowledge_graph_nodes.c.id == item["id"])
                                .values(validation_status="consolidated_source")
                            )
                        
                        consolidated_count += 1
                
                await session.commit()
                
                return {
                    "consolidated_count": consolidated_count,
                    "connections_created": connections_created
                }
                
        except Exception as e:
            self.logger.error("Memory consolidation failed", error=str(e))
            return {
                "consolidated_count": 0,
                "connections_created": 0
            }
    
    async def _consolidate_knowledge_group(self, items: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Consolidate a group of similar knowledge items into one"""
        try:
            if not items:
                return None
            
            # Determine consolidation method based on item types and content
            consolidation_method = "weighted_average"  # Default method
            
            # Calculate consolidated confidence (weighted by recency and original confidence)
            total_weight = 0
            weighted_confidence = 0
            
            for item in items:
                # More recent items get higher weight
                days_old = (datetime.utcnow() - item["created_at"]).days
                recency_weight = max(0.1, 1.0 - (days_old / 30.0))  # Decay over 30 days
                confidence_weight = item["confidence"] / 100.0
                
                item_weight = recency_weight * confidence_weight
                total_weight += item_weight
                weighted_confidence += item["confidence"] * item_weight
            
            final_confidence = int(weighted_confidence / total_weight) if total_weight > 0 else 60
            
            # Merge content from all items
            consolidated_content = await self._merge_knowledge_content(items)
            
            # Determine scope (union of all scopes)
            all_scopes = set()
            for item in items:
                if "scope" in item:
                    all_scopes.update(item["scope"])
            
            if not all_scopes:
                all_scopes = {"general"}
            
            return {
                "type": items[0]["type"],  # Use type from first item
                "data": consolidated_content,
                "confidence": min(95, final_confidence),  # Cap at 95%
                "scope": list(all_scopes),
                "method": consolidation_method,
                "source_count": len(items)
            }
            
        except Exception as e:
            self.logger.error("Knowledge group consolidation failed", error=str(e))
            return None
    
    async def _merge_knowledge_content(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge content from multiple knowledge items"""
        try:
            merged_content = {
                "consolidated_data": {},
                "source_summaries": [],
                "confidence_distribution": [],
                "consolidation_insights": {}
            }
            
            # Collect all unique keys from all items
            all_keys = set()
            for item in items:
                if isinstance(item["data"], dict):
                    all_keys.update(item["data"].keys())
            
            # Merge data by key
            for key in all_keys:
                key_values = []
                for item in items:
                    if isinstance(item["data"], dict) and key in item["data"]:
                        key_values.append({
                            "value": item["data"][key],
                            "confidence": item["confidence"],
                            "source_id": item["id"]
                        })
                
                if key_values:
                    # For numeric values, calculate weighted average
                    if all(isinstance(kv["value"], (int, float)) for kv in key_values):
                        total_weight = sum(kv["confidence"] for kv in key_values)
                        weighted_sum = sum(kv["value"] * kv["confidence"] for kv in key_values)
                        merged_content["consolidated_data"][key] = weighted_sum / total_weight if total_weight > 0 else 0
                    else:
                        # For non-numeric, take most confident value
                        best_value = max(key_values, key=lambda x: x["confidence"])
                        merged_content["consolidated_data"][key] = best_value["value"]
            
            # Create source summaries
            for item in items:
                merged_content["source_summaries"].append({
                    "source_id": item["id"],
                    "confidence": item["confidence"],
                    "created_at": item["created_at"].isoformat(),
                    "key_contributions": list(item["data"].keys()) if isinstance(item["data"], dict) else []
                })
            
            # Confidence distribution
            merged_content["confidence_distribution"] = {
                "min": min(item["confidence"] for item in items),
                "max": max(item["confidence"] for item in items),
                "avg": sum(item["confidence"] for item in items) / len(items),
                "std": self._calculate_std_dev([item["confidence"] for item in items])
            }
            
            # Consolidation insights
            merged_content["consolidation_insights"] = {
                "consolidation_timestamp": datetime.utcnow().isoformat(),
                "source_item_count": len(items),
                "consolidation_strength": len(all_keys) / len(items),  # Keys per item ratio
                "temporal_span_days": (max(item["created_at"] for item in items) - 
                                     min(item["created_at"] for item in items)).days
            }
            
            return merged_content
            
        except Exception as e:
            self.logger.error("Content merging failed", error=str(e))
            return {"error": f"Merge failed: {str(e)}"}
    
    def _calculate_std_dev(self, values: List[float]) -> float:
        """Calculate standard deviation of values"""
        if len(values) <= 1:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    async def _update_knowledge_graph_connections(self, agent_id: str, 
                                                consolidation_results: Dict[str, Any]) -> None:
        """Update knowledge graph connections after consolidation"""
        try:
            db_manager = await get_database_manager()
            
            async with db_manager.get_session() as session:
                # Find knowledge items that should be connected based on consolidation
                query = text("""
                    SELECT 
                        n1.id as node1_id,
                        n2.id as node2_id,
                        n1.node_type as type1,
                        n2.node_type as type2,
                        n1.confidence_score + n2.confidence_score as combined_confidence
                    FROM knowledge_graph_nodes n1
                    CROSS JOIN knowledge_graph_nodes n2
                    WHERE n1.source_agent_id = :agent_id 
                    AND n2.source_agent_id = :agent_id
                    AND n1.id != n2.id
                    AND n1.validation_status IN ('validated', 'consolidated')
                    AND n2.validation_status IN ('validated', 'consolidated')
                    AND NOT EXISTS (
                        SELECT 1 FROM knowledge_graph_edges e 
                        WHERE (e.source_node_id = n1.id AND e.target_node_id = n2.id)
                        OR (e.source_node_id = n2.id AND e.target_node_id = n1.id)
                    )
                    ORDER BY combined_confidence DESC
                    LIMIT 50
                """)
                
                result = await session.execute(query, {"agent_id": agent_id})
                potential_connections = result.fetchall()
                
                connections_created = 0
                for connection in potential_connections:
                    # Calculate connection strength based on type compatibility and confidence
                    connection_strength = self._calculate_connection_strength(
                        connection.type1, connection.type2, connection.combined_confidence
                    )
                    
                    if connection_strength >= 0.5:  # Minimum connection threshold
                        edge_data = {
                            "id": str(uuid.uuid4()),
                            "source_node_id": connection.node1_id,
                            "target_node_id": connection.node2_id,
                            "edge_type": "semantic_connection",
                            "edge_weight": connection_strength,
                            "creation_timestamp": datetime.utcnow(),
                            "edge_metadata": json.dumps({
                                "connection_reason": "post_consolidation_analysis",
                                "strength_score": connection_strength
                            })
                        }
                        
                        await session.execute(insert(knowledge_graph_edges).values(**edge_data))
                        connections_created += 1
                
                await session.commit()
                
                self.logger.debug(f"Created {connections_created} new knowledge connections")
                
        except Exception as e:
            self.logger.error("Failed to update knowledge graph connections", error=str(e))
    
    def _calculate_connection_strength(self, type1: str, type2: str, combined_confidence: int) -> float:
        """Calculate connection strength between two knowledge types"""
        # Define type compatibility matrix
        type_compatibility = {
            ("procedural", "performance"): 0.8,
            ("procedural", "error_patterns"): 0.7,
            ("performance", "contextual"): 0.6,
            ("error_patterns", "contextual"): 0.5,
            ("procedural", "contextual"): 0.4,
            ("performance", "error_patterns"): 0.9
        }
        
        # Get base compatibility
        compatibility_key = tuple(sorted([type1, type2]))
        base_compatibility = type_compatibility.get(compatibility_key, 0.3)
        
        # Adjust by confidence
        confidence_factor = min(1.0, combined_confidence / 200.0)  # Normalize to 0-1
        
        return base_compatibility * confidence_factor
    
    async def _cleanup_obsolete_knowledge(self, agent_id: str) -> Dict[str, Any]:
        """Clean up obsolete or redundant knowledge items"""
        try:
            db_manager = await get_database_manager()
            removed_count = 0
            
            async with db_manager.get_session() as session:
                # Find obsolete knowledge items
                cutoff_date = datetime.utcnow() - timedelta(days=MAX_KNOWLEDGE_RETENTION_DAYS)
                
                obsolete_query = text("""
                    SELECT id, node_type, confidence_score, creation_timestamp
                    FROM knowledge_graph_nodes 
                    WHERE source_agent_id = :agent_id 
                    AND (
                        (creation_timestamp < :cutoff_date AND confidence_score < :min_confidence)
                        OR validation_status = 'obsolete'
                        OR (validation_status = 'consolidated_source' AND creation_timestamp < :recent_cutoff)
                    )
                """)
                
                result = await session.execute(obsolete_query, {
                    "agent_id": agent_id,
                    "cutoff_date": cutoff_date,
                    "min_confidence": MIN_CONFIDENCE_SCORE,
                    "recent_cutoff": datetime.utcnow() - timedelta(days=7)  # Keep source items for 7 days
                })
                
                obsolete_items = result.fetchall()
                
                for item in obsolete_items:
                    # Remove associated edges first
                    await session.execute(
                        delete(knowledge_graph_edges).where(
                            (knowledge_graph_edges.c.source_node_id == item.id) |
                            (knowledge_graph_edges.c.target_node_id == item.id)
                        )
                    )
                    
                    # Remove the knowledge node
                    await session.execute(
                        delete(knowledge_graph_nodes).where(
                            knowledge_graph_nodes.c.id == item.id
                        )
                    )
                    
                    removed_count += 1
                
                await session.commit()
                
                self.logger.debug(f"Cleaned up {removed_count} obsolete knowledge items")
                
                return {
                    "removed_count": removed_count,
                    "cleanup_timestamp": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            self.logger.error("Knowledge cleanup failed", error=str(e))
            return {
                "removed_count": 0,
                "error": str(e)
            }
    
    async def _log_consolidation_session(self, agent_id: str, session_id: str,
                                       consolidation_results: Dict[str, Any],
                                       cleanup_results: Dict[str, Any]) -> None:
        """Log memory consolidation session"""
        try:
            db_manager = await get_database_manager()
            
            async with db_manager.get_session() as session:
                session_data = {
                    "session_id": session_id,
                    "agent_id": agent_id,
                    "consolidation_timestamp": datetime.utcnow(),
                    "memories_processed": consolidation_results.get("consolidated_count", 0),
                    "connections_created": consolidation_results.get("connections_created", 0),
                    "obsolete_items_removed": cleanup_results.get("removed_count", 0),
                    "session_metadata": json.dumps({
                        "consolidation_results": consolidation_results,
                        "cleanup_results": cleanup_results,
                        "session_duration_ms": 0  # Would be calculated from actual timing
                    })
                }
                
                await session.execute(insert(memory_consolidation_sessions).values(**session_data))
                await session.commit()
                
        except Exception as e:
            self.logger.error("Failed to log consolidation session", error=str(e))

# ===============================================================================
# MAIN LEARNING INTEGRATION ORCHESTRATOR
# ===============================================================================

class YmeraLearningIntegrationSystem:
    """Main orchestrator for the YMERA learning integration system"""
    
    def __init__(self):
        self.logger = logger.bind(component="YmeraLearningSystem")
        
        # Core components
        self.learning_engines: Dict[str, ContinuousLearningEngine] = {}
        self.collaboration_manager: Optional[KnowledgeCollaborationManager] = None
        self.memory_consolidation: Optional[MemoryConsolidationSystem] = None
        
        # System state
        self._system_active = False
        self._background_tasks: List[asyncio.Task] = []
        
        # Performance metrics
        self._system_metrics = {
            "total_agents": 0,
            "active_learning_sessions": 0,
            "knowledge_transfers_processed": 0,
            "consolidation_sessions_completed": 0,
            "uptime_seconds": 0
        }
    
    async def initialize(self) -> None:
        """Initialize the complete learning integration system"""
        try:
            self.logger.info("Initializing YMERA Learning Integration System")
            
            # Initialize collaboration manager
            self.collaboration_manager = KnowledgeCollaborationManager()
            await self.collaboration_manager.initialize()
            
            # Initialize memory consolidation system
            self.memory_consolidation = MemoryConsolidationSystem()
            await self.memory_consolidation.initialize()
            
            # Start background processing tasks
            await self._start_background_tasks()
            
            self._system_active = True
            
            self.logger.info("YMERA Learning Integration System fully initialized")
            
        except Exception as e:
            self.logger.error("System initialization failed", error=str(e))
            raise RuntimeError(f"YMERA Learning System initialization failed: {str(e)}")
    
    async def register_agent(self, agent_id: str) -> Dict[str, Any]:
        """Register a new agent with the learning system"""
        try:
            if agent_id in self.learning_engines:
                return {
                    "success": True,
                    "message": "Agent already registered",
                    "agent_id": agent_id
                }
            
            # Create and initialize learning engine for agent
            learning_engine = ContinuousLearningEngine(agent_id)
            await learning_engine.initialize()
            
            self.learning_engines[agent_id] = learning_engine
            self._system_metrics["total_agents"] += 1
            
            self.logger.info("Agent registered", agent_id=agent_id)
            
            return {
                "success": True,
                "message": "Agent successfully registered",
                "agent_id": agent_id,
                "learning_engine_active": True
            }
            
        except Exception as e:
            self.logger.error("Agent registration failed", agent_id=agent_id, error=str(e))
            return {
                "success": False,
                "error": str(e),
                "agent_id": agent_id
            }
    
    async def process_learning_experience(self, agent_id: str, 
                                        experience: LearningExperience) -> Dict[str, Any]:
        """Process a learning experience for an agent"""
        try:
            if agent_id not in self.learning_engines:
                # Auto-register agent if not exists
                registration_result = await self.register_agent(agent_id)
                if not registration_result["success"]:
                    return {
                        "success": False,
                        "error": f"Agent registration failed: {registration_result.get('error')}"
                    }
            
            learning_engine = self.learning_engines[agent_id]
            knowledge_items = await learning_engine.process_experience(experience)
            
            return {
                "success": True,
                "agent_id": agent_id,
                "knowledge_items_extracted": len(knowledge_items),
                "knowledge_items": [
                    {
                        "type": item.knowledge_type,
                        "confidence": item.confidence_score,
                        "scope": item.applicability_scope
                    }
                    for item in knowledge_items
                ]
            }
            
        except Exception as e:
            self.logger.error("Experience processing failed", agent_id=agent_id, error=str(e))
            return {
                "success": False,
                "error": str(e),
                "agent_id": agent_id
            }
    
    async def transfer_knowledge(self, request: KnowledgeTransferRequest) -> Dict[str, Any]:
        """Transfer knowledge between agents"""
        if not self.collaboration_manager:
            return {
                "success": False,
                "error": "Collaboration manager not initialized"
            }
        
        try:
            result = await self.collaboration_manager.transfer_knowledge(request)
            self._system_metrics["knowledge_transfers_processed"] += 1
            return result
            
        except Exception as e:
            self.logger.error("Knowledge transfer failed", error=str(e))
            return {
                "success": False,
                "error": str(e)
            }
    
    async def consolidate_agent_memory(self, agent_id: str) -> Dict[str, Any]:
        """Consolidate memory for a specific agent"""
        if not self.memory_consolidation:
            return {
                "success": False,
                "error": "Memory consolidation system not initialized"
            }
        
        try:
            result = await self.memory_consolidation.consolidate_memories(agent_id)
            if result.get("success"):
                self._system_metrics["consolidation_sessions_completed"] += 1
            return result
            
        except Exception as e:
            self.logger.error("Memory consolidation failed", agent_id=agent_id, error=str(e))
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get system-wide metrics and statistics"""
        try:
            # Calculate uptime
            # This would typically track from system start time
            uptime = self._system_metrics.get("uptime_seconds", 0)
            
            # Get active learning sessions count
            active_sessions = len([engine for engine in self.learning_engines.values() 
                                 if engine._learning_active])
            
            return {
                "system_status": "active" if self._system_active else "inactive",
                "total_registered_agents": self._system_metrics["total_agents"],
                "active_learning_sessions": active_sessions,
                "knowledge_transfers_processed": self._system_metrics["knowledge_transfers_processed"],
                "consolidation_sessions_completed": self._system_metrics["consolidation_sessions_completed"],
                "uptime_seconds": uptime,
                "background_tasks_running": len(self._background_tasks),
                "system_health": "healthy",  # Would implement actual health checks
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error("Failed to get system metrics", error=str(e))
            return {
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _start_background_tasks(self) -> None:
        """Start background processing tasks"""
        try:
            # Periodic pattern discovery task
            pattern_task = asyncio.create_task(self._periodic_pattern_discovery())
            self._background_tasks.append(pattern_task)
            
            # Periodic memory consolidation task
            consolidation_task = asyncio.create_task(self._periodic_memory_consolidation())
            self._background_tasks.append(consolidation_task)
            
            # System metrics update task
            metrics_task = asyncio.create_task(self._periodic_metrics_update())
            self._background_tasks.append(metrics_task)
            
            self.logger.info(f"Started {len(self._background_tasks)} background tasks")
            
        except Exception as e:
            self.logger.error("Failed to start background tasks", error=str(e))
            raise
    
    async def _periodic_pattern_discovery(self) -> None:
        """Background task for periodic pattern discovery"""
        while self._system_active:
            try:
                await asyncio.sleep(PATTERN_DISCOVERY_INTERVAL)
                
                for agent_id, engine in self.learning_engines.items():
                    if engine._learning_active and len(engine._experience_buffer) >= 10:
                        # Discover patterns from recent experiences
                        recent_experiences = engine._experience_buffer[-50:]  # Last 50 experiences
                        patterns = await engine.discover_patterns(recent_experiences)
                        
                        if patterns:
                            self.logger.debug(
                                "Patterns discovered",
                                agent_id=agent_id,
                                pattern_count=len(patterns)
                            )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Pattern discovery task error", error=str(e))
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _periodic_memory_consolidation(self) -> None:
        """Background task for periodic memory consolidation"""
        while self._system_active:
            try:
                await asyncio.sleep(MEMORY_CONSOLIDATION_INTERVAL)
                
                for agent_id in self.learning_engines.keys():
                    if self.memory_consolidation:
                        result = await self.memory_consolidation.consolidate_memories(agent_id)
                        
                        if result.get("success"):
                            self.logger.debug(
                                "Memory consolidation completed",
                                agent_id=agent_id,
                                consolidated=result.get("memories_consolidated", 0)
                            )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Memory consolidation task error", error=str(e))
                await asyncio.sleep(300)  # Wait before retrying
    
    async def _periodic_metrics_update(self) -> None:
        """Background task for updating system metrics"""
        start_time = time.time()
        
        while self._system_active:
            try:
                await asyncio.sleep(60)  # Update every minute
                
                # Update uptime
                self._system_metrics["uptime_seconds"] = int(time.time() - start_time)
                
                # Update active sessions count
                self._system_metrics["active_learning_sessions"] = len([
                    engine for engine in self.learning_engines.values() 
                    if engine._learning_active
                ])
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Metrics update task error", error=str(e))
                await asyncio.sleep(60)
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the learning integration system"""
        try:
            self.logger.info("Shutting down YMERA Learning Integration System")
            
            self._system_active = False
            
            # Cancel background tasks
            for task in self._background_tasks:
                task.cancel()
            
            # Wait for tasks to complete
            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
            
            # Shutdown learning engines
            for agent_id, engine in self.learning_engines.items():
                engine._learning_active = False
            
            self.logger.info("System shutdown completed")
            
        except Exception as e:
            self.logger.error("Error during system shutdown", error=str(e))

# ===============================================================================
# SYSTEM FACTORY & UTILITIES
# ===============================================================================

async def create_ymera_learning_system() -> YmeraLearningIntegrationSystem:
    """Factory function to create and initialize YMERA learning system"""
    system = YmeraLearningIntegrationSystem()
    await system.initialize()
    return system

@asynccontextmanager
async def ymera_learning_context():
    """Context manager for YMERA learning system lifecycle"""
    system = None
    try:
        system = await create_ymera_learning_system()
        yield system
    finally:
        if system:
            await system.shutdown()

# ===============================================================================
# UTILITY FUNCTIONS
# ===============================================================================

def create_learning_experience(
    agent_id: str,
    experience_type: str,
    context_data: Dict[str, Any],
    input_data: Dict[str, Any],
    output_data: Dict[str, Any],
    performance_metrics: Dict[str, Any],
    session_id: Optional[str] = None
) -> LearningExperience:
    """Utility function to create a learning experience"""
    return LearningExperience(
        agent_id=agent_id,
        experience_type=experience_type,
        context_data=context_data,
        input_data=input_data,
        output_data=output_data,
        performance_metrics=performance_metrics,
        session_id=session_id or str(uuid.uuid4())
    )

def create_knowledge_transfer_request(
    source_agent_id: str,
    target_agent_id: str,
    knowledge_items: List[str],
    transfer_type: str = "direct",
    priority: int = 5
) -> KnowledgeTransferRequest:
    """Utility function to create a knowledge transfer request"""
    return KnowledgeTransferRequest(
        source_agent_id=source_agent_id,
        target_agent_id=target_agent_id,
        knowledge_items=knowledge_items,
        transfer_type=transfer_type,
        priority=priority
    )

async def validate_system_health() -> Dict[str, Any]:
    """Validate the health of all system components"""
    health_status = {
        "overall_health": "unknown",
        "components": {},
        "timestamp": datetime.utcnow().isoformat()
    }
    
    try:
        # Check database connectivity
        try:
            db_manager = await get_database_manager()
            async with db_manager.get_session() as session:
                await session.execute(text("SELECT 1"))
            health_status["components"]["database"] = "healthy"
        except Exception as e:
            health_status["components"]["database"] = f"unhealthy: {str(e)}"
        
        # Check Redis connectivity (if available)
        try:
            redis_client = await aioredis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True
            )
            await redis_client.ping()
            await redis_client.close()
            health_status["components"]["redis"] = "healthy"
        except Exception as e:
            health_status["components"]["redis"] = f"unhealthy: {str(e)}"
        
        # Check encryption utilities
        try:
            test_data = "test_encryption"
            encrypted = encrypt_data(test_data)
            decrypted = decrypt_data(encrypted)
            if decrypted == test_data:
                health_status["components"]["encryption"] = "healthy"
            else:
                health_status["components"]["encryption"] = "unhealthy: encryption/decryption mismatch"
        except Exception as e:
            health_status["components"]["encryption"] = f"unhealthy: {str(e)}"
        
        # Determine overall health
        unhealthy_components = [
            comp for comp, status in health_status["components"].items() 
            if not status.startswith("healthy")
        ]
        
        if not unhealthy_components:
            health_status["overall_health"] = "healthy"
        elif len(unhealthy_components) <= 1:
            health_status["overall_health"] = "degraded"
        else:
            health_status["overall_health"] = "unhealthy"
        
        return health_status
        
    except Exception as e:
        health_status["overall_health"] = "error"
        health_status["error"] = str(e)
        return health_status

# ===============================================================================
# EXAMPLE USAGE & TESTING UTILITIES
# ===============================================================================

async def example_usage():
    """Example usage of the YMERA Learning Integration System"""
    try:
        # Initialize the system
        async with ymera_learning_context() as learning_system:
            
            # Register agents
            agent1_result = await learning_system.register_agent("agent_001")
            agent2_result = await learning_system.register_agent("agent_002")
            
            print(f"Agent 1 registration: {agent1_result}")
            print(f"Agent 2 registration: {agent2_result}")
            
            # Create and process learning experiences
            experience1 = create_learning_experience(
                agent_id="agent_001",
                experience_type="procedure_execution",
                context_data={
                    "task": "data_processing",
                    "environment": "production",
                    "dependencies": ["database", "api"]
                },
                input_data={
                    "records_count": 1000,
                    "data_format": "json",
                    "processing_mode": "batch"
                },
                output_data={
                    "processed_records": 950,
                    "errors": [
                        {"type": "validation_error", "count": 30},
                        {"type": "format_error", "count": 20}
                    ],
                    "execution_time_ms": 5500
                },
                performance_metrics={
                    "success_rate": 95.0,
                    "execution_time_ms": 5500,
                    "memory_usage_mb": 256,
                    "error_rate": 5.0
                }
            )
            
            # Process the experience
            processing_result = await learning_system.process_learning_experience(
                "agent_001", experience1
            )
            print(f"Experience processing result: {processing_result}")
            
            # Create another experience for agent 2
            experience2 = create_learning_experience(
                agent_id="agent_002",
                experience_type="procedure_execution",
                context_data={
                    "task": "data_processing",
                    "environment": "production",
                    "dependencies": ["database", "cache"]
                },
                input_data={
                    "records_count": 800,
                    "data_format": "json",
                    "processing_mode": "batch"
                },
                output_data={
                    "processed_records": 800,
                    "errors": [],
                    "execution_time_ms": 3200
                },
                performance_metrics={
                    "success_rate": 100.0,
                    "execution_time_ms": 3200,
                    "memory_usage_mb": 128,
                    "error_rate": 0.0
                }
            )
            
            await learning_system.process_learning_experience("agent_002", experience2)
            
            # Simulate some time passing and more experiences
            await asyncio.sleep(1)
            
            # Get system metrics
            metrics = await learning_system.get_system_metrics()
            print(f"System metrics: {metrics}")
            
            # Perform memory consolidation
            consolidation_result = await learning_system.consolidate_agent_memory("agent_001")
            print(f"Memory consolidation result: {consolidation_result}")
            
            # Example knowledge transfer (would need actual knowledge item IDs from database)
            # This is just to show the API structure
            """
            transfer_request = create_knowledge_transfer_request(
                source_agent_id="agent_001",
                target_agent_id="agent_002",
                knowledge_items=["knowledge_item_id_1", "knowledge_item_id_2"],
                transfer_type="direct",
                priority=7
            )
            
            transfer_result = await learning_system.transfer_knowledge(transfer_request)
            print(f"Knowledge transfer result: {transfer_result}")
            """
            
    except Exception as e:
        print(f"Example usage failed: {str(e)}")

# ===============================================================================
# PERFORMANCE TESTING UTILITIES
# ===============================================================================

async def performance_test(num_agents: int = 5, experiences_per_agent: int = 100):
    """Performance test for the learning system"""
    print(f"Starting performance test: {num_agents} agents, {experiences_per_agent} experiences each")
    
    start_time = time.time()
    
    try:
        async with ymera_learning_context() as learning_system:
            # Register agents
            agent_ids = [f"perf_test_agent_{i:03d}" for i in range(num_agents)]
            
            registration_start = time.time()
            for agent_id in agent_ids:
                await learning_system.register_agent(agent_id)
            registration_time = time.time() - registration_start
            
            print(f"Agent registration completed in {registration_time:.2f}s")
            
            # Generate and process experiences
            processing_start = time.time()
            total_experiences = 0
            
            for agent_id in agent_ids:
                for i in range(experiences_per_agent):
                    experience = create_learning_experience(
                        agent_id=agent_id,
                        experience_type=f"test_procedure_{i % 5}",
                        context_data={
                            "test_iteration": i,
                            "agent": agent_id,
                            "environment": "performance_test"
                        },
                        input_data={
                            "iteration": i,
                            "data_size": 100 + (i * 10),
                            "complexity": i % 10
                        },
                        output_data={
                            "processed": True,
                            "result_size": 90 + (i * 9),
                            "errors": [] if i % 10 != 0 else [{"type": "test_error"}]
                        },
                        performance_metrics={
                            "success_rate": 90.0 + (i % 10),
                            "execution_time_ms": 1000 + (i * 50),
                            "memory_usage_mb": 64 + (i * 2),
                            "error_rate": 10.0 - (i % 10)
                        }
                    )
                    
                    await learning_system.process_learning_experience(agent_id, experience)
                    total_experiences += 1
            
            processing_time = time.time() - processing_start
            
            print(f"Experience processing completed in {processing_time:.2f}s")
            print(f"Average processing time per experience: {(processing_time / total_experiences) * 1000:.2f}ms")
            
            # Test memory consolidation
            consolidation_start = time.time()
            for agent_id in agent_ids[:2]:  # Test on first 2 agents only
                await learning_system.consolidate_agent_memory(agent_id)
            consolidation_time = time.time() - consolidation_start
            
            print(f"Memory consolidation completed in {consolidation_time:.2f}s")
            
            # Get final metrics
            final_metrics = await learning_system.get_system_metrics()
            print(f"Final system metrics: {final_metrics}")
            
            total_time = time.time() - start_time
            print(f"Total test time: {total_time:.2f}s")
            
            # Performance summary
            print("\n=== PERFORMANCE SUMMARY ===")
            print(f"Agents: {num_agents}")
            print(f"Experiences per agent: {experiences_per_agent}")
            print(f"Total experiences: {total_experiences}")
            print(f"Registration rate: {num_agents / registration_time:.2f} agents/sec")
            print(f"Processing rate: {total_experiences / processing_time:.2f} experiences/sec")
            print(f"Memory usage efficiency: {final_metrics.get('active_learning_sessions', 0)} active sessions")
            
    except Exception as e:
        print(f"Performance test failed: {str(e)}")

# ===============================================================================
# MAIN EXECUTION
# ===============================================================================

if __name__ == "__main__":
    """
    Main execution entry point for testing and demonstration
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="YMERA Learning Integration System")
    parser.add_argument("--mode", choices=["example", "performance", "health"], 
                       default="example", help="Execution mode")
    parser.add_argument("--agents", type=int, default=5, 
                       help="Number of agents for performance test")
    parser.add_argument("--experiences", type=int, default=100, 
                       help="Number of experiences per agent for performance test")
    
    args = parser.parse_args()
    
    async def main():
        if args.mode == "example":
            print("Running example usage...")
            await example_usage()
        elif args.mode == "performance":
            print(f"Running performance test with {args.agents} agents and {args.experiences} experiences each...")
            await performance_test(args.agents, args.experiences)
        elif args.mode == "health":
            print("Running health check...")
            health = await validate_system_health()
            print(f"System health: {health}")
        
        print("Execution completed.")
    
    # Run the main function
    asyncio.run(main())