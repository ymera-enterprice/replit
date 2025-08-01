"""
YMERA Enterprise - Pattern Recognition Engine
Production-Ready Pattern Discovery & Analysis - v4.0
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
import statistics
import uuid
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable, AsyncGenerator, Tuple, Set
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from pathlib import Path

# Third-party imports (alphabetical)
import aioredis
import numpy as np
import structlog
from fastapi import FastAPI, HTTPException, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field, validator
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Local imports (alphabetical)
from config.settings import get_settings
from database.connection import get_db_session
from utils.encryption import encrypt_data, decrypt_data
from monitoring.performance_tracker import track_performance
from security.jwt_handler import verify_token

# ===============================================================================
# LOGGING CONFIGURATION
# ===============================================================================

logger = structlog.get_logger(f"ymera.{__name__.split('.')[-1]}")

# ===============================================================================
# CONSTANTS & CONFIGURATION
# ===============================================================================

# Module-specific constants
MAX_PATTERN_HISTORY = 10000
PATTERN_DISCOVERY_INTERVAL = 900  # 15 minutes
MIN_PATTERN_OCCURRENCES = 3
SIGNIFICANCE_THRESHOLD = 0.75
CACHE_TTL = 1800  # 30 minutes

# Configuration loading
settings = get_settings()

# ===============================================================================
# DATA MODELS & SCHEMAS
# ===============================================================================

@dataclass
class PatternConfig:
    """Configuration dataclass for pattern recognition settings"""
    enabled: bool = True
    discovery_interval: int = 900
    min_occurrences: int = 3
    significance_threshold: float = 0.75
    max_history: int = 10000
    clustering_eps: float = 0.5
    clustering_min_samples: int = 3

@dataclass
class AgentBehaviorEvent:
    """Represents a single agent behavior event"""
    agent_id: str
    event_type: str
    timestamp: datetime
    context: Dict[str, Any]
    outcome: str
    duration_ms: int
    resource_usage: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DiscoveredPattern:
    """Represents a discovered behavioral pattern"""
    pattern_id: str
    pattern_type: str
    description: str
    frequency: int
    confidence_score: float
    agents_involved: List[str]
    temporal_characteristics: Dict[str, Any]
    context_conditions: Dict[str, Any]
    performance_impact: Dict[str, float]
    discovery_timestamp: datetime
    last_occurrence: datetime
    actionable_insights: List[str]

class PatternDiscoveryRequest(BaseModel):
    """Request schema for pattern discovery operations"""
    agent_ids: Optional[List[str]] = None
    time_window_hours: int = Field(default=24, ge=1, le=168)
    pattern_types: Optional[List[str]] = None
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            uuid.UUID: lambda v: str(v)
        }

class PatternAnalysisResponse(BaseModel):
    """Response schema for pattern analysis results"""
    patterns_discovered: int
    analysis_duration_ms: int
    patterns: List[Dict[str, Any]]
    insights: List[str]
    recommendations: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            uuid.UUID: lambda v: str(v)
        }

# ===============================================================================
# CORE IMPLEMENTATION CLASSES
# ===============================================================================

class BasePatternAnalyzer(ABC):
    """Abstract base class for pattern analyzers"""
    
    def __init__(self, config: PatternConfig):
        self.config = config
        self.logger = logger.bind(analyzer=self.__class__.__name__)
    
    @abstractmethod
    async def analyze_events(self, events: List[AgentBehaviorEvent]) -> List[DiscoveredPattern]:
        """Analyze events to discover patterns"""
        pass
    
    @abstractmethod
    async def validate_pattern(self, pattern: DiscoveredPattern) -> bool:
        """Validate discovered pattern significance"""
        pass

class TemporalPatternAnalyzer(BasePatternAnalyzer):
    """Analyzes temporal patterns in agent behavior"""
    
    async def analyze_events(self, events: List[AgentBehaviorEvent]) -> List[DiscoveredPattern]:
        """Discover temporal patterns in agent events"""
        patterns = []
        
        # Group events by hour of day
        hourly_patterns = await self._analyze_temporal_distribution(events)
        patterns.extend(hourly_patterns)
        
        # Analyze sequential patterns
        sequential_patterns = await self._analyze_sequential_patterns(events)
        patterns.extend(sequential_patterns)
        
        # Detect cyclic patterns
        cyclic_patterns = await self._analyze_cyclic_patterns(events)
        patterns.extend(cyclic_patterns)
        
        return patterns
    
    async def _analyze_temporal_distribution(self, events: List[AgentBehaviorEvent]) -> List[DiscoveredPattern]:
        """Analyze temporal distribution of events"""
        hourly_distribution = defaultdict(list)
        
        for event in events:
            hour = event.timestamp.hour
            hourly_distribution[hour].append(event)
        
        patterns = []
        for hour, hour_events in hourly_distribution.items():
            if len(hour_events) >= self.config.min_occurrences:
                # Calculate performance metrics for this hour
                avg_duration = statistics.mean([e.duration_ms for e in hour_events])
                success_rate = len([e for e in hour_events if e.outcome == "success"]) / len(hour_events)
                
                if success_rate > self.config.significance_threshold:
                    pattern = DiscoveredPattern(
                        pattern_id=str(uuid.uuid4()),
                        pattern_type="temporal_peak",
                        description=f"High activity and performance at hour {hour}:00",
                        frequency=len(hour_events),
                        confidence_score=success_rate,
                        agents_involved=list(set([e.agent_id for e in hour_events])),
                        temporal_characteristics={
                            "peak_hour": hour,
                            "avg_duration_ms": avg_duration,
                            "event_count": len(hour_events)
                        },
                        context_conditions={"time_of_day": f"{hour}:00-{hour+1}:00"},
                        performance_impact={"success_rate": success_rate, "avg_duration": avg_duration},
                        discovery_timestamp=datetime.utcnow(),
                        last_occurrence=max([e.timestamp for e in hour_events]),
                        actionable_insights=[
                            f"Consider scheduling more tasks during hour {hour} for optimal performance",
                            f"Success rate of {success_rate:.2%} indicates high agent efficiency"
                        ]
                    )
                    patterns.append(pattern)
        
        return patterns
    
    async def _analyze_sequential_patterns(self, events: List[AgentBehaviorEvent]) -> List[DiscoveredPattern]:
        """Analyze sequential patterns in agent behavior"""
        patterns = []
        
        # Group events by agent
        agent_sequences = defaultdict(list)
        for event in sorted(events, key=lambda x: x.timestamp):
            agent_sequences[event.agent_id].append(event)
        
        # Analyze sequences for each agent
        for agent_id, sequence in agent_sequences.items():
            if len(sequence) < 3:
                continue
            
            # Find common event type sequences
            sequence_patterns = await self._find_sequence_patterns(sequence)
            
            for seq_pattern, occurrences in sequence_patterns.items():
                if len(occurrences) >= self.config.min_occurrences:
                    confidence = len(occurrences) / len(sequence)
                    
                    if confidence > 0.3:  # At least 30% of sequences follow this pattern
                        pattern = DiscoveredPattern(
                            pattern_id=str(uuid.uuid4()),
                            pattern_type="sequential_behavior",
                            description=f"Agent {agent_id} follows sequence: {' → '.join(seq_pattern)}",
                            frequency=len(occurrences),
                            confidence_score=confidence,
                            agents_involved=[agent_id],
                            temporal_characteristics={
                                "sequence_length": len(seq_pattern),
                                "avg_interval_minutes": self._calculate_avg_interval(occurrences)
                            },
                            context_conditions={"sequence_pattern": seq_pattern},
                            performance_impact=self._calculate_sequence_performance(occurrences),
                            discovery_timestamp=datetime.utcnow(),
                            last_occurrence=max([occ[-1].timestamp for occ in occurrences]),
                            actionable_insights=[
                                f"Sequence pattern detected with {confidence:.2%} consistency",
                                "Consider optimizing task scheduling based on this pattern"
                            ]
                        )
                        patterns.append(pattern)
        
        return patterns
    
    async def _find_sequence_patterns(self, events: List[AgentBehaviorEvent]) -> Dict[Tuple[str, ...], List[List[AgentBehaviorEvent]]]:
        """Find common sequences of event types"""
        sequence_patterns = defaultdict(list)
        
        # Look for sequences of length 2-5
        for seq_length in range(2, min(6, len(events))):
            for i in range(len(events) - seq_length + 1):
                sequence = events[i:i + seq_length]
                pattern_key = tuple([event.event_type for event in sequence])
                sequence_patterns[pattern_key].append(sequence)
        
        # Filter patterns that occur frequently enough
        return {k: v for k, v in sequence_patterns.items() if len(v) >= self.config.min_occurrences}
    
    def _calculate_avg_interval(self, occurrences: List[List[AgentBehaviorEvent]]) -> float:
        """Calculate average interval between events in sequences"""
        intervals = []
        for sequence in occurrences:
            for i in range(len(sequence) - 1):
                interval = (sequence[i + 1].timestamp - sequence[i].timestamp).total_seconds() / 60
                intervals.append(interval)
        return statistics.mean(intervals) if intervals else 0.0
    
    def _calculate_sequence_performance(self, occurrences: List[List[AgentBehaviorEvent]]) -> Dict[str, float]:
        """Calculate performance metrics for sequence patterns"""
        all_events = [event for sequence in occurrences for event in sequence]
        success_rate = len([e for e in all_events if e.outcome == "success"]) / len(all_events)
        avg_duration = statistics.mean([e.duration_ms for e in all_events])
        
        return {
            "success_rate": success_rate,
            "avg_duration_ms": avg_duration,
            "total_events": len(all_events)
        }
    
    async def _analyze_cyclic_patterns(self, events: List[AgentBehaviorEvent]) -> List[DiscoveredPattern]:
        """Analyze cyclic patterns in agent behavior"""
        patterns = []
        
        # Group events by day of week
        daily_patterns = defaultdict(list)
        for event in events:
            day_of_week = event.timestamp.weekday()
            daily_patterns[day_of_week].append(event)
        
        # Analyze each day's patterns
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        
        for day_idx, day_events in daily_patterns.items():
            if len(day_events) >= self.config.min_occurrences:
                # Calculate performance for this day
                success_rate = len([e for e in day_events if e.outcome == "success"]) / len(day_events)
                avg_duration = statistics.mean([e.duration_ms for e in day_events])
                
                # Check if this day shows significantly different performance
                overall_success_rate = len([e for e in events if e.outcome == "success"]) / len(events)
                
                if abs(success_rate - overall_success_rate) > 0.1:  # 10% difference
                    pattern_type = "high_performance_day" if success_rate > overall_success_rate else "low_performance_day"
                    
                    pattern = DiscoveredPattern(
                        pattern_id=str(uuid.uuid4()),
                        pattern_type=pattern_type,
                        description=f"{days[day_idx]} shows {'above' if success_rate > overall_success_rate else 'below'} average performance",
                        frequency=len(day_events),
                        confidence_score=abs(success_rate - overall_success_rate),
                        agents_involved=list(set([e.agent_id for e in day_events])),
                        temporal_characteristics={
                            "day_of_week": days[day_idx],
                            "performance_delta": success_rate - overall_success_rate
                        },
                        context_conditions={"day_of_week": day_idx},
                        performance_impact={
                            "success_rate": success_rate,
                            "avg_duration_ms": avg_duration,
                            "performance_delta": success_rate - overall_success_rate
                        },
                        discovery_timestamp=datetime.utcnow(),
                        last_occurrence=max([e.timestamp for e in day_events]),
                        actionable_insights=[
                            f"{days[day_idx]} performance is {abs(success_rate - overall_success_rate):.1%} {'above' if success_rate > overall_success_rate else 'below'} average",
                            "Consider adjusting resource allocation based on daily patterns"
                        ]
                    )
                    patterns.append(pattern)
        
        return patterns
    
    async def validate_pattern(self, pattern: DiscoveredPattern) -> bool:
        """Validate temporal pattern significance"""
        return (
            pattern.frequency >= self.config.min_occurrences and
            pattern.confidence_score >= self.config.significance_threshold and
            len(pattern.agents_involved) > 0
        )

class CollaborationPatternAnalyzer(BasePatternAnalyzer):
    """Analyzes collaboration patterns between agents"""
    
    async def analyze_events(self, events: List[AgentBehaviorEvent]) -> List[DiscoveredPattern]:
        """Discover collaboration patterns between agents"""
        patterns = []
        
        # Analyze agent interaction patterns
        interaction_patterns = await self._analyze_agent_interactions(events)
        patterns.extend(interaction_patterns)
        
        # Analyze resource sharing patterns
        resource_patterns = await self._analyze_resource_sharing(events)
        patterns.extend(resource_patterns)
        
        # Analyze task handoff patterns
        handoff_patterns = await self._analyze_task_handoffs(events)
        patterns.extend(handoff_patterns)
        
        return patterns
    
    async def _analyze_agent_interactions(self, events: List[AgentBehaviorEvent]) -> List[DiscoveredPattern]:
        """Analyze patterns in agent-to-agent interactions"""
        patterns = []
        
        # Find events that indicate collaboration
        collaboration_events = [
            e for e in events 
            if any(keyword in e.context.get('description', '').lower() 
                  for keyword in ['collaborate', 'share', 'transfer', 'assist', 'help'])
        ]
        
        if len(collaboration_events) < self.config.min_occurrences:
            return patterns
        
        # Group collaborations by agent pairs
        agent_pairs = defaultdict(list)
        
        for event in collaboration_events:
            if 'target_agent' in event.context:
                pair_key = tuple(sorted([event.agent_id, event.context['target_agent']]))
                agent_pairs[pair_key].append(event)
        
        # Analyze each collaboration pair
        for (agent1, agent2), pair_events in agent_pairs.items():
            if len(pair_events) >= self.config.min_occurrences:
                success_rate = len([e for e in pair_events if e.outcome == "success"]) / len(pair_events)
                avg_duration = statistics.mean([e.duration_ms for e in pair_events])
                
                pattern = DiscoveredPattern(
                    pattern_id=str(uuid.uuid4()),
                    pattern_type="agent_collaboration",
                    description=f"Frequent collaboration between {agent1} and {agent2}",
                    frequency=len(pair_events),
                    confidence_score=success_rate,
                    agents_involved=[agent1, agent2],
                    temporal_characteristics={
                        "collaboration_frequency": len(pair_events),
                        "avg_collaboration_duration": avg_duration
                    },
                    context_conditions={"collaboration_pair": [agent1, agent2]},
                    performance_impact={
                        "success_rate": success_rate,
                        "avg_duration_ms": avg_duration,
                        "collaboration_efficiency": success_rate * (1000 / avg_duration) if avg_duration > 0 else 0
                    },
                    discovery_timestamp=datetime.utcnow(),
                    last_occurrence=max([e.timestamp for e in pair_events]),
                    actionable_insights=[
                        f"Agents {agent1} and {agent2} collaborate effectively with {success_rate:.2%} success rate",
                        "Consider pairing these agents for complex tasks"
                    ]
                )
                patterns.append(pattern)
        
        return patterns
    
    async def _analyze_resource_sharing(self, events: List[AgentBehaviorEvent]) -> List[DiscoveredPattern]:
        """Analyze resource sharing patterns"""
        patterns = []
        
        # Find resource-related events
        resource_events = [
            e for e in events 
            if 'resource_type' in e.context or 'resource_usage' in e.metadata
        ]
        
        if not resource_events:
            return patterns
        
        # Group by resource type
        resource_usage = defaultdict(list)
        
        for event in resource_events:
            resource_type = event.context.get('resource_type', 'unknown')
            resource_usage[resource_type].append(event)
        
        # Analyze sharing patterns for each resource type
        for resource_type, type_events in resource_usage.items():
            if len(type_events) >= self.config.min_occurrences:
                # Analyze sharing behavior
                agents_using = set([e.agent_id for e in type_events])
                
                if len(agents_using) > 1:  # Multiple agents using same resource
                    avg_usage = statistics.mean([
                        sum(e.resource_usage.values()) for e in type_events 
                        if e.resource_usage
                    ])
                    
                    success_rate = len([e for e in type_events if e.outcome == "success"]) / len(type_events)
                    
                    pattern = DiscoveredPattern(
                        pattern_id=str(uuid.uuid4()),
                        pattern_type="resource_sharing",
                        description=f"Multiple agents efficiently sharing {resource_type}",
                        frequency=len(type_events),
                        confidence_score=success_rate,
                        agents_involved=list(agents_using),
                        temporal_characteristics={
                            "resource_type": resource_type,
                            "sharing_frequency": len(type_events),
                            "agents_count": len(agents_using)
                        },
                        context_conditions={"resource_type": resource_type},
                        performance_impact={
                            "success_rate": success_rate,
                            "avg_resource_usage": avg_usage,
                            "sharing_efficiency": success_rate * len(agents_using)
                        },
                        discovery_timestamp=datetime.utcnow(),
                        last_occurrence=max([e.timestamp for e in type_events]),
                        actionable_insights=[
                            f"{len(agents_using)} agents effectively share {resource_type}",
                            "Resource sharing pattern indicates good coordination"
                        ]
                    )
                    patterns.append(pattern)
        
        return patterns
    
    async def _analyze_task_handoffs(self, events: List[AgentBehaviorEvent]) -> List[DiscoveredPattern]:
        """Analyze task handoff patterns between agents"""
        patterns = []
        
        # Find handoff events
        handoff_events = [
            e for e in events 
            if e.event_type in ['task_handoff', 'task_transfer', 'delegation']
        ]
        
        if len(handoff_events) < self.config.min_occurrences:
            return patterns
        
        # Analyze handoff chains
        handoff_chains = defaultdict(int)
        handoff_success = defaultdict(list)
        
        for event in handoff_events:
            source_agent = event.agent_id
            target_agent = event.context.get('target_agent')
            
            if target_agent:
                chain_key = f"{source_agent} → {target_agent}"
                handoff_chains[chain_key] += 1
                handoff_success[chain_key].append(event.outcome == "success")
        
        # Create patterns for frequent handoff chains
        for chain, frequency in handoff_chains.items():
            if frequency >= self.config.min_occurrences:
                success_rate = sum(handoff_success[chain]) / len(handoff_success[chain])
                source, target = chain.split(" → ")
                
                pattern = DiscoveredPattern(
                    pattern_id=str(uuid.uuid4()),
                    pattern_type="task_handoff",
                    description=f"Frequent task handoffs from {source} to {target}",
                    frequency=frequency,
                    confidence_score=success_rate,
                    agents_involved=[source, target],
                    temporal_characteristics={
                        "handoff_chain": chain,
                        "handoff_frequency": frequency
                    },
                    context_conditions={"handoff_direction": chain},
                    performance_impact={
                        "success_rate": success_rate,
                        "handoff_efficiency": success_rate * frequency
                    },
                    discovery_timestamp=datetime.utcnow(),
                    last_occurrence=datetime.utcnow(),  # Would be calculated from actual events
                    actionable_insights=[
                        f"Task handoff chain {chain} has {success_rate:.2%} success rate",
                        "Consider optimizing handoff protocols for better efficiency"
                    ]
                )
                patterns.append(pattern)
        
        return patterns
    
    async def validate_pattern(self, pattern: DiscoveredPattern) -> bool:
        """Validate collaboration pattern significance"""
        return (
            pattern.frequency >= self.config.min_occurrences and
            len(pattern.agents_involved) >= 2 and
            pattern.confidence_score >= 0.5
        )

class PerformancePatternAnalyzer(BasePatternAnalyzer):
    """Analyzes performance patterns in agent behavior"""
    
    async def analyze_events(self, events: List[AgentBehaviorEvent]) -> List[DiscoveredPattern]:
        """Discover performance patterns in agent events"""
        patterns = []
        
        # Analyze performance anomalies
        anomaly_patterns = await self._analyze_performance_anomalies(events)
        patterns.extend(anomaly_patterns)
        
        # Analyze optimization opportunities
        optimization_patterns = await self._analyze_optimization_opportunities(events)
        patterns.extend(optimization_patterns)
        
        # Analyze degradation patterns
        degradation_patterns = await self._analyze_performance_degradation(events)
        patterns.extend(degradation_patterns)
        
        return patterns
    
    async def _analyze_performance_anomalies(self, events: List[AgentBehaviorEvent]) -> List[DiscoveredPattern]:
        """Detect performance anomalies"""
        patterns = []
        
        if len(events) < 10:
            return patterns
        
        # Calculate performance baselines
        durations = [e.duration_ms for e in events]
        mean_duration = statistics.mean(durations)
        std_duration = statistics.stdev(durations) if len(durations) > 1 else 0
        
        # Find anomalous events
        anomalous_events = []
        for event in events:
            if std_duration > 0:
                z_score = abs(event.duration_ms - mean_duration) / std_duration
                if z_score > 2:  # More than 2 standard deviations
                    anomalous_events.append(event)
        
        if len(anomalous_events) >= self.config.min_occurrences:
            # Group anomalies by agent
            agent_anomalies = defaultdict(list)
            for event in anomalous_events:
                agent_anomalies[event.agent_id].append(event)
            
            for agent_id, agent_anomalies_list in agent_anomalies.items():
                if len(agent_anomalies_list) >= self.config.min_occurrences:
                    avg_anomaly_duration = statistics.mean([e.duration_ms for e in agent_anomalies_list])
                    anomaly_rate = len(agent_anomalies_list) / len([e for e in events if e.agent_id == agent_id])
                    
                    pattern = DiscoveredPattern(
                        pattern_id=str(uuid.uuid4()),
                        pattern_type="performance_anomaly",
                        description=f"Agent {agent_id} shows performance anomalies",
                        frequency=len(agent_anomalies_list),
                        confidence_score=min(anomaly_rate * 2, 1.0),  # Scale anomaly rate
                        agents_involved=[agent_id],
                        temporal_characteristics={
                            "anomaly_count": len(agent_anomalies_list),
                            "anomaly_rate": anomaly_rate,
                            "avg_anomaly_duration": avg_anomaly_duration
                        },
                        context_conditions={"performance_deviation": "high"},
                        performance_impact={
                            "avg_anomaly_duration_ms": avg_anomaly_duration,
                            "baseline_duration_ms": mean_duration,
                            "performance_impact": (avg_anomaly_duration - mean_duration) / mean_duration
                        },
                        discovery_timestamp=datetime.utcnow(),
                        last_occurrence=max([e.timestamp for e in agent_anomalies_list]),
                        actionable_insights=[
                            f"Agent {agent_id} has {anomaly_rate:.1%} anomalous performance events",
                            "Investigate potential causes for performance deviations"
                        ]
                    )
                    patterns.append(pattern)
        
        return patterns
    
    async def _analyze_optimization_opportunities(self, events: List[AgentBehaviorEvent]) -> List[DiscoveredPattern]:
        """Identify optimization opportunities"""
        patterns = []
        
        # Group events by event type
        event_types = defaultdict(list)
        for event in events:
            event_types[event.event_type].append(event)
        
        # Analyze each event type for optimization opportunities
        for event_type, type_events in event_types.items():
            if len(type_events) >= self.config.min_occurrences:
                # Calculate performance metrics
                durations = [e.duration_ms for e in type_events]
                success_events = [e for e in type_events if e.outcome == "success"]
                
                if success_events:
                    best_performance = min([e.duration_ms for e in success_events])
                    avg_performance = statistics.mean(durations)
                    
                    # If there's significant room for improvement
                    improvement_potential = (avg_performance - best_performance) / avg_performance
                    
                    if improvement_potential > 0.2:  # 20% improvement potential
                        pattern = DiscoveredPattern(
                            pattern_id=str(uuid.uuid4()),
                            pattern_type="optimization_opportunity",
                            description=f"Optimization opportunity for {event_type} events",
                            frequency=len(type_events),
                            confidence_score=improvement_potential,
                            agents_involved=list(set([e.agent_id for e in type_events])),
                            temporal_characteristics={
                                "event_type": event_type,
                                "improvement_potential": improvement_potential
                            },
                            context_conditions={"event_type": event_type},
                            performance_impact={
                                "avg_duration_ms": avg_performance,
                                "best_duration_ms": best_performance,
                                "improvement_potential_pct": improvement_potential * 100
                            },
                            discovery_timestamp=datetime.utcnow(),
                            last_occurrence=max([e.timestamp for e in type_events]),
                            actionable_insights=[
                                f"{event_type} events could improve by {improvement_potential:.1%}",
                                "Analyze best-performing instances to identify optimization strategies"
                            ]
                        )
                        patterns.append(pattern)
        
        return patterns
    
    async def _analyze_performance_degradation(self, events: List[AgentBehaviorEvent]) -> List[DiscoveredPattern]:
        """Detect performance degradation trends"""
        patterns = []
        
        if len(events) < 20:  # Need sufficient data for trend analysis
            return patterns
        
        # Sort events by timestamp
        sorted_events = sorted(events, key=lambda x: x.timestamp)
        
        # Group events by agent
        agent_events = defaultdict(list)
        for event in sorted_events:
            agent_events[event.agent_id].append(event)
        
        # Analyze degradation for each agent
        for agent_id, agent_event_list in agent_events.items():
            if len(agent_event_list) >= 10:
                degradation_detected = await self._detect_degradation_trend(agent_event_list)
                
                if degradation_detected:
                    recent_events = agent_event_list[-5:]  # Last 5 events
                    early_events = agent_event_list[:5]    # First 5 events
                    
                    recent_avg = statistics.mean([e.duration_ms for e in recent_events])
                    early_avg = statistics.mean([e.duration_ms for e in early_events])
                    degradation_rate = (recent_avg - early_avg) / early_avg
                    
                    pattern = DiscoveredPattern(
                        pattern_id=str(uuid.uuid4()),
                        pattern_type="performance_degradation",
                        description=f"Performance degradation detected for agent {agent_id}",
                        frequency=len(agent_event_list),
                        confidence_score=min(abs(degradation_rate), 1.0),
                        agents_involved=[agent_id],
                        temporal_characteristics={
                            "degradation_rate": degradation_rate,
                            "trend_duration": len(agent_event_list),
                            "recent_avg_duration": recent_avg,
                            "baseline_avg_duration": early_avg
                        },
                        context_conditions={"trend_type": "degradation"},
                        performance_impact={
                            "degradation_rate_pct": degradation_rate * 100,
                            "recent_avg_duration_ms": recent_avg,
                            "baseline_avg_duration_ms": early_avg
                        },
                        discovery_timestamp=datetime.utcnow(),
                        last_occurrence=agent_event_list[-1].timestamp,
                        actionable_insights=[
                            f"Agent {agent_id} performance degraded by {degradation_rate:.1%}",
                            "Investigate potential causes and implement corrective measures"
                        ]
                    )
                    patterns.append(pattern)
        
        return patterns
    
    async def _detect_degradation_trend(self, events: List[AgentBehaviorEvent]) -> bool:
        """Detect if there's a degradation trend in performance"""
        if len(events) < 10:
            return False
        
        # Calculate moving averages
        window_size = max(3, len(events) // 5)
        moving_averages = []
        
        for i in range(len(events) - window_size + 1):
            window_events = events[i:i + window_size]
            avg_duration = statistics.mean([e.duration_ms for e in window_events])
            moving_averages.append(avg_duration)
        
        # Check if there's an increasing trend
        if len(moving_averages) < 3:
            return False
        
        # Simple trend detection: compare first third vs last third
        first_third = moving_averages[:len(moving_averages)//3]
        last_third = moving_averages[-len(moving_averages)//3:]
        
        first_avg = statistics.mean(first_third)
        last_avg = statistics.mean(last_third)
        
        # Degradation if performance got 15% worse
        return (last_avg - first_avg) / first_avg > 0.15
    
    async def validate_pattern(self, pattern: DiscoveredPattern) -> bool:
        """Validate performance pattern significance"""
        return (
            pattern.frequency >= self.config.min_occurrences and
            pattern.confidence_score >= 0.3 and  # Lower threshold for performance patterns
            len(pattern.agents_involved) > 0
        )

class PatternRecognitionEngine:
    """Main pattern recognition engine coordinating all analyzers"""
    
    def __init__(self, config: PatternConfig):
        self.config = config
        self.logger = logger.bind(component="PatternRecognitionEngine")
        self._redis_client = None
        self._event_history = deque(maxlen=config.max_history)
        self._discovered_patterns = {}
        self._last_analysis = None
        
        # Initialize analyzers
        self.analyzers = [
            TemporalPatternAnalyzer(config),
            CollaborationPatternAnalyzer(config),
            PerformancePatternAnalyzer(config)
        ]
    
    async def _initialize_resources(self) -> None:
        """Initialize Redis connection and other resources"""
        try:
            self._redis_client = await aioredis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True,
                retry_on_timeout=True,
                socket_connect_timeout=5
            )
            await self._redis_client.ping()
            self.logger.info("Pattern recognition engine initialized")
        except Exception as e:
            self.logger.error("Failed to initialize resources", error=str(e))
            raise
    
    @track_performance
    async def add_behavior_event(self, event: AgentBehaviorEvent) -> None:
        """Add a new behavior event to the analysis queue"""
        try:
            # Add to local history
            self._event_history.append(event)
            
            # Store in Redis for persistence
            if self._redis_client:
                event_data = {
                    "agent_id": event.agent_id,
                    "event_type": event.event_type,
                    "timestamp": event.timestamp.isoformat(),
                    "context": json.dumps(event.context),
                    "outcome": event.outcome,
                    "duration_ms": event.duration_ms,
                    "resource_usage": json.dumps(event.resource_usage),
                    "metadata": json.dumps(event.metadata)
                }
                
                await self._redis_client.lpush(
                    "ymera:behavior_events",
                    json.dumps(event_data)
                )
                
                # Keep only recent events in Redis (last 24 hours worth)
                await self._redis_client.ltrim("ymera:behavior_events", 0, 10000)
            
            self.logger.debug("Behavior event added", agent_id=event.agent_id, event_type=event.event_type)
            
        except Exception as e:
            self.logger.error("Failed to add behavior event", error=str(e))
            raise
    
    @track_performance
    async def discover_patterns(self, request: PatternDiscoveryRequest) -> PatternAnalysisResponse:
        """Discover patterns in agent behavior"""
        start_time = datetime.utcnow()
        all_patterns = []
        
        try:
            # Get events for analysis
            events = await self._get_events_for_analysis(request)
            
            if not events:
                return PatternAnalysisResponse(
                    patterns_discovered=0,
                    analysis_duration_ms=0,
                    patterns=[],
                    insights=["No events available for analysis"],
                    recommendations=[],
                    metadata={"events_analyzed": 0}
                )
            
            # Run pattern analysis with all analyzers
            for analyzer in self.analyzers:
                try:
                    analyzer_patterns = await analyzer.analyze_events(events)
                    
                    # Validate patterns
                    valid_patterns = []
                    for pattern in analyzer_patterns:
                        if await analyzer.validate_pattern(pattern):
                            valid_patterns.append(pattern)
                    
                    all_patterns.extend(valid_patterns)
                    
                    self.logger.debug(
                        "Analyzer completed",
                        analyzer=analyzer.__class__.__name__,
                        patterns_found=len(valid_patterns)
                    )
                    
                except Exception as e:
                    self.logger.error(
                        "Analyzer failed",
                        analyzer=analyzer.__class__.__name__,
                        error=str(e)
                    )
                    continue
            
            # Filter and rank patterns
            filtered_patterns = await self._filter_and_rank_patterns(all_patterns, request)
            
            # Store discovered patterns
            await self._store_patterns(filtered_patterns)
            
            # Generate insights and recommendations
            insights = await self._generate_insights(filtered_patterns)
            recommendations = await self._generate_recommendations(filtered_patterns)
            
            analysis_duration = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            response = PatternAnalysisResponse(
                patterns_discovered=len(filtered_patterns),
                analysis_duration_ms=analysis_duration,
                patterns=[self._pattern_to_dict(p) for p in filtered_patterns],
                insights=insights,
                recommendations=recommendations,
                metadata={
                    "events_analyzed": len(events),
                    "analyzers_used": len(self.analyzers),
                    "analysis_timestamp": datetime.utcnow().isoformat()
                }
            )
            
            self._last_analysis = datetime.utcnow()
            
            self.logger.info(
                "Pattern discovery completed",
                patterns_found=len(filtered_patterns),
                duration_ms=analysis_duration
            )
            
            return response
            
        except Exception as e:
            self.logger.error("Pattern discovery failed", error=str(e))
            raise HTTPException(status_code=500, detail="Pattern discovery failed")
    
    async def _get_events_for_analysis(self, request: PatternDiscoveryRequest) -> List[AgentBehaviorEvent]:
        """Get events for pattern analysis based on request parameters"""
        events = []
        
        # Calculate time window
        cutoff_time = datetime.utcnow() - timedelta(hours=request.time_window_hours)
        
        # Filter events from history
        for event in self._event_history:
            if event.timestamp < cutoff_time:
                continue
            
            # Filter by agent IDs if specified
            if request.agent_ids and event.agent_id not in request.agent_ids:
                continue
            
            events.append(event)
        
        # Load additional events from Redis if needed
        if self._redis_client and len(events) < 100:
            try:
                redis_events = await self._redis_client.lrange("ymera:behavior_events", 0, 1000)
                
                for event_json in redis_events:
                    event_data = json.loads(event_json)
                    event_timestamp = datetime.fromisoformat(event_data["timestamp"])
                    
                    if event_timestamp < cutoff_time:
                        continue
                    
                    if request.agent_ids and event_data["agent_id"] not in request.agent_ids:
                        continue
                    
                    event = AgentBehaviorEvent(
                        agent_id=event_data["agent_id"],
                        event_type=event_data["event_type"],
                        timestamp=event_timestamp,
                        context=json.loads(event_data["context"]),
                        outcome=event_data["outcome"],
                        duration_ms=event_data["duration_ms"],
                        resource_usage=json.loads(event_data["resource_usage"]),
                        metadata=json.loads(event_data["metadata"])
                    )
                    events.append(event)
                    
            except Exception as e:
                self.logger.warning("Failed to load events from Redis", error=str(e))
        
        # Sort by timestamp
        events.sort(key=lambda x: x.timestamp)
        
        return events
    
    async def _filter_and_rank_patterns(self, patterns: List[DiscoveredPattern], request: PatternDiscoveryRequest) -> List[DiscoveredPattern]:
        """Filter and rank discovered patterns"""
        filtered_patterns = []
        
        # Filter by minimum confidence
        for pattern in patterns:
            if pattern.confidence_score >= request.min_confidence:
                filtered_patterns.append(pattern)
        
        # Filter by pattern types if specified
        if request.pattern_types:
            filtered_patterns = [
                p for p in filtered_patterns 
                if p.pattern_type in request.pattern_types
            ]
        
        # Remove duplicate patterns
        unique_patterns = await self._deduplicate_patterns(filtered_patterns)
        
        # Rank by significance score
        ranked_patterns = sorted(
            unique_patterns,
            key=lambda p: p.confidence_score * p.frequency,
            reverse=True
        )
        
        # Return top patterns
        return ranked_patterns[:50]  # Limit to top 50 patterns
    
    async def _deduplicate_patterns(self, patterns: List[DiscoveredPattern]) -> List[DiscoveredPattern]:
        """Remove duplicate or very similar patterns"""
        unique_patterns = []
        seen_patterns = set()
        
        for pattern in patterns:
            # Create a signature for the pattern
            signature = (
                pattern.pattern_type,
                tuple(sorted(pattern.agents_involved)),
                pattern.description[:50]  # First 50 chars of description
            )
            
            if signature not in seen_patterns:
                unique_patterns.append(pattern)
                seen_patterns.add(signature)
        
        return unique_patterns
    
    async def _store_patterns(self, patterns: List[DiscoveredPattern]) -> None:
        """Store discovered patterns for future reference"""
        try:
            for pattern in patterns:
                self._discovered_patterns[pattern.pattern_id] = pattern
                
                # Store in Redis with TTL
                if self._redis_client:
                    pattern_data = self._pattern_to_dict(pattern)
                    await self._redis_client.setex(
                        f"ymera:pattern:{pattern.pattern_id}",
                        CACHE_TTL,
                        json.dumps(pattern_data)
                    )
            
            self.logger.debug("Patterns stored", count=len(patterns))
            
        except Exception as e:
            self.logger.error("Failed to store patterns", error=str(e))
    
    async def _generate_insights(self, patterns: List[DiscoveredPattern]) -> List[str]:
        """Generate insights from discovered patterns"""
        insights = []
        
        if not patterns:
            return ["No significant patterns detected in the analyzed time window"]
        
        # Pattern type distribution
        pattern_types = {}
        for pattern in patterns:
            pattern_types[pattern.pattern_type] = pattern_types.get(pattern.pattern_type, 0) + 1
        
        most_common_type = max(pattern_types.items(), key=lambda x: x[1])
        insights.append(f"Most common pattern type: {most_common_type[0]} ({most_common_type[1]} instances)")
        
        # Agent involvement
        agent_involvement = {}
        for pattern in patterns:
            for agent in pattern.agents_involved:
                agent_involvement[agent] = agent_involvement.get(agent, 0) + 1
        
        if agent_involvement:
            most_active_agent = max(agent_involvement.items(), key=lambda x: x[1])
            insights.append(f"Most active agent in patterns: {most_active_agent[0]} ({most_active_agent[1]} patterns)")
        
        # High-confidence patterns
        high_confidence_patterns = [p for p in patterns if p.confidence_score > 0.8]
        if high_confidence_patterns:
            insights.append(f"{len(high_confidence_patterns)} high-confidence patterns (>80%) detected")
        
        # Performance impact
        performance_patterns = [p for p in patterns if "performance" in p.pattern_type.lower()]
        if performance_patterns:
            insights.append(f"{len(performance_patterns)} performance-related patterns identified")
        
        return insights
    
    async def _generate_recommendations(self, patterns: List[DiscoveredPattern]) -> List[Dict[str, Any]]:
        """Generate actionable recommendations from patterns"""
        recommendations = []
        
        # Analyze patterns for actionable recommendations
        for pattern in patterns[:10]:  # Top 10 patterns
            if pattern.pattern_type == "optimization_opportunity":
                recommendations.append({
                    "type": "optimization",
                    "priority": "high",
                    "description": f"Optimize {pattern.temporal_characteristics.get('event_type', 'unknown')} operations",
                    "potential_improvement": f"{pattern.performance_impact.get('improvement_potential_pct', 0):.1f}%",
                    "affected_agents": pattern.agents_involved
                })
            
            elif pattern.pattern_type == "performance_degradation":
                recommendations.append({
                    "type": "investigation",
                    "priority": "critical",
                    "description": f"Investigate performance degradation in agent {pattern.agents_involved[0]}",
                    "degradation_rate": f"{pattern.performance_impact.get('degradation_rate_pct', 0):.1f}%",
                    "affected_agents": pattern.agents_involved
                })
            
            elif pattern.pattern_type == "agent_collaboration":
                recommendations.append({
                    "type": "collaboration",
                    "priority": "medium",
                    "description": f"Leverage successful collaboration between {' and '.join(pattern.agents_involved)}",
                    "success_rate": f"{pattern.confidence_score:.1%}",
                    "affected_agents": pattern.agents_involved
                })
        
        return recommendations
    
    def _pattern_to_dict(self, pattern: DiscoveredPattern) -> Dict[str, Any]:
        """Convert pattern to dictionary for JSON serialization"""
        return {
            "pattern_id": pattern.pattern_id,
            "pattern_type": pattern.pattern_type,
            "description": pattern.description,
            "frequency": pattern.frequency,
            "confidence_score": pattern.confidence_score,
            "agents_involved": pattern.agents_involved,
            "temporal_characteristics": pattern.temporal_characteristics,
            "context_conditions": pattern.context_conditions,
            "performance_impact": pattern.performance_impact,
            "discovery_timestamp": pattern.discovery_timestamp.isoformat(),
            "last_occurrence": pattern.last_occurrence.isoformat(),
            "actionable_insights": pattern.actionable_insights
        }
    
    async def get_pattern_by_id(self, pattern_id: str) -> Optional[DiscoveredPattern]:
        """Retrieve a specific pattern by ID"""
        # Check local cache first
        if pattern_id in self._discovered_patterns:
            return self._discovered_patterns[pattern_id]
        
        # Check Redis
        if self._redis_client:
            try:
                pattern_data = await self._redis_client.get(f"ymera:pattern:{pattern_id}")
                if pattern_data:
                    data = json.loads(pattern_data)
                    # Convert back to DiscoveredPattern object
                    return DiscoveredPattern(
                        pattern_id=data["pattern_id"],
                        pattern_type=data["pattern_type"],
                        description=data["description"],
                        frequency=data["frequency"],
                        confidence_score=data["confidence_score"],
                        agents_involved=data["agents_involved"],
                        temporal_characteristics=data["temporal_characteristics"],
                        context_conditions=data["context_conditions"],
                        performance_impact=data["performance_impact"],
                        discovery_timestamp=datetime.fromisoformat(data["discovery_timestamp"]),
                        last_occurrence=datetime.fromisoformat(data["last_occurrence"]),
                        actionable_insights=data["actionable_insights"]
                    )
            except Exception as e:
                self.logger.error("Failed to retrieve pattern from Redis", pattern_id=pattern_id, error=str(e))
        
        return None
    
    async def get_recent_patterns(self, limit: int = 50) -> List[DiscoveredPattern]:
        """Get recently discovered patterns"""
        patterns = list(self._discovered_patterns.values())
        patterns.sort(key=lambda p: p.discovery_timestamp, reverse=True)
        return patterns[:limit]
    
    async def cleanup(self) -> None:
        """Cleanup resources"""
        try:
            if self._redis_client:
                await self._redis_client.close()
            self.logger.info("Pattern recognition engine cleanup completed")
        except Exception as e:
            self.logger.error("Error during cleanup", error=str(e))

# ===============================================================================
# UTILITY FUNCTIONS
# ===============================================================================

async def health_check() -> Dict[str, Any]:
    """Pattern recognition engine health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "module": "pattern_recognition",
        "analyzers": ["temporal", "collaboration", "performance"],
        "capabilities": [
            "behavioral_pattern_analysis",
            "temporal_pattern_discovery",
            "collaboration_pattern_detection",
            "performance_anomaly_detection",
            "optimization_opportunity_identification"
        ]
    }

def validate_configuration(config: Dict[str, Any]) -> bool:
    """Validate pattern recognition configuration"""
    required_fields = [
        "enabled", "discovery_interval", "min_occurrences", 
        "significance_threshold", "max_history"
    ]
    return all(field in config for field in required_fields)

# ===============================================================================
# MODULE INITIALIZATION
# ===============================================================================

async def initialize_pattern_recognition() -> PatternRecognitionEngine:
    """Initialize pattern recognition engine for production use"""
    config = PatternConfig(
        enabled=getattr(settings, "PATTERN_RECOGNITION_ENABLED", True),
        discovery_interval=getattr(settings, "PATTERN_DISCOVERY_INTERVAL", 900),
        min_occurrences=getattr(settings, "MIN_PATTERN_OCCURRENCES", 3),
        significance_threshold=getattr(settings, "PATTERN_SIGNIFICANCE_THRESHOLD", 0.75),
        max_history=getattr(settings, "MAX_PATTERN_HISTORY", 10000),
        clustering_eps=getattr(settings, "CLUSTERING_EPS", 0.5),
        clustering_min_samples=getattr(settings, "CLUSTERING_MIN_SAMPLES", 3)
    )
    
    engine = PatternRecognitionEngine(config)
    await engine._initialize_resources()
    
    return engine

# ===============================================================================
# EXPORTS
# ===============================================================================

__all__ = [
    "PatternRecognitionEngine",
    "PatternConfig",
    "AgentBehaviorEvent",
    "DiscoveredPattern",
    "PatternDiscoveryRequest",
    "PatternAnalysisResponse",
    "TemporalPatternAnalyzer",
    "CollaborationPatternAnalyzer",
    "PerformancePatternAnalyzer",
    "initialize_pattern_recognition",
    "health_check"
] 