"""
YMERA Enterprise API Dependencies - Agent System Dependencies
Production-ready agent system dependencies with orchestration, learning, and monitoring
"""

from fastapi import Depends, HTTPException, status, Request
from typing import Dict, List, Optional, Any, Union, Callable, Set
from datetime import datetime, timedelta
import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
import json
from contextlib import asynccontextmanager
import functools
import uuid

from ymera_agents.orchestrator import AgentOrchestrator
from ymera_agents.registry import AgentRegistry
from ymera_agents.lifecycle_manager import AgentLifecycleManager
from ymera_agents.communication.message_bus import MessageBus
from ymera_agents.learning.learning_engine import LearningEngine
from ymera_agents.learning.knowledge_base import KnowledgeBase
from ymera_agents.learning.feedback_processor import FeedbackProcessor
from ymera_services.ai.multi_llm_manager import MultiLLMManager
from ymera_services.vector_db.pinecone_manager import PineconeManager
from ymera_services.github.repository_analyzer import GitHubRepositoryAnalyzer
from ymera_core.exceptions import YMERAException
from .core import dependency_manager, with_dependency_monitoring
from .auth import SecurityContext, require_authenticated_user


class AgentStatus(Enum):
    """Agent operational status"""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    IDLE = "idle"
    BUSY = "busy"
    LEARNING = "learning"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    SHUTDOWN = "shutdown"


class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    URGENT = 5


class LearningMode(Enum):
    """Learning operation modes"""
    PASSIVE = "passive"        # Learn from observations
    ACTIVE = "active"          # Actively seek learning opportunities
    REINFORCEMENT = "reinforcement"  # Learn from feedback
    COLLABORATIVE = "collaborative"  # Learn from other agents
    SUPERVISED = "supervised"  # Learn from human guidance


@dataclass
class AgentCapabilities:
    """Agent capabilities and limitations"""
    max_concurrent_tasks: int = 5
    supported_file_types: List[str] = field(default_factory=list)
    required_permissions: Set[str] = field(default_factory=set)
    learning_modes: Set[LearningMode] = field(default_factory=set)
    collaboration_enabled: bool = True
    real_time_processing: bool = False
    batch_processing: bool = True
    resource_requirements: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentContext:
    """Complete agent execution context"""
    agent_id: str
    agent_name: str
    agent_type: str
    status: AgentStatus
    capabilities: AgentCapabilities
    current_tasks: List[str]
    performance_metrics: Dict[str, Any]
    learning_progress: Dict[str, Any]
    last_activity: datetime
    user_context: Optional[SecurityContext] = None
    session_id: Optional[str] = None
    resource_usage: Dict[str, float] = field(default_factory=dict)
    collaboration_score: float = 0.0


@dataclass
class TaskRequest:
    """Agent task request structure"""
    task_id: str
    task_type: str
    priority: TaskPriority
    parameters: Dict[str, Any]
    context: Dict[str, Any]
    requester_id: str
    deadline: Optional[datetime] = None
    dependencies: List[str] = field(default_factory=list)
    learning_enabled: bool = True
    collaboration_allowed: bool = True
    expected_duration: Optional[timedelta] = None


@dataclass
class LearningContext:
    """Learning operation context"""
    learning_session_id: str
    mode: LearningMode
    subject_domain: str
    learning_objectives: List[str]
    data_sources: List[str]
    feedback_channels: List[str]
    performance_baseline: Dict[str, Any]
    success_criteria: Dict[str, Any]
    time_budget: Optional[timedelta] = None
    human_supervision: bool = False


class AgentOrchestrationService:
    """Enhanced agent orchestration with intelligent task distribution"""
    
    def __init__(self):
        self.logger = logging.getLogger("ymera.agents.orchestration")
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.task_queue: Dict[TaskPriority, List[TaskRequest]] = {
            priority: [] for priority in TaskPriority
        }
        self.agent_workloads: Dict[str, int] = {}
        self.collaboration_graph: Dict[str, Set[str]] = {}
    
    @with_dependency_monitoring("agent_orchestrator")
    async def get_orchestrator(self) -> AgentOrchestrator:
        """Get agent orchestrator with health validation"""
        orchestrator = await dependency_manager.get_dependency('agent_orchestrator')
        
        # Validate orchestrator health
        if not await self._validate_orchestrator_health(orchestrator):
            raise YMERAException(
                message="Agent orchestrator is not healthy",
                error_code="ORCHESTRATOR_UNHEALTHY"
            )
        
        return orchestrator
    
    async def _validate_orchestrator_health(self, orchestrator: AgentOrchestrator) -> bool:
        """Validate orchestrator health and readiness"""
        try:
            # Check if orchestrator is running
            if not hasattr(orchestrator, 'is_running') or not orchestrator.is_running:
                return False
            
            # Check agent registry connectivity
            registry = await dependency_manager.get_dependency('agent_registry')
            if not registry:
                return False
            
            # Check message bus connectivity
            message_bus = await dependency_manager.get_dependency('message_bus')
            if not message_bus or not await message_bus.is_connected():
                return False
            
            # Verify at least some agents are available
            available_agents = await orchestrator.get_available_agents()
            if len(available_agents) == 0:
                self.logger.warning("No agents available for orchestration")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Orchestrator health check failed: {str(e)}")
            return False
    
    async def create_agent_session(self, user_context: SecurityContext, session_config: Dict[str, Any]) -> str:
        """Create a new agent interaction session"""
        try:
            session_id = str(uuid.uuid4())
            
            # Initialize session
            session_data = {
                'session_id': session_id,
                'user_id': user_context.user_id,
                'created_at': datetime.utcnow(),
                'last_activity': datetime.utcnow(),
                'active_tasks': [],
                'completed_tasks': [],
                'learning_sessions': [],
                'collaboration_requests': [],
                'performance_stats': {},
                'config': session_config
            }
            
            self.active_sessions[session_id] = session_data
            
            # Log session creation
            self.logger.info(
                "Agent session created",
                extra={
                    "session_id": session_id,
                    "user_id": user_context.user_id,
                    "config": session_config
                }
            )
            
            return session_id
            
        except Exception as e:
            self.logger.error(f"Failed to create agent session: {str(e)}")
            raise YMERAException(
                message="Failed to create agent session",
                error_code="SESSION_CREATION_ERROR"
            )
    
    async def get_optimal_agent_for_task(self, task_request: TaskRequest) -> Optional[str]:
        """Find optimal agent for task using AI-driven selection"""
        try:
            orchestrator = await self.get_orchestrator()
            registry = await dependency_manager.get_dependency('agent_registry')
            learning_engine = await dependency_manager.get_dependency('learning_engine')
            
            # Get available agents
            available_agents = await orchestrator.get_available_agents()
            if not available_agents:
                return None
            
            # Filter agents by capabilities
            capable_agents = []
            for agent_id in available_agents:
                agent_info = await registry.get_agent_info(agent_id)
                if await self._agent_can_handle_task(agent_info, task_request):
                    capable_agents.append(agent_id)
            
            if not capable_agents:
                return None
            
            # Use learning engine to select optimal agent
            selection_context = {
                'task_type': task_request.task_type,
                'task_priority': task_request.priority.value,
                'task_parameters': task_request.parameters,
                'available_agents': capable_agents,
                'current_workloads': {
                    agent_id: self.agent_workloads.get(agent_id, 0)
                    for agent_id in capable_agents
                }
            }
            
            # Get agent performance history for this task type
            performance_history = await learning_engine.get_agent_performance_history(
                task_type=task_request.task_type,
                agent_ids=capable_agents
            )
            
            # AI-driven agent selection
            optimal_agent = await self._select_agent_with_ai(
                selection_context, 
                performance_history
            )
            
            return optimal_agent
            
        except Exception as e:
            self.logger.error(f"Error selecting optimal agent: {str(e)}")
            # Fallback to simple load balancing
            return await self._fallback_agent_selection(task_request)
    
    async def _agent_can_handle_task(self, agent_info: Dict[str, Any], task_request: TaskRequest) -> bool:
        """Check if agent can handle the specific task"""
        try:
            capabilities = agent_info.get('capabilities', {})
            
            # Check max concurrent tasks
            current_tasks = agent_info.get('current_task_count', 0)
            max_tasks = capabilities.get('max_concurrent_tasks', 1)
            if current_tasks >= max_tasks:
                return False
            
            # Check supported task types
            supported_types = capabilities.get('supported_task_types', [])
            if task_request.task_type not in supported_types:
                return False
            
            # Check file type support if needed
            if 'file_types' in task_request.parameters:
                required_types = task_request.parameters['file_types']
                supported_file_types = capabilities.get('supported_file_types', [])
                if not all(ft in supported_file_types for ft in required_types):
                    return False
            
            # Check resource requirements
            resource_reqs = task_request.parameters.get('resource_requirements', {})
            agent_resources = capabilities.get('available_resources', {})
            
            for resource, amount in resource_reqs.items():
                if agent_resources.get(resource, 0) < amount:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking agent capabilities: {str(e)}")
            return False
    
    async def _select_agent_with_ai(self, selection_context: Dict[str, Any], performance_history: Dict[str, Any]) -> Optional[str]:
        """Use AI to select optimal agent based on context and history"""
        try:
            llm_manager = await dependency_manager.get_dependency('llm_manager')
            
            # Prepare AI prompt
            prompt = f"""
            Select the optimal agent for the following task:
            
            Task Type: {selection_context['task_type']}
            Priority: {selection_context['task_priority']}
            Parameters: {json.dumps(selection_context['task_parameters'], indent=2)}
            
            Available Agents: {selection_context['available_agents']}
            Current Workloads: {json.dumps(selection_context['current_workloads'], indent=2)}
            
            Performance History: {json.dumps(performance_history, indent=2)}
            
            Consider:
            1. Agent performance history for this task type
            2. Current workload distribution
            3. Task complexity and requirements
            4. Agent specialization and capabilities
            
            Return only the agent_id of the optimal agent.
            """
            
            response = await llm_manager.generate_completion(
                prompt=prompt,
                max_tokens=50,
                temperature=0.1  # Low temperature for consistent selection
            )
            
            selected_agent = response.strip()
            if selected_agent in selection_context['available_agents']:
                return selected_agent
            
            # Fallback if AI selection is invalid
            return await self._fallback_agent_selection_from_context(selection_context)
            
        except Exception as e:
            self.logger.error(f"AI agent selection failed: {str(e)}")
            return await self._fallback_agent_selection_from_context(selection_context)
    
    async def _fallback_agent_selection_from_context(self, context: Dict[str, Any]) -> Optional[str]:
        """Fallback agent selection using load balancing"""
        available_agents = context['available_agents']
        if not available_agents:
            return None
        
        # Select agent with lowest workload
        workloads = context['current_workloads']
        return min(available_agents, key=lambda agent: workloads.get(agent, 0))
    
    async def _fallback_agent_selection(self, task_request: TaskRequest) -> Optional[str]:
        """Simple fallback agent selection"""
        try:
            orchestrator = await self.get_orchestrator()
            available_agents = await orchestrator.get_available_agents()
            
            if not available_agents:
                return None
            
            # Simple round-robin selection based on workload
            return min(available_agents, key=lambda agent: self.agent_workloads.get(agent, 0))
            
        except Exception as e:
            self.logger.error(f"Fallback agent selection failed: {str(e)}")
            return None


class LearningService:
    """Agent learning and knowledge management service"""
    
    def __init__(self):
        self.logger = logging.getLogger("ymera.agents.learning")
        self.active_learning_sessions: Dict[str, LearningContext] = {}
        self.knowledge_cache: Dict[str, Any] = {}
    
    @with_dependency_monitoring("learning_engine")
    async def get_learning_engine(self) -> LearningEngine:
        """Get learning engine with validation"""
        learning_engine = await dependency_manager.get_dependency('learning_engine')
        
        if not learning_engine:
            raise YMERAException(
                message="Learning engine is not available",
                error_code="LEARNING_ENGINE_UNAVAILABLE"
            )
        
        return learning_engine
    
    @with_dependency_monitoring("knowledge_base")
    async def get_knowledge_base(self) -> KnowledgeBase:
        """Get knowledge base with validation"""
        knowledge_base = await dependency_manager.get_dependency('knowledge_base')
        
        if not knowledge_base:
            raise YMERAException(
                message="Knowledge base is not available",
                error_code="KNOWLEDGE_BASE_UNAVAILABLE"
            )
        
        return knowledge_base
    
    async def initiate_learning_session(self, learning_context: LearningContext, user_context: SecurityContext) -> str:
        """Start a new learning session for agents"""
        try:
            learning_engine = await self.get_learning_engine()
            
            # Validate learning context
            await self._validate_learning_context(learning_context)
            
            # Start learning session
            session_id = await learning_engine.start_learning_session(
                context=learning_context,
                user_id=user_context.user_id
            )
            
            # Track active session
            self.active_learning_sessions[session_id] = learning_context
            
            self.logger.info(
                "Learning session initiated",
                extra={
                    "session_id": session_id,
                    "mode": learning_context.mode.value,
                    "domain": learning_context.subject_domain,
                    "user_id": user_context.user_id
                }
            )
            
            return session_id
            
        except Exception as e:
            self.logger.error(f"Failed to initiate learning session: {str(e)}")
            raise YMERAException(
                message="Failed to start learning session",
                error_code="LEARNING_SESSION_ERROR"
            )
    
    async def _validate_learning_context(self, context: LearningContext) -> None:
        """Validate learning context parameters"""
        if not context.learning_objectives:
            raise YMERAException(
                message="Learning objectives are required",
                error_code="INVALID_LEARNING_CONTEXT"
            )
        
        if not context.data_sources:
            raise YMERAException(
                message="At least one data source is required",
                error_code="INVALID_LEARNING_CONTEXT"
            )
        
        # Validate time budget if specified
        if context.time_budget and context.time_budget <= timedelta(0):
            raise YMERAException(
                message="Time budget must be positive",
                error_code="INVALID_LEARNING_CONTEXT"
            )
    
    async def process_agent_feedback(self, agent_id: str, task_id: str, feedback_data: Dict[str, Any]) -> None:
        """Process feedback from agent task execution"""
        try:
            learning_engine = await self.get_learning_engine()
            feedback_processor = await dependency_manager.get_dependency('feedback_processor')
            
            # Process feedback through learning engine
            await learning_engine.process_feedback(
                agent_id=agent_id,
                task_id=task_id,
                feedback=feedback_data
            )
            
            # Update agent performance metrics
            await self._update_agent_performance(agent_id, task_id, feedback_data)
            
            self.logger.info(
                "Agent feedback processed",
                extra={
                    "agent_id": agent_id,
                    "task_id": task_id,
                    "feedback_type": feedback_data.get('type', 'unknown')
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to process agent feedback: {str(e)}")
            # Don't raise exception to avoid breaking task execution
    
    async def _update_agent_performance(self, agent_id: str, task_id: str, feedback_data: Dict[str, Any]) -> None:
        """Update agent performance metrics based on feedback"""
        try:
            registry = await dependency_manager.get_dependency('agent_registry')
            
            # Get current agent performance data
            agent_info = await registry.get_agent_info(agent_id)
            performance_metrics = agent_info.get('performance_metrics', {})
            
            # Update metrics based on feedback
            task_type = feedback_data.get('task_type', 'unknown')
            success = feedback_data.get('success', False)
            execution_time = feedback_data.get('execution_time', 0)
            
            # Initialize task type metrics if not exists
            if task_type not in performance_metrics:
                performance_metrics[task_type] = {
                    'total_tasks': 0,
                    'successful_tasks': 0,
                    'average_execution_time': 0,
                    'success_rate': 0.0
                }
            
            task_metrics = performance_metrics[task_type]
            
            # Update metrics
            task_metrics['total_tasks'] += 1
            if success:
                task_metrics['successful_tasks'] += 1
            
            # Update average execution time
            current_avg = task_metrics['average_execution_time']
            total_tasks = task_metrics['total_tasks']
            task_metrics['average_execution_time'] = (
                (current_avg * (total_tasks - 1) + execution_time) / total_tasks
            )
            
            # Update success rate
            task_metrics['success_rate'] = (
                task_metrics['successful_tasks'] / task_metrics['total_tasks']
            )
            
            # Save updated metrics
            await registry.update_agent_performance(agent_id, performance_metrics)
            
        except Exception as e:
            self.logger.error(f"Failed to update agent performance: {str(e)}")
    
    async def get_knowledge_for_task(self, task_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve relevant knowledge for task execution"""
        try:
            knowledge_base = await self.get_knowledge_base()
            
            # Create cache key
            cache_key = f"{task_type}:{hash(str(sorted(context.items())))}"
            
            # Check cache first
            if cache_key in self.knowledge_cache:
                return self.knowledge_cache[cache_key]
            
            # Query knowledge base
            knowledge = await knowledge_base.get_relevant_knowledge(
                task_type=task_type,
                context=context
            )
            
            # Cache the result
            self.knowledge_cache[cache_key] = knowledge
            
            return knowledge
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve knowledge: {str(e)}")
            return {}


class CollaborationService:
    """Agent collaboration and communication service"""
    
    def __init__(self):
        self.logger = logging.getLogger("ymera.agents.collaboration")
        self.active_collaborations: Dict[str, Dict[str, Any]] = {}
        self.collaboration_history: Dict[str, List[Dict[str, Any]]] = {}
    
    @with_dependency_monitoring("message_bus")
    async def get_message_bus(self) -> MessageBus:
        """Get message bus with validation"""
        message_bus = await dependency_manager.get_dependency('message_bus')
        
        if not message_bus or not await message_bus.is_connected():
            raise YMERAException(
                message="Message bus is not available",
                error_code="MESSAGE_BUS_UNAVAILABLE"
            )
        
        return message_bus
    
    async def request_agent_collaboration(
        self,
        requesting_agent: str,
        target_agents: List[str],
        collaboration_request: Dict[str, Any],
        user_context: SecurityContext
    ) -> str:
        """Request collaboration between agents"""
        try:
            collaboration_id = str(uuid.uuid4())
            message_bus = await self.get_message_bus()
            
            # Create collaboration session
            collaboration_data = {
                'collaboration_id': collaboration_id,
                'requesting_agent': requesting_agent,
                'target_agents': target_agents,
                'request': collaboration_request,
                'status': 'pending',
                'created_at': datetime.utcnow(),
                'user_id': user_context.user_id,
                'responses': {},
                'shared_context': {}
            }
            
            self.active_collaborations[collaboration_id] = collaboration_data
            
            # Send collaboration requests to target agents
            for target_agent in target_agents:
                await message_bus.send_message(
                    recipient=target_agent,
                    message={
                        'type': 'collaboration_request',
                        'collaboration_id': collaboration_id,
                        'requesting_agent': requesting_agent,
                        'request': collaboration_request
                    }
                )
            
            self.logger.info(
                "Collaboration request sent",
                extra={
                    "collaboration_id": collaboration_id,
                    "requesting_agent": requesting_agent,
                    "target_agents": target_agents
                }
            )
            
            return collaboration_id
            
        except Exception as e:
            self.logger.error(f"Failed to request collaboration: {str(e)}")
            raise YMERAException(
                message="Failed to initiate collaboration",
                error_code="COLLABORATION_REQUEST_ERROR"
            )
    
    async def handle_collaboration_response(
        self,
        collaboration_id: str,
        responding_agent: str,
        response: Dict[str, Any]
    ) -> None:
        """Handle response from agent to collaboration request"""
        try:
            if collaboration_id not in self.active_collaborations:
                raise YMERAException(
                    message="Collaboration not found",
                    error_code="COLLABORATION_NOT_FOUND"
                )
            
            collaboration = self.active_collaborations[collaboration_id]
            collaboration['responses'][responding_agent] = {
                'response': response,
                'timestamp': datetime.utcnow()
            }
            
            # Check if all agents have responded
            target_agents = set(collaboration['target_agents'])
            responding_agents = set(collaboration['responses'].keys())
            
            if target_agents.issubset(responding_agents):
                collaboration['status'] = 'active'
                await self._initiate_active_collaboration(collaboration_id)
            
            self.logger.info(
                "Collaboration response received",
                extra={
                    "collaboration_id": collaboration_id,
                    "responding_agent": responding_agent,
                    "status": collaboration['status']
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to handle collaboration response: {str(e)}")
    
    async def _initiate_active_collaboration(self, collaboration_id: str) -> None:
        """Start active collaboration between agents"""
        try:
            collaboration = self.active_collaborations[collaboration_id]
            message_bus = await self.get_message_bus()
            
            # Create shared workspace
            shared_context = {
                'collaboration_id': collaboration_id,
                'participants': [collaboration['requesting_agent']] + collaboration['target_agents'],
                'shared_resources': {},
                'communication_log': [],
                'task_distribution': {}
            }
            
            collaboration['shared_context'] = shared_context
            
            # Notify all participants that collaboration is active
            all_agents = [collaboration['requesting_agent']] + collaboration['target_agents']
            
            for agent_id in all_agents:
                await message_bus.send_message(
                    recipient=agent_id,
                    message={
                        'type': 'collaboration_active',
                        'collaboration_id': collaboration_id,
                        'shared_context': shared_context
                    }
                )
            
            self.logger.info(
                "Active collaboration initiated",
                extra={"collaboration_id": collaboration_id}
            )
            
        except Exception as e:
            self.logger.error(f"Failed to initiate active collaboration: {str(e)}")


# Dependency injection functions
orchestration_service = AgentOrchestrationService()
learning_service = LearningService()
collaboration_service = CollaborationService()


async def get_agent_orchestrator(user_context: SecurityContext = Depends(require_authenticated_user)) -> AgentOrchestrator:
    """Get authenticated agent orchestrator"""
    return await orchestration_service.get_orchestrator()


async def get_learning_engine(user_context: SecurityContext = Depends(require_authenticated_user)) -> LearningEngine:
    """Get authenticated learning engine"""
    return await learning_service.get_learning_engine()


async def get_knowledge_base(user_context: SecurityContext = Depends(require_authenticated_user)) -> KnowledgeBase:
    """Get authenticated knowledge base"""
    return await learning_service.get_knowledge_base()


async def get_message_bus(user_context: SecurityContext = Depends(require_authenticated_user)) -> MessageBus:
    """Get authenticated message bus"""
    return await collaboration_service.get_message_bus()


async def get_agent_registry() -> AgentRegistry:
    """Get agent registry"""
    return await dependency_manager.get_dependency('agent_registry')


async def get_lifecycle_manager() -> AgentLifecycleManager:
    """Get agent lifecycle manager"""
    return await dependency_manager.get_dependency('agent_lifecycle_manager')


async def create_agent_context(
    agent_id: str,
    user_context: SecurityContext = Depends(require_authenticated_user),
    registry: AgentRegistry = Depends(get_agent_registry)
) -> AgentContext:
    """Create complete agent context for request"""
    try:
        # Get agent information
        agent_info = await registry.get_agent_info(agent_id)
        if not agent_info:
            raise YMERAException(
                message=f"Agent {agent_id} not found",
                error_code="AGENT_NOT_FOUND"
            )
        
        # Create agent capabilities
        capabilities_data = agent_info.get('capabilities', {})
        capabilities = AgentCapabilities(
            max_concurrent_tasks=capabilities_data.get('max_concurrent_tasks', 5),
            supported_file_types=capabilities_data.get('supported_file_types', []),
            required_permissions=set(capabilities_data.get('required_permissions', [])),
            learning_modes=set(LearningMode(mode) for mode in capabilities_data.get('learning_modes', ['passive'])),
            collaboration_enabled=capabilities_data.get('collaboration_enabled', True),
            real_time_processing=capabilities_data.get('real_time_processing', False),
            batch_processing=capabilities_data.get('batch_processing', True),
            resource_requirements=capabilities_data.get('resource_requirements', {})
        )
        
        # Create agent context
        agent_context = AgentContext(
            agent_id=agent_id,
            agent_name=agent_info['name'],
            agent_type=agent_info['type'],
            status=AgentStatus(agent_info.get('status', 'idle')),
            capabilities=capabilities,
            current_tasks=agent_info.get('current_tasks', []),
            performance_metrics=agent_info.get('performance_metrics', {}),
            learning_progress=agent_info.get('learning_progress', {}),
            last_activity=datetime.fromisoformat(agent_info.get('last_activity', datetime.utcnow().isoformat())),
            user_context=user_context,
            resource_usage=agent_info.get('resource_usage', {}),
            collaboration_score=agent_info.get('collaboration_score', 0.0)
        )
        
        return agent_context
        
    except Exception as e:
        logging.getLogger("ymera.agents").error(f"Failed to create agent context: {str(e)}")
        raise YMERAException(
            message="Failed to create agent context",
            error_code="AGENT_CONTEXT_ERROR"
        )


async def create_task_request(
    task_data: Dict[str, Any],
    user_context: SecurityContext = Depends(require_authenticated_user)
) -> TaskRequest:
    """Create validated task request"""
    try:
        # Validate required fields
        required_fields = ['task_type', 'parameters']
        for field in required_fields:
            if field not in task_data:
                raise YMERAException(
                    message=f"Missing required field: {field}",
                    error_code="INVALID_TASK_REQUEST"
                )
        
        # Create task request
        task_request = TaskRequest(
            task_id=task_data.get('task_id', str(uuid.uuid4())),
            task_type=task_data['task_type'],
            priority=TaskPriority(task_data.get('priority', TaskPriority.NORMAL.value)),
            parameters=task_data['parameters'],
            context=task_data.get('context', {}),
            requester_id=user_context.user_id,
            deadline=datetime.fromisoformat(task_data['deadline']) if task_data.get('deadline') else None,
            dependencies=task_data.get('dependencies', []),
            learning_enabled=task_data.get('learning_enabled', True),
            collaboration_allowed=task_data.get('collaboration_allowed', True),
            expected_duration=timedelta(seconds=task_data['expected_duration']) if task_data.get('expected_duration') else None
        )
        
        return task_request
        
    except Exception as e:
        logging.getLogger("ymera.agents").error(f"Failed to create task request: {str(e)}")
        raise YMERAException(
            message="Failed to create task request",
            error_code="TASK_REQUEST_ERROR"
        )


async def create_learning_context(
    learning_data: Dict[str, Any],
    user_context: SecurityContext = Depends(require_authenticated_user)
) -> LearningContext:
    """Create validated learning context"""
    try:
        # Validate required fields
        required_fields = ['subject_domain', 'learning_objectives', 'data_sources']
        for field in required_fields:
            if field not in learning_data:
                raise YMERAException(
                    message=f"Missing required field: {field}",
                    error_code="INVALID_LEARNING_CONTEXT"
                )
        
        # Create learning context
        learning_context = LearningContext(
            learning_session_id=learning_data.get('session_id', str(uuid.uuid4())),
            mode=LearningMode(learning_data.get('mode', LearningMode.PASSIVE.value)),
            subject_domain=learning_data['subject_domain'],
            learning_objectives=learning_data['learning_objectives'],
            data_sources=learning_data['data_sources'],
            feedback_channels=learning_data.get('feedback_channels', []),
            performance_baseline=learning_data.get('performance_baseline', {}),
            success_criteria=learning_data.get('success_criteria', {}),
            time_budget=timedelta(seconds=learning_data['time_budget']) if learning_data.get('time_budget') else None,
            human_supervision=learning_data.get('human_supervision', False)
        )
        
        return learning_context
        
    except Exception as e:
        logging.getLogger("ymera.agents").error(f"Failed to create learning context: {str(e)}")
        raise YMERAException(
            message="Failed to create learning context",
            error_code="LEARNING_CONTEXT_ERROR"
        )


# Enhanced dependency providers with monitoring
@asynccontextmanager
async def agent_session_context(user_context: SecurityContext, session_config: Dict[str, Any]):
    """Context manager for agent sessions with proper cleanup"""
    session_id = None
    try:
        session_id = await orchestration_service.create_agent_session(user_context, session_config)
        yield session_id
    except Exception as e:
        logging.getLogger("ymera.agents").error(f"Agent session error: {str(e)}")
        raise
    finally:
        if session_id and session_id in orchestration_service.active_sessions:
            # Cleanup session
            session_data = orchestration_service.active_sessions[session_id]
            session_data['ended_at'] = datetime.utcnow()
            
            # Archive session data if needed
            if session_data.get('config', {}).get('archive', False):
                await _archive_agent_session(session_id, session_data)
            
            # Remove from active sessions
            del orchestration_service.active_sessions[session_id]


async def _archive_agent_session(session_id: str, session_data: Dict[str, Any]) -> None:
    """Archive completed agent session data"""
    try:
        # This would typically save to a persistent storage
        # For now, just log the session completion
        logger = logging.getLogger("ymera.agents.sessions")
        
        session_summary = {
            'session_id': session_id,
            'user_id': session_data['user_id'],
            'duration': (session_data.get('ended_at', datetime.utcnow()) - session_data['created_at']).total_seconds(),
            'total_tasks': len(session_data['completed_tasks']),
            'learning_sessions': len(session_data['learning_sessions']),
            'collaboration_requests': len(session_data['collaboration_requests'])
        }
        
        logger.info(
            "Agent session archived",
            extra=session_summary
        )
        
    except Exception as e:
        logging.getLogger("ymera.agents").error(f"Failed to archive session: {str(e)}")


# Advanced agent monitoring and health checks
class AgentHealthMonitor:
    """Monitor agent health and performance"""
    
    def __init__(self):
        self.logger = logging.getLogger("ymera.agents.health")
        self.health_checks: Dict[str, Dict[str, Any]] = {}
        self.performance_thresholds = {
            'response_time': 5.0,  # seconds
            'success_rate': 0.8,   # 80%
            'error_rate': 0.1,     # 10%
            'memory_usage': 0.8    # 80% of allocated
        }
    
    async def check_agent_health(self, agent_id: str) -> Dict[str, Any]:
        """Comprehensive agent health check"""
        try:
            registry = await dependency_manager.get_dependency('agent_registry')
            agent_info = await registry.get_agent_info(agent_id)
            
            if not agent_info:
                return {
                    'agent_id': agent_id,
                    'healthy': False,
                    'status': 'not_found',
                    'issues': ['Agent not found in registry']
                }
            
            health_status = {
                'agent_id': agent_id,
                'healthy': True,
                'status': agent_info.get('status', 'unknown'),
                'issues': [],
                'metrics': {},
                'last_check': datetime.utcnow()
            }
            
            # Check basic connectivity
            if not await self._check_agent_connectivity(agent_id):
                health_status['healthy'] = False
                health_status['issues'].append('Agent not responding')
            
            # Check performance metrics
            performance_issues = await self._check_performance_metrics(agent_info)
            if performance_issues:
                health_status['healthy'] = False
                health_status['issues'].extend(performance_issues)
            
            # Check resource usage
            resource_issues = await self._check_resource_usage(agent_info)
            if resource_issues:
                health_status['healthy'] = False
                health_status['issues'].extend(resource_issues)
            
            # Store health check result
            self.health_checks[agent_id] = health_status
            
            return health_status
            
        except Exception as e:
            self.logger.error(f"Health check failed for agent {agent_id}: {str(e)}")
            return {
                'agent_id': agent_id,
                'healthy': False,
                'status': 'error',
                'issues': [f'Health check failed: {str(e)}']
            }
    
    async def _check_agent_connectivity(self, agent_id: str) -> bool:
        """Check if agent is responsive"""
        try:
            message_bus = await dependency_manager.get_dependency('message_bus')
            
            # Send ping message with timeout
            response = await asyncio.wait_for(
                message_bus.send_message(
                    recipient=agent_id,
                    message={'type': 'ping', 'timestamp': datetime.utcnow().isoformat()}
                ),
                timeout=5.0
            )
            
            return response is not None
            
        except asyncio.TimeoutError:
            return False
        except Exception as e:
            self.logger.error(f"Connectivity check failed for {agent_id}: {str(e)}")
            return False
    
    async def _check_performance_metrics(self, agent_info: Dict[str, Any]) -> List[str]:
        """Check agent performance against thresholds"""
        issues = []
        performance_metrics = agent_info.get('performance_metrics', {})
        
        # Check average response time
        avg_response_time = performance_metrics.get('average_response_time', 0)
        if avg_response_time > self.performance_thresholds['response_time']:
            issues.append(f'High response time: {avg_response_time}s')
        
        # Check success rate
        success_rate = performance_metrics.get('success_rate', 1.0)
        if success_rate < self.performance_thresholds['success_rate']:
            issues.append(f'Low success rate: {success_rate:.2%}')
        
        # Check error rate
        error_rate = performance_metrics.get('error_rate', 0.0)
        if error_rate > self.performance_thresholds['error_rate']:
            issues.append(f'High error rate: {error_rate:.2%}')
        
        return issues
    
    async def _check_resource_usage(self, agent_info: Dict[str, Any]) -> List[str]:
        """Check agent resource usage"""
        issues = []
        resource_usage = agent_info.get('resource_usage', {})
        
        # Check memory usage
        memory_usage = resource_usage.get('memory_usage', 0.0)
        if memory_usage > self.performance_thresholds['memory_usage']:
            issues.append(f'High memory usage: {memory_usage:.1%}')
        
        # Check CPU usage
        cpu_usage = resource_usage.get('cpu_usage', 0.0)
        if cpu_usage > 0.9:  # 90% CPU threshold
            issues.append(f'High CPU usage: {cpu_usage:.1%}')
        
        return issues


# Global health monitor instance
health_monitor = AgentHealthMonitor()


async def get_agent_health_monitor() -> AgentHealthMonitor:
    """Get agent health monitor"""
    return health_monitor


# Utility functions for agent management
async def get_agent_performance_summary(
    agent_id: str,
    time_period: Optional[timedelta] = None
) -> Dict[str, Any]:
    """Get comprehensive agent performance summary"""
    try:
        registry = await dependency_manager.get_dependency('agent_registry')
        learning_engine = await dependency_manager.get_dependency('learning_engine')
        
        agent_info = await registry.get_agent_info(agent_id)
        if not agent_info:
            raise YMERAException(
                message=f"Agent {agent_id} not found",
                error_code="AGENT_NOT_FOUND"
            )
        
        # Get performance metrics
        performance_metrics = agent_info.get('performance_metrics', {})
        
        # Get learning progress
        learning_progress = await learning_engine.get_agent_learning_progress(
            agent_id=agent_id,
            time_period=time_period
        )
        
        # Get recent health status
        health_status = await health_monitor.check_agent_health(agent_id)
        
        summary = {
            'agent_id': agent_id,
            'agent_name': agent_info['name'],
            'agent_type': agent_info['type'],
            'current_status': agent_info.get('status', 'unknown'),
            'performance_metrics': performance_metrics,
            'learning_progress': learning_progress,
            'health_status': health_status,
            'capabilities': agent_info.get('capabilities', {}),
            'current_workload': len(agent_info.get('current_tasks', [])),
            'last_activity': agent_info.get('last_activity'),
            'collaboration_score': agent_info.get('collaboration_score', 0.0),
            'summary_generated_at': datetime.utcnow()
        }
        
        return summary
        
    except Exception as e:
        logging.getLogger("ymera.agents").error(f"Failed to get performance summary: {str(e)}")
        raise YMERAException(
            message="Failed to get agent performance summary",
            error_code="PERFORMANCE_SUMMARY_ERROR"
        )


async def optimize_agent_workload_distribution() -> Dict[str, Any]:
    """Optimize workload distribution across all agents"""
    try:
        orchestrator = await orchestration_service.get_orchestrator()
        registry = await dependency_manager.get_dependency('agent_registry')
        
        # Get all active agents
        active_agents = await orchestrator.get_available_agents()
        
        # Analyze current workload distribution
        workload_analysis = {}
        total_tasks = 0
        
        for agent_id in active_agents:
            agent_info = await registry.get_agent_info(agent_id)
            current_tasks = len(agent_info.get('current_tasks', []))
            max_tasks = agent_info.get('capabilities', {}).get('max_concurrent_tasks', 5)
            
            workload_analysis[agent_id] = {
                'current_tasks': current_tasks,
                'max_tasks': max_tasks,
                'utilization': current_tasks / max_tasks if max_tasks > 0 else 0,
                'performance_score': agent_info.get('performance_metrics', {}).get('success_rate', 0.8)
            }
            
            total_tasks += current_tasks
        
        # Calculate optimization recommendations
        recommendations = []
        
        # Find overloaded and underloaded agents
        overloaded_agents = [
            agent_id for agent_id, data in workload_analysis.items()
            if data['utilization'] > 0.8
        ]
        
        underloaded_agents = [
            agent_id for agent_id, data in workload_analysis.items()
            if data['utilization'] < 0.3
        ]
        
        if overloaded_agents and underloaded_agents:
            recommendations.append({
                'type': 'rebalance_workload',
                'overloaded_agents': overloaded_agents,
                'underloaded_agents': underloaded_agents,
                'priority': 'high'
            })
        
        # Check for agents with poor performance
        poor_performers = [
            agent_id for agent_id, data in workload_analysis.items()
            if data['performance_score'] < 0.6
        ]
        
        if poor_performers:
            recommendations.append({
                'type': 'performance_review',
                'agents': poor_performers,
                'priority': 'medium'
            })
        
        optimization_result = {
            'analysis_timestamp': datetime.utcnow(),
            'total_active_agents': len(active_agents),
            'total_active_tasks': total_tasks,
            'average_utilization': sum(data['utilization'] for data in workload_analysis.values()) / len(workload_analysis) if workload_analysis else 0,
            'workload_distribution': workload_analysis,
            'recommendations': recommendations,
            'optimization_score': await _calculate_optimization_score(workload_analysis)
        }
        
        return optimization_result
        
    except Exception as e:
        logging.getLogger("ymera.agents").error(f"Workload optimization failed: {str(e)}")
        raise YMERAException(
            message="Failed to optimize agent workload distribution",
            error_code="WORKLOAD_OPTIMIZATION_ERROR"
        )


async def _calculate_optimization_score(workload_analysis: Dict[str, Dict[str, Any]]) -> float:
    """Calculate overall system optimization score"""
    if not workload_analysis:
        return 0.0
    
    utilizations = [data['utilization'] for data in workload_analysis.values()]
    performance_scores = [data['performance_score'] for data in workload_analysis.values()]
    
    # Calculate variance in utilization (lower is better)
    avg_utilization = sum(utilizations) / len(utilizations)
    utilization_variance = sum((u - avg_utilization) ** 2 for u in utilizations) / len(utilizations)
    
    # Calculate average performance
    avg_performance = sum(performance_scores) / len(performance_scores)
    
    # Optimization score (0-1, higher is better)
    utilization_score = max(0, 1 - utilization_variance * 2)  # Penalize high variance
    performance_score = avg_performance
    
    return (utilization_score * 0.4 + performance_score * 0.6)


# Export all dependency functions and services
__all__ = [
    # Enums
    'AgentStatus',
    'TaskPriority', 
    'LearningMode',
    
    # Data classes
    'AgentCapabilities',
    'AgentContext',
    'TaskRequest',
    'LearningContext',
    
    # Services
    'AgentOrchestrationService',
    'LearningService',
    'CollaborationService',
    'AgentHealthMonitor',
    
    # Service instances
    'orchestration_service',
    'learning_service',
    'collaboration_service',
    'health_monitor',
    
    # Dependency functions
    'get_agent_orchestrator',
    'get_learning_engine',
    'get_knowledge_base',
    'get_message_bus',
    'get_agent_registry',
    'get_lifecycle_manager',
    'get_agent_health_monitor',
    
    # Context creators
    'create_agent_context',
    'create_task_request',
    'create_learning_context',
    'agent_session_context',
    
    # Utility functions
    'get_agent_performance_summary',
    'optimize_agent_workload_distribution'
]