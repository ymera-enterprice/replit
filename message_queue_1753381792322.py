"""
YMERA Enterprise Redis Message Queue
Production-ready message queue implementation for multi-agent systems
with learning engine support and enterprise-grade reliability
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Union, Set
from dataclasses import dataclass, asdict
from enum import Enum
from contextlib import asynccontextmanager
import traceback

import redis.asyncio as redis
from redis.asyncio.retry import Retry
from redis.asyncio.backoff import ExponentialBackoff
from redis.exceptions import (
    ConnectionError, TimeoutError, RedisError,
    BusyLoadingError, ReadOnlyError
)


class MessagePriority(Enum):
    """Message priority levels for queue processing"""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


class MessageType(Enum):
    """Types of messages in the system"""
    AGENT_TASK = "agent_task"
    AGENT_RESPONSE = "agent_response"
    ORCHESTRATION = "orchestration"
    LEARNING_UPDATE = "learning_update"
    LEARNING_FEEDBACK = "learning_feedback"
    SYSTEM_EVENT = "system_event"
    HEALTH_CHECK = "health_check"
    MONITORING = "monitoring"
    ERROR_REPORT = "error_report"
    WORKFLOW_EVENT = "workflow_event"


class MessageStatus(Enum):
    """Message processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRY = "retry"
    DEAD_LETTER = "dead_letter"


@dataclass
class QueueMessage:
    """Enhanced message structure for the queue system"""
    id: str
    type: MessageType
    priority: MessagePriority
    payload: Dict[str, Any]
    source_agent: Optional[str] = None
    target_agent: Optional[str] = None
    workflow_id: Optional[str] = None
    correlation_id: Optional[str] = None
    created_at: datetime = None
    expires_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    status: MessageStatus = MessageStatus.PENDING
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization"""
        data = asdict(self)
        data['type'] = self.type.value
        data['priority'] = self.priority.value
        data['status'] = self.status.value
        data['created_at'] = self.created_at.isoformat()
        if self.expires_at:
            data['expires_at'] = self.expires_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QueueMessage':
        """Create message from dictionary"""
        data['type'] = MessageType(data['type'])
        data['priority'] = MessagePriority(data['priority'])
        data['status'] = MessageStatus(data['status'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        if data.get('expires_at'):
            data['expires_at'] = datetime.fromisoformat(data['expires_at'])
        return cls(**data)


class QueueMetrics:
    """Queue performance and health metrics"""
    def __init__(self):
        self.messages_sent = 0
        self.messages_received = 0
        self.messages_processed = 0
        self.messages_failed = 0
        self.processing_times = []
        self.queue_sizes = {}
        self.connection_errors = 0
        self.last_health_check = None
        self.start_time = datetime.utcnow()
    
    def record_message_sent(self):
        self.messages_sent += 1
    
    def record_message_received(self):
        self.messages_received += 1
    
    def record_message_processed(self, processing_time: float):
        self.messages_processed += 1
        self.processing_times.append(processing_time)
        # Keep only last 1000 processing times
        if len(self.processing_times) > 1000:
            self.processing_times = self.processing_times[-1000:]
    
    def record_message_failed(self):
        self.messages_failed += 1
    
    def record_connection_error(self):
        self.connection_errors += 1
    
    def get_average_processing_time(self) -> float:
        if not self.processing_times:
            return 0.0
        return sum(self.processing_times) / len(self.processing_times)
    
    def get_throughput(self) -> float:
        """Messages per second"""
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        if uptime == 0:
            return 0.0
        return self.messages_processed / uptime
    
    def get_error_rate(self) -> float:
        """Error rate percentage"""
        total = self.messages_processed + self.messages_failed
        if total == 0:
            return 0.0
        return (self.messages_failed / total) * 100


class RedisMessageQueue:
    """Enterprise-grade Redis message queue for multi-agent systems"""
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        queue_prefix: str = "ymera",
        max_connections: int = 20,
        retry_on_timeout: bool = True,
        health_check_interval: int = 30,
        message_ttl: int = 3600,
        dead_letter_ttl: int = 86400,
        max_queue_size: int = 10000,
        batch_size: int = 100,
        enable_metrics: bool = True
    ):
        self.redis_url = redis_url
        self.queue_prefix = queue_prefix
        self.max_connections = max_connections
        self.retry_on_timeout = retry_on_timeout
        self.health_check_interval = health_check_interval
        self.message_ttl = message_ttl
        self.dead_letter_ttl = dead_letter_ttl
        self.max_queue_size = max_queue_size
        self.batch_size = batch_size
        self.enable_metrics = enable_metrics
        
        # Redis connections
        self.redis_client: Optional[redis.Redis] = None
        self.subscriber_client: Optional[redis.Redis] = None
        
        # Queue management
        self.queues: Dict[str, str] = {}  # logical_name -> redis_key
        self.subscribers: Dict[str, Set[Callable]] = {}  # queue -> callbacks
        self.processing_tasks: Set[asyncio.Task] = set()
        
        # Learning engine integration
        self.learning_queues = {
            'feedback': f"{queue_prefix}:learning:feedback",
            'updates': f"{queue_prefix}:learning:updates",
            'insights': f"{queue_prefix}:learning:insights",
            'training_data': f"{queue_prefix}:learning:training_data"
        }
        
        # Agent communication queues
        self.agent_queues = {
            'tasks': f"{queue_prefix}:agents:tasks",
            'responses': f"{queue_prefix}:agents:responses",
            'orchestration': f"{queue_prefix}:agents:orchestration",
            'health': f"{queue_prefix}:agents:health"
        }
        
        # System queues
        self.system_queues = {
            'events': f"{queue_prefix}:system:events",
            'monitoring': f"{queue_prefix}:system:monitoring",
            'errors': f"{queue_prefix}:system:errors",
            'workflows': f"{queue_prefix}:system:workflows"
        }
        
        # Metrics and monitoring
        self.metrics = QueueMetrics() if enable_metrics else None
        self.logger = logging.getLogger(self.__class__.__name__)
        self.is_healthy = False
        self.health_check_task: Optional[asyncio.Task] = None
        
        # Message processing
        self._shutdown_event = asyncio.Event()
        self._processing_semaphore = asyncio.Semaphore(max_connections)
    
    async def initialize(self) -> None:
        """Initialize Redis connections and queue system"""
        try:
            self.logger.info("Initializing Redis message queue system...")
            
            # Connection configuration with retry logic
            retry_policy = Retry(
                ExponentialBackoff(),
                retries=3
            )
            
            # Main Redis client
            self.redis_client = redis.from_url(
                self.redis_url,
                max_connections=self.max_connections,
                retry_on_timeout=self.retry_on_timeout,
                retry=retry_policy,
                decode_responses=True
            )
            
            # Separate client for pub/sub
            self.subscriber_client = redis.from_url(
                self.redis_url,
                max_connections=10,
                retry_on_timeout=self.retry_on_timeout,
                retry=retry_policy,
                decode_responses=True
            )
            
            # Test connections
            await self.redis_client.ping()
            await self.subscriber_client.ping()
            
            # Initialize queue structures
            await self._initialize_queues()
            
            # Start health monitoring
            if self.health_check_interval > 0:
                self.health_check_task = asyncio.create_task(
                    self._health_check_loop()
                )
            
            self.is_healthy = True
            self.logger.info("Redis message queue system initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Redis message queue: {str(e)}")
            if self.metrics:
                self.metrics.record_connection_error()
            raise
    
    async def _initialize_queues(self) -> None:
        """Initialize all queue structures in Redis"""
        all_queues = {
            **self.learning_queues,
            **self.agent_queues,
            **self.system_queues
        }
        
        # Create queue structures and set up priority queues
        for queue_name, redis_key in all_queues.items():
            # Priority queues (sorted sets for prioritized processing)
            priority_key = f"{redis_key}:priority"
            
            # Processing queue (list for FIFO processing within priority)
            processing_key = f"{redis_key}:processing"
            
            # Dead letter queue
            dead_letter_key = f"{redis_key}:dead_letter"
            
            # Initialize empty structures if they don't exist
            await self.redis_client.setnx(f"{redis_key}:initialized", "true")
            
            self.queues[queue_name] = redis_key
            self.subscribers[queue_name] = set()
        
        self.logger.info(f"Initialized {len(all_queues)} queue structures")
    
    async def _health_check_loop(self) -> None:
        """Continuous health monitoring"""
        while not self._shutdown_event.is_set():
            try:
                start_time = time.time()
                
                # Test Redis connectivity
                await self.redis_client.ping()
                await self.subscriber_client.ping()
                
                # Update queue size metrics
                if self.metrics:
                    for queue_name, redis_key in self.queues.items():
                        size = await self._get_queue_size(redis_key)
                        self.metrics.queue_sizes[queue_name] = size
                    
                    self.metrics.last_health_check = datetime.utcnow()
                
                self.is_healthy = True
                
                # Check for queue overflows
                await self._check_queue_limits()
                
                health_check_time = time.time() - start_time
                self.logger.debug(f"Health check completed in {health_check_time:.3f}s")
                
            except Exception as e:
                self.is_healthy = False
                self.logger.error(f"Health check failed: {str(e)}")
                if self.metrics:
                    self.metrics.record_connection_error()
            
            # Wait for next check or shutdown
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self.health_check_interval
                )
                break
            except asyncio.TimeoutError:
                continue
    
    async def _check_queue_limits(self) -> None:
        """Monitor and manage queue size limits"""
        for queue_name, redis_key in self.queues.items():
            size = await self._get_queue_size(redis_key)
            
            if size > self.max_queue_size:
                self.logger.warning(
                    f"Queue {queue_name} size ({size}) exceeds limit ({self.max_queue_size})"
                )
                
                # Move oldest messages to dead letter queue
                excess_count = size - self.max_queue_size
                await self._move_to_dead_letter(redis_key, excess_count)
    
    async def _get_queue_size(self, redis_key: str) -> int:
        """Get total queue size across all priority levels"""
        priority_size = await self.redis_client.zcard(f"{redis_key}:priority")
        processing_size = await self.redis_client.llen(f"{redis_key}:processing")
        return priority_size + processing_size
    
    async def send_message(
        self,
        queue_name: str,
        message: QueueMessage,
        delay_seconds: int = 0
    ) -> str:
        """Send message to specified queue with priority handling"""
        try:
            if not self.is_healthy:
                raise ConnectionError("Redis message queue is not healthy")
            
            redis_key = self.queues.get(queue_name)
            if not redis_key:
                raise ValueError(f"Unknown queue: {queue_name}")
            
            # Set message TTL if not set
            if message.expires_at is None:
                message.expires_at = datetime.utcnow() + timedelta(seconds=self.message_ttl)
            
            # Serialize message
            message_data = json.dumps(message.to_dict())
            
            if delay_seconds > 0:
                # Delayed message using sorted set with timestamp
                score = time.time() + delay_seconds
                delayed_key = f"{redis_key}:delayed"
                await self.redis_client.zadd(delayed_key, {message_data: score})
            else:
                # Immediate message to priority queue
                priority_key = f"{redis_key}:priority"
                score = message.priority.value + (time.time() / 1000000)  # Priority + timestamp for FIFO within priority
                await self.redis_client.zadd(priority_key, {message_data: score})
            
            if self.metrics:
                self.metrics.record_message_sent()
            
            self.logger.debug(
                f"Message {message.id} sent to queue {queue_name} "
                f"(priority: {message.priority.name}, delay: {delay_seconds}s)"
            )
            
            return message.id
            
        except Exception as e:
            self.logger.error(f"Failed to send message to {queue_name}: {str(e)}")
            if self.metrics:
                self.metrics.record_connection_error()
            raise
    
    async def receive_message(
        self,
        queue_name: str,
        timeout: int = 5,
        auto_ack: bool = True
    ) -> Optional[QueueMessage]:
        """Receive message from queue with automatic priority handling"""
        try:
            redis_key = self.queues.get(queue_name)
            if not redis_key:
                raise ValueError(f"Unknown queue: {queue_name}")
            
            # Process delayed messages first
            await self._process_delayed_messages(redis_key)
            
            # Get highest priority message
            priority_key = f"{redis_key}:priority"
            
            # Use blocking pop with timeout for efficient waiting
            result = await self.redis_client.bzpopmin(priority_key, timeout)
            
            if result is None:
                return None
            
            _, message_data, _ = result
            message = QueueMessage.from_dict(json.loads(message_data))
            
            # Check message expiration
            if message.expires_at and datetime.utcnow() > message.expires_at:
                self.logger.debug(f"Message {message.id} expired, discarding")
                return await self.receive_message(queue_name, timeout, auto_ack)
            
            if not auto_ack:
                # Move to processing queue for manual acknowledgment
                processing_key = f"{redis_key}:processing"
                await self.redis_client.lpush(processing_key, message_data)
                message.status = MessageStatus.PROCESSING
            else:
                message.status = MessageStatus.COMPLETED
            
            if self.metrics:
                self.metrics.record_message_received()
            
            self.logger.debug(f"Message {message.id} received from queue {queue_name}")
            return message
            
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            self.logger.error(f"Failed to receive message from {queue_name}: {str(e)}")
            if self.metrics:
                self.metrics.record_connection_error()
            raise
    
    async def _process_delayed_messages(self, redis_key: str) -> None:
        """Move ready delayed messages to priority queue"""
        delayed_key = f"{redis_key}:delayed"
        priority_key = f"{redis_key}:priority"
        current_time = time.time()
        
        # Get messages ready for processing
        ready_messages = await self.redis_client.zrangebyscore(
            delayed_key, 0, current_time, withscores=True
        )
        
        if ready_messages:
            pipe = self.redis_client.pipeline()
            
            for message_data, _ in ready_messages:
                # Parse message to get priority
                message = QueueMessage.from_dict(json.loads(message_data))
                score = message.priority.value + (time.time() / 1000000)
                
                # Move to priority queue
                pipe.zadd(priority_key, {message_data: score})
                pipe.zrem(delayed_key, message_data)
            
            await pipe.execute()
            self.logger.debug(f"Moved {len(ready_messages)} delayed messages to processing")
    
    async def acknowledge_message(self, queue_name: str, message: QueueMessage) -> bool:
        """Acknowledge message processing completion"""
        try:
            redis_key = self.queues.get(queue_name)
            if not redis_key:
                raise ValueError(f"Unknown queue: {queue_name}")
            
            processing_key = f"{redis_key}:processing"
            message_data = json.dumps(message.to_dict())
            
            # Remove from processing queue
            removed = await self.redis_client.lrem(processing_key, 1, message_data)
            
            if removed > 0:
                message.status = MessageStatus.COMPLETED
                if self.metrics:
                    processing_time = (datetime.utcnow() - message.created_at).total_seconds()
                    self.metrics.record_message_processed(processing_time)
                
                self.logger.debug(f"Message {message.id} acknowledged")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to acknowledge message: {str(e)}")
            return False
    
    async def reject_message(
        self,
        queue_name: str,
        message: QueueMessage,
        requeue: bool = True
    ) -> bool:
        """Reject message and optionally requeue with retry logic"""
        try:
            redis_key = self.queues.get(queue_name)
            if not redis_key:
                raise ValueError(f"Unknown queue: {queue_name}")
            
            processing_key = f"{redis_key}:processing"
            message_data = json.dumps(message.to_dict())
            
            # Remove from processing queue
            await self.redis_client.lrem(processing_key, 1, message_data)
            
            if requeue and message.retry_count < message.max_retries:
                # Requeue with backoff
                message.retry_count += 1
                message.status = MessageStatus.RETRY
                
                # Exponential backoff delay
                delay_seconds = min(300, 2 ** message.retry_count)  # Max 5 minutes
                
                await self.send_message(queue_name, message, delay_seconds)
                
                self.logger.info(
                    f"Message {message.id} requeued (attempt {message.retry_count}/{message.max_retries})"
                )
            else:
                # Move to dead letter queue
                await self._move_to_dead_letter(redis_key, messages=[message_data])
                message.status = MessageStatus.DEAD_LETTER
                
                self.logger.warning(f"Message {message.id} moved to dead letter queue")
            
            if self.metrics:
                self.metrics.record_message_failed()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to reject message: {str(e)}")
            return False
    
    async def _move_to_dead_letter(
        self,
        redis_key: str,
        count: Optional[int] = None,
        messages: Optional[List[str]] = None
    ) -> None:
        """Move messages to dead letter queue"""
        dead_letter_key = f"{redis_key}:dead_letter"
        
        if messages:
            # Move specific messages
            for message_data in messages:
                await self.redis_client.lpush(dead_letter_key, message_data)
                await self.redis_client.expire(dead_letter_key, self.dead_letter_ttl)
        elif count:
            # Move oldest messages from priority queue
            priority_key = f"{redis_key}:priority"
            oldest_messages = await self.redis_client.zrange(priority_key, 0, count - 1)
            
            if oldest_messages:
                pipe = self.redis_client.pipeline()
                for message_data in oldest_messages:
                    pipe.lpush(dead_letter_key, message_data)
                    pipe.zrem(priority_key, message_data)
                pipe.expire(dead_letter_key, self.dead_letter_ttl)
                await pipe.execute()
    
    async def subscribe(
        self,
        queue_name: str,
        callback: Callable[[QueueMessage], Any],
        max_concurrent: int = 5
    ) -> None:
        """Subscribe to queue with message processing callback"""
        if queue_name not in self.queues:
            raise ValueError(f"Unknown queue: {queue_name}")
        
        self.subscribers[queue_name].add(callback)
        
        # Start processing task
        task = asyncio.create_task(
            self._process_queue_messages(queue_name, callback, max_concurrent)
        )
        self.processing_tasks.add(task)
        task.add_done_callback(self.processing_tasks.discard)
        
        self.logger.info(f"Subscribed to queue {queue_name} with {max_concurrent} concurrent processors")
    
    async def _process_queue_messages(
        self,
        queue_name: str,
        callback: Callable[[QueueMessage], Any],
        max_concurrent: int
    ) -> None:
        """Process messages from queue with concurrency control"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        while not self._shutdown_event.is_set():
            try:
                message = await self.receive_message(queue_name, timeout=1, auto_ack=False)
                if message is None:
                    continue
                
                # Process message with concurrency control
                task = asyncio.create_task(
                    self._process_single_message(semaphore, queue_name, message, callback)
                )
                self.processing_tasks.add(task)
                task.add_done_callback(self.processing_tasks.discard)
                
            except Exception as e:
                self.logger.error(f"Error in queue processor for {queue_name}: {str(e)}")
                await asyncio.sleep(1)  # Brief pause on error
    
    async def _process_single_message(
        self,
        semaphore: asyncio.Semaphore,
        queue_name: str,
        message: QueueMessage,
        callback: Callable[[QueueMessage], Any]
    ) -> None:
        """Process individual message with error handling"""
        async with semaphore:
            try:
                start_time = time.time()
                
                # Execute callback
                if asyncio.iscoroutinefunction(callback):
                    result = await callback(message)
                else:
                    result = callback(message)
                
                # Acknowledge successful processing
                await self.acknowledge_message(queue_name, message)
                
                processing_time = time.time() - start_time
                self.logger.debug(
                    f"Message {message.id} processed successfully in {processing_time:.3f}s"
                )
                
            except Exception as e:
                self.logger.error(
                    f"Error processing message {message.id}: {str(e)}\n"
                    f"Traceback: {traceback.format_exc()}"
                )
                
                # Reject message with retry logic
                await self.reject_message(queue_name, message, requeue=True)
    
    # Learning Engine Integration Methods
    async def send_learning_feedback(self, feedback_data: Dict[str, Any]) -> str:
        """Send feedback data to learning engine"""
        message = QueueMessage(
            id=str(uuid.uuid4()),
            type=MessageType.LEARNING_FEEDBACK,
            priority=MessagePriority.HIGH,
            payload=feedback_data,
            source_agent="learning_feedback_sender"
        )
        return await self.send_message("feedback", message)
    
    async def send_learning_update(self, update_data: Dict[str, Any]) -> str:
        """Send learning update to knowledge base"""
        message = QueueMessage(
            id=str(uuid.uuid4()),
            type=MessageType.LEARNING_UPDATE,
            priority=MessagePriority.HIGH,
            payload=update_data,
            source_agent="learning_updater"
        )
        return await self.send_message("updates", message)
    
    async def send_agent_task(
        self,
        task_data: Dict[str, Any],
        target_agent: str,
        priority: MessagePriority = MessagePriority.NORMAL,
        workflow_id: Optional[str] = None
    ) -> str:
        """Send task to specific agent"""
        message = QueueMessage(
            id=str(uuid.uuid4()),
            type=MessageType.AGENT_TASK,
            priority=priority,
            payload=task_data,
            target_agent=target_agent,
            workflow_id=workflow_id
        )
        return await self.send_message("tasks", message)
    
    async def send_orchestration_command(self, command_data: Dict[str, Any]) -> str:
        """Send orchestration command"""
        message = QueueMessage(
            id=str(uuid.uuid4()),
            type=MessageType.ORCHESTRATION,
            priority=MessagePriority.HIGH,
            payload=command_data,
            source_agent="orchestrator"
        )
        return await self.send_message("orchestration", message)
    
    # Queue Management Methods
    async def get_queue_stats(self, queue_name: str) -> Dict[str, Any]:
        """Get comprehensive queue statistics"""
        redis_key = self.queues.get(queue_name)
        if not redis_key:
            raise ValueError(f"Unknown queue: {queue_name}")
        
        priority_size = await self.redis_client.zcard(f"{redis_key}:priority")
        processing_size = await self.redis_client.llen(f"{redis_key}:processing")
        delayed_size = await self.redis_client.zcard(f"{redis_key}:delayed")
        dead_letter_size = await self.redis_client.llen(f"{redis_key}:dead_letter")
        
        return {
            "queue_name": queue_name,
            "pending_messages": priority_size,
            "processing_messages": processing_size,
            "delayed_messages": delayed_size,
            "dead_letter_messages": dead_letter_size,
            "total_messages": priority_size + processing_size + delayed_size,
            "subscribers": len(self.subscribers.get(queue_name, set()))
        }
    
    async def get_all_queue_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all queues"""
        stats = {}
        for queue_name in self.queues.keys():
            stats[queue_name] = await self.get_queue_stats(queue_name)
        return stats
    
    async def purge_queue(self, queue_name: str, include_dead_letter: bool = False) -> int:
        """Purge all messages from queue"""
        redis_key = self.queues.get(queue_name)
        if not redis_key:
            raise ValueError(f"Unknown queue: {queue_name}")
        
        pipe = self.redis_client.pipeline()
        pipe.delete(f"{redis_key}:priority")
        pipe.delete(f"{redis_key}:processing")
        pipe.delete(f"{redis_key}:delayed")
        
        if include_dead_letter:
            pipe.delete(f"{redis_key}:dead_letter")
        
        results = await pipe.execute()
        total_purged = sum(results)
        
        self.logger.info(f"Purged {total_purged} messages from queue {queue_name}")
        return total_purged
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics"""
        if not self.metrics:
            return {"metrics_disabled": True}
        
        queue_stats = await self.get_all_queue_stats()
        
        return {
            "messages_sent": self.metrics.messages_sent,
            "messages_received": self.metrics.messages_received,
            "messages_processed": self.metrics.messages_processed,
            "messages_failed": self.metrics.messages_failed,
            "connection_errors": self.metrics.connection_errors,
            "average_processing_time": self.metrics.get_average_processing_time(),
            "throughput": self.metrics.get_throughput(),
            "error_rate": self.metrics.get_error_rate(),
            "queue_stats": queue_stats,
            "health_status": self.is_healthy,
            "uptime_seconds": (datetime.utcnow() - self.metrics.start_time).total_seconds(),
            "last_health_check": self.metrics.last_health_check.isoformat() if self.metrics.last_health_check else None
        }
    
    @asynccontextmanager
    async def batch_operations(self):
        """Context manager for batch operations"""
        pipe = self.redis_client.pipeline()
        try:
            yield pipe
            await pipe.execute()
        except Exception as e:
            self.logger.error(f"Batch operation failed: {str(e)}")
            raise
    
    async def requeue_dead_letter_messages(
        self, 
        queue_name: str, 
        max_count: int = 100
    ) -> int:
        """Requeue messages from dead letter queue"""
        redis_key = self.queues.get(queue_name)
        if not redis_key:
            raise ValueError(f"Unknown queue: {queue_name}")
        
        dead_letter_key = f"{redis_key}:dead_letter"
        priority_key = f"{redis_key}:priority"
        
        requeued_count = 0
        
        for _ in range(max_count):
            message_data = await self.redis_client.rpop(dead_letter_key)
            if not message_data:
                break
            
            try:
                message = QueueMessage.from_dict(json.loads(message_data))
                message.retry_count = 0  # Reset retry count
                message.status = MessageStatus.PENDING
                
                # Add back to priority queue
                score = message.priority.value + (time.time() / 1000000)
                await self.redis_client.zadd(
                    priority_key, 
                    {json.dumps(message.to_dict()): score}
                )
                requeued_count += 1
                
            except Exception as e:
                self.logger.error(f"Failed to requeue dead letter message: {str(e)}")
                # Put back in dead letter if requeue fails
                await self.redis_client.lpush(dead_letter_key, message_data)
        
        self.logger.info(f"Requeued {requeued_count} messages from dead letter queue")
        return requeued_count
    
    async def schedule_message(
        self,
        queue_name: str,
        message: QueueMessage,
        schedule_time: datetime
    ) -> str:
        """Schedule a message for future delivery"""
        delay_seconds = max(0, (schedule_time - datetime.utcnow()).total_seconds())
        return await self.send_message(queue_name, message, delay_seconds=int(delay_seconds))
    
    async def cancel_scheduled_message(
        self,
        queue_name: str,
        message_id: str
    ) -> bool:
        """Cancel a scheduled message"""
        redis_key = self.queues.get(queue_name)
        if not redis_key:
            raise ValueError(f"Unknown queue: {queue_name}")
        
        delayed_key = f"{redis_key}:delayed"
        
        # Get all delayed messages
        delayed_messages = await self.redis_client.zrange(delayed_key, 0, -1)
        
        for message_data in delayed_messages:
            try:
                message = QueueMessage.from_dict(json.loads(message_data))
                if message.id == message_id:
                    await self.redis_client.zrem(delayed_key, message_data)
                    self.logger.info(f"Cancelled scheduled message {message_id}")
                    return True
            except json.JSONDecodeError:
                continue
        
        return False
    
    async def get_message_status(
        self,
        queue_name: str,
        message_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get status of a specific message"""
        redis_key = self.queues.get(queue_name)
        if not redis_key:
            raise ValueError(f"Unknown queue: {queue_name}")
        
        # Check in all possible locations
        locations = [
            (f"{redis_key}:priority", "pending"),
            (f"{redis_key}:processing", "processing"),
            (f"{redis_key}:delayed", "scheduled"),
            (f"{redis_key}:dead_letter", "dead_letter")
        ]
        
        for location_key, status in locations:
            if "priority" in location_key or "delayed" in location_key:
                # Sorted set
                messages = await self.redis_client.zrange(location_key, 0, -1, withscores=True)
                for message_data, score in messages:
                    try:
                        message = QueueMessage.from_dict(json.loads(message_data))
                        if message.id == message_id:
                            return {
                                "message_id": message_id,
                                "status": status,
                                "location": location_key,
                                "score": score,
                                "message": message.to_dict()
                            }
                    except json.JSONDecodeError:
                        continue
            else:
                # List
                messages = await self.redis_client.lrange(location_key, 0, -1)
                for message_data in messages:
                    try:
                        message = QueueMessage.from_dict(json.loads(message_data))
                        if message.id == message_id:
                            return {
                                "message_id": message_id,
                                "status": status,
                                "location": location_key,
                                "message": message.to_dict()
                            }
                    except json.JSONDecodeError:
                        continue
        
        return None
    
    async def replay_messages(
        self,
        queue_name: str,
        from_time: datetime,
        to_time: datetime,
        target_queue: Optional[str] = None
    ) -> int:
        """Replay messages from a time range (requires message logging)"""
        # This would require persistent message logging
        # Implementation depends on your logging strategy
        self.logger.warning("Message replay requires persistent logging implementation")
        return 0
    
    # Advanced Learning Engine Integration
    async def create_learning_workflow(
        self,
        workflow_id: str,
        steps: List[Dict[str, Any]]
    ) -> str:
        """Create a learning workflow with multiple steps"""
        workflow_message = QueueMessage(
            id=str(uuid.uuid4()),
            type=MessageType.WORKFLOW_EVENT,
            priority=MessagePriority.HIGH,
            payload={
                "workflow_id": workflow_id,
                "action": "create_workflow",
                "steps": steps,
                "created_at": datetime.utcnow().isoformat()
            },
            workflow_id=workflow_id
        )
        
        return await self.send_message("workflows", workflow_message)
    
    async def send_training_data(
        self,
        data: Dict[str, Any],
        data_type: str = "general"
    ) -> str:
        """Send training data to learning engine"""
        message = QueueMessage(
            id=str(uuid.uuid4()),
            type=MessageType.LEARNING_UPDATE,
            priority=MessagePriority.NORMAL,
            payload={
                "data": data,
                "data_type": data_type,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        return await self.send_message("training_data", message)
    
    async def broadcast_to_agents(
        self,
        message_data: Dict[str, Any],
        agent_filter: Optional[List[str]] = None,
        priority: MessagePriority = MessagePriority.NORMAL
    ) -> List[str]:
        """Broadcast message to multiple agents"""
        message_ids = []
        
        broadcast_message = QueueMessage(
            id=str(uuid.uuid4()),
            type=MessageType.SYSTEM_EVENT,
            priority=priority,
            payload={
                "broadcast": True,
                "agent_filter": agent_filter,
                "data": message_data,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        message_id = await self.send_message("orchestration", broadcast_message)
        message_ids.append(message_id)
        
        return message_ids
    
    async def send_system_alert(
        self,
        alert_type: str,
        severity: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Send system alert message"""
        priority_map = {
            "critical": MessagePriority.CRITICAL,
            "high": MessagePriority.HIGH,
            "medium": MessagePriority.NORMAL,
            "low": MessagePriority.LOW
        }
        
        alert_message = QueueMessage(
            id=str(uuid.uuid4()),
            type=MessageType.SYSTEM_EVENT,
            priority=priority_map.get(severity.lower(), MessagePriority.NORMAL),
            payload={
                "alert_type": alert_type,
                "severity": severity,
                "message": message,
                "metadata": metadata or {},
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        return await self.send_message("events", alert_message)
    
    # Performance Optimization Methods
    async def optimize_queues(self) -> Dict[str, Any]:
        """Optimize queue performance by cleaning expired messages"""
        optimization_results = {}
        current_time = datetime.utcnow()
        
        for queue_name, redis_key in self.queues.items():
            cleaned_count = 0
            
            # Clean priority queue of expired messages
            priority_key = f"{redis_key}:priority"
            messages = await self.redis_client.zrange(priority_key, 0, -1)
            
            for message_data in messages:
                try:
                    message = QueueMessage.from_dict(json.loads(message_data))
                    if message.expires_at and current_time > message.expires_at:
                        await self.redis_client.zrem(priority_key, message_data)
                        cleaned_count += 1
                except (json.JSONDecodeError, KeyError):
                    # Remove malformed messages
                    await self.redis_client.zrem(priority_key, message_data)
                    cleaned_count += 1
            
            optimization_results[queue_name] = {
                "expired_messages_removed": cleaned_count
            }
        
        self.logger.info(f"Queue optimization completed: {optimization_results}")
        return optimization_results
    
    async def get_processing_agents(self, queue_name: str) -> List[str]:
        """Get list of agents currently processing from a queue"""
        processing_agents = []
        redis_key = self.queues.get(queue_name)
        
        if redis_key:
            processing_key = f"{redis_key}:processing"
            messages = await self.redis_client.lrange(processing_key, 0, -1)
            
            for message_data in messages:
                try:
                    message = QueueMessage.from_dict(json.loads(message_data))
                    if message.target_agent and message.target_agent not in processing_agents:
                        processing_agents.append(message.target_agent)
                except json.JSONDecodeError:
                    continue
        
        return processing_agents
    
    async def pause_queue(self, queue_name: str) -> bool:
        """Pause message processing for a queue"""
        pause_key = f"{self.queue_prefix}:paused:{queue_name}"
        await self.redis_client.set(pause_key, "true", ex=3600)  # 1 hour expiry
        self.logger.info(f"Queue {queue_name} paused")
        return True
    
    async def resume_queue(self, queue_name: str) -> bool:
        """Resume message processing for a queue"""
        pause_key = f"{self.queue_prefix}:paused:{queue_name}"
        deleted = await self.redis_client.delete(pause_key)
        if deleted:
            self.logger.info(f"Queue {queue_name} resumed")
        return deleted > 0
    
    async def is_queue_paused(self, queue_name: str) -> bool:
        """Check if a queue is paused"""
        pause_key = f"{self.queue_prefix}:paused:{queue_name}"
        return await self.redis_client.exists(pause_key) > 0
    
    # Cleanup and Shutdown
    async def close(self) -> None:
        """Graceful shutdown of the message queue system"""
        self.logger.info("Initiating Redis message queue shutdown...")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Wait for processing tasks to complete
        if self.processing_tasks:
            self.logger.info(f"Waiting for {len(self.processing_tasks)} processing tasks to complete...")
            await asyncio.gather(*self.processing_tasks, return_exceptions=True)
        
        # Stop health check task
        if self.health_check_task and not self.health_check_task.done():
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
        
        # Close Redis connections
        try:
            if self.redis_client:
                await self.redis_client.aclose()
            if self.subscriber_client:
                await self.subscriber_client.aclose()
        except Exception as e:
            self.logger.error(f"Error closing Redis connections: {str(e)}")
        
        self.is_healthy = False
        self.logger.info("Redis message queue shutdown completed")
    
    def __repr__(self) -> str:
        return (
            f"RedisMessageQueue(prefix={self.queue_prefix}, "
            f"queues={len(self.queues)}, healthy={self.is_healthy})"
        )