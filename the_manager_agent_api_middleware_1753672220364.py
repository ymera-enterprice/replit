“””
YMERA Enterprise Manager Agent - FastAPI Server
Complete API Gateway with File Handling, Live Chat, and Agent Integration
“””

import asyncio
import json
import logging
import os
import shutil
import tempfile
import uuid
import zipfile
import mimetypes
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, BinaryIO
from contextlib import asynccontextmanager
import aiofiles
import os
import re
from fastapi.responses import FileResponse

# FastAPI imports

from fastapi import (
FastAPI, HTTPException, Depends, UploadFile, File,
Form, WebSocket, WebSocketDisconnect, BackgroundTasks,
Request, Response, status
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import (
JSONResponse, FileResponse, StreamingResponse, HTMLResponse
)
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.responses import Response as StarletteResponse

# YMERA imports (your existing modules)

from ymera_enterprise_manager import TheManagerAgent, TaskPriority, TaskType
from ymera_core.config import ConfigManager
from ymera_core.database.manager import DatabaseManager
from ymera_core.logging.structured_logger import StructuredLogger
from ymera_core.cache.redis_cache import RedisCacheManager
from ymera_core.security.auth_manager import AuthManager

# Pydantic models for API

class ProcessingRequest(BaseModel):
data: Dict[str, Any]
data_type: str = “auto_detect”
priority: TaskPriority = TaskPriority.MEDIUM
workflow_preference: Optional[str] = None
user_id: str
session_id: Optional[str] = None

class TaskStatusResponse(BaseModel):
task_id: str
status: str
progress: float
message: str
results: Optional[Dict[str, Any]] = None
created_at: datetime
updated_at: datetime

class FileProcessingRequest(BaseModel):
file_type: str
processing_options: Dict[str, Any] = {}
priority: TaskPriority = TaskPriority.MEDIUM
workflow_preference: Optional[str] = None

class ChatMessage(BaseModel):
message: str
user_id: str
session_id: Optional[str] = None
message_type: str = “user_message”

class AgentCommunicationRequest(BaseModel):
agent_name: str
action: str
payload: Dict[str, Any]
priority: TaskPriority = TaskPriority.MEDIUM

class SystemMetricsResponse(BaseModel):
system_health: Dict[str, Any]
active_tasks: List[Dict[str, Any]]
agent_status: Dict[str, Any]
performance_metrics: Dict[str, Any]

# Custom middleware classes

class RequestTrackingMiddleware(BaseHTTPMiddleware):
“”“Track requests for monitoring and analytics”””

```
async def dispatch(self, request: Request, call_next):
    start_time = datetime.utcnow()
    request_id = str(uuid.uuid4())
    
    # Add request ID to headers
    request.state.request_id = request_id
    
    # Process request
    response = await call_next(request)
    
    # Calculate processing time
    process_time = (datetime.utcnow() - start_time).total_seconds()
    
    # Add custom headers
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = str(process_time)
    
    # Log request info (implement your logging logic)
    logging.info(f"Request {request_id}: {request.method} {request.url} - {process_time}s")
    
    return response
```

class SecurityMiddleware(BaseHTTPMiddleware):
“”“Basic security middleware”””

```
async def dispatch(self, request: Request, call_next):
    # Add security headers
    response = await call_next(request)
    
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    
    return response
```

# File utilities

class FileManager:
“”“Centralized file management for uploads and downloads”””

```
def __init__(self, upload_dir: str = "uploads", download_dir: str = "downloads"):
    self.upload_dir = Path(upload_dir)
    self.download_dir = Path(download_dir)
    self.temp_dir = Path("temp")
    
    # Create directories
    for dir_path in [self.upload_dir, self.download_dir, self.temp_dir]:
        dir_path.mkdir(exist_ok=True)
    
    # Supported file types
    self.supported_extensions = {
        'text': ['.txt', '.md', '.json', '.yaml', '.yml', '.xml', '.csv'],
        'code': ['.py', '.js', '.html', '.css', '.java', '.cpp', '.c', '.php'],
        'document': ['.pdf', '.docx', '.doc', '.xlsx', '.xls', '.pptx', '.ppt'],
        'image': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp'],
        'archive': ['.zip', '.tar', '.gz', '.rar', '.7z'],
        'media': ['.mp4', '.avi', '.mov', '.mp3', '.wav', '.flac']
    }

async def save_uploaded_file(self, file: UploadFile, user_id: str) -> Dict[str, Any]:
    """Save uploaded file and return metadata"""
    try:
        # Generate unique filename
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        file_id = str(uuid.uuid4())[:8]
        filename = f"{timestamp}_{file_id}_{file.filename}"
        
        # Create user directory
        user_dir = self.upload_dir / user_id
        user_dir.mkdir(exist_ok=True)
        
        file_path = user_dir / filename
        
        # Save file
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        # Get file metadata
        file_size = len(content)
        file_type = self._determine_file_type(file.filename)
        mime_type = mimetypes.guess_type(file.filename)[0] or 'application/octet-stream'
        
        metadata = {
            'file_id': file_id,
            'original_filename': file.filename,
            'saved_filename': filename,
            'file_path': str(file_path),
            'file_size': file_size,
            'file_type': file_type,
            'mime_type': mime_type,
            'uploaded_at': datetime.utcnow().isoformat(),
            'user_id': user_id,
            'is_archive': file_type == 'archive'
        }
        
        # If it's an archive, extract and analyze
        if file_type == 'archive' and file.filename.endswith('.zip'):
            extract_info = await self._extract_zip_file(file_path, user_dir)
            metadata['extracted_files'] = extract_info
        
        return metadata
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")

def _determine_file_type(self, filename: str) -> str:
    """Determine file category from extension"""
    ext = Path(filename).suffix.lower()
    
    for file_type, extensions in self.supported_extensions.items():
        if ext in extensions:
            return file_type
    
    return 'unknown'

async def _extract_zip_file(self, zip_path: Path, extract_to: Path) -> Dict[str, Any]:
    """Extract ZIP file and return information about contents"""
    try:
        extract_dir = extract_to / f"extracted_{zip_path.stem}"
        extract_dir.mkdir(exist_ok=True)
        
        extracted_files = []
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            
            # Extract all files
            zip_ref.extractall(extract_dir)
            
            # Analyze extracted files
            for file_name in file_list:
                file_path = extract_dir / file_name
                if file_path.is_file():
                    extracted_files.append({
                        'name': file_name,
                        'path': str(file_path),
                        'size': file_path.stat().st_size,
                        'type': self._determine_file_type(file_name)
                    })
        
        return {
            'extract_path': str(extract_dir),
            'total_files': len(extracted_files),
            'files': extracted_files
        }
        
    except Exception as e:
        logging.error(f"ZIP extraction failed: {str(e)}")
        return {'error': str(e)}

async def create_download_file(self, data: Any, filename: str, user_id: str) -> str:
    """Create a file for download"""
    try:
        # Create user download directory
        user_dir = self.download_dir / user_id
        user_dir.mkdir(exist_ok=True)
        
        file_path = user_dir / filename
        
        # Write data based on type
        if isinstance(data, (dict, list)):
            async with aiofiles.open(file_path, 'w') as f:
                await f.write(json.dumps(data, indent=2, default=str))
        elif isinstance(data, str):
            async with aiofiles.open(file_path, 'w') as f:
                await f.write(data)
        else:
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(data)
        
        return str(file_path)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File creation failed: {str(e)}")
```

# WebSocket manager for live chat

class ConnectionManager:
“”“Manage WebSocket connections for live chat”””

```
def __init__(self):
    self.active_connections: Dict[str, WebSocket] = {}
    self.user_sessions: Dict[str, Dict[str, Any]] = {}

async def connect(self, websocket: WebSocket, user_id: str):
    await websocket.accept()
    self.active_connections[user_id] = websocket
    self.user_sessions[user_id] = {
        'connected_at': datetime.utcnow(),
        'last_activity': datetime.utcnow()
    }
    logging.info(f"User {user_id} connected to WebSocket")

def disconnect(self, user_id: str):
    if user_id in self.active_connections:
        del self.active_connections[user_id]
    if user_id in self.user_sessions:
        del self.user_sessions[user_id]
    logging.info(f"User {user_id} disconnected from WebSocket")

async def send_personal_message(self, message: str, user_id: str):
    if user_id in self.active_connections:
        websocket = self.active_connections[user_id]
        try:
            await websocket.send_text(message)
            self.user_sessions[user_id]['last_activity'] = datetime.utcnow()
        except Exception as e:
            logging.error(f"Error sending message to {user_id}: {str(e)}")
            self.disconnect(user_id)

async def broadcast(self, message: str):
    disconnected_users = []
    for user_id, websocket in self.active_connections.items():
        try:
            await websocket.send_text(message)
        except Exception as e:
            logging.error(f"Error broadcasting to {user_id}: {str(e)}")
            disconnected_users.append(user_id)
    
    # Clean up disconnected users
    for user_id in disconnected_users:
        self.disconnect(user_id)
```

# Global instances

file_manager = FileManager()
connection_manager = ConnectionManager()

# Dependency functions

async def get_manager_agent():
“”“Get the manager agent instance”””
# Initialize your manager agent here
# This is a placeholder - implement based on your initialization
config = ConfigManager()
db_manager = DatabaseManager(config)
cache_manager = RedisCacheManager(config)

```
manager_agent = TheManagerAgent(
    config=config,
    db_manager=db_manager,
    cache_manager=cache_manager
)
await manager_agent.initialize()
return manager_agent
```

security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
“”“Validate JWT token and return user info”””
# Implement your JWT validation logic here
# For now, return a mock user
return {“user_id”: “user_123”, “username”: “demo_user”}

# Lifespan context manager

@asynccontextmanager
async def lifespan(app: FastAPI):
“”“Manage application startup and shutdown”””
# Startup
logging.info(“🚀 Starting YMERA FastAPI Server…”)

```
# Initialize any startup tasks here
yield

# Shutdown
logging.info("🛑 Shutting down YMERA FastAPI Server...")
```

# Create FastAPI app

app = FastAPI(
title=“YMERA Enterprise Manager API”,
description=“Production-Ready Central Orchestration and Management System API”,
version=“1.0.0”,
lifespan=lifespan
)

# Add middleware

app.add_middleware(
CORSMiddleware,
allow_origins=[”*”],  # Configure based on your needs
allow_credentials=True,
allow_methods=[”*”],
allow_headers=[”*”],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=[”*”])  # Configure for production
app.add_middleware(RequestTrackingMiddleware)
app.add_middleware(SecurityMiddleware)

# Mount static files

static_dir = Path(“static”)
static_dir.mkdir(exist_ok=True)
app.mount(”/static”, StaticFiles(directory=str(static_dir)), name=“static”)

# API Routes

@app.get(”/”)
async def root():
“”“API root endpoint”””
return {
“message”: “YMERA Enterprise Manager API”,
“version”: “1.0.0”,
“status”: “operational”,
“timestamp”: datetime.utcnow().isoformat()
}

@app.get(”/health”)
async def health_check():
“”“Health check endpoint”””
return {
“status”: “healthy”,
“timestamp”: datetime.utcnow().isoformat(),
“services”: {
“api”: “operational”,
“database”: “operational”,  # Add actual checks
“cache”: “operational”,
“websocket”: “operational”
}
}

# Core processing endpoints

@app.post(”/api/v1/process”)
async def process_data(
request: ProcessingRequest,
background_tasks: BackgroundTasks,
current_user: dict = Depends(get_current_user),
manager: TheManagerAgent = Depends(get_manager_agent)
):
“”“Process data through the manager agent”””
try:
# Add user context to request
enhanced_data = {
**request.data,
“_user_context”: {
“user_id”: current_user[“user_id”],
“session_id”: request.session_id,
“request_time”: datetime.utcnow().isoformat()
}
}

```
    # Process through manager agent
    result = await manager.process_incoming_data(
        data=enhanced_data,
        data_type=request.data_type,
        priority=request.priority,
        workflow_preference=request.workflow_preference
    )
    
    return {
        "success": True,
        "process_id": result.get("process_id"),
        "result": result,
        "timestamp": datetime.utcnow().isoformat()
    }
    
except Exception as e:
    logging.error(f"Processing error: {str(e)}")
    raise HTTPException(status_code=500, detail=str(e))
```

@app.get(”/api/v1/task/{task_id}/status”)
async def get_task_status(
task_id: str,
current_user: dict = Depends(get_current_user),
manager: TheManagerAgent = Depends(get_manager_agent)
):
“”“Get task execution status”””
try:
status = await manager.get_task_status(task_id)
return TaskStatusResponse(**status)
except Exception as e:
raise HTTPException(status_code=404, detail=f”Task not found: {str(e)}”)

@app.post(”/api/v1/agent/communicate”)
async def communicate_with_agent(
request: AgentCommunicationRequest,
current_user: dict = Depends(get_current_user),
manager: TheManagerAgent = Depends(get_manager_agent)
):
“”“Direct communication with specific agents”””
try:
result = await manager.communicate_with_agent(
agent_name=request.agent_name,
action=request.action,
payload=request.payload,
user_context=current_user
)

```
    return {
        "success": True,
        "agent": request.agent_name,
        "result": result,
        "timestamp": datetime.utcnow().isoformat()
    }
    
except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))
```

# File upload endpoints

@app.post(”/api/v1/files/upload”)
async def upload_file(
file: UploadFile = File(…),
processing_options: str = Form(”{}”),
priority: TaskPriority = Form(TaskPriority.MEDIUM),
workflow_preference: Optional[str] = Form(None),
current_user: dict = Depends(get_current_user),
manager: TheManagerAgent = Depends(get_manager_agent)
):
“”“Upload and process a single file”””
try:
# Parse processing options
options = json.loads(processing_options) if processing_options else {}

```
    # Save file
    file_metadata = await file_manager.save_uploaded_file(file, current_user["user_id"])
    
    # Process file through manager
    processing_request = {
        "file_metadata": file_metadata,
        "processing_options": options,
        "user_context": current_user
    }
    
    result = await manager.process_incoming_data(
        data=processing_request,
        data_type="file",
        priority=priority,
        workflow_preference=workflow_preference
    )
    
    return {
        "success": True,
        "file_metadata": file_metadata,
        "processing_result": result,
        "timestamp": datetime.utcnow().isoformat()
    }
    
except Exception as e:
    logging.error(f"File upload error: {str(e)}")
    raise HTTPException(status_code=500, detail=str(e))
```

@app.post(”/api/v1/files/upload-multiple”)
async def upload_multiple_files(
files: List[UploadFile] = File(…),
processing_options: str = Form(”{}”),
priority: TaskPriority = Form(TaskPriority.MEDIUM),
current_user: dict = Depends(get_current_user),
manager: TheManagerAgent = Depends(get_manager_agent)
):
“”“Upload and process multiple files”””
try:
options = json.loads(processing_options) if processing_options else {}
uploaded_files = []
processing_results = []

```
    # Process each file
    for file in files:
        # Save file
        file_metadata = await file_manager.save_uploaded_file(file, current_user["user_id"])
        uploaded_files.append(file_metadata)
        
        # Process file
        processing_request = {
            "file_metadata": file_metadata,
            "processing_options": options,
            "user_context": current_user,
            "batch_processing": True,
            "total_files": len(files)
        }
        
        result = await manager.process_incoming_data(
            data=processing_request,
            data_type="file",
            priority=priority
        )
        
        processing_results.append(result)
    
    return {
        "success": True,
        "total_files": len(files),
        "uploaded_files": uploaded_files,
        "processing_results": processing_results,
        "timestamp": datetime.utcnow().isoformat()
    }
    
except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))
```

# Download endpoints

@app.get(”/api/v1/files/download/{file_id}”)
async def download_file(
file_id: str,
current_user: dict = Depends(get_current_user)
):
“”“Download a specific file”””
try:
# Construct file path
user_dir = file_manager.download_dir / current_user[“user_id”]

```
    # Find file by ID (you might want to store file mappings in database)
    for file_path in user_dir.glob(f"*{file_id}*"):
        if file_path.is_file():
            return FileResponse(
                path=str(file_path),
                filename=file_path.name,
                media_type='application/octet-stream'
            )
    
    raise HTTPException(status_code=404, detail="File not found")
    
except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))
```

@app.post(”/api/v1/files/generate”)
async def generate_download_file(
data: Dict[str, Any],
filename: str,
format_type: str = “json”,
current_user: dict = Depends(get_current_user)
):
“”“Generate a downloadable file from data”””
try:
# Format data based on type
if format_type == “json”:
file_content = json.dumps(data, indent=2, default=str)
filename = f”{filename}.json”
elif format_type == “txt”:
file_content = str(data)
filename = f”{filename}.txt”
else:
file_content = data

```
    # Create file
    file_path = await file_manager.create_download_file(
        file_content, filename, current_user["user_id"]
    )
    
    return {
        "success": True,
        "filename": filename,
        "download_url": f"/api/v1/files/download/{filename}",
        "created_at": datetime.utcnow().isoformat()
    }
    
except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))
```

# System monitoring endpoints

@app.get(”/api/v1/system/metrics”)
async def get_system_metrics(
current_user: dict = Depends(get_current_user),
manager: TheManagerAgent = Depends(get_manager_agent)
) -> SystemMetricsResponse:
“”“Get comprehensive system metrics”””
try:
metrics = await manager.get_system_metrics(current_user[“user_id”])
return SystemMetricsResponse(**metrics)
except Exception as e:
raise HTTPException(status_code=500, detail=str(e))

@app.get(”/api/v1/system/agents/status”)
async def get_agents_status(
current_user: dict = Depends(get_current_user),
manager: TheManagerAgent = Depends(get_manager_agent)
):
“”“Get status of all agents”””
try:
status = await manager.get_all_agents_status()
return {
“success”: True,
“agents”: status,
“timestamp”: datetime.utcnow().isoformat()
}
except Exception as e:
raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoints for live chat

@app.websocket(”/ws/chat/{user_id}”)
async def websocket_chat_endpoint(
websocket: WebSocket,
user_id: str,
manager: TheManagerAgent = Depends(get_manager_agent)
):
“”“WebSocket endpoint for live chat”””
await connection_manager.connect(websocket, user_id)

```
try:
    while True:
        # Receive message from client
        data = await websocket.receive_text()
        message_data = json.loads(data)
        
        # Process message through manager
        chat_response = await manager.process_chat_message(
            message=message_data.get("message", ""),
            user_id=user_id,
            session_id=message_data.get("session_id"),
            message_type=message_data.get("type", "user_message")
        )
        
        # Send response back to client
        response = {
            "type": "chat_response",
            "message": chat_response.get("response", ""),
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": chat_response.get("metadata", {})
        }
        
        await connection_manager.send_personal_message(
            json.dumps(response), user_id
        )
        
except WebSocketDisconnect:
    connection_manager.disconnect(user_id)
    logging.info(f"Client {user_id} disconnected from chat")
except Exception as e:
    logging.error(f"WebSocket error for {user_id}: {str(e)}")
    connection_manager.disconnect(user_id)
```

@app.post(”/api/v1/chat/broadcast”)
async def broadcast_message(
message: str,
message_type: str = “system_announcement”,
current_user: dict = Depends(get_current_user)
):
“”“Broadcast message to all connected users”””
try:
broadcast_data = {
“type”: message_type,
“message”: message,
“from”: current_user[“username”],
“timestamp”: datetime.utcnow().isoformat()
}

```
    await connection_manager.broadcast(json.dumps(broadcast_data))
    
    return {
        "success": True,
        "message": "Message broadcasted",
        "recipients": len(connection_manager.active_connections)
    }
    
except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))
```

# Chat frontend endpoint

@app.get(”/chat”, response_class=HTMLResponse)
async def chat_frontend():
“”“Serve the chat frontend”””
html_content = “””
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>YMERA Live Chat</title>
<style>
body {
font-family: ‘Segoe UI’, Tahoma, Geneva, Verdana, sans-serif;
margin: 0;
padding: 0;
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
height: 100vh;
display: flex;
justify-content: center;
align-items: center;
}

```
        .chat-container {
            width: 90%;
            max-width: 800px;
            height: 80vh;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        
        .chat-header {
            background: #4a5568;
            color: white;
            padding: 20px;
            text-align: center;
            font-size: 1.5rem;
            font-weight: bold;
        }
        
        .chat-messages {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            background: #f8f9fa;
        }
        
        .message {
            margin-bottom: 15px;
            padding: 12px 16px;
            border-radius: 12px;
            max-width: 70%;
            word-wrap: break-word;
        }
        
        .message.user {
            background: #007bff;
            color: white;
            margin-left: auto;
            text-align: right;
        }
        
        .message.system {
            background: #28a745;
            color: white;
            margin: 0 auto;
            text-align: center;
            max-width: 90%;
        }
        
        .message.agent {
            background: #6c757d;
            color: white;
        }
        
        .message-time {
            font-size: 0.8rem;
            opacity: 0.7;
            margin-top: 5px;
        }
        
        .chat-input-container {
            padding: 20px;
            background: white;
            border-top: 1px solid #e9ecef;
            display: flex;
            gap: 10px;
        }
        
        #messageInput {
            flex: 1;
            padding: 12px 16px;
            border: 2px solid #dee2e6;
            border-radius: 25px;
            outline: none;
            font-size: 1rem;
            transition: border-color 0.3s;
        }
        
        #messageInput:focus {
            border-color: #007bff;
        }
        
        #sendButton {
            padding: 12px 24px;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-size: 1rem;
            font-weight: bold;
            transition: background-color 0.3s;
        }
        
        #sendButton:hover {
            background: #0056b3;
        }
        
        #sendButton:disabled {
            background: #6c757d;
            cursor: not-allowed;
        }
        
        .connection-status {
            padding: 10px;
            text-align: center;
            font-weight: bold;
        }
        
        .connected {
            background: #d4edda;
            color: #155724;
        } ```html
        
        .disconnected {
            background: #f8d7da;
            color: #721c24;
        }
        
        .connecting {
            background: #fff3cd;
            color: #856404;
        }
        
        .typing-indicator {
            padding: 10px 16px;
            background: #e9ecef;
            border-radius: 12px;
            margin-bottom: 15px;
            font-style: italic;
            color: #6c757d;
        }
        
        @keyframes pulse {
            0% { opacity: 0.6; }
            50% { opacity: 1; }
            100% { opacity: 0.6; }
        }
        
        .typing-indicator::after {
            content: '...';
            animation: pulse 1.5s infinite;
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            🤖 YMERA Live Chat
        </div>
        <div id="connectionStatus" class="connection-status connecting">
            Connecting to server...
        </div>
        <div id="messages" class="chat-messages">
            <div class="message system">
                <div>Welcome to YMERA Live Chat!</div>
                <div class="message-time" id="welcomeTime"></div>
            </div>
        </div>
        <div class="chat-input-container">
            <input type="text" id="messageInput" placeholder="Type your message..." disabled>
            <button id="sendButton" disabled>Send</button>
        </div>
    </div>

    <script>
        class YMERAChat {
            constructor() {
                this.ws = null;
                this.userId = 'user_' + Math.random().toString(36).substr(2, 9);
                this.sessionId = 'session_' + Date.now();
                this.isConnected = false;
                this.reconnectAttempts = 0;
                this.maxReconnectAttempts = 5;
                this.reconnectDelay = 1000;
                
                this.initializeElements();
                this.connect();
                this.setupEventListeners();
                this.setWelcomeTime();
            }
            
            initializeElements() {
                this.messagesContainer = document.getElementById('messages');
                this.messageInput = document.getElementById('messageInput');
                this.sendButton = document.getElementById('sendButton');
                this.connectionStatus = document.getElementById('connectionStatus');
            }
            
            setWelcomeTime() {
                document.getElementById('welcomeTime').textContent = 
                    new Date().toLocaleTimeString();
            }
            
            connect() {
                try {
                    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                    const wsUrl = `${protocol}//${window.location.host}/ws/chat/${this.userId}`;
                    
                    this.ws = new WebSocket(wsUrl);
                    this.setupWebSocketEventListeners();
                    
                } catch (error) {
                    console.error('Connection error:', error);
                    this.updateConnectionStatus('disconnected', 'Connection failed');
                    this.scheduleReconnect();
                }
            }
            
            setupWebSocketEventListeners() {
                this.ws.onopen = () => {
                    console.log('Connected to YMERA Chat');
                    this.isConnected = true;
                    this.reconnectAttempts = 0;
                    this.updateConnectionStatus('connected', 'Connected to YMERA');
                    this.enableInput();
                    
                    // Send initial connection message
                    this.sendMessage('Hello YMERA!', 'connection');
                };
                
                this.ws.onmessage = (event) => {
                    try {
                        const data = JSON.parse(event.data);
                        this.handleIncomingMessage(data);
                    } catch (error) {
                        console.error('Error parsing message:', error);
                    }
                };
                
                this.ws.onclose = (event) => {
                    console.log('Disconnected from YMERA Chat:', event.code, event.reason);
                    this.isConnected = false;
                    this.updateConnectionStatus('disconnected', 'Disconnected from server');
                    this.disableInput();
                    
                    if (event.code !== 1000) { // Not a normal closure
                        this.scheduleReconnect();
                    }
                };
                
                this.ws.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    this.updateConnectionStatus('disconnected', 'Connection error');
                };
            }
            
            setupEventListeners() {
                this.sendButton.addEventListener('click', () => this.handleSendMessage());
                
                this.messageInput.addEventListener('keypress', (e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        this.handleSendMessage();
                    }
                });
                
                this.messageInput.addEventListener('input', () => {
                    this.sendButton.disabled = !this.messageInput.value.trim();
                });
            }
            
            handleSendMessage() {
                const message = this.messageInput.value.trim();
                if (message && this.isConnected) {
                    this.sendMessage(message, 'user_message');
                    this.addMessageToUI(message, 'user');
                    this.messageInput.value = '';
                    this.sendButton.disabled = true;
                    this.showTypingIndicator();
                }
            }
            
            sendMessage(message, type = 'user_message') {
                if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                    const messageData = {
                        message: message,
                        type: type,
                        session_id: this.sessionId,
                        timestamp: new Date().toISOString()
                    };
                    
                    this.ws.send(JSON.stringify(messageData));
                }
            }
            
            handleIncomingMessage(data) {
                this.hideTypingIndicator();
                
                switch (data.type) {
                    case 'chat_response':
                        this.addMessageToUI(data.message, 'agent', data.metadata);
                        break;
                    case 'system_announcement':
                        this.addMessageToUI(data.message, 'system');
                        break;
                    case 'typing':
                        this.showTypingIndicator();
                        break;
                    case 'error':
                        this.addMessageToUI(`Error: ${data.message}`, 'system');
                        break;
                    default:
                        console.log('Unknown message type:', data);
                }
            }
            
            addMessageToUI(message, type, metadata = {}) {
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${type}`;
                
                const messageContent = document.createElement('div');
                messageContent.textContent = message;
                messageDiv.appendChild(messageContent);
                
                const timeDiv = document.createElement('div');
                timeDiv.className = 'message-time';
                timeDiv.textContent = new Date().toLocaleTimeString();
                messageDiv.appendChild(timeDiv);
                
                // Add metadata if available
                if (metadata && Object.keys(metadata).length > 0) {
                    const metadataDiv = document.createElement('div');
                    metadataDiv.style.fontSize = '0.8rem';
                    metadataDiv.style.opacity = '0.7';
                    metadataDiv.style.marginTop = '5px';
                    metadataDiv.textContent = `Agent: ${metadata.agent || 'Manager'} | Task: ${metadata.task_id || 'N/A'}`;
                    messageDiv.appendChild(metadataDiv);
                }
                
                this.messagesContainer.appendChild(messageDiv);
                this.scrollToBottom();
            }
            
            showTypingIndicator() {
                this.hideTypingIndicator(); // Remove existing indicator
                
                const typingDiv = document.createElement('div');
                typingDiv.className = 'typing-indicator';
                typingDiv.id = 'typingIndicator';
                typingDiv.textContent = 'YMERA is thinking';
                
                this.messagesContainer.appendChild(typingDiv);
                this.scrollToBottom();
            }
            
            hideTypingIndicator() {
                const indicator = document.getElementById('typingIndicator');
                if (indicator) {
                    indicator.remove();
                }
            }
            
            updateConnectionStatus(status, message) {
                this.connectionStatus.className = `connection-status ${status}`;
                this.connectionStatus.textContent = message;
            }
            
            enableInput() {
                this.messageInput.disabled = false;
                this.messageInput.focus();
            }
            
            disableInput() {
                this.messageInput.disabled = true;
                this.sendButton.disabled = true;
            }
            
            scrollToBottom() {
                this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
            }
            
            scheduleReconnect() {
                if (this.reconnectAttempts < this.maxReconnectAttempts) {
                    this.reconnectAttempts++;
                    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
                    
                    this.updateConnectionStatus('connecting', 
                        `Reconnecting in ${Math.ceil(delay / 1000)}s... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
                    
                    setTimeout(() => {
                        if (!this.isConnected) {
                            this.connect();
                        }
                    }, delay);
                } else {
                    this.updateConnectionStatus('disconnected', 
                        'Connection failed. Please refresh the page.');
                }
            }
        }
        
        // Initialize chat when page loads
        document.addEventListener('DOMContentLoaded', () => {
            new YMERAChat();
        });
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)

# Administrative endpoints

@app.get("/api/v1/admin/system/logs")
async def get_system_logs(
    limit: int = 100,
    level: str = "INFO",
    current_user: dict = Depends(get_current_user),
    manager: TheManagerAgent = Depends(get_manager_agent)
):
    """Get system logs (admin only)"""
    try:
        # Check admin permissions (implement your logic)
        if not current_user.get("is_admin", False):
            raise HTTPException(status_code=403, detail="Admin access required")
        
        logs = await manager.get_system_logs(limit=limit, level=level)
        return {
            "success": True,
            "logs": logs,
            "total_count": len(logs),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/admin/system/restart-agent")
async def restart_agent(
    agent_name: str,
    current_user: dict = Depends(get_current_user),
    manager: TheManagerAgent = Depends(get_manager_agent)
):
    """Restart a specific agent (admin only)"""
    try:
        if not current_user.get("is_admin", False):
            raise HTTPException(status_code=403, detail="Admin access required")
        
        result = await manager.restart_agent(agent_name)
        return {
            "success": True,
            "agent": agent_name,
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/admin/system/clear-cache")
async def clear_system_cache(
    cache_type: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
    manager: TheManagerAgent = Depends(get_manager_agent)
):
    """Clear system cache (admin only)"""
    try:
        if not current_user.get("is_admin", False):
            raise HTTPException(status_code=403, detail="Admin access required")
        
        result = await manager.clear_cache(cache_type)
        return {
            "success": True,
            "cache_cleared": cache_type or "all",
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Data export endpoints

@app.post("/api/v1/export/conversation-history")
async def export_conversation_history(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    format_type: str = "json",
    current_user: dict = Depends(get_current_user),
    manager: TheManagerAgent = Depends(get_manager_agent)
):
    """Export conversation history"""
    try:
        # Parse dates if provided
        start_dt = datetime.fromisoformat(start_date) if start_date else None
        end_dt = datetime.fromisoformat(end_date) if end_date else None
        
        # Get conversation history
        history = await manager.get_conversation_history(
            user_id=current_user["user_id"],
            start_date=start_dt,
            end_date=end_dt
        )
        
        # Generate filename
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_history_{current_user['user_id']}_{timestamp}"
        
        # Create downloadable file
        file_path = await file_manager.create_download_file(
            history, filename, current_user["user_id"]
        )
        
        return {
            "success": True,
            "filename": f"{filename}.{format_type}",
            "download_url": f"/api/v1/files/download/{filename}.{format_type}",
            "record_count": len(history),
            "created_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/export/task-results")
async def export_task_results(
    task_ids: Optional[List[str]] = None,
    status_filter: Optional[str] = None,
    format_type: str = "json",
    current_user: dict = Depends(get_current_user),
    manager: TheManagerAgent = Depends(get_manager_agent)
):
    """Export task results"""
    try:
        results = await manager.export_task_results(
            user_id=current_user["user_id"],
            task_ids=task_ids,
            status_filter=status_filter
        )
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"task_results_{current_user['user_id']}_{timestamp}"
        
        file_path = await file_manager.create_download_file(
            results, filename, current_user["user_id"]
        )
        
        return {
            "success": True,
            "filename": f"{filename}.{format_type}",
            "download_url": f"/api/v1/files/download/{filename}.{format_type}",
            "task_count": len(results),
            "created_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Workflow management endpoints

@app.get("/api/v1/workflows/templates")
async def get_workflow_templates(
    category: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
    manager: TheManagerAgent = Depends(get_manager_agent)
):
    """Get available workflow templates"""
    try:
        templates = await manager.get_workflow_templates(category=category)
        return {
            "success": True,
            "templates": templates,
            "count": len(templates),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/workflows/create")
async def create_custom_workflow(
    workflow_config: Dict[str, Any],
    current_user: dict = Depends(get_current_user),
    manager: TheManagerAgent = Depends(get_manager_agent)
):
    """Create a custom workflow"""
    try:
        workflow_id = await manager.create_workflow(
            config=workflow_config,
            user_id=current_user["user_id"]
        )
        
        return {
            "success": True,
            "workflow_id": workflow_id,
            "message": "Workflow created successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/workflows/{workflow_id}/status")
async def get_workflow_status(
    workflow_id: str,
    current_user: dict = Depends(get_current_user),
    manager: TheManagerAgent = Depends(get_manager_agent)
):
    """Get workflow execution status"""
    try:
        status = await manager.get_workflow_status(
            workflow_id=workflow_id,
            user_id=current_user["user_id"]
        )
        return {
            "success": True,
            "workflow_id": workflow_id,
            "status": status,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Workflow not found: {str(e)}")

# Analytics endpoints

@app.get("/api/v1/analytics/usage-stats")
async def get_usage_statistics(
    period: str = "24h",
    current_user: dict = Depends(get_current_user),
    manager: TheManagerAgent = Depends(get_manager_agent)
):
    """Get usage analytics for the user"""
    try:
        stats = await manager.get_usage_statistics(
            user_id=current_user["user_id"],
            period=period
        )
        
        return {
            "success": True,
            "period": period,
            "statistics": stats,
            "generated_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/analytics/performance-metrics")
async def get_performance_metrics(
    metric_type: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
    manager: TheManagerAgent = Depends(get_manager_agent)
):
    """Get system performance metrics"""
    try:
        metrics = await manager.get_performance_metrics(
            user_id=current_user["user_id"],
            metric_type=metric_type
        )
        
        return {
            "success": True,
            "metrics": metrics,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Notification endpoints

@app.post("/api/v1/notifications/subscribe")
async def subscribe_to_notifications(
    notification_types: List[str],
    webhook_url: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
    manager: TheManagerAgent = Depends(get_manager_agent)
):
    """Subscribe to system notifications"""
    try:
        subscription_id = await manager.subscribe_to_notifications(
            user_id=current_user["user_id"],
            notification_types=notification_types,
            webhook_url=webhook_url
        )
        
        return {
            "success": True,
            "subscription_id": subscription_id,
            "notification_types": notification_types,
            "webhook_url": webhook_url,
            "created_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/notifications/history")
async def get_notification_history(
    limit: int = 50,
    notification_type: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
    manager: TheManagerAgent = Depends(get_manager_agent)
):
    """Get notification history"""
    try:
        history = await manager.get_notification_history(
            user_id=current_user["user_id"],
            limit=limit,
            notification_type=notification_type
        )
        
        return {
            "success": True,
            "notifications": history,
            "count": len(history),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "code": exc.status_code,
                "message": exc.detail,
                "timestamp": datetime.utcnow().isoformat(),
                "request_id": getattr(request.state, "request_id", "unknown")
            }
        }
    )

@app.exception_handler(500)
async def internal_server_error_handler(request: Request, exc: Exception):
    """Handle internal server errors"""
    logging.error(f"Internal server error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": {
                "code": 500,
                "message": "Internal server error occurred",
                "timestamp": datetime.utcnow().isoformat(),
                "request_id": getattr(request.state, "request_id", "unknown")
            }
        }
    )

# Development and debugging endpoints (remove in production)

@app.get("/api/v1/debug/active-connections")
async def debug_active_connections(current_user: dict = Depends(get_current_user)):
    """Debug: Get active WebSocket connections"""
    if not current_user.get("is_admin", False):
        raise HTTPException(status_code=403, detail="Admin access required")
    
    return {
        "active_connections": len(connection_manager.active_connections),
        "user_sessions": {
            user_id: {
                "connected_at": session["connected_at"].isoformat(),
                "last_activity": session["last_activity"].isoformat()
            }
            for user_id, session in connection_manager.user_sessions.items()
        },
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/api/v1/debug/test-broadcast")
async def debug_test_broadcast(
    message: str = "Test broadcast message",
    current_user: dict = Depends(get_current_user)
):
    """Debug: Test broadcast functionality"""
    if not current_user.get("is_admin", False):
        raise HTTPException(status_code=403, detail="Admin access required")
    
    broadcast_data = {
        "type": "debug_broadcast",
        "message": message,
        "from": "System Debug",
        "timestamp": datetime.utcnow().isoformat()
    }
    
    await connection_manager.broadcast(json.dumps(broadcast_data))
    
    return {
        "success": True,
        "message": "Broadcast sent",
        "recipients": len(connection_manager.active_connections),
        "data": broadcast_data
    }

# Startup message

if __name__ == "__main__":
    import uvicorn
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("ymera_api.log"),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("🚀 Starting YMERA Enterprise Manager API Server...")
    
    # Run the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True,
        use_colors=True,
        reload_dirs=["./"],
        reload_includes=["*.py"],
        reload_excludes=["*.log", "*.tmp", "__pycache__/*"]
    )
```from your_file_manager import FileManager
   file_manager = FileManager() @app.on_event("startup")
   async def startup_cleanup():
       import asyncio
       async def periodic_cleanup():
           while True:
               await asyncio.sleep(3600)  # Run every hour
               await file_manager.cleanup_expired_files()
       
       asyncio.create_task(periodic_cleanup()) # Add these imports to your main.py file if not already present:

import os
import re
from fastapi.responses import FileResponse

# Add these endpoints to your main.py file

@app.get(”/api/v1/files/download/{filename}”)
async def download_file(
filename: str,
current_user: dict = Depends(get_current_user),
manager: TheManagerAgent = Depends(get_manager_agent)
):
“”“Download exported files”””
try:
# Security: Validate filename and check user permissions
if not re.match(r’^[a-zA-Z0-9_-.]+$’, filename):
raise HTTPException(status_code=400, detail=“Invalid filename”)

```
    # Get file info and verify user ownership
    file_info = await file_manager.get_file_info(filename, current_user["user_id"])
    
    if not file_info:
        raise HTTPException(status_code=404, detail="File not found")
    
    # Check if file exists on disk
    file_path = file_info.get("file_path")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found on disk")
    
    # Determine content type based on file extension
    content_type_map = {
        '.json': 'application/json',
        '.csv': 'text/csv',
        '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        '.pdf': 'application/pdf',
        '.txt': 'text/plain',
        '.zip': 'application/zip'
    }
    
    file_ext = os.path.splitext(filename)[1].lower()
    content_type = content_type_map.get(file_ext, 'application/octet-stream')
    
    # Log download activity
    await manager.log_download_activity(
        user_id=current_user["user_id"],
        filename=filename,
        file_size=os.path.getsize(file_path)
    )
    
    # Return file response
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type=content_type,
        headers={
            "Content-Disposition": f"attachment; filename={filename}",
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )
    
except HTTPException:
    raise
except Exception as e:
    logging.error(f"Download error for {filename}: {str(e)}")
    raise HTTPException(status_code=500, detail="Download failed")
```

@app.get(”/api/v1/files/download/{filename}/info”)
async def get_download_file_info(
filename: str,
current_user: dict = Depends(get_current_user)
):
“”“Get information about a downloadable file”””
try:
file_info = await file_manager.get_file_info(filename, current_user[“user_id”])

```
    if not file_info:
        raise HTTPException(status_code=404, detail="File not found")
    
    return {
        "success": True,
        "file_info": {
            "filename": filename,
            "size": file_info.get("size", 0),
            "created_at": file_info.get("created_at"),
            "expires_at": file_info.get("expires_at"),
            "download_count": file_info.get("download_count", 0),
            "content_type": file_info.get("content_type"),
            "is_expired": file_info.get("is_expired", False)
        },
        "timestamp": datetime.utcnow().isoformat()
    }
    
except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))
```

@app.delete(”/api/v1/files/download/{filename}”)
async def delete_download_file(
filename: str,
current_user: dict = Depends(get_current_user)
):
“”“Delete a download file”””
try:
success = await file_manager.delete_file(filename, current_user[“user_id”])

```
    if not success:
        raise HTTPException(status_code=404, detail="File not found")
    
    return {
        "success": True,
        "message": f"File {filename} deleted successfully",
        "timestamp": datetime.utcnow().isoformat()
    }
    
except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))
```

@app.get(”/api/v1/files/downloads”)
async def list_download_files(
limit: int = 50,
offset: int = 0,
current_user: dict = Depends(get_current_user)
):
“”“List all download files for the current user”””
try:
files = await file_manager.list_user_files(
user_id=current_user[“user_id”],
limit=limit,
offset=offset
)

```
    return {
        "success": True,
        "files": files,
        "count": len(files),
        "limit": limit,
        "offset": offset,
        "timestamp": datetime.utcnow().isoformat()
    }
    
except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))
```import os
import json
import aiofiles
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

class FileManager:
def **init**(self, download_dir: str = “downloads”, max_file_age_hours: int = 24):
self.download_dir = Path(download_dir)
self.max_file_age = timedelta(hours=max_file_age_hours)
self.file_registry = {}  # In production, use a database

```
    # Create download directory if it doesn't exist
    self.download_dir.mkdir(exist_ok=True, parents=True)

async def create_download_file(
    self, 
    data: Any, 
    filename: str, 
    user_id: str,
    content_type: str = "application/json"
) -> str:
    """Create a downloadable file and return its path"""
    try:
        # Ensure filename is safe
        safe_filename = self._sanitize_filename(filename)
        file_path = self.download_dir / f"{user_id}_{safe_filename}"
        
        # Write data to file based on content type
        if content_type == "application/json":
            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(data, indent=2, default=str))
        elif content_type == "text/csv":
            # Handle CSV data
            if isinstance(data, list) and len(data) > 0:
                import csv
                import io
                
                output = io.StringIO()
                if isinstance(data[0], dict):
                    writer = csv.DictWriter(output, fieldnames=data[0].keys())
                    writer.writeheader()
                    writer.writerows(data)
                else:
                    writer = csv.writer(output)
                    writer.writerows(data)
                
                async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                    await f.write(output.getvalue())
        else:
            # Handle other content types (binary data, etc.)
            async with aiofiles.open(file_path, 'wb') as f:
                if isinstance(data, str):
                    await f.write(data.encode('utf-8'))
                else:
                    await f.write(data)
        
        # Register file in registry
        file_info = {
            "filename": safe_filename,
            "file_path": str(file_path),
            "user_id": user_id,
            "created_at": datetime.utcnow().isoformat(),
            "expires_at": (datetime.utcnow() + self.max_file_age).isoformat(),
            "size": os.path.getsize(file_path),
            "content_type": content_type,
            "download_count": 0
        }
        
        self.file_registry[safe_filename] = file_info
        
        return str(file_path)
        
    except Exception as e:
        raise Exception(f"Failed to create download file: {str(e)}")

async def get_file_info(self, filename: str, user_id: str) -> Optional[Dict]:
    """Get file information"""
    file_info = self.file_registry.get(filename)
    
    if not file_info or file_info["user_id"] != user_id:
        return None
    
    # Check if file is expired
    expires_at = datetime.fromisoformat(file_info["expires_at"])
    file_info["is_expired"] = datetime.utcnow() > expires_at
    
    return file_info

async def delete_file(self, filename: str, user_id: str) -> bool:
    """Delete a file"""
    try:
        file_info = await self.get_file_info(filename, user_id)
        if not file_info:
            return False
        
        file_path = file_info["file_path"]
        if os.path.exists(file_path):
            os.remove(file_path)
        
        # Remove from registry
        if filename in self.file_registry:
            del self.file_registry[filename]
        
        return True
        
    except Exception as e:
        print(f"Error deleting file {filename}: {str(e)}")
        return False

async def list_user_files(
    self, 
    user_id: str, 
    limit: int = 50, 
    offset: int = 0
) -> List[Dict]:
    """List files for a specific user"""
    user_files = [
        file_info for file_info in self.file_registry.values()
        if file_info["user_id"] == user_id
    ]
    
    # Sort by creation date (newest first)
    user_files.sort(
        key=lambda x: datetime.fromisoformat(x["created_at"]), 
        reverse=True
    )
    
    # Apply pagination
    return user_files[offset:offset + limit]

async def cleanup_expired_files(self):
    """Clean up expired files"""
    now = datetime.utcnow()
    expired_files = []
    
    for filename, file_info in self.file_registry.items():
        expires_at = datetime.fromisoformat(file_info["expires_at"])
        if now > expires_at:
            expired_files.append((filename, file_info))
    
    for filename, file_info in expired_files:
        file_path = file_info["file_path"]
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
            del self.file_registry[filename]
            print(f"Cleaned up expired file: {filename}")
        except Exception as e:
            print(f"Error cleaning up {filename}: {str(e)}")
    
    return len(expired_files)

def _sanitize_filename(self, filename: str) -> str:
    """Sanitize filename to prevent path traversal"""
    # Remove any path separators and keep only safe characters
    safe_chars = re.sub(r'[^\w\-_\.]', '_', filename)
    return safe_chars[:100]  # Limit length
```

# Initialize file manager globally

file_manager = FileManager()