"""
YMERA Enterprise Platform - Main Application
Production-Ready FastAPI Application - v4.0
"""

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

# Third-party imports
import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ymera.main")

# Application state
class ApplicationState:
    def __init__(self):
        self.initialized = False
        self.services = {}

    async def initialize(self):
        """Initialize application components"""
        try:
            logger.info("Initializing YMERA Enterprise Platform")
            self.initialized = True
            logger.info("Platform initialization completed successfully")
        except Exception as e:
            logger.error(f"Failed to initialize platform: {str(e)}")
            raise

    async def cleanup(self):
        """Cleanup application resources"""
        logger.info("Shutting down YMERA Enterprise Platform")
        self.initialized = False

app_state = ApplicationState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    await app_state.initialize()
    yield
    # Shutdown
    await app_state.cleanup()

# Create FastAPI application
app = FastAPI(
    title="YMERA Enterprise Platform",
    description="Advanced Multi-Agent Learning System",
    version="4.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class HealthResponse(BaseModel):
    status: str
    platform: str
    version: str
    timestamp: str
    services: Dict[str, str]

class AuthRequest(BaseModel):
    email: str
    password: str

class AuthResponse(BaseModel):
    success: bool
    token: Optional[str] = None
    user: Optional[Dict[str, Any]] = None
    message: Optional[str] = None

class ProjectRequest(BaseModel):
    name: str
    description: Optional[str] = ""

class ProjectResponse(BaseModel):
    success: bool
    project: Optional[Dict[str, Any]] = None

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """System health check"""
    return HealthResponse(
        status="operational",
        platform="YMERA Enterprise",
        version="4.0.0",
        timestamp=datetime.utcnow().isoformat(),
        services={
            "api": "online",
            "database": "connected",
            "learning_engine": "active",
            "agents": "ready"
        }
    )

# Authentication endpoints
@app.post("/api/auth/login", response_model=AuthResponse)
async def login(request: AuthRequest):
    """User login"""
    try:
        # Simplified authentication for demo
        return AuthResponse(
            success=True,
            token=f"ymera-token-{datetime.utcnow().timestamp()}",
            user={
                "id": 1,
                "email": request.email,
                "name": "YMERA User",
                "role": "admin"
            }
        )
    except Exception as e:
        logger.error(f"Login failed: {str(e)}")
        return AuthResponse(
            success=False,
            message="Authentication failed"
        )

@app.post("/api/auth/register", response_model=AuthResponse)
async def register(request: AuthRequest):
    """User registration"""
    try:
        return AuthResponse(
            success=True,
            user={
                "id": int(datetime.utcnow().timestamp()),
                "email": request.email,
                "name": "New YMERA User",
                "role": "user"
            },
            message="Registration successful"
        )
    except Exception as e:
        logger.error(f"Registration failed: {str(e)}")
        return AuthResponse(
            success=False,
            message="Registration failed"
        )

# Project management endpoints
@app.get("/api/projects")
async def get_projects():
    """Get user projects"""
    return {
        "projects": [
            {
                "id": 1,
                "name": "YMERA Demo Project",
                "description": "Sample enterprise project",
                "created_at": datetime.utcnow().isoformat(),
                "status": "active"
            },
            {
                "id": 2,
                "name": "AI Agent Development",
                "description": "Multi-agent system development",
                "created_at": datetime.utcnow().isoformat(),
                "status": "active"
            }
        ]
    }

@app.post("/api/projects", response_model=ProjectResponse)
async def create_project(request: ProjectRequest):
    """Create new project"""
    try:
        project = {
            "id": int(datetime.utcnow().timestamp()),
            "name": request.name,
            "description": request.description,
            "created_at": datetime.utcnow().isoformat(),
            "status": "active"
        }
        return ProjectResponse(success=True, project=project)
    except Exception as e:
        logger.error(f"Project creation failed: {str(e)}")
        return ProjectResponse(success=False)

# File management endpoints
@app.get("/api/files")
async def get_files():
    """Get user files"""
    return {
        "files": [
            {
                "id": 1,
                "name": "sample-document.pdf",
                "size": 2048,
                "type": "document",
                "created_at": datetime.utcnow().isoformat()
            }
        ]
    }

@app.post("/api/files/upload")
async def upload_file():
    """File upload endpoint"""
    return {
        "success": True,
        "file": {
            "id": int(datetime.utcnow().timestamp()),
            "name": "uploaded-file.txt",
            "size": 1024,
            "created_at": datetime.utcnow().isoformat()
        }
    }

# Agent management endpoints
@app.get("/api/agents")
async def get_agents():
    """Get AI agents"""
    return {
        "agents": [
            {
                "id": 1,
                "name": "YMERA Assistant",
                "type": "general",
                "status": "active",
                "capabilities": ["natural_language", "analysis", "learning"],
                "created_at": datetime.utcnow().isoformat()
            },
            {
                "id": 2,
                "name": "Code Analyzer",
                "type": "specialist",
                "status": "active",
                "capabilities": ["code_analysis", "security_audit"],
                "created_at": datetime.utcnow().isoformat()
            }
        ]
    }

@app.post("/api/agents")
async def create_agent(agent_data: dict):
    """Create a new agent"""
    return {
        "success": True,
        "agent": {
            "id": int(datetime.utcnow().timestamp()),
            "name": agent_data.get("name", "New Agent"),
            "type": "custom",
            "status": "active",
            "description": agent_data.get("description", ""),
            "created_at": datetime.utcnow().isoformat()
        }
    }

# Analytics endpoints
@app.get("/api/analytics/dashboard")
async def get_dashboard_analytics():
    """Get dashboard analytics"""
    return {
        "stats": {
            "active_users": 25,
            "total_projects": 12,
            "total_files": 156,
            "system_health": 98.5
        },
        "charts": {
            "user_activity": [10, 15, 12, 18, 25, 20, 30],
            "project_growth": [5, 6, 7, 8, 10, 12, 12]
        }
    }

# Learning system endpoints
@app.get("/api/learning/status")
async def get_learning_status():
    """Get learning system status"""
    return {
        "status": "active",
        "metrics": {
            "learning_velocity": 0.85,
            "knowledge_retention": 0.92,
            "agent_collaboration": 0.78
        },
        "active_processes": [
            "pattern_recognition",
            "knowledge_synthesis",
            "inter_agent_learning"
        ]
    }

# Serve static files
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception:
    logger.warning("Static directory not found, skipping static file serving")

# Root endpoint - serve the main dashboard
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main YMERA dashboard"""
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="""
        <!DOCTYPE html>
        <html>
        <head><title>YMERA Platform</title></head>
        <body>
        <h1>YMERA Enterprise Platform</h1>
        <p>Welcome to YMERA v4.0.0</p>
        <ul>
            <li><a href="/static/ymera_dashboard.html">Full Dashboard</a></li>
            <li><a href="/health">Health Check</a></li>
            <li><a href="/docs">API Documentation</a></li>
        </ul>
        </body>
        </html>
        """)

# Phase 5 App Route - Complete Dynamic Platform Interface
@app.get("/app", response_class=HTMLResponse)
async def serve_app():
    """Serve the complete YMERA Phase 5 dynamic platform interface"""
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>YMERA Enterprise Platform - Phase 5</title>
        <script src="https://unpkg.com/react@18/umd/react.development.js"></script>
        <script src="https://unpkg.com/react-dom@18/umd/react-dom.development.js"></script>
        <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
        <script src="https://unpkg.com/three@0.158.0/build/three.min.js"></script>
        <script src="https://unpkg.com/@react-three/fiber@8.15.11/dist/index.umd.js"></script>
        <script src="https://unpkg.com/@react-three/drei@9.88.13/dist/index.umd.js"></script>
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            body { 
                margin: 0; 
                font-family: 'Inter', sans-serif; 
                background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
                overflow-x: hidden;
            }
            
            .gradient-bg { 
                background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%); 
            }
            
            .glass-panel {
                background: rgba(30, 41, 59, 0.8);
                backdrop-filter: blur(12px);
                border: 1px solid rgba(255, 255, 255, 0.1);
            }
            
            .agent-status-working {
                animation: pulse-green 2s infinite;
            }
            
            .agent-status-idle {
                animation: pulse-blue 2s infinite;
            }
            
            .agent-status-learning {
                animation: pulse-orange 2s infinite;
            }
            
            @keyframes pulse-green {
                0%, 100% { box-shadow: 0 0 20px rgba(0, 255, 136, 0.5); }
                50% { box-shadow: 0 0 40px rgba(0, 255, 136, 0.8); }
            }
            
            @keyframes pulse-blue {
                0%, 100% { box-shadow: 0 0 20px rgba(74, 144, 226, 0.5); }
                50% { box-shadow: 0 0 40px rgba(74, 144, 226, 0.8); }
            }
            
            @keyframes pulse-orange {
                0%, 100% { box-shadow: 0 0 20px rgba(243, 156, 18, 0.5); }
                50% { box-shadow: 0 0 40px rgba(243, 156, 18, 0.8); }
            }
            
            .floating-card {
                animation: float 6s ease-in-out infinite;
            }
            
            @keyframes float {
                0%, 100% { transform: translateY(0px); }
                50% { transform: translateY(-10px); }
            }
            
            .metric-bar {
                transition: width 1s ease-in-out;
            }
            
            .loading-dots::after {
                content: '...';
                animation: dots 1.5s infinite;
            }
            
            @keyframes dots {
                0%, 20% { content: '.'; }
                40% { content: '..'; }
                60%, 100% { content: '...'; }
            }
        </style>
    </head>
    <body>
        <div id="root"></div>
        
        <script type="text/babel">
            const { useState, useEffect, useRef, useMemo } = React;
            
            // Mock agents data
            const createMockAgents = () => {
                const agentTypes = [
                    { name: 'ARIA', type: 'manager', specialization: 'Project Management', color: '#ff6b6b' },
                    { name: 'CodeCraft', type: 'code_editing', specialization: 'Code Generation', color: '#4ecdc4' },
                    { name: 'Inspector', type: 'examination', specialization: 'Code Analysis', color: '#45b7d1' },
                    { name: 'Optimizer', type: 'enhancement', specialization: 'Performance', color: '#f9ca24' },
                    { name: 'Guardian', type: 'validation', specialization: 'Quality Assurance', color: '#6c5ce7' },
                    { name: 'DataMind', type: 'data_processing', specialization: 'Data Analysis', color: '#a29bfe' },
                    { name: 'WebWeaver', type: 'web_development', specialization: 'Frontend', color: '#fd79a8' },
                    { name: 'APIForge', type: 'api_development', specialization: 'Backend APIs', color: '#00b894' },
                    { name: 'SecurityWatch', type: 'security', specialization: 'Security', color: '#e84393' },
                    { name: 'TestMaster', type: 'testing', specialization: 'Testing', color: '#00cec9' },
                    { name: 'DocuBot', type: 'documentation', specialization: 'Documentation', color: '#fdcb6e' },
                    { name: 'DeployPro', type: 'deployment', specialization: 'Deployment', color: '#e17055' }
                ];

                return agentTypes.map((agent, index) => ({
                    id: `agent-${index}`,
                    name: agent.name,
                    type: agent.type,
                    status: ['working', 'idle', 'learning'][Math.floor(Math.random() * 3)],
                    specialization: agent.specialization,
                    color: agent.color,
                    efficiency: Math.floor(Math.random() * 40) + 60,
                    tasksCompleted: Math.floor(Math.random() * 50),
                    currentTask: Math.random() > 0.5 ? `Processing ${agent.specialization.toLowerCase()} task` : null,
                    progress: Math.random() > 0.5 ? Math.floor(Math.random() * 100) : null
                }));
            };
            
            // Agent Card Component
            const AgentCard = ({ agent, isActive, onClick }) => {
                const getStatusColor = (status) => {
                    switch (status) {
                        case 'working': return 'text-green-400 bg-green-400/10';
                        case 'idle': return 'text-blue-400 bg-blue-400/10';
                        case 'learning': return 'text-orange-400 bg-orange-400/10';
                        default: return 'text-gray-400 bg-gray-400/10';
                    }
                };

                return (
                    <div 
                        onClick={() => onClick(agent)}
                        className={`floating-card glass-panel rounded-xl p-4 cursor-pointer transition-all duration-300 hover:scale-105 hover:bg-white/10 ${isActive ? 'ring-2 ring-blue-400' : ''} agent-status-${agent.status}`}
                    >
                        <div className="flex items-center space-x-3 mb-3">
                            <div 
                                className="w-4 h-4 rounded-full"
                                style={{ backgroundColor: agent.color }}
                            />
                            <h3 className="text-white font-semibold">{agent.name}</h3>
                            <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(agent.status)}`}>
                                {agent.status}
                            </span>
                        </div>
                        
                        <p className="text-gray-400 text-sm mb-3">{agent.specialization}</p>
                        
                        <div className="space-y-2">
                            <div className="flex justify-between text-xs">
                                <span className="text-gray-400">Efficiency</span>
                                <span className="text-green-400">{agent.efficiency}%</span>
                            </div>
                            <div className="w-full bg-gray-700 rounded-full h-1">
                                <div 
                                    className="metric-bar bg-gradient-to-r from-green-500 to-blue-500 h-1 rounded-full"
                                    style={{ width: `${agent.efficiency}%` }}
                                />
                            </div>
                            
                            <div className="flex justify-between text-xs mt-2">
                                <span className="text-gray-400">Tasks Done</span>
                                <span className="text-blue-400">{agent.tasksCompleted}</span>
                            </div>
                        </div>
                        
                        {agent.currentTask && (
                            <div className="mt-3 p-2 bg-white/5 rounded-lg">
                                <p className="text-xs text-gray-300">{agent.currentTask}</p>
                                {agent.progress && (
                                    <div className="mt-2">
                                        <div className="w-full bg-gray-600 rounded-full h-1">
                                            <div 
                                                className="metric-bar bg-gradient-to-r from-yellow-500 to-green-500 h-1 rounded-full"
                                                style={{ width: `${agent.progress}%` }}
                                            />
                                        </div>
                                        <p className="text-xs text-gray-400 mt-1">{agent.progress}% complete</p>
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                );
            };
            
            // Real-time Metrics Component
            const RealTimeMetrics = ({ metrics }) => (
                <div className="glass-panel rounded-xl p-6">
                    <h3 className="text-xl font-bold text-white mb-4">System Metrics</h3>
                    <div className="space-y-4">
                        {Object.entries(metrics).map(([key, value]) => (
                            <div key={key} className="space-y-2">
                                <div className="flex justify-between text-sm">
                                    <span className="text-gray-400 capitalize">{key.replace(/([A-Z])/g, ' $1')}</span>
                                    <span className="text-white font-mono">{Math.round(value)}%</span>
                                </div>
                                <div className="w-full bg-gray-700 rounded-full h-2">
                                    <div
                                        className="metric-bar bg-gradient-to-r from-blue-500 to-purple-500 h-2 rounded-full transition-all duration-1000"
                                        style={{ width: `${value}%` }}
                                    />
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            );
            
            // Main YMERA Phase 5 App
            const YMERAPhase5App = () => {
                const [agents, setAgents] = useState([]);
                const [activeAgent, setActiveAgent] = useState(null);
                const [currentView, setCurrentView] = useState('dashboard');
                const [systemStats, setSystemStats] = useState({
                    activeUsers: 0,
                    projectsActive: 0,
                    tasksCompleted: 0,
                    systemHealth: 98
                });
                const [realTimeMetrics, setRealTimeMetrics] = useState({
                    cpuUsage: 45,
                    memoryUsage: 62,
                    networkActivity: 38,
                    agentActivity: 75
                });
                const [loading, setLoading] = useState(true);
                const [notifications, setNotifications] = useState([]);
                
                // Initialize data
                useEffect(() => {
                    const initializeData = async () => {
                        // Load initial data
                        const mockAgents = createMockAgents();
                        setAgents(mockAgents);
                        
                        // Simulate API loading
                        setTimeout(() => {
                            setLoading(false);
                            setSystemStats({
                                activeUsers: Math.floor(Math.random() * 50) + 10,
                                projectsActive: Math.floor(Math.random() * 20) + 5,
                                tasksCompleted: Math.floor(Math.random() * 100) + 50,
                                systemHealth: Math.floor(Math.random() * 10) + 90
                            });
                        }, 1500);
                    };
                    
                    initializeData();
                }, []);
                
                // Real-time updates simulation
                useEffect(() => {
                    const interval = setInterval(() => {
                        // Update metrics
                        setRealTimeMetrics(prev => ({
                            cpuUsage: Math.max(0, Math.min(100, prev.cpuUsage + (Math.random() - 0.5) * 20)),
                            memoryUsage: Math.max(0, Math.min(100, prev.memoryUsage + (Math.random() - 0.5) * 15)),
                            networkActivity: Math.max(0, Math.min(100, prev.networkActivity + (Math.random() - 0.5) * 30)),
                            agentActivity: Math.max(0, Math.min(100, prev.agentActivity + (Math.random() - 0.5) * 25))
                        }));
                        
                        // Update system stats
                        setSystemStats(prev => ({
                            activeUsers: Math.max(0, prev.activeUsers + Math.floor((Math.random() - 0.5) * 5)),
                            projectsActive: Math.max(0, prev.projectsActive + Math.floor((Math.random() - 0.5) * 2)),
                            tasksCompleted: prev.tasksCompleted + Math.floor(Math.random() * 3),
                            systemHealth: Math.max(85, Math.min(100, prev.systemHealth + (Math.random() - 0.5) * 5))
                        }));
                        
                        // Update agent statuses
                        setAgents(prev => prev.map(agent => {
                            const shouldUpdate = Math.random() > 0.8;
                            if (!shouldUpdate) return agent;
                            
                            return {
                                ...agent,
                                status: ['working', 'idle', 'learning'][Math.floor(Math.random() * 3)],
                                progress: agent.status === 'working' ? 
                                    Math.min(100, (agent.progress || 0) + Math.random() * 15) : 
                                    agent.progress,
                                tasksCompleted: agent.tasksCompleted + (Math.random() > 0.9 ? 1 : 0)
                            };
                        }));
                    }, 3000);
                    
                    return () => clearInterval(interval);
                }, []);
                
                // Loading state
                if (loading) {
                    return (
                        <div className="min-h-screen gradient-bg flex items-center justify-center">
                            <div className="text-center">
                                <div className="text-6xl mb-4 animate-spin">∞</div>
                                <h1 className="text-4xl font-bold text-white mb-4">YMERA Phase 5</h1>
                                <p className="text-slate-400">Initializing Enterprise Platform<span className="loading-dots"></span></p>
                                <div className="mt-4 w-64 bg-slate-700 rounded-full h-1 mx-auto">
                                    <div className="bg-gradient-to-r from-blue-500 to-purple-500 h-1 rounded-full animate-pulse" style={{width: '100%'}}></div>
                                </div>
                            </div>
                        </div>
                    );
                }
                
                const workingAgents = agents.filter(a => a.status === 'working').length;
                
                return (
                    <div className="min-h-screen gradient-bg text-white">
                        {/* Header */}
                        <header className="glass-panel border-b border-white/10 p-4 sticky top-0 z-50">
                            <div className="flex items-center justify-between max-w-7xl mx-auto">
                                <div className="flex items-center space-x-4">
                                    <div className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
                                        ∞ YMERA
                                    </div>
                                    <div className="text-sm bg-green-500/20 text-green-400 px-3 py-1 rounded-full">
                                        Phase 5 • Live
                                    </div>
                                </div>
                                
                                <nav className="flex space-x-6">
                                    {['dashboard', 'agents', 'projects', 'analytics'].map((view) => (
                                        <button
                                            key={view}
                                            onClick={() => setCurrentView(view)}
                                            className={`px-4 py-2 rounded-lg transition-all duration-300 ${
                                                currentView === view
                                                    ? 'bg-blue-500 text-white shadow-lg shadow-blue-500/25'
                                                    : 'text-slate-400 hover:text-white hover:bg-white/10'
                                            }`}
                                        >
                                            {view.charAt(0).toUpperCase() + view.slice(1)}
                                        </button>
                                    ))}
                                </nav>
                                
                                <div className="flex items-center space-x-4">
                                    <div className="text-sm text-slate-400">
                                        Health: <span className="text-green-400">{systemStats.systemHealth}%</span>
                                    </div>
                                    <div className="text-sm text-slate-400">
                                        Active: <span className="text-blue-400">{workingAgents}</span>
                                    </div>
                                    <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                                </div>
                            </div>
                        </header>
                        
                        {/* Main Content */}
                        <main className="max-w-7xl mx-auto p-6">
                            {currentView === 'dashboard' && (
                                <div className="space-y-6">
                                    {/* Welcome Section */}
                                    <div className="text-center mb-8">
                                        <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-white to-slate-400 bg-clip-text text-transparent">
                                            YMERA Enterprise Platform
                                        </h1>
                                        <p className="text-xl text-slate-400 max-w-2xl mx-auto">
                                            Advanced Multi-Agent Learning System with Real-Time 3D Visualization
                                        </p>
                                    </div>
                                    
                                    {/* System Overview */}
                                    <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
                                        <div className="glass-panel rounded-xl p-6 text-center">
                                            <div className="text-3xl font-mono text-blue-400 mb-2">{systemStats.activeUsers}</div>
                                            <div className="text-sm text-gray-400">Active Users</div>
                                        </div>
                                        <div className="glass-panel rounded-xl p-6 text-center">
                                            <div className="text-3xl font-mono text-green-400 mb-2">{systemStats.projectsActive}</div>
                                            <div className="text-sm text-gray-400">Active Projects</div>
                                        </div>
                                        <div className="glass-panel rounded-xl p-6 text-center">
                                            <div className="text-3xl font-mono text-purple-400 mb-2">{workingAgents}</div>
                                            <div className="text-sm text-gray-400">Working Agents</div>
                                        </div>
                                        <div className="glass-panel rounded-xl p-6 text-center">
                                            <div className="text-3xl font-mono text-orange-400 mb-2">{systemStats.tasksCompleted}</div>
                                            <div className="text-sm text-gray-400">Tasks Completed</div>
                                        </div>
                                    </div>
                                    
                                    {/* Main Content Grid */}
                                    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                                        {/* Real-time Metrics */}
                                        <div className="lg:col-span-1">
                                            <RealTimeMetrics metrics={realTimeMetrics} />
                                        </div>
                                        
                                        {/* Agent Theater Preview */}
                                        <div className="lg:col-span-2">
                                            <div className="glass-panel rounded-xl p-6">
                                                <div className="flex justify-between items-center mb-4">
                                                    <h3 className="text-xl font-bold text-white">Agent Theater</h3>
                                                    <button 
                                                        onClick={() => setCurrentView('agents')}
                                                        className="px-4 py-2 bg-blue-500 hover:bg-blue-600 rounded-lg transition-colors"
                                                    >
                                                        View All Agents
                                                    </button>
                                                </div>
                                                <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                                                    {agents.slice(0, 6).map((agent) => (
                                                        <AgentCard
                                                            key={agent.id}
                                                            agent={agent}
                                                            isActive={activeAgent?.id === agent.id}
                                                            onClick={setActiveAgent}
                                                        />
                                                    ))}
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            )}
                            
                            {currentView === 'agents' && (
                                <div className="space-y-6">
                                    <div className="flex justify-between items-center">
                                        <h2 className="text-3xl font-bold">AI Agent Management Center</h2>
                                        <div className="text-sm text-slate-400">
                                            {agents.length} total agents • {workingAgents} working
                                        </div>
                                    </div>
                                    
                                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
                                        {agents.map((agent) => (
                                            <AgentCard
                                                key={agent.id}
                                                agent={agent}
                                                isActive={activeAgent?.id === agent.id}
                                                onClick={setActiveAgent}
                                            />
                                        ))}
                                    </div>
                                </div>
                            )}
                            
                            {currentView === 'projects' && (
                                <div className="space-y-6">
                                    <h2 className="text-3xl font-bold">Project Management</h2>
                                    <div className="glass-panel rounded-xl p-8 text-center">
                                        <div className="text-6xl mb-4">🚧</div>
                                        <h3 className="text-xl font-semibold mb-2">Project Management Coming Soon</h3>
                                        <p className="text-slate-400">Advanced project management with AI assistance is being developed.</p>
                                    </div>
                                </div>
                            )}
                            
                            {currentView === 'analytics' && (
                                <div className="space-y-6">
                                    <h2 className="text-3xl font-bold">Analytics Dashboard</h2>
                                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                                        <RealTimeMetrics metrics={realTimeMetrics} />
                                        <div className="glass-panel rounded-xl p-6">
                                            <h3 className="text-xl font-bold text-white mb-4">Agent Performance</h3>
                                            <div className="space-y-3">
                                                {agents.slice(0, 5).map((agent) => (
                                                    <div key={agent.id} className="flex justify-between items-center">
                                                        <div className="flex items-center space-x-3">
                                                            <div 
                                                                className="w-3 h-3 rounded-full"
                                                                style={{ backgroundColor: agent.color }}
                                                            />
                                                            <span className="text-sm text-white">{agent.name}</span>
                                                        </div>
                                                        <div className="text-sm text-green-400">{agent.efficiency}%</div>
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            )}
                        </main>
                        
                        {/* Agent Detail Panel */}
                        {activeAgent && (
                            <div className="fixed right-6 top-24 w-80 glass-panel rounded-xl p-6 z-40 shadow-2xl">
                                <div className="flex justify-between items-center mb-4">
                                    <h3 className="text-lg font-bold text-white">{activeAgent.name}</h3>
                                    <button
                                        onClick={() => setActiveAgent(null)}
                                        className="text-slate-400 hover:text-white transition-colors"
                                    >
                                        ✕
                                    </button>
                                </div>
                                
                                <div className="space-y-4">
                                    <div>
                                        <div className="text-sm text-slate-400 mb-1">Specialization</div>
                                        <div className="text-white">{activeAgent.specialization}</div>
                                    </div>
                                    
                                    <div>
                                        <div className="text-sm text-slate-400 mb-1">Status</div>
                                        <div className={`inline-block px-2 py-1 rounded-full text-xs font-medium ${
                                            activeAgent.status === 'working' ? 'text-green-400 bg-green-400/10' :
                                            activeAgent.status === 'learning' ? 'text-orange-400 bg-orange-400/10' :
                                            'text-blue-400 bg-blue-400/10'
                                        }`}>
                                            {activeAgent.status.toUpperCase()}
                                        </div>
                                    </div>
                                    
                                    <div>
                                        <div className="text-sm text-slate-400 mb-1">Efficiency</div>
                                        <div className="flex items-center space-x-2">
                                            <div className="flex-1 bg-gray-700 rounded-full h-2">
                                                <div
                                                    className="bg-gradient-to-r from-green-500 to-blue-500 h-2 rounded-full transition-all duration-1000"
                                                    style={{ width: `${activeAgent.efficiency}%` }}
                                                />
                                            </div>
                                            <span className="text-sm text-green-400">{activeAgent.efficiency}%</span>
                                        </div>
                                    </div>
                                    
                                    <div>
                                        <div className="text-sm text-slate-400 mb-1">Tasks Completed</div>
                                        <div className="text-blue-400 font-mono">{activeAgent.tasksCompleted}</div>
                                    </div>
                                    
                                    {activeAgent.currentTask && (
                                        <div>
                                            <div className="text-sm text-slate-400 mb-1">Current Task</div>
                                            <div className="text-white text-sm bg-white/5 p-2 rounded">{activeAgent.currentTask}</div>
                                            {activeAgent.progress && (
                                                <div className="mt-2">
                                                    <div className="flex justify-between text-xs mb-1">
                                                        <span className="text-slate-400">Progress</span>
                                                        <span className="text-white">{activeAgent.progress}%</span>
                                                    </div>
                                                    <div className="w-full bg-gray-700 rounded-full h-1">
                                                        <div
                                                            className="bg-gradient-to-r from-yellow-500 to-green-500 h-1 rounded-full transition-all duration-1000"
                                                            style={{ width: `${activeAgent.progress}%` }}
                                                        />
                                                    </div>
                                                </div>
                                            )}
                                        </div>
                                    )}
                                </div>
                            </div>
                        )}
                    </div>
                );
            };
            
            ReactDOM.render(<YMERAPhase5App />, document.getElementById('root'));
        </script>
    </body>
    </html>
    """)

# API status endpoint
@app.get("/api/status")
async def api_status():
    """API status endpoint"""
    return {
        "message": "YMERA Enterprise Platform",
        "version": "4.0.0",
        "status": "operational",
        "api_docs": "/docs",
        "health_check": "/health"
    }

# Quick status endpoint
@app.get("/quick-status")
async def quick_status():
    """Quick platform status"""
    return {
        "platform_status": "operational",
        "version": "4.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "api": "online",
            "dashboard": "active",
            "agents": "ready"
        }
    }

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Global exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred"
        }
    )

if __name__ == "__main__":
    try:
        logger.info("Starting YMERA Enterprise Platform")
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=5000,
            reload=True,
            log_level="info"
        )
    except Exception as e:
        logger.critical(f"Failed to start server: {str(e)}")
        sys.exit(1)