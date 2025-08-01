import express from 'express';
import { createServer } from 'http';
import { storage } from './storage.js';

// Create YMERA Enterprise Platform server
const app = express();
const httpServer = createServer(app);

// Basic middleware
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: false }));

// Simple CORS handling
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept, Authorization');
  if (req.method === 'OPTIONS') {
    res.sendStatus(200);
  } else {
    next();
  }
});

// Request logging
app.use((req, res, next) => {
  const start = Date.now();
  res.on('finish', () => {
    const duration = Date.now() - start;
    if (req.path.startsWith('/api')) {
      console.log(`${req.method} ${req.path} ${res.statusCode} in ${duration}ms`);
    }
  });
  next();
});

// Health check endpoint
app.get("/api/health", (req, res) => {
  res.json({ 
    status: "healthy", 
    timestamp: new Date().toISOString(),
    version: "2.0.0"
  });
});

// Dashboard endpoint
app.get("/api/dashboard", async (req, res) => {
  try {
    const stats = await storage.getDashboardStats();
    
    res.json({
      success: true,
      data: {
        stats,
        systemMetrics: []
      }
    });
  } catch (error) {
    console.error("Dashboard error:", error);
    res.status(500).json({ 
      success: false, 
      error: "Failed to load dashboard data" 
    });
  }
});

// Agents endpoints
app.get("/api/agents", async (req, res) => {
  try {
    const result = await storage.getAgents({ 
      limit: 20,
      offset: 0 
    });
    
    res.json({
      success: true,
      data: result.agents,
      pagination: {
        page: 1,
        limit: 20,
        total: result.total,
        totalPages: Math.ceil(result.total / 20)
      }
    });
  } catch (error) {
    console.error("Get agents error:", error);
    res.status(500).json({ 
      success: false, 
      error: "Failed to fetch agents" 
    });
  }
});

// Projects endpoints
app.get("/api/projects", async (req, res) => {
  try {
    const result = await storage.getProjects({ 
      limit: 20,
      offset: 0 
    });
    
    res.json({
      success: true,
      data: result.projects,
      pagination: {
        page: 1,
        limit: 20,
        total: result.total,
        totalPages: Math.ceil(result.total / 20)
      }
    });
  } catch (error) {
    console.error("Get projects error:", error);
    res.status(500).json({ 
      success: false, 
      error: "Failed to fetch projects" 
    });
  }
});

// WebSocket placeholder
app.get("/api/websocket/connections", (req, res) => {
  res.json({
    success: true,
    data: {
      activeConnections: 0,
      totalConnections: 0
    }
  });
});

// Basic HTML page for testing
app.get("/", (req, res) => {
  res.send(`
    <!DOCTYPE html>
    <html>
    <head>
      <title>YMERA Enterprise Platform</title>
      <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .status { color: green; }
        .api-list { background: #f5f5f5; padding: 20px; border-radius: 5px; }
        .endpoint { margin: 10px 0; }
      </style>
    </head>
    <body>
      <h1>🚀 YMERA Enterprise Platform</h1>
      <p class="status">✅ Server is running successfully!</p>
      
      <h2>Available API Endpoints:</h2>
      <div class="api-list">
        <div class="endpoint">GET /api/health - Health check</div>
        <div class="endpoint">GET /api/dashboard - Dashboard data</div>
        <div class="endpoint">GET /api/agents - List agents</div>
        <div class="endpoint">GET /api/projects - List projects</div>
        <div class="endpoint">GET /api/websocket/connections - WebSocket status</div>
      </div>
      
      <h2>System Information:</h2>
      <ul>
        <li>Environment: ${process.env.NODE_ENV || 'development'}</li>
        <li>Version: 2.0.0</li>
        <li>Database: ${process.env.DATABASE_URL ? 'Connected' : 'Not configured'}</li>
        <li>Port: ${process.env.PORT || 5000}</li>
      </ul>
    </body>
    </html>
  `);
});

// Error handler
app.use((error: any, req: express.Request, res: express.Response, next: express.NextFunction) => {
  console.error('Server error:', error);
  res.status(500).json({
    success: false,
    error: 'Internal server error'
  });
});

// 404 handler
app.use((req, res) => {
  res.status(404).json({
    success: false,
    error: 'Endpoint not found'
  });
});

const port = parseInt(process.env.PORT || '5000', 10);
httpServer.listen(port, '0.0.0.0', () => {
  console.log(`🚀 YMERA Enterprise Platform running on port ${port}`);
  console.log(`📊 Dashboard: http://localhost:${port}`);
  console.log(`🔌 API Base: http://localhost:${port}/api`);
  console.log(`💾 Database: ${process.env.DATABASE_URL ? 'Connected' : 'Not configured'}`);
});
