import { useState, useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { useToast } from "@/hooks/use-toast";
import { useWebSocket } from "@/hooks/useWebSocket";
import { isUnauthorizedError } from "@/lib/authUtils";
import { apiRequest } from "@/lib/queryClient";
import { 
  Bot, 
  Brain, 
  Network, 
  GraduationCap,
  Activity,
  Settings,
  Zap,
  MessageCircle,
  Play,
  Pause,
  RotateCcw,
  TrendingUp,
  Database,
  Cpu,
  HardDrive
} from "lucide-react";

export default function Phase3Dashboard() {
  const [selectedAgent, setSelectedAgent] = useState<string>('');
  const [agentMessage, setAgentMessage] = useState('');
  const [knowledgeFilter, setKnowledgeFilter] = useState('all');
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const { sendMessage: sendWsMessage, messages: wsMessages } = useWebSocket();

  // Fetch AI agents
  const { data: agentsData, isLoading: agentsLoading } = useQuery({
    queryKey: ['/api/agents'],
    refetchInterval: 15000,
    retry: (failureCount, error) => {
      if (isUnauthorizedError(error as Error)) {
        toast({
          title: "Unauthorized",
          description: "You are logged out. Logging in again...",
          variant: "destructive",
        });
        setTimeout(() => {
          window.location.href = "/api/login";
        }, 500);
        return false;
      }
      return failureCount < 3;
    },
  });

  // Fetch agent tasks
  const { data: tasksData } = useQuery({
    queryKey: ['/api/agent-tasks'],
    refetchInterval: 10000,
  });

  // Fetch knowledge nodes
  const { data: knowledgeData } = useQuery({
    queryKey: ['/api/knowledge/nodes', knowledgeFilter === 'all' ? undefined : knowledgeFilter],
    refetchInterval: 30000,
  });

  // Fetch knowledge relationships
  const { data: relationshipsData } = useQuery({
    queryKey: ['/api/knowledge/relationships'],
    refetchInterval: 30000,
  });

  // Create agent mutation
  const createAgentMutation = useMutation({
    mutationFn: async (agentData: any) => {
      return await apiRequest('/api/agents', {
        method: 'POST',
        body: JSON.stringify(agentData),
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/agents'] });
      toast({
        title: "Success",
        description: "Agent created successfully",
      });
    },
    onError: (error) => {
      if (isUnauthorizedError(error as Error)) {
        toast({
          title: "Unauthorized",
          description: "You are logged out. Logging in again...",
          variant: "destructive",
        });
        setTimeout(() => {
          window.location.href = "/api/login";
        }, 500);
        return;
      }
      toast({
        title: "Error",
        description: "Failed to create agent",
        variant: "destructive",
      });
    },
  });

  // Update agent status mutation
  const updateAgentMutation = useMutation({
    mutationFn: async ({ id, updates }: { id: string; updates: any }) => {
      return await apiRequest(`/api/agents/${id}`, {
        method: 'PATCH',
        body: JSON.stringify(updates),
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/agents'] });
    },
  });

  const agents = agentsData?.data || [];
  const tasks = tasksData?.data || [];
  const knowledgeNodes = knowledgeData?.data || [];
  const relationships = relationshipsData?.data || [];

  const handleSendAgentMessage = () => {
    if (selectedAgent && agentMessage.trim()) {
      // Send via WebSocket for real-time agent communication
      sendWsMessage('agent_message', {
        agentId: selectedAgent,
        message: agentMessage,
      });
      
      setAgentMessage('');
      toast({
        title: "Message sent",
        description: "Message sent to agent successfully",
      });
    }
  };

  const handleToggleAgent = (agentId: string, currentStatus: string) => {
    const newStatus = currentStatus === 'active' ? 'idle' : 'active';
    updateAgentMutation.mutate({
      id: agentId,
      updates: { status: newStatus }
    });
  };

  const getAgentStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'bg-green-400';
      case 'learning': return 'bg-blue-400';
      case 'idle': return 'bg-yellow-400';
      case 'error': return 'bg-red-400';
      default: return 'bg-gray-400';
    }
  };

  const getAgentIcon = (type: string) => {
    switch (type) {
      case 'code_analyzer': return Cpu;
      case 'security_scanner': return Shield;
      case 'quality_assurance': return Settings;
      case 'module_manager': return Database;
      case 'performance_monitor': return Activity;
      default: return Bot;
    }
  };

  // Calculate metrics
  const activeAgents = agents.filter((agent: any) => agent.status === 'active').length;
  const totalTasks = tasks.length;
  const avgEfficiency = agents.length > 0 
    ? Math.round(agents.reduce((sum: number, agent: any) => sum + (agent.healthScore || 0), 0) / agents.length)
    : 0;
  const learningNodes = knowledgeNodes.length;

  return (
    <div className="space-y-8">
      {/* AI Agents Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card className="glass-card interactive-card text-center">
          <CardContent className="p-6">
            <Bot className="w-8 h-8 text-primary mx-auto mb-3" />
            <h4 className="font-semibold">Active Agents</h4>
            <p className="text-2xl font-bold gradient-text">{activeAgents}</p>
          </CardContent>
        </Card>
        
        <Card className="glass-card interactive-card text-center">
          <CardContent className="p-6">
            <Zap className="w-8 h-8 text-secondary mx-auto mb-3" />
            <h4 className="font-semibold">Tasks Completed</h4>
            <p className="text-2xl font-bold gradient-text">{totalTasks}</p>
          </CardContent>
        </Card>
        
        <Card className="glass-card interactive-card text-center">
          <CardContent className="p-6">
            <Network className="w-8 h-8 text-accent mx-auto mb-3" />
            <h4 className="font-semibold">Knowledge Nodes</h4>
            <p className="text-2xl font-bold gradient-text">{learningNodes}</p>
          </CardContent>
        </Card>
        
        <Card className="glass-card interactive-card text-center">
          <CardContent className="p-6">
            <TrendingUp className="w-8 h-8 text-purple-400 mx-auto mb-3" />
            <h4 className="font-semibold">Avg Efficiency</h4>
            <p className="text-2xl font-bold gradient-text">{avgEfficiency}%</p>
          </CardContent>
        </Card>
      </div>

      {/* AI Agents Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {agentsLoading ? (
          Array.from({ length: 6 }).map((_, i) => (
            <Skeleton key={i} className="h-48 glass-card" />
          ))
        ) : agents.length === 0 ? (
          <Card className="glass-card lg:col-span-3 p-12 text-center">
            <Bot className="w-16 h-16 mx-auto mb-4 text-white/50" />
            <h3 className="text-xl font-semibold mb-2">No AI Agents Found</h3>
            <p className="text-white/70 mb-6">Deploy your first AI agent to start automation</p>
            <Button 
              onClick={() => createAgentMutation.mutate({
                name: 'Data Processor Agent',
                type: 'code_analyzer',
                description: 'Analyzes code for optimization opportunities',
                status: 'idle'
              })}
              className="ymera-gradient text-black"
              disabled={createAgentMutation.isPending}
            >
              <Bot className="w-4 h-4 mr-2" />
              Deploy Sample Agent
            </Button>
          </Card>
        ) : (
          agents.map((agent: any) => {
            const AgentIcon = getAgentIcon(agent.type);
            return (
              <Card key={agent.id} className="glass-card interactive-card">
                <CardContent className="p-6">
                  <div className="flex items-center justify-between mb-4">
                    <h4 className="font-semibold flex items-center gap-2">
                      <AgentIcon className="w-5 h-5" />
                      {agent.name}
                    </h4>
                    <div className="flex items-center gap-2">
                      <div className={`w-3 h-3 ${getAgentStatusColor(agent.status)} rounded-full status-dot`}></div>
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => handleToggleAgent(agent.id, agent.status)}
                        className="p-1 h-8 w-8"
                      >
                        {agent.status === 'active' ? (
                          <Pause className="w-4 h-4" />
                        ) : (
                          <Play className="w-4 h-4" />
                        )}
                      </Button>
                    </div>
                  </div>
                  
                  <p className="text-white/70 text-sm mb-3">
                    {agent.description || 'AI agent performing automated tasks'}
                  </p>
                  
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Health Score</span>
                      <span className="font-semibold">{agent.healthScore || 100}%</span>
                    </div>
                    <div className="w-full bg-white/20 rounded-full h-2">
                      <div 
                        className="bg-gradient-to-r from-primary to-secondary h-2 rounded-full transition-all duration-300" 
                        style={{ width: `${agent.healthScore || 100}%` }}
                      ></div>
                    </div>
                    
                    <div className="flex justify-between text-xs text-white/60 mt-3">
                      <span>Tasks: {agent.taskCount || 0}</span>
                      <span>Success: {Math.round(agent.successRate || 100)}%</span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            );
          })
        )}
      </div>

      {/* Knowledge Graph Visualization */}
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Network className="w-5 h-5" />
              Knowledge Graph Network
            </div>
            <Select value={knowledgeFilter} onValueChange={setKnowledgeFilter}>
              <SelectTrigger className="w-40 glass-card border-white/20">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Types</SelectItem>
                <SelectItem value="concept">Concepts</SelectItem>
                <SelectItem value="relationship">Relationships</SelectItem>
                <SelectItem value="entity">Entities</SelectItem>
              </SelectContent>
            </Select>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="bg-black/20 rounded-lg p-8 h-64 flex items-center justify-center relative overflow-hidden">
            {knowledgeNodes.length === 0 ? (
              <div className="text-center text-white/60">
                <Network className="w-16 h-16 mx-auto mb-4 opacity-50" />
                <h3 className="text-lg font-semibold mb-2">Knowledge Graph Empty</h3>
                <p>AI agents will populate the knowledge graph as they learn</p>
              </div>
            ) : (
              <div className="grid grid-cols-5 gap-8 opacity-80">
                {knowledgeNodes.slice(0, 5).map((node: any, index: number) => {
                  const icons = [Database, Brain, Network, Settings, Activity];
                  const colors = ['bg-primary', 'bg-secondary', 'bg-accent', 'bg-purple-400', 'bg-green-400'];
                  const Icon = icons[index % icons.length];
                  const color = colors[index % colors.length];
                  
                  return (
                    <div 
                      key={node.id}
                      className={`w-12 h-12 ${color} rounded-full animate-float flex items-center justify-center cursor-pointer hover:scale-110 transition-transform`}
                      style={{ animationDelay: `${index}s` }}
                      title={`${node.type}: ${JSON.stringify(node.data).substring(0, 50)}...`}
                    >
                      <Icon className="w-6 h-6 text-white" />
                    </div>
                  );
                })}
              </div>
            )}
            
            {/* Connection lines between nodes */}
            <svg className="absolute inset-0 w-full h-full pointer-events-none">
              {relationships.slice(0, 3).map((rel: any, index: number) => (
                <line
                  key={rel.id}
                  x1={`${20 + index * 15}%`}
                  y1="50%"
                  x2={`${60 + index * 10}%`}
                  y2="50%"
                  stroke="rgba(251, 191, 36, 0.3)"
                  strokeWidth="2"
                  strokeDasharray="5,5"
                  className="animate-pulse-slow"
                />
              ))}
            </svg>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6 text-center">
            <div>
              <div className="text-lg font-bold gradient-text">{knowledgeNodes.length}</div>
              <div className="text-sm text-white/70">Knowledge Nodes</div>
            </div>
            <div>
              <div className="text-lg font-bold gradient-text">{relationships.length}</div>
              <div className="text-sm text-white/70">Relationships</div>
            </div>
            <div>
              <div className="text-lg font-bold gradient-text">
                {relationships.length > 0 
                  ? Math.round(relationships.reduce((sum: number, rel: any) => sum + (rel.strength || 1), 0) / relationships.length * 100)
                  : 0}%
              </div>
              <div className="text-sm text-white/70">Avg Connection Strength</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Agent Communication Panel */}
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-3">
            <MessageCircle className="w-5 h-5" />
            Agent-to-Agent Communication
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Communication Log */}
          <div className="bg-black/20 rounded-lg p-4 h-40 overflow-y-auto custom-scrollbar">
            <div className="space-y-2 text-sm font-mono">
              {/* Real-time agent communications from WebSocket */}
              {wsMessages
                .filter(msg => msg.type === 'agent_communication')
                .slice(-10)
                .map((msg, index) => (
                  <div key={index} className="text-primary">
                    [{new Date().toLocaleTimeString()}] Agent Communication: {msg.data.message}
                  </div>
                ))}
              
              {/* Simulated system messages */}
              {agents.filter((agent: any) => agent.status === 'active').map((agent: any, index: number) => (
                <div key={agent.id} className="text-secondary">
                  [{new Date().toLocaleTimeString()}] {agent.name}: System health check OK
                </div>
              ))}
              
              {agents.length === 0 && wsMessages.length === 0 && (
                <div className="text-white/60 text-center py-8">
                  <MessageCircle className="w-8 h-8 mx-auto mb-2 opacity-50" />
                  <p>No agent communications yet</p>
                </div>
              )}
            </div>
          </div>
          
          {/* Message Input */}
          <div className="flex space-x-3">
            <Select value={selectedAgent} onValueChange={setSelectedAgent}>
              <SelectTrigger className="w-48 glass-card border-white/20">
                <SelectValue placeholder="Select Agent" />
              </SelectTrigger>
              <SelectContent>
                {agents.map((agent: any) => (
                  <SelectItem key={agent.id} value={agent.id}>
                    {agent.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            
            <Input
              value={agentMessage}
              onChange={(e) => setAgentMessage(e.target.value)}
              placeholder="Send message to agent..."
              className="flex-1 glass-card border-white/20 text-white placeholder-white/50 focus:border-primary"
              onKeyPress={(e) => {
                if (e.key === 'Enter') {
                  e.preventDefault();
                  handleSendAgentMessage();
                }
              }}
              disabled={!selectedAgent}
            />
            
            <Button
              onClick={handleSendAgentMessage}
              disabled={!selectedAgent || !agentMessage.trim()}
              className="ymera-gradient text-black hover:scale-105 transition-transform"
            >
              Send
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Learning Analytics */}
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-3">
            <GraduationCap className="w-5 h-5" />
            Learning Analytics
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            <div className="text-center">
              <div className="text-2xl font-bold gradient-text">
                {tasks.filter((task: any) => task.status === 'completed').length}
              </div>
              <div className="text-sm text-muted-foreground">Learning Sessions</div>
            </div>
            
            <div className="text-center">
              <div className="text-2xl font-bold gradient-text">
                {tasks.length > 0 
                  ? Math.round((tasks.filter((task: any) => task.status === 'completed').length / tasks.length) * 100)
                  : 0}%
              </div>
              <div className="text-sm text-muted-foreground">Success Rate</div>
            </div>
            
            <div className="text-center">
              <div className="text-2xl font-bold gradient-text">
                {agents.length > 0
                  ? Math.round(agents.reduce((sum: number, agent: any) => sum + (agent.taskCount || 0), 0) / agents.length)
                  : 0}
              </div>
              <div className="text-sm text-muted-foreground">Avg Tasks/Agent</div>
            </div>
            
            <div className="text-center">
              <div className="text-2xl font-bold gradient-text">
                {knowledgeNodes.filter((node: any) => 
                  new Date(node.createdAt) > new Date(Date.now() - 24 * 60 * 60 * 1000)
                ).length}
              </div>
              <div className="text-sm text-muted-foreground">New Patterns (24h)</div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
