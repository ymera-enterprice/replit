import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Plus, Search, FolderOpen, Calendar, Users, TrendingUp } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Progress } from '@/components/ui/progress';
import { Skeleton } from '@/components/ui/skeleton';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import type { Project } from '@/types';

export default function Projects() {
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [page, setPage] = useState(1);

  const { data: projectsData, isLoading, error } = useQuery({
    queryKey: ['/api/projects', { 
      page, 
      limit: 12, 
      status: statusFilter !== 'all' ? statusFilter : undefined,
    }],
    refetchInterval: 30000,
  });

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active':
        return 'bg-success/20 text-success border-success/30';
      case 'planning':
        return 'bg-primary/20 text-primary border-primary/30';
      case 'paused':
        return 'bg-warning/20 text-warning border-warning/30';
      case 'completed':
        return 'bg-blue-500/20 text-blue-400 border-blue-500/30';
      case 'cancelled':
        return 'bg-error/20 text-error border-error/30';
      default:
        return 'bg-dark-500/20 text-dark-300 border-dark-500/30';
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'urgent':
        return 'bg-error text-white';
      case 'high':
        return 'bg-warning text-dark-900';
      case 'medium':
        return 'bg-primary text-white';
      case 'low':
        return 'bg-dark-500 text-white';
      default:
        return 'bg-dark-500 text-white';
    }
  };

  if (isLoading) {
    return (
      <div className="p-6 space-y-8">
        <div className="flex justify-between items-center">
          <Skeleton className="h-8 w-48" />
          <Skeleton className="h-10 w-32" />
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {Array.from({ length: 6 }).map((_, i) => (
            <Skeleton key={i} className="h-64" />
          ))}
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6 flex items-center justify-center min-h-[60vh]">
        <Card className="glass-card p-6">
          <div className="text-center">
            <div className="text-error text-xl mb-2">Error Loading Projects</div>
            <p className="text-muted-foreground">Failed to load project data. Please try again.</p>
          </div>
        </Card>
      </div>
    );
  }

  const projects = projectsData?.data || [];
  const pagination = projectsData?.pagination;

  // Filter projects based on search term
  const filteredProjects = projects.filter((project: Project) => {
    const matchesSearch = !searchTerm || 
      project.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      (project.description && project.description.toLowerCase().includes(searchTerm.toLowerCase()));
    return matchesSearch;
  });

  const statusCounts = {
    all: projects.length,
    active: projects.filter((p: Project) => p.status === 'active').length,
    planning: projects.filter((p: Project) => p.status === 'planning').length,
    paused: projects.filter((p: Project) => p.status === 'paused').length,
    completed: projects.filter((p: Project) => p.status === 'completed').length,
  };

  return (
    <div className="p-6 space-y-8 animate-fade-in">
      {/* Header */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div>
          <h1 className="text-3xl font-bold mb-2">Project Management</h1>
          <p className="text-muted-foreground">
            Manage your AI-powered projects and collaborations
          </p>
        </div>
        
        <Button className="ymera-gradient">
          <Plus className="w-4 h-4 mr-2" />
          Create Project
        </Button>
      </div>

      {/* Status Overview */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        {Object.entries(statusCounts).map(([status, count]) => (
          <Card
            key={status}
            className={`glass-card cursor-pointer transition-all hover:bg-white/5 ${
              statusFilter === status ? 'ring-2 ring-primary' : ''
            }`}
            onClick={() => setStatusFilter(status)}
          >
            <CardContent className="p-4 text-center">
              <div className="text-2xl font-bold mb-1">{count}</div>
              <div className="text-sm text-muted-foreground capitalize">
                {status === 'all' ? 'Total' : status}
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Search and Filters */}
      <div className="flex flex-col md:flex-row gap-4">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground w-4 h-4" />
          <Input
            placeholder="Search projects..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-10 glass-card border-input"
          />
        </div>
        
        <Select value={statusFilter} onValueChange={setStatusFilter}>
          <SelectTrigger className="w-40 glass-card">
            <SelectValue placeholder="Status" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Statuses</SelectItem>
            <SelectItem value="planning">Planning</SelectItem>
            <SelectItem value="active">Active</SelectItem>
            <SelectItem value="paused">Paused</SelectItem>
            <SelectItem value="completed">Completed</SelectItem>
            <SelectItem value="cancelled">Cancelled</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {/* Projects Grid */}
      {filteredProjects.length === 0 ? (
        <Card className="glass-card p-12 text-center">
          <FolderOpen className="w-16 h-16 text-dark-500 mx-auto mb-4" />
          <div className="text-muted-foreground text-lg mb-2">No projects found</div>
          <p className="text-muted-foreground">
            {searchTerm || statusFilter !== 'all'
              ? 'Try adjusting your search or filters'
              : 'Create your first project to get started'
            }
          </p>
          {!searchTerm && statusFilter === 'all' && (
            <Button className="mt-4 ymera-gradient">
              <Plus className="w-4 h-4 mr-2" />
              Create Project
            </Button>
          )}
        </Card>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredProjects.map((project: Project) => (
            <Card key={project.id} className="glass-card hover:border-primary/50 transition-all duration-300">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle className="text-lg truncate">{project.name}</CardTitle>
                  <Badge className={getPriorityColor(project.priority)}>
                    {project.priority}
                  </Badge>
                </div>
                <div className="flex items-center space-x-2">
                  <Badge className={getStatusColor(project.status)}>
                    {project.status}
                  </Badge>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                {project.description && (
                  <p className="text-sm text-dark-300 line-clamp-2">
                    {project.description}
                  </p>
                )}

                {/* Progress */}
                <div>
                  <div className="flex justify-between text-sm mb-2">
                    <span className="text-dark-300">Progress</span>
                    <span className="font-medium">{project.progress}%</span>
                  </div>
                  <Progress value={project.progress} className="h-2" />
                </div>

                {/* Project Metadata */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-sm">
                    <div className="flex items-center space-x-1 text-dark-400">
                      <Calendar className="w-3 h-3" />
                      <span>Created</span>
                    </div>
                    <span className="text-dark-300">
                      {new Date(project.createdAt).toLocaleDateString()}
                    </span>
                  </div>
                  
                  <div className="flex items-center justify-between text-sm">
                    <div className="flex items-center space-x-1 text-dark-400">
                      <Users className="w-3 h-3" />
                      <span>Owner</span>
                    </div>
                    <span className="text-dark-300">User #{project.ownerId}</span>
                  </div>

                  <div className="flex items-center justify-between text-sm">
                    <div className="flex items-center space-x-1 text-dark-400">
                      <TrendingUp className="w-3 h-3" />
                      <span>Last Updated</span>
                    </div>
                    <span className="text-dark-300">
                      {new Date(project.updatedAt).toLocaleDateString()}
                    </span>
                  </div>
                </div>

                {/* Action Buttons */}
                <div className="flex space-x-2 pt-2">
                  <Button variant="outline" size="sm" className="flex-1">
                    View Details
                  </Button>
                  {project.status === 'active' && (
                    <Button variant="outline" size="sm">
                      Pause
                    </Button>
                  )}
                  {project.status === 'paused' && (
                    <Button variant="outline" size="sm">
                      Resume
                    </Button>
                  )}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* Pagination */}
      {pagination && pagination.totalPages > 1 && (
        <div className="flex justify-center items-center space-x-2">
          <Button
            variant="outline"
            disabled={page === 1}
            onClick={() => setPage(page - 1)}
          >
            Previous
          </Button>
          <span className="text-sm text-muted-foreground">
            Page {page} of {pagination.totalPages}
          </span>
          <Button
            variant="outline"
            disabled={page === pagination.totalPages}
            onClick={() => setPage(page + 1)}
          >
            Next
          </Button>
        </div>
      )}
    </div>
  );
}
