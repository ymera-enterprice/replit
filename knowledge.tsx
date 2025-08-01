import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Brain, Search, Plus, GitBranch, Database, BookOpen, Lightbulb, Share2, Filter } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import KnowledgeGraph from '@/components/knowledge/knowledge-graph';
import type { KnowledgeItem } from '@/types';

export default function Knowledge() {
  const [searchTerm, setSearchTerm] = useState('');
  const [typeFilter, setTypeFilter] = useState<string>('all');
  const [page, setPage] = useState(1);

  const { data: knowledgeData, isLoading, error } = useQuery({
    queryKey: ['/api/knowledge', { 
      page, 
      limit: 12, 
      type: typeFilter !== 'all' ? typeFilter : undefined,
      search: searchTerm || undefined 
    }],
    refetchInterval: 30000,
  });

  const getTypeIcon = (type: string) => {
    switch (type.toLowerCase()) {
      case 'model':
        return Brain;
      case 'pattern':
        return GitBranch;
      case 'insight':
        return Lightbulb;
      case 'rule':
        return BookOpen;
      case 'data':
        return Database;
      default:
        return Brain;
    }
  };

  const getTypeColor = (type: string) => {
    switch (type.toLowerCase()) {
      case 'model':
        return 'bg-primary/20 text-primary border-primary/30';
      case 'pattern':
        return 'bg-secondary/20 text-secondary border-secondary/30';
      case 'insight':
        return 'bg-accent/20 text-accent border-accent/30';
      case 'rule':
        return 'bg-success/20 text-success border-success/30';
      case 'data':
        return 'bg-warning/20 text-warning border-warning/30';
      default:
        return 'bg-dark-500/20 text-dark-300 border-dark-500/30';
    }
  };

  if (isLoading) {
    return (
      <div className="p-6 space-y-8">
        <div className="flex justify-between items-center">
          <Skeleton className="h-8 w-48" />
          <Skeleton className="h-10 w-32" />
        </div>
        
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <Skeleton className="h-96" />
          </div>
          <Skeleton className="h-96" />
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6 flex items-center justify-center min-h-[60vh]">
        <Card className="glass-card p-6">
          <div className="text-center">
            <div className="text-error text-xl mb-2">Error Loading Knowledge Base</div>
            <p className="text-muted-foreground">Failed to load knowledge data. Please try again.</p>
          </div>
        </Card>
      </div>
    );
  }

  const knowledgeItems = knowledgeData?.data || [];
  const pagination = knowledgeData?.pagination;

  // Filter knowledge items based on search term
  const filteredItems = knowledgeItems.filter((item: KnowledgeItem) => {
    const matchesSearch = !searchTerm || 
      item.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
      item.content.toLowerCase().includes(searchTerm.toLowerCase()) ||
      item.tags.some(tag => tag.toLowerCase().includes(searchTerm.toLowerCase()));
    return matchesSearch;
  });

  const typeCounts = {
    all: knowledgeItems.length,
    model: knowledgeItems.filter((k: KnowledgeItem) => k.type === 'model').length,
    pattern: knowledgeItems.filter((k: KnowledgeItem) => k.type === 'pattern').length,
    insight: knowledgeItems.filter((k: KnowledgeItem) => k.type === 'insight').length,
    rule: knowledgeItems.filter((k: KnowledgeItem) => k.type === 'rule').length,
    data: knowledgeItems.filter((k: KnowledgeItem) => k.type === 'data').length,
  };

  return (
    <div className="p-6 space-y-8 animate-fade-in">
      {/* Header */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div>
          <h1 className="text-3xl font-bold mb-2">Knowledge Graph</h1>
          <p className="text-muted-foreground">
            Explore and manage the collective intelligence of your AI agents
          </p>
        </div>
        
        <Button className="ymera-gradient">
          <Plus className="w-4 h-4 mr-2" />
          Add Knowledge
        </Button>
      </div>

      {/* Knowledge Graph Visualization */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <KnowledgeGraph nodeCount={1847} connectionCount={3291} />
        </div>

        {/* Knowledge Stats */}
        <div className="space-y-6">
          <Card className="glass-card">
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Database className="h-5 w-5 text-primary" />
                <span>Knowledge Statistics</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="text-center">
                  <div className="text-2xl font-bold text-primary">1,847</div>
                  <div className="text-xs text-dark-400">Total Items</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-secondary">3,291</div>
                  <div className="text-xs text-dark-400">Connections</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-success">94.2%</div>
                  <div className="text-xs text-dark-400">Graph Density</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-accent">247</div>
                  <div className="text-xs text-dark-400">Recent Adds</div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="glass-card">
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <GitBranch className="h-5 w-5 text-primary" />
                <span>Knowledge Types</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {Object.entries(typeCounts).filter(([type]) => type !== 'all').map(([type, count]) => {
                const TypeIcon = getTypeIcon(type);
                return (
                  <div key={type} className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <TypeIcon className="w-4 h-4 text-dark-400" />
                      <span className="text-sm text-dark-300 capitalize">{type}</span>
                    </div>
                    <span className="text-sm font-medium text-dark-200">{count}</span>
                  </div>
                );
              })}
            </CardContent>
          </Card>

          <Card className="glass-card">
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Share2 className="h-5 w-5 text-primary" />
                <span>Recent Activity</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {[
                { action: 'New pattern discovered', time: '2 min ago', type: 'pattern' },
                { action: 'Model updated', time: '5 min ago', type: 'model' },
                { action: 'Knowledge consolidated', time: '12 min ago', type: 'insight' },
                { action: 'Rule refined', time: '18 min ago', type: 'rule' }
              ].map((activity, index) => (
                <div key={index} className="flex items-start space-x-3">
                  <div className="w-2 h-2 rounded-full bg-primary mt-2"></div>
                  <div className="flex-1">
                    <p className="text-sm text-dark-200">{activity.action}</p>
                    <p className="text-xs text-dark-400">{activity.time}</p>
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Knowledge Items List */}
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-semibold text-dark-100">Knowledge Items</h2>
          <div className="flex items-center space-x-2">
            <span className="text-sm text-dark-400">
              {filteredItems.length} of {knowledgeItems.length} items
            </span>
          </div>
        </div>

        {/* Search and Filters */}
        <div className="flex flex-col md:flex-row gap-4">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground w-4 h-4" />
            <Input
              placeholder="Search knowledge items..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-10 glass-card border-input"
            />
          </div>
          
          <Select value={typeFilter} onValueChange={setTypeFilter}>
            <SelectTrigger className="w-40 glass-card">
              <SelectValue placeholder="Type" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Types</SelectItem>
              <SelectItem value="model">Models</SelectItem>
              <SelectItem value="pattern">Patterns</SelectItem>
              <SelectItem value="insight">Insights</SelectItem>
              <SelectItem value="rule">Rules</SelectItem>
              <SelectItem value="data">Data</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* Knowledge Items Grid */}
        {filteredItems.length === 0 ? (
          <Card className="glass-card p-12 text-center">
            <Brain className="w-16 h-16 text-dark-500 mx-auto mb-4" />
            <div className="text-muted-foreground text-lg mb-2">No knowledge items found</div>
            <p className="text-muted-foreground">
              {searchTerm || typeFilter !== 'all'
                ? 'Try adjusting your search or filters'
                : 'Add your first knowledge item to get started'
              }
            </p>
          </Card>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredItems.map((item: KnowledgeItem) => {
              const TypeIcon = getTypeIcon(item.type);
              return (
                <Card key={item.id} className="glass-card hover:border-primary/50 transition-all duration-300">
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-lg truncate flex items-center space-x-2">
                        <TypeIcon className="w-5 h-5" />
                        <span>{item.title}</span>
                      </CardTitle>
                      <Badge className={getTypeColor(item.type)}>
                        {item.type}
                      </Badge>
                    </div>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <p className="text-sm text-dark-300 line-clamp-3">
                      {item.content}
                    </p>

                    {/* Tags */}
                    {item.tags.length > 0 && (
                      <div className="flex flex-wrap gap-1">
                        {item.tags.slice(0, 3).map((tag, index) => (
                          <Badge 
                            key={index} 
                            variant="outline" 
                            className="text-xs px-2 py-0.5"
                          >
                            {tag}
                          </Badge>
                        ))}
                        {item.tags.length > 3 && (
                          <Badge variant="outline" className="text-xs px-2 py-0.5">
                            +{item.tags.length - 3}
                          </Badge>
                        )}
                      </div>
                    )}

                    {/* Metadata */}
                    <div className="text-xs text-dark-400 space-y-1">
                      <div>Created: {new Date(item.createdAt).toLocaleDateString()}</div>
                      {item.sourceAgentId && (
                        <div>Source: Agent {item.sourceAgentId.slice(0, 8)}...</div>
                      )}
                      {item.isPublic && (
                        <Badge className="bg-success/20 text-success text-xs">
                          Public
                        </Badge>
                      )}
                    </div>

                    {/* Action Buttons */}
                    <div className="flex space-x-2 pt-2">
                      <Button variant="outline" size="sm" className="flex-1">
                        View Details
                      </Button>
                      <Button variant="outline" size="sm">
                        Share
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              );
            })}
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
    </div>
  );
}
