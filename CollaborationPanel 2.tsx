import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { X, Eye, Share2, History, ExternalLink, Paperclip, AtSign, Users, MessageCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Textarea } from '@/components/ui/textarea';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Separator } from '@/components/ui/separator';
import { formatDistanceToNow } from 'date-fns';
import { useToast } from '@/hooks/use-toast';
import { apiRequest } from '@/lib/queryClient';

interface CollaborationPanelProps {
  file: {
    id: string;
    name: string;
    originalName: string;
    mimeType: string;
    size: number;
    createdAt: string;
    updatedAt: string;
    status: string;
    metadata?: {
      tags?: string[];
      wordCount?: number;
      pageCount?: number;
      thumbnailPath?: string;
    };
  };
  onClose: () => void;
}

export function CollaborationPanel({ file, onClose }: CollaborationPanelProps) {
  const [newComment, setNewComment] = useState('');
  const { toast } = useToast();
  const queryClient = useQueryClient();

  // Fetch file details with collaboration data
  const { data: fileDetails, isLoading } = useQuery({
    queryKey: ['/api/files', file.id],
    queryFn: async () => {
      const response = await fetch(`/api/files/${file.id}`, {
        headers: {
          'X-User-Id': 'user123',
          'X-User-Name': 'John Doe',
        },
      });
      if (!response.ok) throw new Error('Failed to fetch file details');
      return response.json();
    },
  });

  // Add comment mutation
  const addCommentMutation = useMutation({
    mutationFn: async (content: string) => {
      return apiRequest(`/api/files/${file.id}/comments`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-User-Id': 'user123',
          'X-User-Name': 'John Doe',
        },
        body: JSON.stringify({ content })
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/files', file.id] });
      setNewComment('');
      toast({
        title: "Comment Added",
        description: "Your comment has been added successfully.",
      });
    },
    onError: (error) => {
      toast({
        title: "Error",
        description: "Failed to add comment. Please try again.",
        variant: "destructive",
      });
    },
  });

  const handleAddComment = () => {
    if (!newComment.trim()) return;
    addCommentMutation.mutate(newComment);
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  };

  const getFileIcon = () => {
    if (file.mimeType.includes('pdf')) return '📄';
    if (file.mimeType.startsWith('image/')) return '🖼️';
    if (file.mimeType.includes('document')) return '📝';
    if (file.mimeType.startsWith('text/')) return '📋';
    return '📄';
  };

  return (
    <aside className="w-80 bg-white dark:bg-gray-800 border-l border-gray-200 dark:border-gray-700 flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">File Details</h2>
        <Button variant="ghost" size="sm" onClick={onClose}>
          <X className="w-4 h-4" />
        </Button>
      </div>

      {/* File Details Panel */}
      <div className="p-6 border-b border-gray-200 dark:border-gray-700">
        {isLoading ? (
          <div className="animate-pulse space-y-4">
            <div className="aspect-video bg-gray-200 dark:bg-gray-700 rounded-lg"></div>
            <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded"></div>
            <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-3/4"></div>
          </div>
        ) : (
          <>
            {/* File Preview */}
            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4 mb-4">
              <div className="aspect-video bg-gradient-to-br from-gray-100 to-gray-200 dark:from-gray-600 dark:to-gray-700 rounded-lg flex items-center justify-center mb-3">
                {file.metadata?.thumbnailPath ? (
                  <img 
                    src={`/thumbnails/${file.metadata.thumbnailPath}`} 
                    alt={file.originalName}
                    className="w-full h-full object-cover rounded-lg"
                  />
                ) : (
                  <div className="text-4xl">{getFileIcon()}</div>
                )}
              </div>
              
              <h3 className="font-medium text-gray-900 dark:text-gray-100 mb-2 truncate" title={file.originalName}>
                {file.originalName}
              </h3>
              
              <div className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
                <div className="flex justify-between">
                  <span>Size:</span>
                  <span>{formatFileSize(file.size)}</span>
                </div>
                <div className="flex justify-between">
                  <span>Type:</span>
                  <span>{file.mimeType}</span>
                </div>
                <div className="flex justify-between">
                  <span>Modified:</span>
                  <span>{formatDistanceToNow(new Date(file.updatedAt), { addSuffix: true })}</span>
                </div>
                {fileDetails?.versions && fileDetails.versions.length > 0 && (
                  <div className="flex justify-between">
                    <span>Version:</span>
                    <span>v{fileDetails.versions[0].version}</span>
                  </div>
                )}
              </div>
            </div>

            {/* Metadata Tags */}
            {file.metadata?.tags && file.metadata.tags.length > 0 && (
              <div className="mb-4">
                <h4 className="text-sm font-medium text-gray-900 dark:text-gray-100 mb-2">Extracted Metadata</h4>
                <div className="flex flex-wrap gap-2">
                  {file.metadata.tags.map((tag) => (
                    <Badge key={tag} variant="outline" className="text-xs">
                      {tag}
                    </Badge>
                  ))}
                </div>
              </div>
            )}

            {/* Action Buttons */}
            <div className="space-y-2">
              <Button className="w-full bg-blue-600 hover:bg-blue-700 text-white">
                <ExternalLink className="w-4 h-4 mr-2" />
                Open File
              </Button>
              <div className="grid grid-cols-2 gap-2">
                <Button variant="outline">
                  <Share2 className="w-4 h-4 mr-1" />
                  Share
                </Button>
                <Button variant="outline">
                  <History className="w-4 h-4 mr-1" />
                  History
                </Button>
              </div>
            </div>
          </>
        )}
      </div>

      {/* Collaboration Panel */}
      <div className="flex-1 p-6 overflow-y-auto custom-scrollbar">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">Collaboration</h2>
          <Button variant="outline" size="sm">
            <Users className="w-4 h-4 mr-1" />
            Invite
          </Button>
        </div>

        {/* Active Collaborators */}
        {fileDetails?.activeCollaborators && fileDetails.activeCollaborators.length > 0 && (
          <div className="space-y-3 mb-6">
            <h3 className="text-sm font-medium text-gray-900 dark:text-gray-100">Active Collaborators</h3>
            {fileDetails.activeCollaborators.map((collaborator: any, index: number) => (
              <div key={index} className="flex items-center space-x-3">
                <div className="relative">
                  <div className="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center text-white text-sm font-medium">
                    {collaborator.userName.charAt(0)}
                  </div>
                  <div className="absolute -bottom-1 -right-1 w-3 h-3 bg-green-500 border-2 border-white rounded-full"></div>
                </div>
                <div className="flex-1">
                  <p className="text-sm font-medium text-gray-900 dark:text-gray-100">{collaborator.userName}</p>
                  <p className="text-xs text-green-600 dark:text-green-400">
                    Active • {formatDistanceToNow(new Date(collaborator.lastActivity), { addSuffix: true })}
                  </p>
                </div>
                <div className="flex items-center space-x-1">
                  <Eye className="w-4 h-4 text-green-500" />
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Recent Comments */}
        <div className="mb-6">
          <h3 className="text-sm font-medium text-gray-900 dark:text-gray-100 mb-3 flex items-center">
            <MessageCircle className="w-4 h-4 mr-2" />
            Recent Comments
          </h3>
          <div className="space-y-3">
            {fileDetails?.comments && fileDetails.comments.length > 0 ? (
              fileDetails.comments.map((comment: any) => (
                <Card key={comment.id} className="bg-gray-50 dark:bg-gray-700">
                  <CardContent className="p-3">
                    <div className="flex items-start space-x-2">
                      <div className="w-6 h-6 bg-blue-500 rounded-full flex items-center justify-center text-white text-xs font-medium">
                        {comment.authorName.charAt(0)}
                      </div>
                      <div className="flex-1">
                        <p className="text-sm text-gray-900 dark:text-gray-100 mb-1">
                          "{comment.content}"
                        </p>
                        <p className="text-xs text-gray-500 dark:text-gray-400">
                          {comment.authorName} • {formatDistanceToNow(new Date(comment.createdAt), { addSuffix: true })}
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))
            ) : (
              <p className="text-sm text-gray-500 dark:text-gray-400 text-center py-4">
                No comments yet. Be the first to comment!
              </p>
            )}
          </div>
        </div>

        {/* Add Comment */}
        <div className="space-y-3">
          <Textarea
            placeholder="Add a comment..."
            value={newComment}
            onChange={(e) => setNewComment(e.target.value)}
            rows={3}
            className="resize-none"
          />
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Button variant="ghost" size="sm">
                <Paperclip className="w-4 h-4" />
              </Button>
              <Button variant="ghost" size="sm">
                <AtSign className="w-4 h-4" />
              </Button>
            </div>
            <Button 
              onClick={handleAddComment}
              disabled={!newComment.trim() || addCommentMutation.isPending}
              className="bg-blue-600 hover:bg-blue-700 text-white"
            >
              {addCommentMutation.isPending ? 'Adding...' : 'Comment'}
            </Button>
          </div>
        </div>
      </div>
    </aside>
  );
}
