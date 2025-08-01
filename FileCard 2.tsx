import { useState } from 'react';
import { 
  FileText, 
  Image, 
  Code, 
  Eye, 
  Share2, 
  MoreHorizontal, 
  Download,
  Edit,
  Users,
  Clock
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent } from '@/components/ui/card';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { formatDistanceToNow } from 'date-fns';

interface FileCardProps {
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
    collaborators?: number;
    shares?: number;
  };
  onOpen?: (file: any) => void;
  onShare?: (file: any) => void;
  onDelete?: (file: any) => void;
}

export function FileCard({ file, onOpen, onShare, onDelete }: FileCardProps) {
  const [isHovered, setIsHovered] = useState(false);

  const getFileIcon = () => {
    if (file.mimeType.startsWith('image/')) {
      return <Image className="w-8 h-8 text-purple-600 dark:text-purple-400" />;
    }
    if (file.mimeType.includes('pdf')) {
      return <FileText className="w-8 h-8 text-red-600 dark:text-red-400" />;
    }
    if (file.mimeType.includes('document') || file.mimeType.includes('word')) {
      return <FileText className="w-8 h-8 text-blue-600 dark:text-blue-400" />;
    }
    if (file.mimeType.startsWith('text/') || file.originalName.match(/\.(js|ts|py|java|cpp|css|html)$/)) {
      return <Code className="w-8 h-8 text-green-600 dark:text-green-400" />;
    }
    return <FileText className="w-8 h-8 text-gray-600 dark:text-gray-400" />;
  };

  const getFileTypeColor = () => {
    if (file.mimeType.startsWith('image/')) return 'from-purple-100 to-purple-200 dark:from-purple-900 dark:to-purple-800';
    if (file.mimeType.includes('pdf')) return 'from-red-100 to-red-200 dark:from-red-900 dark:to-red-800';
    if (file.mimeType.includes('document')) return 'from-blue-100 to-blue-200 dark:from-blue-900 dark:to-blue-800';
    if (file.mimeType.startsWith('text/')) return 'from-green-100 to-green-200 dark:from-green-900 dark:to-green-800';
    return 'from-gray-100 to-gray-200 dark:from-gray-700 dark:to-gray-600';
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  };

  const getStatusBadge = () => {
    switch (file.status) {
      case 'processing':
        return <Badge variant="secondary" className="text-orange-600 bg-orange-100 dark:bg-orange-900">Processing</Badge>;
      case 'ready':
        return <Badge variant="secondary" className="text-green-600 bg-green-100 dark:bg-green-900">Ready</Badge>;
      case 'error':
        return <Badge variant="destructive">Error</Badge>;
      default:
        return null;
    }
  };

  const hasLiveCollaboration = file.collaborators && file.collaborators > 0;

  return (
    <Card 
      className="file-card group overflow-hidden hover:shadow-xl transition-all duration-200"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <CardContent className="p-4 space-y-3">
        {/* File Preview */}
        <div className={`aspect-square bg-gradient-to-br ${getFileTypeColor()} rounded-lg flex items-center justify-center relative overflow-hidden`}>
          {file.metadata?.thumbnailPath ? (
            <img 
              src={`/thumbnails/${file.metadata.thumbnailPath}`} 
              alt={file.originalName}
              className="w-full h-full object-cover"
            />
          ) : (
            getFileIcon()
          )}
          
          {/* Hover Overlay */}
          {isHovered && (
            <div className="absolute inset-0 bg-black bg-opacity-20 flex items-center justify-center transition-all duration-200">
              <Button
                variant="secondary"
                size="sm"
                className="bg-white bg-opacity-90 hover:bg-opacity-100"
                onClick={() => onOpen?.(file)}
              >
                <Eye className="w-4 h-4 mr-1" />
                Preview
              </Button>
            </div>
          )}

          {/* Live Collaboration Indicator */}
          {hasLiveCollaboration && (
            <div className="absolute top-2 right-2 flex -space-x-1">
              <div className="w-4 h-4 bg-green-500 rounded-full border-2 border-white animate-pulse" title="Live collaboration"></div>
              {file.collaborators && file.collaborators > 1 && (
                <div className="w-4 h-4 bg-blue-500 rounded-full border-2 border-white" title={`${file.collaborators} collaborators`}></div>
              )}
            </div>
          )}

          {/* Status Badge */}
          <div className="absolute top-2 left-2">
            {getStatusBadge()}
          </div>
        </div>

        {/* File Info */}
        <div className="space-y-2">
          <h3 className="font-medium text-gray-900 dark:text-gray-100 truncate" title={file.originalName}>
            {file.originalName}
          </h3>
          
          <div className="flex items-center justify-between text-sm text-gray-500 dark:text-gray-400">
            <span>{formatFileSize(file.size)}</span>
            <span>{formatDistanceToNow(new Date(file.updatedAt), { addSuffix: true })}</span>
          </div>

          {/* Metadata */}
          {file.metadata && (
            <div className="space-y-2">
              {file.metadata.wordCount && (
                <div className="text-xs text-gray-500 dark:text-gray-400">
                  {file.metadata.wordCount} words
                  {file.metadata.pageCount && ` • ${file.metadata.pageCount} pages`}
                </div>
              )}
              
              {file.metadata.tags && file.metadata.tags.length > 0 && (
                <div className="flex flex-wrap gap-1">
                  {file.metadata.tags.slice(0, 3).map((tag) => (
                    <Badge key={tag} variant="outline" className="text-xs">
                      {tag}
                    </Badge>
                  ))}
                  {file.metadata.tags.length > 3 && (
                    <Badge variant="outline" className="text-xs">
                      +{file.metadata.tags.length - 3}
                    </Badge>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Collaboration Info */}
          {((file.collaborators && file.collaborators > 0) || (file.shares && file.shares > 0)) && (
            <div className="flex items-center space-x-4 text-xs text-gray-500 dark:text-gray-400">
              {file.collaborators && file.collaborators > 0 && (
                <div className="flex items-center space-x-1">
                  <Users className="w-3 h-3" />
                  <span>{file.collaborators} active</span>
                </div>
              )}
              {file.shares && file.shares > 0 && (
                <div className="flex items-center space-x-1">
                  <Share2 className="w-3 h-3" />
                  <span>{file.shares} shared</span>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Action Buttons */}
        <div className="flex items-center space-x-2 pt-2">
          <Button 
            className="flex-1 bg-blue-600 hover:bg-blue-700 text-white text-sm"
            onClick={() => onOpen?.(file)}
          >
            <Eye className="w-4 h-4 mr-1" />
            Open
          </Button>
          
          <Button
            variant="outline"
            size="sm"
            onClick={() => onShare?.(file)}
          >
            <Share2 className="w-4 h-4" />
          </Button>
          
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" size="sm">
                <MoreHorizontal className="w-4 h-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem onClick={() => onOpen?.(file)}>
                <Eye className="w-4 h-4 mr-2" />
                Preview
              </DropdownMenuItem>
              <DropdownMenuItem>
                <Download className="w-4 h-4 mr-2" />
                Download
              </DropdownMenuItem>
              <DropdownMenuItem>
                <Edit className="w-4 h-4 mr-2" />
                Rename
              </DropdownMenuItem>
              <DropdownMenuItem>
                <Clock className="w-4 h-4 mr-2" />
                Version History
              </DropdownMenuItem>
              <DropdownMenuItem 
                className="text-red-600 dark:text-red-400"
                onClick={() => onDelete?.(file)}
              >
                Delete
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </CardContent>
    </Card>
  );
}
