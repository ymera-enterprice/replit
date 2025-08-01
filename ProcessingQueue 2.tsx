import { useState, useEffect } from 'react';
import { X, FileText, Image, Code, AlertCircle, CheckCircle } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';

interface ProcessingFile {
  id: string;
  name: string;
  size: number;
  mimeType: string;
  status: 'uploading' | 'processing' | 'extracting' | 'indexing' | 'complete' | 'error';
  progress: number;
  stage: string;
  error?: string;
}

export function ProcessingQueue() {
  const [processingFiles, setProcessingFiles] = useState<ProcessingFile[]>([]);

  // Simulate processing files (in real app, this would come from WebSocket)
  useEffect(() => {
    const mockFiles: ProcessingFile[] = [
      {
        id: '1',
        name: 'quarterly-report-2024.pdf',
        size: 2400000,
        mimeType: 'application/pdf',
        status: 'extracting',
        progress: 67,
        stage: 'Extracting metadata and text content...',
      },
      {
        id: '2',
        name: 'project-presentation.pptx',
        size: 8900000,
        mimeType: 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
        status: 'processing',
        progress: 34,
        stage: 'Processing slides and extracting content...',
      },
    ];

    setProcessingFiles(mockFiles);

    // Simulate progress updates
    const interval = setInterval(() => {
      setProcessingFiles(prev => 
        prev.map(file => {
          if (file.status === 'complete' || file.status === 'error') return file;
          
          const newProgress = Math.min(file.progress + Math.random() * 15, 100);
          let newStatus = file.status;
          let newStage = file.stage;
          
          if (newProgress >= 100) {
            newStatus = 'complete' as 'uploading' | 'processing' | 'extracting' | 'indexing';
            newStage = 'Processing complete';
          } else if (newProgress >= 80 && file.status === 'extracting') {
            newStatus = 'indexing';
            newStage = 'Building search index...';
          }
          
          return {
            ...file,
            progress: newProgress,
            status: newStatus,
            stage: newStage,
          };
        })
      );
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  const getFileIcon = (mimeType: string) => {
    if (mimeType.startsWith('image/')) return <Image className="w-5 h-5 text-purple-600" />;
    if (mimeType.includes('pdf')) return <FileText className="w-5 h-5 text-red-600" />;
    if (mimeType.includes('document') || mimeType.includes('presentation')) return <FileText className="w-5 h-5 text-blue-600" />;
    if (mimeType.startsWith('text/') || mimeType.includes('code')) return <Code className="w-5 h-5 text-green-600" />;
    return <FileText className="w-5 h-5 text-gray-600" />;
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'complete':
        return <CheckCircle className="w-4 h-4 text-green-600" />;
      case 'error':
        return <AlertCircle className="w-4 h-4 text-red-600" />;
      default:
        return null;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'uploading': return 'text-blue-600';
      case 'processing': return 'text-orange-600';
      case 'extracting': return 'text-purple-600';
      case 'indexing': return 'text-indigo-600';
      case 'complete': return 'text-green-600';
      case 'error': return 'text-red-600';
      default: return 'text-gray-600';
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case 'uploading': return 'Uploading...';
      case 'processing': return 'Processing...';
      case 'extracting': return 'Extracting...';
      case 'indexing': return 'Indexing...';
      case 'complete': return 'Complete';
      case 'error': return 'Failed';
      default: return status;
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  };

  const removeFile = (id: string) => {
    setProcessingFiles(prev => prev.filter(file => file.id !== id));
  };

  if (processingFiles.length === 0) {
    return null;
  }

  return (
    <div className="mb-6 space-y-3">
      <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 flex items-center">
        <div className="w-2 h-2 bg-orange-500 rounded-full animate-pulse mr-2"></div>
        Processing Queue
      </h2>
      
      {processingFiles.map((file) => (
        <Card key={file.id} className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700">
          <CardContent className="p-4">
            <div className="flex items-center space-x-4">
              <div className="flex-shrink-0">
                {getFileIcon(file.mimeType)}
              </div>
              
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="font-medium text-gray-900 dark:text-gray-100 truncate" title={file.name}>
                    {file.name}
                  </h3>
                  <div className="flex items-center space-x-2">
                    {getStatusIcon(file.status)}
                    <span className={`text-sm font-medium ${getStatusColor(file.status)}`}>
                      {getStatusText(file.status)}
                    </span>
                    {(file.status === 'complete' || file.status === 'error') && (
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => removeFile(file.id)}
                        className="h-6 w-6 p-0"
                      >
                        <X className="w-4 h-4" />
                      </Button>
                    )}
                  </div>
                </div>
                
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-xs text-gray-500 dark:text-gray-400">
                    <span>{file.stage}</span>
                    <span>{formatFileSize(file.size)}</span>
                  </div>
                  
                  {file.status !== 'error' && file.status !== 'complete' && (
                    <div className="space-y-1">
                      <Progress 
                        value={file.progress} 
                        className="h-2"
                      />
                      <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400">
                        <span>{Math.round(file.progress)}% complete</span>
                        <Badge variant="outline" className="text-xs px-2 py-0">
                          {file.status}
                        </Badge>
                      </div>
                    </div>
                  )}
                  
                  {file.status === 'error' && file.error && (
                    <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-2">
                      <p className="text-sm text-red-700 dark:text-red-400">
                        {file.error}
                      </p>
                    </div>
                  )}
                  
                  {file.status === 'complete' && (
                    <div className="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg p-2">
                      <p className="text-sm text-green-700 dark:text-green-400">
                        File processed successfully and ready for use
                      </p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}
