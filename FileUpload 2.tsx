import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, X, FileText, Image, Code, FileArchive } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { useToast } from '@/hooks/use-toast';
import { apiRequest } from '@/lib/queryClient';

interface FileUploadProps {
  onUploadComplete?: (file: any) => void;
  onUploadStart?: (file: File) => void;
}

interface UploadingFile {
  file: File;
  progress: number;
  status: 'uploading' | 'processing' | 'complete' | 'error';
  id?: string;
  error?: string;
}

export function FileUpload({ onUploadComplete, onUploadStart }: FileUploadProps) {
  const [uploadingFiles, setUploadingFiles] = useState<UploadingFile[]>([]);
  const { toast } = useToast();

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    for (const file of acceptedFiles) {
      const uploadingFile: UploadingFile = {
        file,
        progress: 0,
        status: 'uploading',
      };

      setUploadingFiles(prev => [...prev, uploadingFile]);
      onUploadStart?.(file);

      try {
        await uploadFile(file, uploadingFile);
      } catch (error) {
        console.error('Upload failed:', error);
        setUploadingFiles(prev => 
          prev.map(f => 
            f.file === file 
              ? { ...f, status: 'error', error: error instanceof Error ? error.message : 'Upload failed' }
              : f
          )
        );
        toast({
          title: "Upload Failed",
          description: `Failed to upload ${file.name}`,
          variant: "destructive",
        });
      }
    }
  }, [onUploadStart, toast]);

  const uploadFile = async (file: File, uploadingFile: UploadingFile) => {
    const formData = new FormData();
    formData.append('file', file);

    try {
      // Simulate progress updates
      const progressInterval = setInterval(() => {
        setUploadingFiles(prev => 
          prev.map(f => 
            f.file === file && f.progress < 90
              ? { ...f, progress: f.progress + Math.random() * 20 }
              : f
          )
        );
      }, 500);

      const response = await fetch('/api/files/upload', {
        method: 'POST',
        body: formData,
        headers: {
          'X-User-Id': 'user123',
          'X-User-Name': 'Demo User',
        },
      });
      
      if (!response.ok) {
        throw new Error('Upload failed');
      }
      
      const result = await response.json();

      clearInterval(progressInterval);

      setUploadingFiles(prev => 
        prev.map(f => 
          f.file === file 
            ? { ...f, progress: 100, status: 'processing', id: result.file.id }
            : f
        )
      );

      // Simulate processing delay
      setTimeout(() => {
        setUploadingFiles(prev => 
          prev.map(f => 
            f.file === file 
              ? { ...f, status: 'complete' }
              : f
          )
        );

        onUploadComplete?.(result.file);
        
        toast({
          title: "Upload Complete",
          description: `${file.name} has been uploaded and is being processed.`,
        });

        // Remove from list after a delay
        setTimeout(() => {
          setUploadingFiles(prev => prev.filter(f => f.file !== file));
        }, 3000);
      }, 2000);

    } catch (error) {
      throw error;
    }
  };

  const removeUploadingFile = (file: File) => {
    setUploadingFiles(prev => prev.filter(f => f.file !== file));
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    multiple: true,
    maxSize: 100 * 1024 * 1024, // 100MB
  });

  const getFileIcon = (file: File) => {
    if (file.type.startsWith('image/')) return <Image className="w-6 h-6" />;
    if (file.type.startsWith('text/') || file.type.includes('document')) return <FileText className="w-6 h-6" />;
    if (file.type.includes('code') || file.name.match(/\.(js|ts|py|java|cpp|css|html)$/)) return <Code className="w-6 h-6" />;
    if (file.type.includes('zip') || file.type.includes('archive')) return <FileArchive className="w-6 h-6" />;
    return <FileText className="w-6 h-6" />;
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'uploading': return 'text-blue-600';
      case 'processing': return 'text-orange-600';
      case 'complete': return 'text-green-600';
      case 'error': return 'text-red-600';
      default: return 'text-gray-600';
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case 'uploading': return 'Uploading...';
      case 'processing': return 'Processing...';
      case 'complete': return 'Complete';
      case 'error': return 'Failed';
      default: return '';
    }
  };

  return (
    <div className="space-y-6">
      {/* Drop Zone */}
      <div
        {...getRootProps()}
        className={`
          drop-zone border-2 border-dashed rounded-xl p-8 text-center transition-all duration-200 cursor-pointer
          ${isDragActive 
            ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20 drag-over' 
            : 'border-gray-300 dark:border-gray-600 bg-gray-50 dark:bg-gray-800 hover:border-blue-400 hover:bg-blue-50 dark:hover:bg-blue-900/10'
          }
        `}
      >
        <input {...getInputProps()} />
        <div className="space-y-4">
          <div className="w-16 h-16 bg-gradient-to-br from-blue-600 to-orange-500 rounded-full mx-auto flex items-center justify-center animate-float">
            <Upload className="w-8 h-8 text-white" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-2">
              Drop files here to upload
            </h3>
            <p className="text-gray-600 dark:text-gray-400 mb-4">
              or click to browse • Supports PDF, DOCX, TXT, Images, Code files
            </p>
            <Button className="bg-blue-600 hover:bg-blue-700 text-white">
              Choose Files
            </Button>
          </div>
        </div>
      </div>

      {/* Uploading Files */}
      {uploadingFiles.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
            Processing Queue
          </h3>
          {uploadingFiles.map((uploadingFile, index) => (
            <div key={index} className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
              <div className="flex items-center space-x-4">
                <div className="flex-shrink-0">
                  {getFileIcon(uploadingFile.file)}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="text-sm font-medium text-gray-900 dark:text-gray-100 truncate">
                      {uploadingFile.file.name}
                    </h4>
                    <div className="flex items-center space-x-2">
                      <span className={`text-sm font-medium ${getStatusColor(uploadingFile.status)}`}>
                        {getStatusText(uploadingFile.status)}
                      </span>
                      {uploadingFile.status === 'error' && (
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => removeUploadingFile(uploadingFile.file)}
                        >
                          <X className="w-4 h-4" />
                        </Button>
                      )}
                    </div>
                  </div>
                  
                  {uploadingFile.status !== 'error' ? (
                    <div className="space-y-1">
                      <Progress 
                        value={uploadingFile.progress} 
                        className="h-2"
                      />
                      <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400">
                        <span>{Math.round(uploadingFile.progress)}% complete</span>
                        <span>{(uploadingFile.file.size / 1024 / 1024).toFixed(1)} MB</span>
                      </div>
                    </div>
                  ) : (
                    <p className="text-sm text-red-600 dark:text-red-400">
                      {uploadingFile.error || 'Upload failed'}
                    </p>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
