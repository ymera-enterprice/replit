import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { useToast } from "@/hooks/use-toast";
import { isUnauthorizedError } from "@/lib/authUtils";
import { apiRequest } from "@/lib/queryClient";
import { 
  Shield, 
  FolderOpen, 
  Upload, 
  Download, 
  Plus, 
  Edit, 
  Trash2,
  FileText,
  Image as ImageIcon,
  Video,
  File
} from "lucide-react";

export default function Phase1Dashboard() {
  const [showCreateProject, setShowCreateProject] = useState(false);
  const [projectForm, setProjectForm] = useState({ name: '', description: '' });
  const [selectedFiles, setSelectedFiles] = useState<FileList | null>(null);
  const { toast } = useToast();
  const queryClient = useQueryClient();

  // Fetch projects
  const { data: projectsData, isLoading: projectsLoading } = useQuery({
    queryKey: ['/api/projects'],
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

  // Fetch files
  const { data: filesData, isLoading: filesLoading } = useQuery({
    queryKey: ['/api/files'],
    retry: (failureCount, error) => {
      if (isUnauthorizedError(error as Error)) {
        return false;
      }
      return failureCount < 3;
    },
  });

  // Create project mutation
  const createProjectMutation = useMutation({
    mutationFn: async (projectData: { name: string; description: string }) => {
      return await apiRequest('/api/projects', {
        method: 'POST',
        body: JSON.stringify(projectData),
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/projects'] });
      setShowCreateProject(false);
      setProjectForm({ name: '', description: '' });
      toast({
        title: "Success",
        description: "Project created successfully",
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
        description: "Failed to create project",
        variant: "destructive",
      });
    },
  });

  // Upload files mutation
  const uploadFilesMutation = useMutation({
    mutationFn: async (files: FileList) => {
      const formData = new FormData();
      Array.from(files).forEach(file => {
        formData.append('files', file);
      });
      
      return await fetch('/api/files/upload', {
        method: 'POST',
        body: formData,
      }).then(res => {
        if (!res.ok) throw new Error('Upload failed');
        return res.json();
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/files'] });
      setSelectedFiles(null);
      toast({
        title: "Success",
        description: "Files uploaded successfully",
      });
    },
    onError: (error) => {
      toast({
        title: "Error",
        description: "Failed to upload files",
        variant: "destructive",
      });
    },
  });

  const projects = projectsData?.data || [];
  const files = filesData?.data || [];

  const getFileIcon = (mimeType: string) => {
    if (mimeType?.startsWith('image/')) return ImageIcon;
    if (mimeType?.startsWith('video/')) return Video;
    if (mimeType?.includes('text/') || mimeType?.includes('application/json')) return FileText;
    return File;
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const handleFileUpload = () => {
    if (selectedFiles && selectedFiles.length > 0) {
      uploadFilesMutation.mutate(selectedFiles);
    }
  };

  const handleDownload = async (fileId: string, fileName: string) => {
    try {
      const response = await fetch(`/api/files/${fileId}/download`);
      if (!response.ok) throw new Error('Download failed');
      
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = fileName;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to download file",
        variant: "destructive",
      });
    }
  };

  return (
    <div className="space-y-8">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Authentication System */}
        <Card className="glass-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-3">
              <Shield className="w-5 h-5 text-primary" />
              Authentication System
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex justify-between items-center p-3 bg-white/5 rounded-lg">
              <span>JWT Token Management</span>
              <Badge variant="secondary" className="bg-green-500/20 text-green-400">Active</Badge>
            </div>
            <div className="flex justify-between items-center p-3 bg-white/5 rounded-lg">
              <span>User Sessions</span>
              <span className="text-primary font-semibold">Protected</span>
            </div>
            <div className="flex justify-between items-center p-3 bg-white/5 rounded-lg">
              <span>Security Level</span>
              <Badge variant="secondary" className="bg-green-500/20 text-green-400">High</Badge>
            </div>
          </CardContent>
        </Card>

        {/* Project Management */}
        <Card className="glass-card">
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <FolderOpen className="w-5 h-5 text-secondary" />
                Project Management
              </div>
              <Button
                size="sm"
                className="ymera-gradient text-black"
                onClick={() => setShowCreateProject(true)}
              >
                <Plus className="w-4 h-4 mr-2" />
                New Project
              </Button>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {projectsLoading ? (
              <div className="space-y-3">
                {Array.from({ length: 3 }).map((_, i) => (
                  <Skeleton key={i} className="h-12 w-full" />
                ))}
              </div>
            ) : projects.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">
                <FolderOpen className="w-12 h-12 mx-auto mb-4 opacity-50" />
                <p>No projects yet. Create your first project to get started.</p>
              </div>
            ) : (
              projects.slice(0, 3).map((project: any) => (
                <div key={project.id} className="flex justify-between items-center p-3 bg-white/5 rounded-lg">
                  <div>
                    <span className="font-medium">{project.name}</span>
                    {project.description && (
                      <p className="text-sm text-muted-foreground">{project.description}</p>
                    )}
                  </div>
                  <Badge 
                    variant={project.status === 'active' ? 'default' : 'secondary'}
                    className={project.status === 'active' ? 'bg-green-500/20 text-green-400' : ''}
                  >
                    {project.status}
                  </Badge>
                </div>
              ))
            )}
          </CardContent>
        </Card>
      </div>

      {/* Create Project Modal */}
      {showCreateProject && (
        <Card className="glass-card max-w-md mx-auto">
          <CardHeader>
            <CardTitle>Create New Project</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <label className="text-sm font-medium mb-2 block">Project Name</label>
              <Input
                value={projectForm.name}
                onChange={(e) => setProjectForm(prev => ({ ...prev, name: e.target.value }))}
                placeholder="Enter project name"
                className="glass-card border-input"
              />
            </div>
            <div>
              <label className="text-sm font-medium mb-2 block">Description</label>
              <Textarea
                value={projectForm.description}
                onChange={(e) => setProjectForm(prev => ({ ...prev, description: e.target.value }))}
                placeholder="Enter project description"
                className="glass-card border-input"
                rows={3}
              />
            </div>
            <div className="flex gap-3">
              <Button
                onClick={() => createProjectMutation.mutate(projectForm)}
                disabled={!projectForm.name.trim() || createProjectMutation.isPending}
                className="ymera-gradient text-black flex-1"
              >
                {createProjectMutation.isPending ? 'Creating...' : 'Create'}
              </Button>
              <Button
                variant="outline"
                onClick={() => setShowCreateProject(false)}
                className="border-white/30"
              >
                Cancel
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* File Management System */}
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-3">
            <Upload className="w-5 h-5 text-accent" />
            File Management System
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* File Upload Zone */}
          <div className="border-2 border-dashed border-white/30 rounded-xl p-8 text-center upload-zone">
            <Upload className="w-12 h-12 text-white/50 mx-auto mb-4" />
            <p className="text-lg mb-2">Drag & drop files here</p>
            <p className="text-white/70 mb-4">or click to browse</p>
            <input
              type="file"
              multiple
              onChange={(e) => setSelectedFiles(e.target.files)}
              className="hidden"
              id="file-upload"
            />
            <label htmlFor="file-upload">
              <Button
                variant="outline"
                className="border-white/30 cursor-pointer"
                asChild
              >
                <span>Choose Files</span>
              </Button>
            </label>
            {selectedFiles && selectedFiles.length > 0 && (
              <div className="mt-4">
                <p className="text-sm text-muted-foreground mb-2">
                  {selectedFiles.length} file(s) selected
                </p>
                <Button
                  onClick={handleFileUpload}
                  disabled={uploadFilesMutation.isPending}
                  className="ymera-gradient text-black"
                >
                  {uploadFilesMutation.isPending ? 'Uploading...' : 'Upload Files'}
                </Button>
              </div>
            )}
          </div>
          
          {/* File List */}
          <div>
            <h4 className="text-lg font-semibold mb-4">Recent Files</h4>
            {filesLoading ? (
              <div className="space-y-3">
                {Array.from({ length: 5 }).map((_, i) => (
                  <Skeleton key={i} className="h-16 w-full" />
                ))}
              </div>
            ) : files.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">
                <FileText className="w-12 h-12 mx-auto mb-4 opacity-50" />
                <p>No files uploaded yet. Start by uploading your first file.</p>
              </div>
            ) : (
              <div className="space-y-3">
                {files.slice(0, 5).map((file: any) => {
                  const FileIcon = getFileIcon(file.mimeType);
                  return (
                    <div key={file.id} className="flex items-center justify-between p-4 bg-white/5 rounded-lg">
                      <div className="flex items-center gap-3">
                        <FileIcon className="w-6 h-6 text-primary" />
                        <div>
                          <p className="font-medium">{file.originalName}</p>
                          <p className="text-sm text-muted-foreground">
                            {formatFileSize(file.size)} • {file.downloadCount} downloads
                          </p>
                        </div>
                      </div>
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => handleDownload(file.id, file.originalName)}
                        className="border-white/30"
                      >
                        <Download className="w-4 h-4 mr-2" />
                        Download
                      </Button>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {/* File Statistics */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 pt-6 border-t border-white/20">
            <div className="text-center">
              <div className="text-2xl font-bold gradient-text">{files.length}</div>
              <p className="text-white/70">Total Files</p>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold gradient-text">
                {formatFileSize(files.reduce((sum: number, file: any) => sum + (file.size || 0), 0))}
              </div>
              <p className="text-white/70">Storage Used</p>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold gradient-text">
                {files.reduce((sum: number, file: any) => sum + (file.downloadCount || 0), 0)}
              </div>
              <p className="text-white/70">Downloads</p>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold gradient-text">
                {files.filter((file: any) => file.isShared).length}
              </div>
              <p className="text-white/70">Shared Files</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
