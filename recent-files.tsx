import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { FileText, MoreHorizontal } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { File } from "@shared/schema";
import { formatDistanceToNow } from "date-fns";

const getFileIcon = (mimeType: string) => {
  if (mimeType.includes('pdf')) return { icon: FileText, color: "bg-red-100 text-red-600" };
  if (mimeType.includes('word')) return { icon: FileText, color: "bg-blue-100 text-blue-600" };
  if (mimeType.includes('sheet')) return { icon: FileText, color: "bg-green-100 text-green-600" };
  return { icon: FileText, color: "bg-gray-100 text-gray-600" };
};

const getStatusBadge = (status: File['status']) => {
  const statusConfig = {
    synced: { label: "Synced", variant: "default", className: "bg-success/10 text-success" },
    processing: { label: "Processing", variant: "secondary", className: "bg-warning/10 text-warning" },
    uploading: { label: "Uploading", variant: "secondary", className: "bg-primary/10 text-primary" },
    failed: { label: "Failed", variant: "destructive", className: "bg-error/10 text-error" },
  } as const;

  const config = statusConfig[status];
  return (
    <Badge variant={config.variant} className={config.className}>
      {config.label}
    </Badge>
  );
};

export function RecentFiles() {
  const { data: files, isLoading } = useQuery({
    queryKey: ['/api/files/recent'],
  });

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Recent Files</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {[1, 2, 3].map((i) => (
              <div key={i} className="flex items-center space-x-3 animate-pulse">
                <div className="w-10 h-10 bg-muted rounded-lg" />
                <div className="flex-1 space-y-2">
                  <div className="h-4 bg-muted rounded w-3/4" />
                  <div className="h-3 bg-muted rounded w-1/2" />
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle>Recent Files</CardTitle>
          <Button variant="ghost" size="sm">
            View All
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {files?.map((file: File) => {
            const { icon: Icon, color } = getFileIcon(file.mimeType);
            
            return (
              <div key={file.id} className="flex items-center justify-between py-3 border-b border-border last:border-b-0">
                <div className="flex items-center space-x-3">
                  <div className={`file-icon ${color}`}>
                    <Icon className="h-5 w-5" />
                  </div>
                  <div>
                    <p className="font-medium text-foreground">{file.name}</p>
                    <p className="text-sm text-muted-foreground">
                      Modified {formatDistanceToNow(new Date(file.updatedAt), { addSuffix: true })}
                    </p>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  {getStatusBadge(file.status)}
                  <Button variant="ghost" size="icon" className="h-8 w-8">
                    <MoreHorizontal className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
}
