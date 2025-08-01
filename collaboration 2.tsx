import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import CodeEditor from "@/components/collaboration/code-editor";
import ChatPanel from "@/components/collaboration/chat-panel";
import { useWebSocket } from "@/hooks/useWebSocket";
import { useAuth } from "@/hooks/useAuth";

export default function Collaboration() {
  const { user } = useAuth();
  const { socket, isConnected, sendMessage } = useWebSocket();
  const [activeUsers, setActiveUsers] = useState<any[]>([]);
  const [codeContent, setCodeContent] = useState({
    'main.py': `import numpy as np
from sklearn.model_selection import train_test_split

def preprocess_data(data):
    # Normalize the input data
    return np.array(data)`,
    'utils.py': `class DataValidator:
    def __init__(self, schema):
        self.schema = schema
    
    def validate(self, data):
        # Validation logic here
        pass`
  });

  useEffect(() => {
    if (socket && isConnected) {
      socket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        switch (data.type) {
          case 'code_change':
            if (data.data.fileId && data.data.changes) {
              setCodeContent(prev => ({
                ...prev,
                [data.data.fileId]: data.data.changes
              }));
            }
            break;
          case 'user_joined':
            setActiveUsers(prev => [...prev, data.data.user]);
            break;
          case 'user_left':
            setActiveUsers(prev => prev.filter(u => u.id !== data.data.userId));
            break;
        }
      };
    }
  }, [socket, isConnected]);

  const handleCodeChange = (fileId: string, content: string) => {
    setCodeContent(prev => ({
      ...prev,
      [fileId]: content
    }));
    
    if (socket && isConnected) {
      sendMessage({
        type: 'code_change',
        fileId,
        changes: content,
        user: user?.firstName || 'Anonymous'
      });
    }
  };

  return (
    <div className="h-full p-6 overflow-hidden">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-full">
        
        {/* Code Collaboration Panels */}
        <div className="lg:col-span-2 grid grid-rows-2 gap-6">
          
          {/* Code Editor 1 */}
          <Card className="glass-card">
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <div className="w-8 h-8 bg-gradient-to-r from-purple-500 to-blue-500 rounded-full flex items-center justify-center text-white text-sm font-medium">
                    {user?.firstName?.[0] || 'U'}
                  </div>
                  <div>
                    <CardTitle className="text-sm font-medium text-foreground">
                      {user?.firstName} {user?.lastName}
                    </CardTitle>
                    <div className="text-xs text-muted-foreground">main.py</div>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-success pulse-dot' : 'bg-error'}`}></div>
                  <span className={`text-xs ${isConnected ? 'text-success' : 'text-error'}`}>
                    {isConnected ? 'Live' : 'Disconnected'}
                  </span>
                </div>
              </div>
            </CardHeader>
            <CardContent className="p-4">
              <CodeEditor
                content={codeContent['main.py']}
                onChange={(content) => handleCodeChange('main.py', content)}
                language="python"
                readOnly={!isConnected}
              />
            </CardContent>
          </Card>
          
          {/* Code Editor 2 */}
          <Card className="glass-card">
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <div className="w-8 h-8 bg-gradient-to-r from-pink-500 to-red-500 rounded-full flex items-center justify-center text-white text-sm font-medium">
                    S
                  </div>
                  <div>
                    <CardTitle className="text-sm font-medium text-foreground">Sarah Kim</CardTitle>
                    <div className="text-xs text-muted-foreground">utils.py</div>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-success rounded-full pulse-dot"></div>
                  <span className="text-xs text-success">Live</span>
                </div>
              </div>
            </CardHeader>
            <CardContent className="p-4">
              <CodeEditor
                content={codeContent['utils.py']}
                onChange={(content) => handleCodeChange('utils.py', content)}
                language="python"
                readOnly={true}
              />
            </CardContent>
          </Card>
        </div>
        
        {/* Chat Panel */}
        <div className="h-full">
          <ChatPanel />
        </div>
      </div>
    </div>
  );
}
