import { useState, useEffect, useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useWebSocket } from "@/hooks/useWebSocket";
import { useAuth } from "@/hooks/useAuth";
import { formatTimeAgo } from "@/lib/utils";

interface ChatMessage {
  id: string;
  message: string;
  user: string;
  timestamp: string;
  avatar?: string;
}

export default function ChatPanel() {
  const { user } = useAuth();
  const { socket, isConnected, sendMessage } = useWebSocket();
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: '1',
      message: 'Working on the data preprocessing function. Should we normalize all features or just the numerical ones?',
      user: 'Alex Chen',
      timestamp: new Date(Date.now() - 2 * 60 * 1000).toISOString(),
      avatar: 'A'
    },
    {
      id: '2',
      message: "I'd suggest normalizing all features for consistency. Let me check the validation logic...",
      user: 'Sarah Kim',
      timestamp: new Date(Date.now() - 1 * 60 * 1000).toISOString(),
      avatar: 'S'
    },
    {
      id: '3',
      message: 'Agreed! 👍 Also, we should add error handling for edge cases.',
      user: 'Mike Torres',
      timestamp: new Date(Date.now() - 30 * 1000).toISOString(),
      avatar: 'M'
    }
  ]);
  const [newMessage, setNewMessage] = useState("");
  const [isTyping, setIsTyping] = useState<string[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    if (socket && isConnected) {
      socket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        switch (data.type) {
          case 'chat_message':
            setMessages(prev => [...prev, {
              id: data.data.id,
              message: data.data.message,
              user: data.data.user,
              timestamp: data.data.timestamp,
              avatar: data.data.user.charAt(0).toUpperCase()
            }]);
            break;
          case 'user_typing':
            setIsTyping(prev => [...prev.filter(u => u !== data.data.user), data.data.user]);
            setTimeout(() => {
              setIsTyping(prev => prev.filter(u => u !== data.data.user));
            }, 3000);
            break;
        }
      };
    }
  }, [socket, isConnected]);

  const handleSendMessage = (e: React.FormEvent) => {
    e.preventDefault();
    if (!newMessage.trim() || !isConnected) return;

    const message = {
      id: Date.now().toString(),
      message: newMessage,
      user: user?.firstName ? `${user.firstName} ${user.lastName}` : 'Anonymous',
      timestamp: new Date().toISOString(),
      avatar: user?.firstName?.charAt(0).toUpperCase() || 'U'
    };

    // Add to local messages immediately
    setMessages(prev => [...prev, message]);

    // Send via WebSocket
    if (socket) {
      sendMessage({
        type: 'chat_message',
        message: newMessage,
        user: message.user
      });
    }

    setNewMessage("");
  };

  const handleTyping = () => {
    if (socket && isConnected) {
      sendMessage({
        type: 'user_typing',
        user: user?.firstName ? `${user.firstName} ${user.lastName}` : 'Anonymous'
      });
    }
  };

  const getAvatarColor = (name: string) => {
    const colors = [
      'bg-gradient-to-r from-purple-500 to-blue-500',
      'bg-gradient-to-r from-pink-500 to-red-500',
      'bg-gradient-to-r from-green-500 to-teal-500',
      'bg-gradient-to-r from-orange-500 to-yellow-500',
      'bg-gradient-to-r from-indigo-500 to-purple-500',
    ];
    const index = name.charCodeAt(0) % colors.length;
    return colors[index];
  };

  return (
    <Card className="glass-card flex flex-col h-full">
      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg font-semibold text-foreground">Team Chat</CardTitle>
          <div className="flex items-center space-x-2">
            <div className="flex -space-x-1">
              <div className="w-2 h-2 bg-success rounded-full"></div>
              <div className="w-2 h-2 bg-secondary rounded-full"></div>
              <div className="w-2 h-2 bg-accent rounded-full"></div>
            </div>
            <span className="text-xs text-muted-foreground ml-2">
              {isConnected ? '3 online' : 'Offline'}
            </span>
          </div>
        </div>
      </CardHeader>
      
      <CardContent className="flex-1 flex flex-col p-0">
        {/* Messages */}
        <div className="flex-1 p-4 overflow-y-auto scroll-container space-y-4">
          {messages.map((message) => (
            <div key={message.id} className="flex items-start space-x-3">
              <div className={`w-8 h-8 rounded-full flex items-center justify-center text-white text-sm font-medium flex-shrink-0 ${getAvatarColor(message.user)}`}>
                {message.avatar}
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center space-x-2 mb-1">
                  <span className="text-sm font-medium text-foreground">{message.user}</span>
                  <span className="text-xs text-muted-foreground">
                    {formatTimeAgo(message.timestamp)}
                  </span>
                </div>
                <div className="text-sm text-muted-foreground">{message.message}</div>
              </div>
            </div>
          ))}
          
          {/* Typing Indicators */}
          {isTyping.map((typingUser) => (
            <div key={typingUser} className="flex items-center space-x-2 text-xs text-muted-foreground">
              <div className={`w-6 h-6 rounded-full flex items-center justify-center text-white text-xs font-medium ${getAvatarColor(typingUser)}`}>
                {typingUser.charAt(0)}
              </div>
              <span className="typing-indicator">{typingUser} is typing...</span>
            </div>
          ))}
          
          <div ref={messagesEndRef} />
        </div>
        
        {/* Input */}
        <div className="p-4 border-t border-border">
          <form onSubmit={handleSendMessage} className="flex items-center space-x-3">
            <Input
              type="text"
              placeholder={isConnected ? "Type a message..." : "Connecting..."}
              value={newMessage}
              onChange={(e) => {
                setNewMessage(e.target.value);
                handleTyping();
              }}
              disabled={!isConnected}
              className="flex-1 glass-effect"
            />
            <Button 
              type="submit"
              disabled={!newMessage.trim() || !isConnected}
              className="btn-gradient"
              size="sm"
            >
              <i className="fas fa-paper-plane"></i>
            </Button>
          </form>
          
          {!isConnected && (
            <div className="flex items-center justify-center mt-2 text-xs text-warning">
              <i className="fas fa-exclamation-triangle mr-1"></i>
              Connection lost. Attempting to reconnect...
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
