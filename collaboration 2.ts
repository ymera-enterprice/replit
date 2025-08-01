export interface CollaborationSession {
  id: string;
  name: string;
  description?: string;
  participants: CollaborationParticipant[];
  files: CollaborationFile[];
  status: 'active' | 'paused' | 'completed' | 'archived';
  createdAt: string;
  updatedAt: string;
  createdBy: string;
}

export interface CollaborationParticipant {
  id: string;
  userId: string;
  name: string;
  email?: string;
  avatar?: string;
  role: 'owner' | 'editor' | 'viewer';
  status: 'online' | 'offline' | 'away';
  lastActive: string;
  permissions: CollaborationPermissions;
}

export interface CollaborationPermissions {
  canEdit: boolean;
  canComment: boolean;
  canShare: boolean;
  canDelete: boolean;
  canManageParticipants: boolean;
}

export interface CollaborationFile {
  id: string;
  name: string;
  path: string;
  content: string;
  language: string;
  lastModifiedBy: string;
  lastModifiedAt: string;
  version: number;
  isLocked: boolean;
  lockedBy?: string;
  changes: CollaborationChange[];
}

export interface CollaborationChange {
  id: string;
  userId: string;
  userName: string;
  type: 'insert' | 'delete' | 'replace';
  position: {
    line: number;
    column: number;
  };
  content: string;
  timestamp: string;
}

export interface ChatMessage {
  id: string;
  sessionId: string;
  userId: string;
  userName: string;
  content: string;
  type: 'message' | 'system' | 'code_reference';
  timestamp: string;
  metadata?: {
    fileId?: string;
    lineNumber?: number;
    codeSnippet?: string;
  };
}

export interface CodeCursor {
  userId: string;
  userName: string;
  fileId: string;
  position: {
    line: number;
    column: number;
  };
  selection?: {
    start: {
      line: number;
      column: number;
    };
    end: {
      line: number;
      column: number;
    };
  };
  color: string;
}

export interface CollaborationEvent {
  id: string;
  sessionId: string;
  type: 'user_joined' | 'user_left' | 'file_opened' | 'file_closed' | 'file_modified' | 'chat_message' | 'cursor_moved';
  userId: string;
  userName: string;
  timestamp: string;
  data: Record<string, any>;
}

export interface WebSocketMessage {
  type: 'auth' | 'join_session' | 'leave_session' | 'code_change' | 'cursor_move' | 'chat_message' | 'user_typing' | 'file_lock' | 'file_unlock';
  sessionId?: string;
  userId?: string;
  data: Record<string, any>;
}

export interface CollaborationSettings {
  autoSave: boolean;
  autoSaveInterval: number; // seconds
  showCursors: boolean;
  showUserNames: boolean;
  enableChat: boolean;
  enableVoiceChat: boolean;
  enableVideoCall: boolean;
  enableScreenShare: boolean;
  maxParticipants: number;
  allowAnonymous: boolean;
  requireApproval: boolean;
}

export type CollaborationRole = CollaborationParticipant['role'];
export type CollaborationStatus = CollaborationSession['status'];
export type ParticipantStatus = CollaborationParticipant['status'];
export type ChangeType = CollaborationChange['type'];
export type MessageType = ChatMessage['type'];
export type EventType = CollaborationEvent['type'];
export type WebSocketMessageType = WebSocketMessage['type'];
