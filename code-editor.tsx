import { useState, useEffect } from "react";

interface CodeEditorProps {
  content: string;
  onChange: (content: string) => void;
  language?: string;
  readOnly?: boolean;
}

export default function CodeEditor({ 
  content, 
  onChange, 
  language = "python", 
  readOnly = false 
}: CodeEditorProps) {
  const [localContent, setLocalContent] = useState(content);
  const [cursorPosition, setCursorPosition] = useState({ line: 0, column: 0 });

  useEffect(() => {
    setLocalContent(content);
  }, [content]);

  const handleContentChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    if (readOnly) return;
    
    const newContent = e.target.value;
    setLocalContent(newContent);
    onChange(newContent);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (readOnly) return;
    
    // Handle tab key for indentation
    if (e.key === 'Tab') {
      e.preventDefault();
      const textarea = e.currentTarget;
      const start = textarea.selectionStart;
      const end = textarea.selectionEnd;
      
      const newContent = localContent.substring(0, start) + '    ' + localContent.substring(end);
      setLocalContent(newContent);
      onChange(newContent);
      
      // Set cursor position after tab
      setTimeout(() => {
        textarea.selectionStart = textarea.selectionEnd = start + 4;
      }, 0);
    }
  };

  const lines = localContent.split('\n');

  return (
    <div className="relative bg-black/40 rounded-lg h-40 overflow-hidden">
      {/* Line Numbers */}
      <div className="absolute left-0 top-0 bottom-0 w-12 bg-black/20 p-2 text-xs font-mono text-muted-foreground select-none">
        {lines.map((_, index) => (
          <div key={index} className="text-right leading-6">
            {index + 1}
          </div>
        ))}
      </div>
      
      {/* Code Content */}
      <div className="ml-12 p-2 h-full overflow-auto scroll-container">
        <textarea
          value={localContent}
          onChange={handleContentChange}
          onKeyDown={handleKeyDown}
          readOnly={readOnly}
          className={`w-full h-full bg-transparent text-sm font-mono text-foreground resize-none outline-none leading-6 ${
            readOnly ? 'cursor-default' : 'cursor-text'
          }`}
          style={{ minHeight: '100%' }}
          spellCheck={false}
        />
      </div>
      
      {/* Syntax Highlighting Overlay */}
      <div className="absolute ml-12 p-2 top-0 left-0 right-0 bottom-0 pointer-events-none overflow-hidden">
        <div className="code-editor text-sm">
          {lines.map((line, lineIndex) => (
            <div key={lineIndex} className="leading-6 whitespace-pre">
              {highlightSyntax(line, language)}
            </div>
          ))}
        </div>
      </div>
      
      {/* Typing Indicator */}
      {!readOnly && (
        <div className="absolute bottom-2 right-2 text-xs text-muted-foreground">
          <span className="typing-indicator">|</span>
        </div>
      )}
      
      {/* Read-only Indicator */}
      {readOnly && (
        <div className="absolute top-2 right-2 text-xs text-warning bg-warning/20 px-2 py-1 rounded">
          Read Only
        </div>
      )}
    </div>
  );
}

// Simple syntax highlighting for Python
function highlightSyntax(line: string, language: string): React.ReactNode {
  if (language !== 'python') {
    return <span className="text-foreground">{line}</span>;
  }

  // Keywords
  const keywords = ['import', 'from', 'def', 'class', 'if', 'else', 'elif', 'for', 'while', 'try', 'except', 'finally', 'with', 'as', 'return', 'yield', 'break', 'continue', 'pass', 'raise', 'assert', 'del', 'global', 'nonlocal', 'lambda', 'and', 'or', 'not', 'in', 'is'];
  
  // Comments
  if (line.trim().startsWith('#')) {
    return <span className="text-muted-foreground">{line}</span>;
  }
  
  // String literals
  const stringRegex = /(["'])((?:(?!\1)[^\\]|\\.)*)(\1)/g;
  let result: React.ReactNode[] = [];
  let lastIndex = 0;
  let match;
  
  while ((match = stringRegex.exec(line)) !== null) {
    // Add text before string
    if (match.index > lastIndex) {
      const beforeString = line.substring(lastIndex, match.index);
      result.push(highlightKeywords(beforeString, keywords));
    }
    
    // Add string
    result.push(
      <span key={match.index} className="text-green-400">
        {match[0]}
      </span>
    );
    
    lastIndex = match.index + match[0].length;
  }
  
  // Add remaining text
  if (lastIndex < line.length) {
    const remaining = line.substring(lastIndex);
    result.push(highlightKeywords(remaining, keywords));
  }
  
  return result.length > 0 ? <>{result}</> : highlightKeywords(line, keywords);
}

function highlightKeywords(text: string, keywords: string[]): React.ReactNode {
  const keywordRegex = new RegExp(`\\b(${keywords.join('|')})\\b`, 'g');
  const parts = text.split(keywordRegex);
  
  return parts.map((part, index) => {
    if (keywords.includes(part)) {
      return (
        <span key={index} className="text-blue-400 font-medium">
          {part}
        </span>
      );
    }
    return <span key={index} className="text-foreground">{part}</span>;
  });
}
