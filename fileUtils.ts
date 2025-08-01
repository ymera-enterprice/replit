export interface FileInfo {
  id: string;
  name: string;
  originalName: string;
  mimeType: string;
  size: number;
  extension: string;
  category: 'document' | 'image' | 'code' | 'archive' | 'video' | 'audio' | 'other';
}

export function getFileCategory(mimeType: string, fileName: string): FileInfo['category'] {
  const extension = getFileExtension(fileName).toLowerCase();
  
  // Document types
  if (mimeType.includes('pdf') || 
      mimeType.includes('document') || 
      mimeType.includes('text') ||
      mimeType.includes('presentation') ||
      mimeType.includes('spreadsheet') ||
      ['doc', 'docx', 'pdf', 'txt', 'rtf', 'odt', 'ppt', 'pptx', 'xls', 'xlsx'].includes(extension)) {
    return 'document';
  }
  
  // Image types
  if (mimeType.startsWith('image/') ||
      ['jpg', 'jpeg', 'png', 'gif', 'svg', 'bmp', 'webp', 'ico', 'tiff'].includes(extension)) {
    return 'image';
  }
  
  // Code types
  if (mimeType.includes('javascript') ||
      mimeType.includes('json') ||
      mimeType.includes('xml') ||
      ['js', 'ts', 'jsx', 'tsx', 'py', 'java', 'cpp', 'c', 'h', 'css', 'scss', 'html', 'xml', 'json', 'yaml', 'yml', 'php', 'rb', 'go', 'rs', 'swift', 'kt'].includes(extension)) {
    return 'code';
  }
  
  // Archive types
  if (mimeType.includes('zip') ||
      mimeType.includes('archive') ||
      mimeType.includes('compressed') ||
      ['zip', 'rar', '7z', 'tar', 'gz', 'bz2', 'xz'].includes(extension)) {
    return 'archive';
  }
  
  // Video types
  if (mimeType.startsWith('video/') ||
      ['mp4', 'avi', 'mov', 'wmv', 'flv', 'webm', 'mkv', 'm4v'].includes(extension)) {
    return 'video';
  }
  
  // Audio types
  if (mimeType.startsWith('audio/') ||
      ['mp3', 'wav', 'flac', 'aac', 'ogg', 'm4a', 'wma'].includes(extension)) {
    return 'audio';
  }
  
  return 'other';
}

export function getFileExtension(fileName: string): string {
  const parts = fileName.split('.');
  return parts.length > 1 ? parts[parts.length - 1] : '';
}

export function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 Bytes';
  
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

export function getFileIcon(category: FileInfo['category'], mimeType?: string): string {
  switch (category) {
    case 'document':
      if (mimeType?.includes('pdf')) return '📄';
      if (mimeType?.includes('presentation')) return '📊';
      if (mimeType?.includes('spreadsheet')) return '📈';
      return '📝';
    case 'image':
      return '🖼️';
    case 'code':
      return '💻';
    case 'archive':
      return '📦';
    case 'video':
      return '🎥';
    case 'audio':
      return '🎵';
    default:
      return '📄';
  }
}

export function getFileIconColor(category: FileInfo['category'], mimeType?: string): string {
  switch (category) {
    case 'document':
      if (mimeType?.includes('pdf')) return 'text-red-600 dark:text-red-400';
      if (mimeType?.includes('presentation')) return 'text-orange-600 dark:text-orange-400';
      if (mimeType?.includes('spreadsheet')) return 'text-green-600 dark:text-green-400';
      return 'text-blue-600 dark:text-blue-400';
    case 'image':
      return 'text-purple-600 dark:text-purple-400';
    case 'code':
      return 'text-green-600 dark:text-green-400';
    case 'archive':
      return 'text-yellow-600 dark:text-yellow-400';
    case 'video':
      return 'text-pink-600 dark:text-pink-400';
    case 'audio':
      return 'text-indigo-600 dark:text-indigo-400';
    default:
      return 'text-gray-600 dark:text-gray-400';
  }
}

export function getFileBackgroundGradient(category: FileInfo['category'], mimeType?: string): string {
  switch (category) {
    case 'document':
      if (mimeType?.includes('pdf')) return 'from-red-100 to-red-200 dark:from-red-900 dark:to-red-800';
      if (mimeType?.includes('presentation')) return 'from-orange-100 to-orange-200 dark:from-orange-900 dark:to-orange-800';
      if (mimeType?.includes('spreadsheet')) return 'from-green-100 to-green-200 dark:from-green-900 dark:to-green-800';
      return 'from-blue-100 to-blue-200 dark:from-blue-900 dark:to-blue-800';
    case 'image':
      return 'from-purple-100 to-purple-200 dark:from-purple-900 dark:to-purple-800';
    case 'code':
      return 'from-green-100 to-green-200 dark:from-green-900 dark:to-green-800';
    case 'archive':
      return 'from-yellow-100 to-yellow-200 dark:from-yellow-900 dark:to-yellow-800';
    case 'video':
      return 'from-pink-100 to-pink-200 dark:from-pink-900 dark:to-pink-800';
    case 'audio':
      return 'from-indigo-100 to-indigo-200 dark:from-indigo-900 dark:to-indigo-800';
    default:
      return 'from-gray-100 to-gray-200 dark:from-gray-700 dark:to-gray-600';
  }
}

export function isImageFile(mimeType: string): boolean {
  return mimeType.startsWith('image/');
}

export function isDocumentFile(mimeType: string): boolean {
  return mimeType.includes('pdf') ||
         mimeType.includes('document') ||
         mimeType.includes('text') ||
         mimeType.includes('presentation') ||
         mimeType.includes('spreadsheet');
}

export function isCodeFile(fileName: string, mimeType: string): boolean {
  const extension = getFileExtension(fileName).toLowerCase();
  const codeExtensions = ['js', 'ts', 'jsx', 'tsx', 'py', 'java', 'cpp', 'c', 'h', 'css', 'scss', 'html', 'xml', 'json', 'yaml', 'yml', 'php', 'rb', 'go', 'rs', 'swift', 'kt'];
  
  return mimeType.includes('javascript') ||
         mimeType.includes('json') ||
         mimeType.includes('xml') ||
         codeExtensions.includes(extension);
}

export function canPreview(mimeType: string, fileName: string): boolean {
  const category = getFileCategory(mimeType, fileName);
  
  // These file types can typically be previewed
  return category === 'image' || 
         mimeType.includes('pdf') ||
         mimeType.includes('text') ||
         category === 'code';
}

export function getPreviewUrl(fileId: string, mimeType: string, fileName: string): string | null {
  if (!canPreview(mimeType, fileName)) return null;
  
  if (isImageFile(mimeType)) {
    return `/thumbnails/${fileId}.png`;
  }
  
  if (mimeType.includes('pdf')) {
    return `/thumbnails/${fileId}.png`;
  }
  
  return `/uploads/${fileName}`;
}

export function validateFileType(file: File, allowedTypes?: string[]): boolean {
  if (!allowedTypes || allowedTypes.length === 0) return true;
  
  return allowedTypes.some(type => {
    if (type.endsWith('/*')) {
      const baseType = type.replace('/*', '');
      return file.type.startsWith(baseType);
    }
    return file.type === type;
  });
}

export function validateFileSize(file: File, maxSizeBytes: number): boolean {
  return file.size <= maxSizeBytes;
}

export interface FileValidationResult {
  valid: boolean;
  errors: string[];
}

export function validateFile(
  file: File,
  options: {
    maxSize?: number;
    allowedTypes?: string[];
    allowedExtensions?: string[];
  } = {}
): FileValidationResult {
  const errors: string[] = [];
  
  // Check file size
  if (options.maxSize && !validateFileSize(file, options.maxSize)) {
    errors.push(`File size exceeds maximum limit of ${formatFileSize(options.maxSize)}`);
  }
  
  // Check file type
  if (options.allowedTypes && !validateFileType(file, options.allowedTypes)) {
    errors.push(`File type "${file.type}" is not allowed`);
  }
  
  // Check file extension
  if (options.allowedExtensions) {
    const extension = getFileExtension(file.name).toLowerCase();
    if (!options.allowedExtensions.includes(extension)) {
      errors.push(`File extension "${extension}" is not allowed`);
    }
  }
  
  return {
    valid: errors.length === 0,
    errors,
  };
}

export function generateThumbnail(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    if (!isImageFile(file.type)) {
      reject(new Error('Cannot generate thumbnail for non-image files'));
      return;
    }
    
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const img = new Image();
    
    img.onload = () => {
      // Set thumbnail dimensions
      const maxSize = 300;
      let { width, height } = img;
      
      if (width > height) {
        if (width > maxSize) {
          height = (height * maxSize) / width;
          width = maxSize;
        }
      } else {
        if (height > maxSize) {
          width = (width * maxSize) / height;
          height = maxSize;
        }
      }
      
      canvas.width = width;
      canvas.height = height;
      
      ctx?.drawImage(img, 0, 0, width, height);
      
      canvas.toBlob((blob) => {
        if (blob) {
          const url = URL.createObjectURL(blob);
          resolve(url);
        } else {
          reject(new Error('Failed to generate thumbnail'));
        }
      }, 'image/png', 0.8);
    };
    
    img.onerror = () => {
      reject(new Error('Failed to load image for thumbnail generation'));
    };
    
    img.src = URL.createObjectURL(file);
  });
}

export function downloadFile(fileId: string, fileName: string): void {
  const link = document.createElement('a');
  link.href = `/uploads/${fileName}`;
  link.download = fileName;
  link.target = '_blank';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

export function copyToClipboard(text: string): Promise<void> {
  if (navigator.clipboard && window.isSecureContext) {
    return navigator.clipboard.writeText(text);
  } else {
    // Fallback for older browsers
    const textArea = document.createElement('textarea');
    textArea.value = text;
    textArea.style.position = 'fixed';
    textArea.style.left = '-999999px';
    textArea.style.top = '-999999px';
    document.body.appendChild(textArea);
    textArea.focus();
    textArea.select();
    
    return new Promise((resolve, reject) => {
      if (document.execCommand('copy')) {
        resolve();
      } else {
        reject(new Error('Failed to copy to clipboard'));
      }
      document.body.removeChild(textArea);
    });
  }
}
