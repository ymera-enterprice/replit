#!/bin/bash

# YMERA Unified AI Deployment - Dependency Installation Script
# Execute the exact YMERA Unified Deployment strategy with 20-minute timeline

set -e  # Exit on any error

echo "🚀 Starting YMERA Unified AI Enhancement..."
echo "📦 Phase 1: Installing Python AI dependencies with exact versions"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_info() {
    echo -e "${BLUE}📦 $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check Python version
echo "🔍 Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    print_error "Python not found. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
echo "Python version: $PYTHON_VERSION"

# Check pip
if ! $PYTHON_CMD -m pip --version &> /dev/null; then
    print_error "pip not found. Please install pip."
    exit 1
fi

# Upgrade pip first
print_info "Upgrading pip to latest version..."
$PYTHON_CMD -m pip install --upgrade pip
print_status "pip: UPGRADED successfully"

# Install exact dependency versions
echo ""
echo "📦 Installing AI packages with exact versions..."

# Core AI packages
print_info "Installing pinecone-client==3.0.0..."
$PYTHON_CMD -m pip install pinecone-client==3.0.0
print_status "pinecone: INSTALLED"

print_info "Installing groq==0.4.2..."
$PYTHON_CMD -m pip install groq==0.4.2
print_status "groq: INSTALLED"

print_info "Installing anthropic==0.21.3..."
$PYTHON_CMD -m pip install anthropic==0.21.3
print_status "anthropic: INSTALLED"

print_info "Installing sentence-transformers==2.2.2..."
$PYTHON_CMD -m pip install sentence-transformers==2.2.2
print_status "sentence-transformers: INSTALLED"

# Machine Learning packages
print_info "Installing numpy==1.24.3..."
$PYTHON_CMD -m pip install numpy==1.24.3
print_status "numpy: INSTALLED"

print_info "Installing pandas==1.5.3..."
$PYTHON_CMD -m pip install pandas==1.5.3
print_status "pandas: INSTALLED"

print_info "Installing scikit-learn==1.3.0..."
$PYTHON_CMD -m pip install scikit-learn==1.3.0
print_status "scikit-learn: INSTALLED"

print_info "Installing torch==2.0.1..."
$PYTHON_CMD -m pip install torch==2.0.1
print_status "torch: INSTALLED"

print_info "Installing transformers==4.30.0..."
$PYTHON_CMD -m pip install transformers==4.30.0
print_status "transformers: INSTALLED"

print_info "Installing tiktoken==0.5.1..."
$PYTHON_CMD -m pip install tiktoken==0.5.1
print_status "tiktoken: INSTALLED"

# Web and API packages
print_info "Installing fastapi==0.104.1..."
$PYTHON_CMD -m pip install fastapi==0.104.1
print_status "fastapi: INSTALLED"

print_info "Installing uvicorn==0.24.0..."
$PYTHON_CMD -m pip install uvicorn==0.24.0
print_status "uvicorn: INSTALLED"

print_info "Installing python-multipart==0.0.6..."
$PYTHON_CMD -m pip install python-multipart==0.0.6
print_status "python-multipart: INSTALLED"

print_info "Installing websockets==11.0.3..."
$PYTHON_CMD -m pip install websockets==11.0.3
print_status "websockets: INSTALLED"

print_info "Installing httpx==0.25.2..."
$PYTHON_CMD -m pip install httpx==0.25.2
print_status "httpx: INSTALLED"

# Database and caching
print_info "Installing redis==4.6.0..."
$PYTHON_CMD -m pip install redis==4.6.0
print_status "redis: INSTALLED"

print_info "Installing psutil==5.9.0..."
$PYTHON_CMD -m pip install psutil==5.9.0
print_status "psutil: INSTALLED"

# Security packages
print_info "Installing PyJWT==2.8.0..."
$PYTHON_CMD -m pip install PyJWT==2.8.0
print_status "PyJWT: INSTALLED"

print_info "Installing cryptography==41.0.7..."
$PYTHON_CMD -m pip install cryptography==41.0.7
print_status "cryptography: INSTALLED"

print_info "Installing bcrypt==4.1.2..."
$PYTHON_CMD -m pip install bcrypt==4.1.2
print_status "bcrypt: INSTALLED"

# Utility packages
print_info "Installing python-dotenv==1.0.0..."
$PYTHON_CMD -m pip install python-dotenv==1.0.0
print_status "python-dotenv: INSTALLED"

print_info "Installing structlog==23.2.0..."
$PYTHON_CMD -m pip install structlog==23.2.0
print_status "structlog: INSTALLED"

print_info "Installing aiofiles==23.2.1..."
$PYTHON_CMD -m pip install aiofiles==23.2.1
print_status "aiofiles: INSTALLED"

echo ""
echo "🧪 Running verification script..."

# Create verification script
cat > verify_installation.py << 'EOF'
#!/usr/bin/env python3
"""
YMERA AI Installation Verification Script
Verifies all required packages are installed with correct versions
"""

import sys
import importlib
from typing import Dict, Tuple

def check_package(package_name: str, expected_version: str = None) -> Tuple[bool, str]:
    """Check if a package is installed with optional version check"""
    try:
        module = importlib.import_module(package_name)
        
        if expected_version and hasattr(module, '__version__'):
            actual_version = module.__version__
            if actual_version != expected_version:
                return False, f"Version mismatch: expected {expected_version}, got {actual_version}"
            return True, f"✅ {package_name}=={actual_version}"
        elif hasattr(module, '__version__'):
            return True, f"✅ {package_name}=={module.__version__}"
        else:
            return True, f"✅ {package_name} (version unknown)"
    except ImportError as e:
        return False, f"❌ {package_name}: {str(e)}"

def main():
    """Main verification function"""
    print("🔍 Verifying YMERA AI package installation...")
    print("=" * 50)
    
    # Define required packages with expected versions
    required_packages = {
        'pinecone': '3.0.0',
        'groq': '0.4.2',
        'anthropic': '0.21.3',
        'sentence_transformers': '2.2.2',
        'numpy': '1.24.3',
        'pandas': '1.5.3',
        'sklearn': '1.3.0',
        'torch': '2.0.1',
        'transformers': '4.30.0',
        'tiktoken': '0.5.1',
        'fastapi': '0.104.1',
        'uvicorn': '0.24.0',
        'redis': '4.6.0',
        'psutil': '5.9.0',
        'jwt': '2.8.0',
        'cryptography': '41.0.7',
        'bcrypt': '4.1.2',
        'dotenv': '1.0.0',
        'structlog': '23.2.0',
        'aiofiles': '23.2.1'
    }
    
    # Special cases for package names that differ from import names
    import_name_mapping = {
        'pinecone': 'pinecone',
        'groq': 'groq',
        'anthropic': 'anthropic',
        'sentence_transformers': 'sentence_transformers',
        'sklearn': 'sklearn',
        'jwt': 'jwt',
        'dotenv': 'dotenv'
    }
    
    success_count = 0
    total_count = len(required_packages)
    
    for package, expected_version in required_packages.items():
        import_name = import_name_mapping.get(package, package)
        success, message = check_package(import_name, expected_version)
        print(message)
        if success:
            success_count += 1
    
    print("=" * 50)
    print(f"📊 Verification Results: {success_count}/{total_count} packages verified successfully")
    
    if success_count == total_count:
        print("🎉 All packages installed correctly!")
        return 0
    else:
        print("⚠️ Some packages failed verification. Please check the installation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
EOF

# Run verification
$PYTHON_CMD verify_installation.py
verification_result=$?

# Clean up verification script
rm -f verify_installation.py

echo ""
if [ $verification_result -eq 0 ]; then
    print_status "Package verification completed successfully!"
    echo ""
    echo "🧠 Initializing AI services..."
    
    # Test AI services initialization
    if [ -f "server/ai_unified.py" ]; then
        print_info "Testing AI services..."
        cd server && $PYTHON_CMD ai_unified.py status && cd ..
        print_status "AI services initialized successfully"
    else
        print_warning "AI services script not found. Will be created in Phase 2."
    fi
    
    echo ""
    echo "✅ Phase 1 Complete: Environment Setup"
    echo "🚀 Ready for Phase 2: AI Services Integration"
    echo ""
    echo "Summary of installed packages:"
    echo "• Pinecone Vector Database: 3.0.0"
    echo "• Groq Fast LLM: 0.4.2" 
    echo "• Claude AI: 0.21.3"
    echo "• SentenceTransformers: 2.2.2"
    echo "• Core ML packages: NumPy, Pandas, Scikit-learn, PyTorch"
    echo "• Web framework: FastAPI, Uvicorn"
    echo "• Security: JWT, Cryptography, Bcrypt"
    echo ""
    echo "🎯 YMERA Unified AI environment ready for deployment!"
    
else
    print_error "Package verification failed. Please check the installation logs."
    exit 1
fi
