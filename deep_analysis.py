
#!/usr/bin/env python3
"""
YMERA Enterprise Platform - Deep Analysis Tool
Comprehensive analysis of missing dependencies and commands
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Any
import re
import ast

class YMERADeepAnalyzer:
    def __init__(self):
        self.project_root = Path(".")
        self.missing_imports = set()
        self.missing_commands = set()
        self.missing_system_deps = set()
        self.python_files = []
        self.js_ts_files = []
        
    def analyze_all(self) -> Dict[str, Any]:
        """Run comprehensive analysis"""
        print("🔍 Starting Deep Analysis of YMERA Platform...")
        
        results = {
            "python_analysis": self.analyze_python_dependencies(),
            "js_ts_analysis": self.analyze_js_ts_dependencies(),
            "system_commands": self.analyze_system_commands(),
            "file_structure": self.analyze_file_structure(),
            "missing_imports": list(self.missing_imports),
            "missing_commands": list(self.missing_commands),
            "recommendations": []
        }
        
        results["recommendations"] = self.generate_recommendations(results)
        return results
    
    def analyze_python_dependencies(self) -> Dict[str, Any]:
        """Analyze Python files for missing imports and dependencies"""
        print("📋 Analyzing Python dependencies...")
        
        # Find all Python files
        self.python_files = list(self.project_root.rglob("*.py"))
        
        # Standard library modules (partial list of commonly used ones)
        stdlib_modules = {
            'os', 'sys', 'json', 'asyncio', 'datetime', 'pathlib', 'typing',
            'collections', 're', 'uuid', 'logging', 'time', 'subprocess',
            'threading', 'multiprocessing', 'pickle', 'base64', 'hashlib',
            'urllib', 'http', 'socket', 'ssl', 'email', 'xml', 'html',
            'csv', 'sqlite3', 'decimal', 'fractions', 'random', 'math',
            'statistics', 'itertools', 'functools', 'operator', 'copy',
            'pprint', 'textwrap', 'string', 'struct', 'codecs', 'unicodedata'
        }
        
        # Extract imports from requirements.txt
        installed_packages = set()
        req_file = self.project_root / "requirements.txt"
        if req_file.exists():
            with open(req_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Extract package name (before ==, >=, etc.)
                        package = re.split(r'[><=!]', line)[0].strip()
                        installed_packages.add(package.lower())
                        # Add common aliases
                        if package.lower() == 'pillow':
                            installed_packages.add('pil')
                        elif package.lower() == 'python-dotenv':
                            installed_packages.add('dotenv')
        
        # Common package name mappings
        package_mappings = {
            'cv2': 'opencv-python',
            'pil': 'pillow',
            'yaml': 'pyyaml',
            'jwt': 'pyjwt',
            'dotenv': 'python-dotenv',
            'redis': 'redis',
            'psycopg2': 'psycopg2-binary',
            'sklearn': 'scikit-learn',
            'torch': 'torch',
            'tensorflow': 'tensorflow',
            'np': 'numpy',
            'pd': 'pandas'
        }
        
        all_imports = set()
        missing_imports = set()
        
        for py_file in self.python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse AST to extract imports
                try:
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                module_name = alias.name.split('.')[0]
                                all_imports.add(module_name)
                        elif isinstance(node, ast.ImportFrom):
                            if node.module:
                                module_name = node.module.split('.')[0]
                                all_imports.add(module_name)
                except SyntaxError:
                    # If AST parsing fails, use regex as fallback
                    import_pattern = r'^\s*(?:from\s+(\w+)|import\s+(\w+))'
                    for match in re.finditer(import_pattern, content, re.MULTILINE):
                        module = match.group(1) or match.group(2)
                        if module:
                            all_imports.add(module.split('.')[0])
                            
            except Exception as e:
                print(f"⚠️  Error analyzing {py_file}: {e}")
        
        # Check which imports are missing
        for imp in all_imports:
            imp_lower = imp.lower()
            if (imp_lower not in stdlib_modules and 
                imp_lower not in installed_packages and
                package_mappings.get(imp_lower, imp_lower) not in installed_packages):
                missing_imports.add(imp)
        
        self.missing_imports.update(missing_imports)
        
        return {
            "total_python_files": len(self.python_files),
            "all_imports": sorted(list(all_imports)),
            "installed_packages": sorted(list(installed_packages)),
            "missing_imports": sorted(list(missing_imports)),
            "package_mappings": package_mappings
        }
    
    def analyze_js_ts_dependencies(self) -> Dict[str, Any]:
        """Analyze JavaScript/TypeScript files for missing dependencies"""
        print("📋 Analyzing JavaScript/TypeScript dependencies...")
        
        # Find all JS/TS files
        self.js_ts_files = (
            list(self.project_root.rglob("*.js")) +
            list(self.project_root.rglob("*.ts")) +
            list(self.project_root.rglob("*.tsx")) +
            list(self.project_root.rglob("*.jsx"))
        )
        
        installed_packages = set()
        package_file = self.project_root / "package.json"
        if package_file.exists():
            with open(package_file, 'r') as f:
                package_data = json.load(f)
                deps = package_data.get("dependencies", {})
                dev_deps = package_data.get("devDependencies", {})
                installed_packages.update(deps.keys())
                installed_packages.update(dev_deps.keys())
        
        all_imports = set()
        missing_imports = set()
        
        for js_file in self.js_ts_files:
            try:
                with open(js_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract imports using regex
                import_patterns = [
                    r'import\s+.*?\s+from\s+[\'"]([^/\'"]+)',  # import ... from 'package'
                    r'require\s*\(\s*[\'"]([^/\'"]+)',        # require('package')
                    r'import\s*\(\s*[\'"]([^/\'"]+)'          # import('package')
                ]
                
                for pattern in import_patterns:
                    for match in re.finditer(pattern, content):
                        package = match.group(1)
                        if not package.startswith('.') and not package.startswith('/'):
                            # Extract base package name (before /)
                            base_package = package.split('/')[0]
                            if base_package.startswith('@'):
                                # Scoped package like @types/node
                                parts = package.split('/')
                                if len(parts) >= 2:
                                    base_package = f"{parts[0]}/{parts[1]}"
                            all_imports.add(base_package)
                            
            except Exception as e:
                print(f"⚠️  Error analyzing {js_file}: {e}")
        
        # Check which imports are missing
        for imp in all_imports:
            if imp not in installed_packages:
                missing_imports.add(imp)
        
        return {
            "total_js_ts_files": len(self.js_ts_files),
            "all_imports": sorted(list(all_imports)),
            "installed_packages": sorted(list(installed_packages)),
            "missing_imports": sorted(list(missing_imports))
        }
    
    def analyze_system_commands(self) -> Dict[str, Any]:
        """Analyze for missing system commands"""
        print("🔧 Analyzing system commands...")
        
        # Common commands used in the project
        commands_to_check = [
            'python', 'python3', 'pip', 'pip3', 'poetry', 'npm', 'node',
            'yarn', 'git', 'curl', 'wget', 'docker', 'docker-compose',
            'redis-server', 'redis-cli', 'psql', 'sqlite3'
        ]
        
        available_commands = {}
        missing_commands = []
        
        for cmd in commands_to_check:
            try:
                result = subprocess.run(['which', cmd], capture_output=True, text=True)
                if result.returncode == 0:
                    available_commands[cmd] = result.stdout.strip()
                else:
                    missing_commands.append(cmd)
            except Exception:
                missing_commands.append(cmd)
        
        self.missing_commands.update(missing_commands)
        
        return {
            "available_commands": available_commands,
            "missing_commands": missing_commands
        }
    
    def analyze_file_structure(self) -> Dict[str, Any]:
        """Analyze project file structure"""
        print("📁 Analyzing file structure...")
        
        structure = {
            "python_files": len(self.python_files),
            "js_ts_files": len(self.js_ts_files),
            "config_files": [],
            "missing_essential_files": []
        }
        
        # Check for essential config files
        essential_files = [
            "requirements.txt", "pyproject.toml", "package.json",
            ".env", ".gitignore", "README.md"
        ]
        
        for file in essential_files:
            if (self.project_root / file).exists():
                structure["config_files"].append(file)
            else:
                structure["missing_essential_files"].append(file)
        
        return structure
    
    def generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        # Python recommendations
        python_missing = analysis["python_analysis"]["missing_imports"]
        if python_missing:
            recommendations.append(f"Install missing Python packages: {', '.join(python_missing)}")
        
        # JS/TS recommendations
        js_missing = analysis["js_ts_analysis"]["missing_imports"]
        if js_missing:
            recommendations.append(f"Install missing Node.js packages: {', '.join(js_missing)}")
        
        # System commands
        cmd_missing = analysis["system_commands"]["missing_commands"]
        if cmd_missing:
            recommendations.append(f"Missing system commands: {', '.join(cmd_missing)}")
        
        # File structure
        missing_files = analysis["file_structure"]["missing_essential_files"]
        if missing_files:
            recommendations.append(f"Consider adding missing files: {', '.join(missing_files)}")
        
        return recommendations
    
    def print_report(self, analysis: Dict[str, Any]):
        """Print detailed analysis report"""
        print("\n" + "="*60)
        print("🚀 YMERA ENTERPRISE PLATFORM - DEEP ANALYSIS REPORT")
        print("="*60)
        
        print(f"\n📊 PROJECT OVERVIEW:")
        print(f"   Python files: {analysis['python_analysis']['total_python_files']}")
        print(f"   JS/TS files: {analysis['js_ts_analysis']['total_js_ts_files']}")
        
        print(f"\n🐍 PYTHON ANALYSIS:")
        python_missing = analysis["python_analysis"]["missing_imports"]
        if python_missing:
            print(f"   ❌ Missing imports: {', '.join(python_missing)}")
        else:
            print(f"   ✅ All Python dependencies appear to be satisfied")
        
        print(f"\n📦 NODE.JS ANALYSIS:")
        js_missing = analysis["js_ts_analysis"]["missing_imports"]
        if js_missing:
            print(f"   ❌ Missing packages: {', '.join(js_missing)}")
        else:
            print(f"   ✅ All Node.js dependencies appear to be satisfied")
        
        print(f"\n🔧 SYSTEM COMMANDS:")
        cmd_missing = analysis["system_commands"]["missing_commands"]
        if cmd_missing:
            print(f"   ❌ Missing commands: {', '.join(cmd_missing)}")
        else:
            print(f"   ✅ All required system commands are available")
        
        print(f"\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(analysis["recommendations"], 1):
            print(f"   {i}. {rec}")
        
        print("\n" + "="*60)
        
        # Save detailed report to file
        with open("ymera_analysis_report.json", "w") as f:
            json.dump(analysis, f, indent=2)
        print("📄 Detailed report saved to: ymera_analysis_report.json")

if __name__ == "__main__":
    analyzer = YMERADeepAnalyzer()
    analysis = analyzer.analyze_all()
    analyzer.print_report(analysis)
