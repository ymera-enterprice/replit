"""
YMERA Syntax Fixer
Automated syntax error detection and fixing for Phase 3 AI components
"""

import os
import ast
import re
import glob
import logging
from typing import Dict, List, Any, Tuple, Optional
import tempfile
import shutil

class SyntaxFixer:
    def __init__(self):
        self.logger = logging.getLogger('syntax_fixer')
        self.common_fixes = {
            'missing_imports': {
                'asyncio': 'import asyncio',
                'logging': 'import logging',
                'typing': 'from typing import Dict, List, Any, Optional',
                'json': 'import json',
                'os': 'import os',
                'sys': 'import sys',
                'datetime': 'from datetime import datetime',
                'pathlib': 'from pathlib import Path',
                'uuid': 'import uuid',
                'time': 'import time'
            },
            'relative_import_fixes': {
                r'from \.\.': 'from ymera_core.',
                r'from learning_engine': 'from .learning_engine',
                r'from ymera_agents': 'from .ymera_agents',
                r'from ai_services': 'from .ai_services'
            }
        }
        
    def fix_syntax_errors(self) -> Dict[str, Any]:
        """Main syntax fixing workflow"""
        self.logger.info("Starting syntax error detection and fixes")
        
        results = {
            'success': False,
            'files_processed': 0,
            'syntax_errors_found': 0,
            'syntax_errors_fixed': 0,
            'fixes_applied': {},
            'remaining_errors': [],
            'message': ''
        }
        
        try:
            # Find Phase 3 files
            phase3_files = self._find_phase3_files()
            results['files_processed'] = len(phase3_files)
            
            if not phase3_files:
                results['message'] = 'No Phase 3 files found for syntax fixing'
                results['success'] = True
                return results
            
            # Process each file
            for file_path in phase3_files:
                file_result = self._fix_file_syntax(file_path)
                
                if file_result['syntax_errors']:
                    results['syntax_errors_found'] += len(file_result['syntax_errors'])
                
                if file_result['fixes_applied']:
                    results['fixes_applied'][file_path] = file_result['fixes_applied']
                    results['syntax_errors_fixed'] += len(file_result['fixes_applied'])
                
                if file_result['remaining_errors']:
                    results['remaining_errors'].extend([
                        {'file': file_path, 'errors': file_result['remaining_errors']}
                    ])
            
            # Additional fixes
            additional_fixes = self._apply_additional_fixes(phase3_files)
            results['additional_fixes'] = additional_fixes
            
            results['success'] = True
            results['message'] = f'Syntax fixing completed. {results["syntax_errors_fixed"]} errors fixed, {len(results["remaining_errors"])} remaining.'
            
            self.logger.info(f"Syntax fixing completed: {results['syntax_errors_fixed']} fixed, {len(results['remaining_errors'])} remaining")
            
        except Exception as e:
            self.logger.error(f"Syntax fixing failed: {e}")
            results['message'] = f'Syntax fixing failed: {str(e)}'
            results['success'] = False
            
        return results
    
    def _find_phase3_files(self) -> List[str]:
        """Find all Phase 3 Python files"""
        phase3_files = []
        
        # Search patterns for Phase 3 files
        patterns = [
            'learning_engine/**/*.py',
            'ymera_agents/**/*.py', 
            'ai_services/**/*.py',
            '**/learning*.py',
            '**/agent*.py',
            '**/ai_*.py'
        ]
        
        for pattern in patterns:
            files = glob.glob(pattern, recursive=True)
            phase3_files.extend(files)
        
        # Remove duplicates and filter out __pycache__
        phase3_files = list(set(phase3_files))
        phase3_files = [f for f in phase3_files if '__pycache__' not in f]
        phase3_files.sort()
        
        return phase3_files
    
    def _fix_file_syntax(self, file_path: str) -> Dict[str, Any]:
        """Fix syntax errors in a single file"""
        result = {
            'syntax_errors': [],
            'fixes_applied': [],
            'remaining_errors': []
        }
        
        try:
            # Read original file
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # Check for syntax errors
            syntax_errors = self._check_syntax_errors(original_content, file_path)
            result['syntax_errors'] = syntax_errors
            
            if not syntax_errors:
                return result
            
            # Apply fixes
            fixed_content = original_content
            
            # Fix 1: Missing imports
            fixed_content, import_fixes = self._fix_missing_imports(fixed_content, file_path)
            result['fixes_applied'].extend(import_fixes)
            
            # Fix 2: Relative imports
            fixed_content, relative_fixes = self._fix_relative_imports(fixed_content)
            result['fixes_applied'].extend(relative_fixes)
            
            # Fix 3: Indentation errors
            fixed_content, indent_fixes = self._fix_indentation_errors(fixed_content)
            result['fixes_applied'].extend(indent_fixes)
            
            # Fix 4: Common syntax issues
            fixed_content, syntax_fixes = self._fix_common_syntax_issues(fixed_content)
            result['fixes_applied'].extend(syntax_fixes)
            
            # Check if fixes resolved the issues
            remaining_errors = self._check_syntax_errors(fixed_content, file_path)
            result['remaining_errors'] = remaining_errors
            
            # Write fixed content if improvements were made
            if result['fixes_applied'] and len(remaining_errors) < len(syntax_errors):
                # Create backup
                backup_path = f"{file_path}.backup"
                shutil.copy2(file_path, backup_path)
                
                # Write fixed content
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(fixed_content)
                
                self.logger.info(f"Applied {len(result['fixes_applied'])} fixes to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to fix syntax in {file_path}: {e}")
            result['remaining_errors'].append(f"Fix failed: {str(e)}")
        
        return result
    
    def _check_syntax_errors(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """Check for syntax errors in Python code"""
        errors = []
        
        try:
            ast.parse(content)
        except SyntaxError as e:
            errors.append({
                'type': 'SyntaxError',
                'line': e.lineno,
                'column': e.offset,
                'message': e.msg,
                'text': e.text
            })
        except Exception as e:
            errors.append({
                'type': 'ParseError',
                'line': 0,
                'column': 0,
                'message': str(e),
                'text': ''
            })
        
        return errors
    
    def _fix_missing_imports(self, content: str, file_path: str) -> Tuple[str, List[str]]:
        """Fix missing import statements"""
        fixes_applied = []
        lines = content.splitlines()
        
        # Detect missing imports based on usage
        for module, import_statement in self.common_fixes['missing_imports'].items():
            if module in content and not self._has_import(content, module):
                # Add import at the top
                lines.insert(0, import_statement)
                fixes_applied.append(f"Added missing import: {import_statement}")
        
        return '\n'.join(lines), fixes_applied
    
    def _fix_relative_imports(self, content: str) -> Tuple[str, List[str]]:
        """Fix relative import statements"""
        fixes_applied = []
        
        for pattern, replacement in self.common_fixes['relative_import_fixes'].items():
            if re.search(pattern, content):
                content = re.sub(pattern, replacement, content)
                fixes_applied.append(f"Fixed relative import: {pattern} -> {replacement}")
        
        return content, fixes_applied
    
    def _fix_indentation_errors(self, content: str) -> Tuple[str, List[str]]:
        """Fix common indentation errors"""
        fixes_applied = []
        lines = content.splitlines()
        fixed_lines = []
        
        for i, line in enumerate(lines):
            original_line = line
            
            # Fix mixed tabs and spaces
            if '\t' in line and '    ' in line:
                line = line.replace('\t', '    ')
                if line != original_line:
                    fixes_applied.append(f"Line {i+1}: Fixed mixed tabs and spaces")
            
            # Fix trailing whitespace
            stripped_line = line.rstrip()
            if len(stripped_line) != len(line):
                line = stripped_line
                if line != original_line:
                    fixes_applied.append(f"Line {i+1}: Removed trailing whitespace")
            
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines), fixes_applied
    
    def _fix_common_syntax_issues(self, content: str) -> Tuple[str, List[str]]:
        """Fix common syntax issues"""
        fixes_applied = []
        
        # Fix missing colons in function/class definitions
        colon_fixes = [
            (r'^(\s*def\s+\w+\([^)]*\))\s*$', r'\1:'),
            (r'^(\s*class\s+\w+(?:\([^)]*\))?)\s*$', r'\1:'),
            (r'^(\s*if\s+[^:]+)\s*$', r'\1:'),
            (r'^(\s*for\s+[^:]+)\s*$', r'\1:'),
            (r'^(\s*while\s+[^:]+)\s*$', r'\1:'),
            (r'^(\s*try)\s*$', r'\1:'),
            (r'^(\s*except[^:]*)\s*$', r'\1:'),
            (r'^(\s*finally)\s*$', r'\1:'),
            (r'^(\s*else)\s*$', r'\1:'),
            (r'^(\s*elif\s+[^:]+)\s*$', r'\1:')
        ]
        
        for pattern, replacement in colon_fixes:
            if re.search(pattern, content, re.MULTILINE):
                content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
                fixes_applied.append(f"Added missing colon: {pattern}")
        
        # Fix string quote mismatches (basic)
        quote_fixes = [
            (r"'([^']*)'([^']*)'", r'"\1\2"'),  # Fix mixed quotes
        ]
        
        for pattern, replacement in quote_fixes:
            if re.search(pattern, content):
                content = re.sub(pattern, replacement, content)
                fixes_applied.append(f"Fixed quote mismatch: {pattern}")
        
        return content, fixes_applied
    
    def _has_import(self, content: str, module: str) -> bool:
        """Check if a module is already imported"""
        import_patterns = [
            f'import {module}',
            f'from {module} import',
            f'import {module} as',
            f'from {module}.'
        ]
        
        for pattern in import_patterns:
            if pattern in content:
                return True
        
        return False
    
    def _apply_additional_fixes(self, files: List[str]) -> Dict[str, Any]:
        """Apply additional fixes across all files"""
        additional_fixes = {
            'encoding_fixes': 0,
            'shebang_fixes': 0,
            'docstring_fixes': 0
        }
        
        try:
            for file_path in files:
                # Add encoding declaration if missing
                if self._add_encoding_declaration(file_path):
                    additional_fixes['encoding_fixes'] += 1
                
                # Add shebang for executable files
                if self._add_shebang_if_needed(file_path):
                    additional_fixes['shebang_fixes'] += 1
                
                # Fix docstring formatting
                if self._fix_docstring_formatting(file_path):
                    additional_fixes['docstring_fixes'] += 1
                    
        except Exception as e:
            self.logger.error(f"Additional fixes failed: {e}")
        
        return additional_fixes
    
    def _add_encoding_declaration(self, file_path: str) -> bool:
        """Add encoding declaration if missing"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Check if encoding is already declared
            for line in lines[:3]:  # Check first 3 lines
                if 'coding' in line or 'encoding' in line:
                    return False
            
            # Add encoding declaration
            encoding_line = '# -*- coding: utf-8 -*-\n'
            
            # Insert after shebang if present
            insert_index = 0
            if lines and lines[0].startswith('#!'):
                insert_index = 1
            
            lines.insert(insert_index, encoding_line)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add encoding to {file_path}: {e}")
            return False
    
    def _add_shebang_if_needed(self, file_path: str) -> bool:
        """Add shebang for executable Python files"""
        try:
            # Only add to main files or scripts
            if not (file_path.endswith('main.py') or 'script' in file_path.lower()):
                return False
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if content.startswith('#!'):
                return False
            
            # Add shebang
            shebang = '#!/usr/bin/env python3\n'
            content = shebang + content
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add shebang to {file_path}: {e}")
            return False
    
    def _fix_docstring_formatting(self, file_path: str) -> bool:
        """Fix basic docstring formatting issues"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Fix single quotes in docstrings to triple quotes
            fixed_content = re.sub(
                r'^(\s*)(\'[^\']*\')\s*$',
                r'\1"""\2"""',
                content,
                flags=re.MULTILINE
            )
            
            if fixed_content != content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(fixed_content)
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to fix docstrings in {file_path}: {e}")
            return False
    
    def validate_fixed_files(self, files: List[str]) -> Dict[str, Any]:
        """Validate that fixed files have correct syntax"""
        validation_results = {
            'valid_files': [],
            'invalid_files': [],
            'total_files': len(files)
        }
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Try to parse the file
                ast.parse(content)
                validation_results['valid_files'].append(file_path)
                
            except SyntaxError as e:
                validation_results['invalid_files'].append({
                    'file': file_path,
                    'error': str(e),
                    'line': e.lineno
                })
            except Exception as e:
                validation_results['invalid_files'].append({
                    'file': file_path,
                    'error': str(e),
                    'line': 0
                })
        
        return validation_results
    
    def create_fix_report(self, results: Dict[str, Any]) -> str:
        """Create a detailed syntax fix report"""
        report_lines = [
            "YMERA Syntax Fix Report",
            "=" * 24,
            f"Generated: {self._get_timestamp()}",
            "",
            f"Files Processed: {results.get('files_processed', 0)}",
            f"Syntax Errors Found: {results.get('syntax_errors_found', 0)}",
            f"Syntax Errors Fixed: {results.get('syntax_errors_fixed', 0)}",
            f"Remaining Errors: {len(results.get('remaining_errors', []))}",
            ""
        ]
        
        # List fixes applied
        if results.get('fixes_applied'):
            report_lines.extend([
                "FIXES APPLIED:",
                "-" * 14
            ])
            
            for file_path, fixes in results['fixes_applied'].items():
                report_lines.append(f"\n{file_path}:")
                for fix in fixes:
                    report_lines.append(f"  ✅ {fix}")
        
        # List remaining errors
        if results.get('remaining_errors'):
            report_lines.extend([
                "",
                "REMAINING ERRORS:",
                "-" * 17
            ])
            
            for error_info in results['remaining_errors']:
                file_path = error_info['file']
                errors = error_info['errors']
                report_lines.append(f"\n{file_path}:")
                for error in errors:
                    line = error.get('line', 'unknown')
                    message = error.get('message', 'Unknown error')
                    report_lines.append(f"  ❌ Line {line}: {message}")
        
        # Additional fixes
        if results.get('additional_fixes'):
            additional = results['additional_fixes']
            report_lines.extend([
                "",
                "ADDITIONAL FIXES:",
                "-" * 17,
                f"  Encoding declarations added: {additional.get('encoding_fixes', 0)}",
                f"  Shebang lines added: {additional.get('shebang_fixes', 0)}",
                f"  Docstring fixes: {additional.get('docstring_fixes', 0)}"
            ])
        
        return "\n".join(report_lines)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
