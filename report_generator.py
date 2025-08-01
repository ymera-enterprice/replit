"""
YMERA Report Generator
Comprehensive report generation system for YMERA Platform integration
"""

import os
import json
import html
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

class ReportGenerator:
    """Generate comprehensive reports for YMERA Platform integration"""
    
    def __init__(self):
        self.logger = logging.getLogger('report_generator')
        self.reports_dir = Path('reports')
        self.reports_dir.mkdir(exist_ok=True)
        
    def generate_comprehensive_report(self, integration_state: Dict[str, Any]) -> str:
        """Generate comprehensive integration report in multiple formats"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Generate text report
        txt_report_path = self.reports_dir / f'ymera_integration_report_{timestamp}.txt'
        self._generate_text_report(integration_state, txt_report_path)
        
        # Generate JSON report
        json_report_path = self.reports_dir / f'ymera_integration_report_{timestamp}.json'
        self._generate_json_report(integration_state, json_report_path)
        
        # Generate HTML report
        html_report_path = self.reports_dir / f'ymera_integration_report_{timestamp}.html'
        self._generate_html_report(integration_state, html_report_path)
        
        self.logger.info(f"Comprehensive reports generated: {txt_report_path.name}")
        return str(txt_report_path)
    
    def _generate_text_report(self, state: Dict[str, Any], output_path: Path):
        """Generate detailed text report"""
        lines = [
            "YMERA PLATFORM COMPREHENSIVE INTEGRATION REPORT",
            "=" * 50,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Report ID: {output_path.stem}",
            ""
        ]
        
        # Executive Summary
        lines.extend(self._generate_executive_summary(state))
        lines.append("")
        
        # Phase 1-2 Validation Results
        if state.get('validation_results'):
            lines.extend(self._generate_phase12_section(state['validation_results']))
            lines.append("")
        
        # Phase 3 Analysis Results
        if state.get('phase3_analysis'):
            lines.extend(self._generate_phase3_section(state['phase3_analysis']))
            lines.append("")
        
        # Integration Timeline
        lines.extend(self._generate_timeline_section(state))
        lines.append("")
        
        # Error Analysis
        if state.get('errors'):
            lines.extend(self._generate_error_analysis(state['errors']))
            lines.append("")
        
        # Logs Summary
        if state.get('logs'):
            lines.extend(self._generate_logs_summary(state['logs']))
            lines.append("")
        
        # Recommendations
        lines.extend(self._generate_recommendations(state))
        lines.append("")
        
        # Technical Details
        lines.extend(self._generate_technical_details(state))
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
    
    def _generate_json_report(self, state: Dict[str, Any], output_path: Path):
        """Generate structured JSON report"""
        report_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'report_version': '1.0',
                'ymera_platform': 'Phase 3 Integration System',
                'report_id': output_path.stem
            },
            'summary': self._get_summary_data(state),
            'validation_results': state.get('validation_results', {}),
            'phase3_analysis': state.get('phase3_analysis', {}),
            'timeline': self._get_timeline_data(state),
            'errors': state.get('errors', []),
            'logs': state.get('logs', [])[-100:],  # Last 100 log entries
            'recommendations': self._get_recommendations_data(state),
            'technical_metrics': self._get_technical_metrics(state)
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, default=str)
    
    def _generate_html_report(self, state: Dict[str, Any], output_path: Path):
        """Generate interactive HTML report"""
        html_content = self._build_html_report(state)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _generate_executive_summary(self, state: Dict[str, Any]) -> List[str]:
        """Generate executive summary section"""
        lines = [
            "📋 EXECUTIVE SUMMARY",
            "-" * 20
        ]
        
        # Overall status
        overall_status = self._determine_overall_status(state)
        status_emoji = "✅" if overall_status == "SUCCESS" else "⚠️" if overall_status == "WARNING" else "❌"
        lines.append(f"Overall Status: {status_emoji} {overall_status}")
        
        # Integration phase
        current_phase = state.get('current_phase', 'unknown')
        progress = state.get('progress', 0)
        lines.append(f"Current Phase: {current_phase.title()} ({progress}%)")
        
        # Runtime
        start_time = state.get('start_time')
        end_time = state.get('end_time')
        if start_time:
            start_dt = datetime.fromisoformat(start_time)
            if end_time:
                end_dt = datetime.fromisoformat(end_time)
                duration = end_dt - start_dt
                lines.append(f"Total Runtime: {self._format_duration(duration.total_seconds())}")
            else:
                current_duration = datetime.now() - start_dt
                lines.append(f"Runtime (In Progress): {self._format_duration(current_duration.total_seconds())}")
        
        # Component status summary
        validation_results = state.get('validation_results', {})
        if validation_results:
            total_components = len([k for k in validation_results.keys() if k != 'overall_status'])
            successful_components = len([v for k, v in validation_results.items() 
                                       if k != 'overall_status' and isinstance(v, dict) and v.get('success')])
            lines.append(f"Components Validated: {successful_components}/{total_components}")
        
        # Error summary
        errors = state.get('errors', [])
        if errors:
            lines.append(f"Critical Errors: {len(errors)}")
        
        return lines
    
    def _generate_phase12_section(self, validation_results: Dict[str, Any]) -> List[str]:
        """Generate Phase 1-2 validation section"""
        lines = [
            "🔐 PHASE 1-2 VALIDATION RESULTS",
            "-" * 32
        ]
        
        overall_status = validation_results.get('overall_status', 'UNKNOWN')
        status_emoji = "✅" if overall_status == "READY" else "❌"
        lines.append(f"Overall Status: {status_emoji} {overall_status}")
        lines.append("")
        
        # Component results
        components = {
            'architecture': '🏗️ System Architecture',
            'backend': '🔧 Backend Services',
            'frontend': '🌐 Frontend Application',
            'api': '🔗 API Endpoints',
            'security': '🛡️ Security Features'
        }
        
        for component_key, component_name in components.items():
            if component_key in validation_results:
                result = validation_results[component_key]
                if isinstance(result, dict):
                    success = result.get('success', False)
                    message = result.get('message', 'No message')
                    status_emoji = "✅" if success else "❌"
                    lines.append(f"{component_name}: {status_emoji} {message}")
                    
                    # Add detailed results if available
                    if 'tests' in result:
                        for test_name, test_result in result['tests'].items():
                            if isinstance(test_result, dict):
                                test_success = test_result.get('success', False)
                                test_message = test_result.get('message', '')
                                test_emoji = "  ✅" if test_success else "  ❌"
                                lines.append(f"{test_emoji} {test_name.replace('_', ' ').title()}: {test_message}")
        
        return lines
    
    def _generate_phase3_section(self, phase3_analysis: Dict[str, Any]) -> List[str]:
        """Generate Phase 3 analysis section"""
        lines = [
            "🤖 PHASE 3 AI AGENTS ANALYSIS",
            "-" * 29
        ]
        
        # File analysis
        if 'files' in phase3_analysis:
            files_data = phase3_analysis['files']
            files_found = len(files_data.get('files_found', []))
            syntax_errors = len(files_data.get('syntax_errors', []))
            lines.append(f"📁 Files Found: {files_found}")
            lines.append(f"🔧 Syntax Errors: {syntax_errors}")
            
            if files_data.get('ai_imports'):
                ai_imports_count = sum(len(imports) for imports in files_data['ai_imports'].values())
                lines.append(f"🧠 AI Imports Detected: {ai_imports_count}")
        
        # Dependencies
        if 'dependencies' in phase3_analysis:
            deps_data = phase3_analysis['dependencies']
            detected = len(deps_data.get('detected_dependencies', []))
            missing = len(deps_data.get('missing_dependencies', []))
            installed = len(deps_data.get('installed_dependencies', []))
            lines.append(f"📦 Dependencies - Detected: {detected}, Missing: {missing}, Installed: {installed}")
        
        # Syntax fixes
        if 'syntax_fixes' in phase3_analysis:
            fixes_data = phase3_analysis['syntax_fixes']
            fixes_applied = fixes_data.get('syntax_errors_fixed', 0)
            remaining_errors = len(fixes_data.get('remaining_errors', []))
            lines.append(f"🔧 Syntax Fixes Applied: {fixes_applied}, Remaining Errors: {remaining_errors}")
        
        # Integration results
        if 'integration' in phase3_analysis:
            integration_data = phase3_analysis['integration']
            if 'tests' in integration_data:
                test_results = integration_data['tests']
                successful_tests = len([t for t in test_results.values() if isinstance(t, dict) and t.get('success')])
                total_tests = len(test_results)
                lines.append(f"🧪 Integration Tests: {successful_tests}/{total_tests} passed")
        
        return lines
    
    def _generate_timeline_section(self, state: Dict[str, Any]) -> List[str]:
        """Generate timeline section"""
        lines = [
            "⏱️ INTEGRATION TIMELINE",
            "-" * 20
        ]
        
        start_time = state.get('start_time')
        end_time = state.get('end_time')
        current_phase = state.get('current_phase', 'unknown')
        progress = state.get('progress', 0)
        
        if start_time:
            start_dt = datetime.fromisoformat(start_time)
            lines.append(f"Started: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
            
            if end_time:
                end_dt = datetime.fromisoformat(end_time)
                lines.append(f"Completed: {end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
                duration = end_dt - start_dt
                lines.append(f"Total Duration: {self._format_duration(duration.total_seconds())}")
            else:
                lines.append(f"Status: In Progress ({current_phase} - {progress}%)")
                current_duration = datetime.now() - start_dt
                lines.append(f"Elapsed Time: {self._format_duration(current_duration.total_seconds())}")
        
        # Phase breakdown
        phases = [
            ('idle', 'System Ready'),
            ('validation', 'Phase 1-2 Validation'),
            ('phase3_analysis', 'Phase 3 Analysis'),
            ('completed', 'Integration Complete')
        ]
        
        lines.append("")
        lines.append("Phase Progress:")
        for phase_key, phase_name in phases:
            if phase_key == current_phase:
                lines.append(f"  ▶️ {phase_name} ({progress}%)")
            elif self._is_phase_completed(phase_key, current_phase, state):
                lines.append(f"  ✅ {phase_name}")
            else:
                lines.append(f"  ⏳ {phase_name}")
        
        return lines
    
    def _generate_error_analysis(self, errors: List[Dict[str, Any]]) -> List[str]:
        """Generate error analysis section"""
        lines = [
            "❌ ERROR ANALYSIS",
            "-" * 15
        ]
        
        if not errors:
            lines.append("No critical errors detected.")
            return lines
        
        lines.append(f"Total Errors: {len(errors)}")
        lines.append("")
        
        # Group errors by type
        error_types = {}
        for error in errors:
            error_level = error.get('level', 'unknown')
            if error_level not in error_types:
                error_types[error_level] = []
            error_types[error_level].append(error)
        
        for error_type, error_list in error_types.items():
            lines.append(f"{error_type.upper()} ({len(error_list)}):")
            for i, error in enumerate(error_list[:5], 1):  # Show first 5 errors
                timestamp = error.get('timestamp', 'Unknown time')
                message = error.get('message', 'No message')
                lines.append(f"  {i}. [{timestamp}] {message}")
            
            if len(error_list) > 5:
                lines.append(f"  ... and {len(error_list) - 5} more {error_type} errors")
            lines.append("")
        
        return lines
    
    def _generate_logs_summary(self, logs: List[Dict[str, Any]]) -> List[str]:
        """Generate logs summary section"""
        lines = [
            "📜 LOGS SUMMARY",
            "-" * 13
        ]
        
        if not logs:
            lines.append("No logs available.")
            return lines
        
        lines.append(f"Total Log Entries: {len(logs)}")
        
        # Log level breakdown
        log_levels = {}
        for log in logs:
            level = log.get('level', 'unknown')
            log_levels[level] = log_levels.get(level, 0) + 1
        
        lines.append("")
        lines.append("Log Level Breakdown:")
        for level, count in sorted(log_levels.items()):
            emoji = {"error": "❌", "warning": "⚠️", "info": "ℹ️", "debug": "🔍"}.get(level.lower(), "📝")
            lines.append(f"  {emoji} {level.upper()}: {count}")
        
        # Recent logs
        lines.append("")
        lines.append("Recent Log Entries:")
        recent_logs = logs[-10:] if len(logs) > 10 else logs
        for log in recent_logs:
            timestamp = log.get('timestamp', 'Unknown')
            level = log.get('level', 'INFO')
            message = log.get('message', 'No message')
            lines.append(f"  [{timestamp}] {level}: {message}")
        
        return lines
    
    def _generate_recommendations(self, state: Dict[str, Any]) -> List[str]:
        """Generate recommendations section"""
        lines = [
            "💡 RECOMMENDATIONS",
            "-" * 17
        ]
        
        recommendations = []
        
        # Based on validation results
        validation_results = state.get('validation_results', {})
        if validation_results.get('overall_status') == 'NEEDS_FIXES':
            recommendations.append("🔧 Address Phase 1-2 validation failures before proceeding to Phase 3")
        
        # Based on errors
        errors = state.get('errors', [])
        if errors:
            recommendations.append(f"❌ Resolve {len(errors)} critical errors identified during integration")
        
        # Based on Phase 3 analysis
        phase3_analysis = state.get('phase3_analysis', {})
        if phase3_analysis:
            syntax_fixes = phase3_analysis.get('syntax_fixes', {})
            remaining_errors = syntax_fixes.get('remaining_errors', [])
            if remaining_errors:
                recommendations.append(f"🔧 Fix {len(remaining_errors)} remaining syntax errors in Phase 3 files")
            
            dependencies = phase3_analysis.get('dependencies', {})
            missing_deps = dependencies.get('missing_dependencies', [])
            if missing_deps:
                recommendations.append(f"📦 Install {len(missing_deps)} missing dependencies for full functionality")
        
        # Security recommendations
        if validation_results.get('security', {}).get('success') != True:
            recommendations.append("🛡️ Review and strengthen security configurations")
        
        # Performance recommendations
        recommendations.extend([
            "📊 Monitor system performance after integration completion",
            "🧪 Run comprehensive end-to-end tests on the integrated system",
            "📝 Document any configuration changes made during integration",
            "🔄 Set up automated health checks for ongoing monitoring"
        ])
        
        # AI service configuration
        phase3_integration = phase3_analysis.get('integration', {})
        if phase3_integration:
            ai_services = phase3_integration.get('tests', {}).get('ai_services', {})
            if ai_services and not ai_services.get('any_configured', False):
                recommendations.append("🤖 Configure AI service API keys (OpenAI, Anthropic, Pinecone) for full AI functionality")
        
        if not recommendations:
            recommendations.append("✅ All systems appear to be functioning correctly")
        
        lines.extend(recommendations)
        return lines
    
    def _generate_technical_details(self, state: Dict[str, Any]) -> List[str]:
        """Generate technical details section"""
        lines = [
            "🔧 TECHNICAL DETAILS",
            "-" * 18
        ]
        
        # System Information
        lines.append("System Information:")
        lines.append(f"  Platform: YMERA Phase 3 Integration System")
        lines.append(f"  Python Version: {self._get_python_version()}")
        lines.append(f"  Working Directory: {os.getcwd()}")
        lines.append("")
        
        # File Statistics
        validation_results = state.get('validation_results', {})
        phase3_analysis = state.get('phase3_analysis', {})
        
        if phase3_analysis.get('files'):
            files_data = phase3_analysis['files']
            files_found = files_data.get('files_found', [])
            lines.append("File Analysis:")
            lines.append(f"  Phase 3 Files Discovered: {len(files_found)}")
            
            if files_data.get('file_analysis'):
                total_lines = sum(analysis.get('line_count', 0) 
                                for analysis in files_data['file_analysis'].values())
                total_functions = sum(len(analysis.get('functions', [])) 
                                    for analysis in files_data['file_analysis'].values())
                total_classes = sum(len(analysis.get('classes', [])) 
                                  for analysis in files_data['file_analysis'].values())
                
                lines.append(f"  Total Lines of Code: {total_lines}")
                lines.append(f"  Total Functions: {total_functions}")
                lines.append(f"  Total Classes: {total_classes}")
            lines.append("")
        
        # Integration Metrics
        if phase3_analysis.get('integration'):
            integration_data = phase3_analysis['integration']
            if 'tests' in integration_data:
                test_results = integration_data['tests']
                lines.append("Integration Test Results:")
                for test_name, test_result in test_results.items():
                    if isinstance(test_result, dict):
                        success = test_result.get('success', False)
                        message = test_result.get('message', 'No details')
                        status = "✅" if success else "❌"
                        lines.append(f"  {status} {test_name.replace('_', ' ').title()}: {message}")
                lines.append("")
        
        # Environment Configuration
        lines.append("Environment Configuration:")
        env_vars = ['DATABASE_URL', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'PINECONE_API_KEY']
        for var in env_vars:
            value = os.getenv(var, '')
            status = "✅ Configured" if value and value != 'test_key' else "❌ Not configured"
            lines.append(f"  {var}: {status}")
        
        return lines
    
    def _build_html_report(self, state: Dict[str, Any]) -> str:
        """Build interactive HTML report"""
        overall_status = self._determine_overall_status(state)
        status_class = "success" if overall_status == "SUCCESS" else "warning" if overall_status == "WARNING" else "danger"
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YMERA Integration Report</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/feather-icons@4.28.0/dist/feather.css" rel="stylesheet">
    <style>
        body {{ background-color: #f8f9fa; }}
        .report-header {{ background: linear-gradient(135deg, #0066cc, #0052a3); color: white; }}
        .status-badge {{ font-size: 1.1rem; }}
        .metric-card {{ border-left: 4px solid #0066cc; }}
        .log-entry {{ font-family: 'Courier New', monospace; font-size: 0.875rem; }}
        .timeline-item {{ border-left: 3px solid #dee2e6; padding-left: 1rem; margin-bottom: 1rem; }}
        .timeline-item.active {{ border-left-color: #0066cc; }}
        .timeline-item.completed {{ border-left-color: #28a745; }}
    </style>
</head>
<body>
    <div class="container-fluid">
        <div class="report-header py-4 mb-4">
            <div class="container">
                <h1 class="display-6 mb-2">YMERA Platform Integration Report</h1>
                <p class="lead mb-0">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        </div>
        
        <div class="container">
            <!-- Executive Summary -->
            <div class="row mb-4">
                <div class="col">
                    <div class="card">
                        <div class="card-header">
                            <h5 class="mb-0"><i data-feather="clipboard"></i> Executive Summary</h5>
                        </div>
                        <div class="card-body">
                            <div class="row">
                                <div class="col-md-6">
                                    <h6>Overall Status</h6>
                                    <span class="badge bg-{status_class} status-badge">{overall_status}</span>
                                </div>
                                <div class="col-md-6">
                                    <h6>Current Phase</h6>
                                    <div class="progress">
                                        <div class="progress-bar" style="width: {state.get('progress', 0)}%"></div>
                                    </div>
                                    <small class="text-muted">{state.get('current_phase', 'unknown').title()} ({state.get('progress', 0)}%)</small>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Metrics Row -->
            {self._build_metrics_section(state)}
            
            <!-- Validation Results -->
            {self._build_validation_section(state)}
            
            <!-- Phase 3 Analysis -->
            {self._build_phase3_section(state)}
            
            <!-- Timeline -->
            {self._build_timeline_section(state)}
            
            <!-- Errors and Logs -->
            {self._build_errors_logs_section(state)}
            
            <!-- Recommendations -->
            {self._build_recommendations_section(state)}
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/feather-icons@4.28.0/dist/feather.min.js"></script>
    <script>feather.replace();</script>
</body>
</html>
        """
        
        return html
    
    def _build_metrics_section(self, state: Dict[str, Any]) -> str:
        """Build metrics section for HTML report"""
        validation_results = state.get('validation_results', {})
        phase3_analysis = state.get('phase3_analysis', {})
        errors = state.get('errors', [])
        
        # Calculate metrics
        total_components = len([k for k in validation_results.keys() if k != 'overall_status'])
        successful_components = len([v for k, v in validation_results.items() 
                                   if k != 'overall_status' and isinstance(v, dict) and v.get('success')])
        
        files_found = 0
        if phase3_analysis.get('files'):
            files_found = len(phase3_analysis['files'].get('files_found', []))
        
        dependencies_installed = 0
        if phase3_analysis.get('dependencies'):
            dependencies_installed = len(phase3_analysis['dependencies'].get('installed_dependencies', []))
        
        return f"""
            <div class="row mb-4">
                <div class="col-md-3">
                    <div class="card metric-card">
                        <div class="card-body text-center">
                            <h3 class="text-primary">{successful_components}/{total_components}</h3>
                            <p class="card-text">Components Validated</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card metric-card">
                        <div class="card-body text-center">
                            <h3 class="text-info">{files_found}</h3>
                            <p class="card-text">Phase 3 Files</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card metric-card">
                        <div class="card-body text-center">
                            <h3 class="text-success">{dependencies_installed}</h3>
                            <p class="card-text">Dependencies Installed</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card metric-card">
                        <div class="card-body text-center">
                            <h3 class="text-danger">{len(errors)}</h3>
                            <p class="card-text">Critical Errors</p>
                        </div>
                    </div>
                </div>
            </div>
        """
    
    def _build_validation_section(self, state: Dict[str, Any]) -> str:
        """Build validation section for HTML report"""
        validation_results = state.get('validation_results', {})
        if not validation_results:
            return ""
        
        components_html = ""
        components = {
            'architecture': ('🏗️', 'System Architecture'),
            'backend': ('🔧', 'Backend Services'),
            'frontend': ('🌐', 'Frontend Application'),
            'api': ('🔗', 'API Endpoints'),
            'security': ('🛡️', 'Security Features')
        }
        
        for component_key, (emoji, component_name) in components.items():
            if component_key in validation_results:
                result = validation_results[component_key]
                if isinstance(result, dict):
                    success = result.get('success', False)
                    message = result.get('message', 'No message')
                    status_class = "success" if success else "danger"
                    status_icon = "✅" if success else "❌"
                    
                    components_html += f"""
                        <div class="col-md-6 mb-3">
                            <div class="card">
                                <div class="card-body">
                                    <h6>{emoji} {component_name}</h6>
                                    <span class="badge bg-{status_class}">{status_icon} {message}</span>
                                </div>
                            </div>
                        </div>
                    """
        
        return f"""
            <div class="row mb-4">
                <div class="col">
                    <div class="card">
                        <div class="card-header">
                            <h5 class="mb-0"><i data-feather="shield"></i> Phase 1-2 Validation Results</h5>
                        </div>
                        <div class="card-body">
                            <div class="row">
                                {components_html}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        """
    
    def _build_phase3_section(self, state: Dict[str, Any]) -> str:
        """Build Phase 3 section for HTML report"""
        phase3_analysis = state.get('phase3_analysis', {})
        if not phase3_analysis:
            return ""
        
        return f"""
            <div class="row mb-4">
                <div class="col">
                    <div class="card">
                        <div class="card-header">
                            <h5 class="mb-0"><i data-feather="cpu"></i> Phase 3 AI Agents Analysis</h5>
                        </div>
                        <div class="card-body">
                            <div class="row">
                                <div class="col-md-6">
                                    <h6>File Analysis</h6>
                                    <ul class="list-unstyled">
                                        <li>📁 Files Found: {len(phase3_analysis.get('files', {}).get('files_found', []))}</li>
                                        <li>🔧 Syntax Errors: {len(phase3_analysis.get('files', {}).get('syntax_errors', []))}</li>
                                    </ul>
                                </div>
                                <div class="col-md-6">
                                    <h6>Dependencies</h6>
                                    <ul class="list-unstyled">
                                        <li>📦 Installed: {len(phase3_analysis.get('dependencies', {}).get('installed_dependencies', []))}</li>
                                        <li>🔧 Fixes Applied: {phase3_analysis.get('syntax_fixes', {}).get('syntax_errors_fixed', 0)}</li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        """
    
    def _build_timeline_section(self, state: Dict[str, Any]) -> str:
        """Build timeline section for HTML report"""
        current_phase = state.get('current_phase', 'unknown')
        
        phases = [
            ('idle', 'System Ready'),
            ('validation', 'Phase 1-2 Validation'),
            ('phase3_analysis', 'Phase 3 Analysis'),
            ('completed', 'Integration Complete')
        ]
        
        timeline_html = ""
        for phase_key, phase_name in phases:
            if phase_key == current_phase:
                class_name = "timeline-item active"
                icon = "▶️"
            elif self._is_phase_completed(phase_key, current_phase, state):
                class_name = "timeline-item completed"
                icon = "✅"
            else:
                class_name = "timeline-item"
                icon = "⏳"
            
            timeline_html += f"""
                <div class="{class_name}">
                    <h6>{icon} {phase_name}</h6>
                </div>
            """
        
        return f"""
            <div class="row mb-4">
                <div class="col">
                    <div class="card">
                        <div class="card-header">
                            <h5 class="mb-0"><i data-feather="clock"></i> Integration Timeline</h5>
                        </div>
                        <div class="card-body">
                            {timeline_html}
                        </div>
                    </div>
                </div>
            </div>
        """
    
    def _build_errors_logs_section(self, state: Dict[str, Any]) -> str:
        """Build errors and logs section for HTML report"""
        errors = state.get('errors', [])
        logs = state.get('logs', [])
        
        errors_html = ""
        if errors:
            for error in errors[-5:]:  # Show last 5 errors
                timestamp = error.get('timestamp', 'Unknown')
                message = html.escape(error.get('message', 'No message'))
                errors_html += f"""
                    <div class="alert alert-danger alert-sm mb-2">
                        <small class="text-muted">[{timestamp}]</small><br>
                        {message}
                    </div>
                """
        else:
            errors_html = "<p class='text-muted'>No critical errors detected.</p>"
        
        logs_html = ""
        if logs:
            for log in logs[-10:]:  # Show last 10 logs
                timestamp = log.get('timestamp', 'Unknown')
                level = log.get('level', 'INFO')
                message = html.escape(log.get('message', 'No message'))
                level_class = {
                    'ERROR': 'text-danger',
                    'WARNING': 'text-warning',
                    'INFO': 'text-info',
                    'DEBUG': 'text-muted'
                }.get(level.upper(), 'text-dark')
                
                logs_html += f"""
                    <div class="log-entry {level_class} mb-1">
                        <small>[{timestamp}] {level}:</small> {message}
                    </div>
                """
        else:
            logs_html = "<p class='text-muted'>No logs available.</p>"
        
        return f"""
            <div class="row mb-4">
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">
                            <h5 class="mb-0"><i data-feather="alert-triangle"></i> Recent Errors</h5>
                        </div>
                        <div class="card-body">
                            {errors_html}
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">
                            <h5 class="mb-0"><i data-feather="file-text"></i> Recent Logs</h5>
                        </div>
                        <div class="card-body" style="max-height: 300px; overflow-y: auto;">
                            {logs_html}
                        </div>
                    </div>
                </div>
            </div>
        """
    
    def _build_recommendations_section(self, state: Dict[str, Any]) -> str:
        """Build recommendations section for HTML report"""
        recommendations = self._generate_recommendations(state)
        recommendations_content = recommendations[2:]  # Skip header
        
        recommendations_html = ""
        for rec in recommendations_content:
            recommendations_html += f"<li>{html.escape(rec)}</li>"
        
        return f"""
            <div class="row mb-4">
                <div class="col">
                    <div class="card">
                        <div class="card-header">
                            <h5 class="mb-0"><i data-feather="lightbulb"></i> Recommendations</h5>
                        </div>
                        <div class="card-body">
                            <ul>
                                {recommendations_html}
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
        """
    
    # Helper methods
    
    def _determine_overall_status(self, state: Dict[str, Any]) -> str:
        """Determine overall integration status"""
        errors = state.get('errors', [])
        if errors:
            return "ERROR"
        
        validation_results = state.get('validation_results', {})
        if validation_results.get('overall_status') == 'NEEDS_FIXES':
            return "WARNING"
        
        current_phase = state.get('current_phase', 'idle')
        status = state.get('status', 'ready')
        
        if status == 'completed':
            return "SUCCESS"
        elif status == 'error':
            return "ERROR"
        elif status == 'running':
            return "IN_PROGRESS"
        else:
            return "WARNING"
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format"""
        if seconds < 60:
            return f"{seconds:.1f} seconds"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f} minutes"
        else:
            hours = seconds / 3600
            return f"{hours:.1f} hours"
    
    def _is_phase_completed(self, phase: str, current_phase: str, state: Dict[str, Any]) -> bool:
        """Check if a phase has been completed"""
        phase_order = ['idle', 'validation', 'phase3_analysis', 'completed']
        
        try:
            phase_index = phase_order.index(phase)
            current_index = phase_order.index(current_phase)
            return phase_index < current_index
        except ValueError:
            return False
    
    def _get_summary_data(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Get summary data for JSON report"""
        return {
            'overall_status': self._determine_overall_status(state),
            'current_phase': state.get('current_phase', 'unknown'),
            'progress': state.get('progress', 0),
            'start_time': state.get('start_time'),
            'end_time': state.get('end_time'),
            'total_errors': len(state.get('errors', [])),
            'total_logs': len(state.get('logs', []))
        }
    
    def _get_timeline_data(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Get timeline data for JSON report"""
        start_time = state.get('start_time')
        end_time = state.get('end_time')
        
        timeline = {
            'start_time': start_time,
            'end_time': end_time,
            'current_phase': state.get('current_phase'),
            'progress': state.get('progress', 0)
        }
        
        if start_time:
            start_dt = datetime.fromisoformat(start_time)
            if end_time:
                end_dt = datetime.fromisoformat(end_time)
                timeline['duration_seconds'] = (end_dt - start_dt).total_seconds()
            else:
                timeline['elapsed_seconds'] = (datetime.now() - start_dt).total_seconds()
        
        return timeline
    
    def _get_recommendations_data(self, state: Dict[str, Any]) -> List[str]:
        """Get recommendations data for JSON report"""
        recommendations = self._generate_recommendations(state)
        return recommendations[2:]  # Skip header
    
    def _get_technical_metrics(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Get technical metrics for JSON report"""
        metrics = {
            'python_version': self._get_python_version(),
            'working_directory': os.getcwd(),
            'report_generated_at': datetime.now().isoformat()
        }
        
        # Add file metrics
        phase3_analysis = state.get('phase3_analysis', {})
        if phase3_analysis.get('files'):
            files_data = phase3_analysis['files']
            metrics['files'] = {
                'total_files': len(files_data.get('files_found', [])),
                'syntax_errors': len(files_data.get('syntax_errors', []))
            }
            
            if files_data.get('file_analysis'):
                metrics['code_metrics'] = {
                    'total_lines': sum(analysis.get('line_count', 0) 
                                     for analysis in files_data['file_analysis'].values()),
                    'total_functions': sum(len(analysis.get('functions', [])) 
                                         for analysis in files_data['file_analysis'].values()),
                    'total_classes': sum(len(analysis.get('classes', [])) 
                                       for analysis in files_data['file_analysis'].values())
                }
        
        return metrics
    
    def _get_python_version(self) -> str:
        """Get Python version"""
        import sys
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
