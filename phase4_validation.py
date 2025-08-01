
#!/usr/bin/env python3
"""
YMERA Phase 4 Integration - Step 1: Phase 1-3 Validation
Validates all existing platform components before Phase 4 integration
"""

import asyncio
import httpx
import json
import time
from datetime import datetime
from typing import Dict, List, Any

class Phase123Validator:
    def __init__(self, base_url: str = "http://0.0.0.0:5000"):
        self.base_url = base_url
        self.validation_results = {
            "phase1": {},
            "phase2": {},
            "phase3": {},
            "websocket": False,
            "total_endpoints": 0,
            "successful_endpoints": 0,
            "failed_endpoints": []
        }
    
    async def validate_phase123_endpoints(self):
        """Comprehensive Phase 1-3 endpoint validation"""
        
        print("🚀 Starting YMERA Phase 1-3 Validation...")
        
        # Phase 1: Foundation Infrastructure Tests
        phase1_endpoints = [
            "/",                               # Main dashboard interface  
            "/health",                         # System health check
            "/api/projects",                   # Project management system
            "/api/users",                      # User management system
            "/api/files",                      # File management system
        ]
        
        # Phase 2: Core Functionality Tests
        phase2_endpoints = [
            "/api/dashboard/summary",          # Dashboard analytics
            "/api/agents/status",              # AI agent system status
            "/api/agents",                     # Agent management
            "/api/tasks",                      # Task management
        ]
        
        # Phase 3: Advanced Features Tests
        phase3_endpoints = [
            "/api/learning/metrics",           # Learning engine status
            "/api/learning/progress",          # Learning progress tracking
            "/api/analytics",                  # Advanced analytics
            "/api/reports",                    # System reporting
            "/api/search",                     # Semantic search capabilities
        ]
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Test Phase 1 endpoints
            print("🔍 Testing Phase 1: Foundation Infrastructure...")
            await self._test_endpoint_group(client, phase1_endpoints, "phase1")
            
            # Test Phase 2 endpoints
            print("🤖 Testing Phase 2: Core AI Functionality...")
            await self._test_endpoint_group(client, phase2_endpoints, "phase2")
            
            # Test Phase 3 endpoints
            print("🧠 Testing Phase 3: Advanced AI Features...")
            await self._test_endpoint_group(client, phase3_endpoints, "phase3")
        
        # Test WebSocket connection
        await self._test_websocket_connection()
        
        return self.validation_results
    
    async def _test_endpoint_group(self, client: httpx.AsyncClient, endpoints: List[str], phase: str):
        """Test a group of endpoints for a specific phase"""
        
        phase_results = {}
        
        for endpoint in endpoints:
            try:
                start_time = time.time()
                response = await client.get(f"{self.base_url}{endpoint}")
                response_time = time.time() - start_time
                
                success = response.status_code in [200, 201, 202]
                
                phase_results[endpoint] = {
                    "status_code": response.status_code,
                    "success": success,
                    "response_time": response_time,
                    "data": response.json() if success else None
                }
                
                if success:
                    self.validation_results["successful_endpoints"] += 1
                    print(f"  ✅ {endpoint} - OK ({response.status_code}) - {response_time:.3f}s")
                else:
                    self.validation_results["failed_endpoints"].append(endpoint)
                    print(f"  ❌ {endpoint} - FAILED ({response.status_code}) - {response_time:.3f}s")
                    
            except Exception as e:
                self.validation_results["failed_endpoints"].append(endpoint)
                phase_results[endpoint] = {
                    "status_code": 0,
                    "success": False,
                    "response_time": 0,
                    "error": str(e)
                }
                print(f"  ❌ {endpoint} - ERROR: {str(e)}")
            
            self.validation_results["total_endpoints"] += 1
        
        self.validation_results[phase] = phase_results
    
    async def _test_websocket_connection(self):
        """Test WebSocket connection"""
        print("🔌 Testing WebSocket Real-time Communication...")
        
        try:
            import websockets
            uri = f"ws://0.0.0.0:5000/ws"
            
            async with websockets.connect(uri, timeout=5.0) as websocket:
                # Send test message
                test_message = json.dumps({
                    "type": "test", 
                    "message": "Phase 4 validation"
                })
                await websocket.send(test_message)
                
                # Wait for response
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                self.validation_results["websocket"] = True
                print("  ✅ WebSocket - OK (Real-time communication active)")
                
        except Exception as e:
            self.validation_results["websocket"] = False
            print(f"  ❌ WebSocket - ERROR: {str(e)}")
    
    def generate_validation_report(self):
        """Generate comprehensive validation report"""
        
        success_rate = (self.validation_results["successful_endpoints"] / 
                       max(1, self.validation_results["total_endpoints"]) * 100)
        
        report = f"""
# 🎯 YMERA PHASES 1-3 VALIDATION REPORT

## 📊 VALIDATION SUMMARY
- **Validation Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Total Endpoints Tested**: {self.validation_results["total_endpoints"]}
- **Successful Endpoints**: {self.validation_results["successful_endpoints"]}
- **Failed Endpoints**: {len(self.validation_results["failed_endpoints"])}
- **Success Rate**: {success_rate:.1f}%
- **WebSocket Status**: {"✅ ACTIVE" if self.validation_results["websocket"] else "❌ FAILED"}

## 🏗️ PHASE 1: FOUNDATION INFRASTRUCTURE
"""
        
        # Phase 1 details
        for endpoint, data in self.validation_results["phase1"].items():
            status = "✅ OPERATIONAL" if data["success"] else "❌ FAILED"
            response_time = f"{data['response_time']:.3f}s" if data.get("response_time") else "N/A"
            report += f"- **{endpoint}**: {status} (Response: {response_time})\n"
        
        report += f"""
## 🤖 PHASE 2: CORE AI FUNCTIONALITY
"""
        
        # Phase 2 details
        for endpoint, data in self.validation_results["phase2"].items():
            status = "✅ OPERATIONAL" if data["success"] else "❌ FAILED"
            response_time = f"{data['response_time']:.3f}s" if data.get("response_time") else "N/A"
            report += f"- **{endpoint}**: {status} (Response: {response_time})\n"
        
        report += f"""
## 🧠 PHASE 3: ADVANCED AI FEATURES
"""
        
        # Phase 3 details
        for endpoint, data in self.validation_results["phase3"].items():
            status = "✅ OPERATIONAL" if data["success"] else "❌ FAILED"
            response_time = f"{data['response_time']:.3f}s" if data.get("response_time") else "N/A"
            report += f"- **{endpoint}**: {status} (Response: {response_time})\n"
        
        # Phase 4 Readiness Assessment
        phase4_ready = success_rate >= 85 and self.validation_results["websocket"]
        
        report += f"""
## 🔗 PHASE 4 INTEGRATION READINESS

### Readiness Criteria
- **Minimum 85% endpoint success rate**: {"✅ PASSED" if success_rate >= 85 else f"❌ FAILED ({success_rate:.1f}%)"}
- **WebSocket communication**: {"✅ PASSED" if self.validation_results["websocket"] else "❌ FAILED"}
- **Core system stability**: {"✅ STABLE" if len(self.validation_results["failed_endpoints"]) <= 2 else "⚠️ ISSUES"}

## 🚀 PHASE 4 READINESS ASSESSMENT

{"✅ READY FOR PHASE 4 INTEGRATION" if phase4_ready else "⚠️ REQUIRES FIXES BEFORE PHASE 4"}

### Failed Components (Must be fixed before Phase 4)
"""
        
        if self.validation_results["failed_endpoints"]:
            for endpoint in self.validation_results["failed_endpoints"]:
                report += f"- {endpoint} - Requires immediate attention\n"
        else:
            report += "- None - All systems operational\n"
        
        return report, phase4_ready

async def main():
    """Main validation execution"""
    
    print("🎯 YMERA PHASE 4 INTEGRATION - STEP 1: VALIDATION")
    print("=" * 60)
    
    validator = Phase123Validator()
    
    # Execute validation
    validation_results = await validator.validate_phase123_endpoints()
    
    # Generate report
    report, phase4_ready = validator.generate_validation_report()
    
    print(report)
    
    # Save report to file
    with open("phase123_validation_report.md", "w") as f:
        f.write(report)
    
    print(f"\n📄 Validation report saved to: phase123_validation_report.md")
    
    if phase4_ready:
        print("\n🎉 VALIDATION COMPLETE - READY FOR PHASE 4 INTEGRATION!")
        print("✅ All Phase 1-3 systems operational")
        print("✅ WebSocket communication active")
        print("✅ Proceeding to Phase 4 file processing...")
        return True
    else:
        print("\n⚠️ VALIDATION INCOMPLETE - FIXES REQUIRED")
        print("🔧 Please address failed components before Phase 4")
        print("🚨 Phase 4 integration cannot proceed until validation passes")
        return False

if __name__ == "__main__":
    asyncio.run(main())
