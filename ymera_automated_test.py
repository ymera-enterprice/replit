#!/usr/bin/env python3
"""
YMERA Platform Automated Testing Script
Run this script in your Replit environment to test all platform components
"""

import asyncio
import aiohttp
import websockets
import json
import time
from datetime import datetime
from typing import Dict, List, Any
import sqlite3
import os

class YMERATestSuite:
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.ws_url = base_url.replace("http", "ws") + "/ws"
        self.test_results = {
            "phase1": {"tests": [], "success_rate": 0},
            "phase2": {"tests": [], "success_rate": 0},
            "phase3": {"tests": [], "success_rate": 0},
            "phase4": {"tests": [], "success_rate": 0},
            "overall": {"success_rate": 0, "critical_issues": []}
        }
    
    async def test_phase1_foundation(self) -> Dict[str, Any]:
        """Test Phase 1: Server Infrastructure, Database, Basic API"""
        print("🔍 Testing Phase 1: Foundation...")
        phase1_results = []
        
        # Test 1: Server Health Check
        try:
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                async with session.get(f"{self.base_url}/health") as response:
                    response_time = time.time() - start_time
                    
                    if response.status == 200:
                        data = await response.json()
                        phase1_results.append({
                            "test": "Health Endpoint",
                            "status": "✅ PASSED",
                            "response_time": f"{response_time:.2f}s",
                            "details": f"Status: {data.get('status', 'unknown')}"
                        })
                    else:
                        phase1_results.append({
                            "test": "Health Endpoint", 
                            "status": "❌ FAILED",
                            "response_time": f"{response_time:.2f}s",
                            "details": f"HTTP {response.status}"
                        })
                        
        except Exception as e:
            phase1_results.append({
                "test": "Health Endpoint",
                "status": "❌ FAILED", 
                "response_time": "timeout",
                "details": f"Error: {str(e)}"
            })
        
        # Test 2: Main Page Load
        try:
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                async with session.get(self.base_url) as response:
                    response_time = time.time() - start_time
                    
                    if response.status == 200:
                        content = await response.text()
                        has_content = len(content) > 100  # Basic content check
                        
                        phase1_results.append({
                            "test": "Main Page Load",
                            "status": "✅ PASSED" if has_content else "⚠️ PARTIAL",
                            "response_time": f"{response_time:.2f}s",
                            "details": f"Content length: {len(content)} chars"
                        })
                    else:
                        phase1_results.append({
                            "test": "Main Page Load",
                            "status": "❌ FAILED",
                            "response_time": f"{response_time:.2f}s", 
                            "details": f"HTTP {response.status}"
                        })
                        
        except Exception as e:
            phase1_results.append({
                "test": "Main Page Load",
                "status": "❌ FAILED",
                "response_time": "timeout",
                "details": f"Error: {str(e)}"
            })
        
        # Test 3: API Documentation
        try:
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                async with session.get(f"{self.base_url}/docs") as response:
                    response_time = time.time() - start_time
                    
                    phase1_results.append({
                        "test": "API Documentation",
                        "status": "✅ PASSED" if response.status == 200 else "❌ FAILED",
                        "response_time": f"{response_time:.2f}s",
                        "details": f"HTTP {response.status}"
                    })
                    
        except Exception as e:
            phase1_results.append({
                "test": "API Documentation",
                "status": "❌ FAILED",
                "response_time": "timeout", 
                "details": f"Error: {str(e)}"
            })
        
        # Test 4: Database Connection
        try:
            if os.path.exists("ymera_platform.db") or os.path.exists("database.db"):
                db_path = "ymera_platform.db" if os.path.exists("ymera_platform.db") else "database.db"
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # Check if demo_user exists
                cursor.execute("SELECT * FROM users WHERE username = 'demo_user'")
                demo_user = cursor.fetchone()
                
                conn.close()
                
                phase1_results.append({
                    "test": "Database Connection",
                    "status": "✅ PASSED" if demo_user else "⚠️ PARTIAL",
                    "response_time": "< 0.1s",
                    "details": f"demo_user {'found' if demo_user else 'missing'}"
                })
            else:
                phase1_results.append({
                    "test": "Database Connection",
                    "status": "❌ FAILED",
                    "response_time": "N/A",
                    "details": "Database file not found"
                })
                
        except Exception as e:
            phase1_results.append({
                "test": "Database Connection",
                "status": "❌ FAILED",
                "response_time": "N/A",
                "details": f"Error: {str(e)}"
            })
        
        # Test 5: File Operations API
        try:
            async with aiohttp.ClientSession() as session:
                # Test file upload endpoint
                start_time = time.time()
                test_data = {"test": "data"}
                
                async with session.post(f"{self.base_url}/api/files/upload", 
                                      json=test_data) as response:
                    response_time = time.time() - start_time
                    
                    phase1_results.append({
                        "test": "File Operations API",
                        "status": "✅ PASSED" if response.status in [200, 201, 422] else "❌ FAILED",
                        "response_time": f"{response_time:.2f}s",
                        "details": f"HTTP {response.status} (422 expected for test data)"
                    })
                    
        except Exception as e:
            phase1_results.append({
                "test": "File Operations API",
                "status": "❌ FAILED",
                "response_time": "timeout",
                "details": f"Error: {str(e)}"
            })
        
        # Calculate Phase 1 success rate
        passed_tests = len([t for t in phase1_results if t["status"] == "✅ PASSED"])
        partial_tests = len([t for t in phase1_results if t["status"] == "⚠️ PARTIAL"])
        total_tests = len(phase1_results)
        
        success_rate = ((passed_tests + partial_tests * 0.5) / total_tests) * 100 if total_tests > 0 else 0
        
        self.test_results["phase1"] = {
            "tests": phase1_results,
            "success_rate": success_rate,
            "passed": passed_tests,
            "partial": partial_tests,
            "failed": total_tests - passed_tests - partial_tests,
            "total": total_tests
        }
        
        return self.test_results["phase1"]
    
    async def test_phase2_functionality(self) -> Dict[str, Any]:
        """Test Phase 2: WebSocket, Real-time Features, Multi-Agent System"""
        print("🔍 Testing Phase 2: Core Functionality...")
        phase2_results = []
        
        # Test 1: WebSocket Connection
        try:
            start_time = time.time()
            async with websockets.connect(self.ws_url) as websocket:
                connection_time = time.time() - start_time
                
                # Test basic message
                await websocket.send(json.dumps({"type": "test", "message": "hello"}))
                
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    phase2_results.append({
                        "test": "WebSocket Connection",
                        "status": "✅ PASSED",
                        "response_time": f"{connection_time:.2f}s",
                        "details": f"Connected and received: {response[:50]}..."
                    })
                except asyncio.TimeoutError:
                    phase2_results.append({
                        "test": "WebSocket Connection", 
                        "status": "⚠️ PARTIAL",
                        "response_time": f"{connection_time:.2f}s",
                        "details": "Connected but no response received"
                    })
                    
        except Exception as e:
            phase2_results.append({
                "test": "WebSocket Connection",
                "status": "❌ FAILED",
                "response_time": "timeout",
                "details": f"Error: {str(e)}"
            })
        
        # Test 2: Dashboard API
        try:
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                async with session.get(f"{self.base_url}/api/dashboard/summary") as response:
                    response_time = time.time() - start_time
                    
                    if response.status == 200:
                        data = await response.json()
                        phase2_results.append({
                            "test": "Dashboard API",
                            "status": "✅ PASSED",
                            "response_time": f"{response_time:.2f}s",
                            "details": f"Data keys: {list(data.keys()) if isinstance(data, dict) else 'Non-dict response'}"
                        })
                    else:
                        phase2_results.append({
                            "test": "Dashboard API",
                            "status": "❌ FAILED", 
                            "response_time": f"{response_time:.2f}s",
                            "details": f"HTTP {response.status}"
                        })
                        
        except Exception as e:
            phase2_results.append({
                "test": "Dashboard API",
                "status": "❌ FAILED",
                "response_time": "timeout",
                "details": f"Error: {str(e)}"
            })
        
        # Test 3: Agent System API
        try:
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                async with session.get(f"{self.base_url}/api/agents/status") as response:
                    response_time = time.time() - start_time
                    
                    phase2_results.append({
                        "test": "Agent System API",
                        "status": "✅ PASSED" if response.status == 200 else "❌ FAILED",
                        "response_time": f"{response_time:.2f}s",
                        "details": f"HTTP {response.status}"
                    })
                    
        except Exception as e:
            phase2_results.append({
                "test": "Agent System API",
                "status": "❌ FAILED",
                "response_time": "timeout",
                "details": f"Error: {str(e)}"
            })
        
        # Calculate Phase 2 success rate
        passed_tests = len([t for t in phase2_results if t["status"] == "✅ PASSED"])
        partial_tests = len([t for t in phase2_results if t["status"] == "⚠️ PARTIAL"])
        total_tests = len(phase2_results)
        
        success_rate = ((passed_tests + partial_tests * 0.5) / total_tests) * 100 if total_tests > 0 else 0
        
        self.test_results["phase2"] = {
            "tests": phase2_results,
            "success_rate": success_rate,
            "passed": passed_tests,
            "partial": partial_tests,
            "failed": total_tests - passed_tests - partial_tests,
            "total": total_tests
        }
        
        return self.test_results["phase2"]
    
    async def test_phase3_advanced(self) -> Dict[str, Any]:
        """Test Phase 3: Learning Engine, AI Services, Performance"""
        print("🔍 Testing Phase 3: Advanced Features...")
        phase3_results = []
        
        # Test 1: Learning Metrics API
        try:
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                async with session.get(f"{self.base_url}/api/learning/metrics") as response:
                    response_time = time.time() - start_time
                    
                    if response.status == 200:
                        data = await response.json()
                        phase3_results.append({
                            "test": "Learning Metrics API",
                            "status": "✅ PASSED",
                            "response_time": f"{response_time:.2f}s",
                            "details": f"Metrics received: {len(data) if isinstance(data, (list, dict)) else 'Invalid format'}"
                        })
                    else:
                        phase3_results.append({
                            "test": "Learning Metrics API",
                            "status": "❌ FAILED",
                            "response_time": f"{response_time:.2f}s",
                            "details": f"HTTP {response.status}"
                        })
                        
        except Exception as e:
            phase3_results.append({
                "test": "Learning Metrics API",
                "status": "❌ FAILED",
                "response_time": "timeout",
                "details": f"Error: {str(e)}"
            })
        
        # Test 2: Projects API
        try:
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                async with session.get(f"{self.base_url}/api/projects") as response:
                    response_time = time.time() - start_time
                    
                    phase3_results.append({
                        "test": "Projects API",
                        "status": "✅ PASSED" if response.status == 200 else "❌ FAILED",
                        "response_time": f"{response_time:.2f}s",
                        "details": f"HTTP {response.status}"
                    })
                    
        except Exception as e:
            phase3_results.append({
                "test": "Projects API",
                "status": "❌ FAILED",
                "response_time": "timeout",
                "details": f"Error: {str(e)}"
            })
        
        # Test 3: Messages API
        try:
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                async with session.get(f"{self.base_url}/api/messages") as response:
                    response_time = time.time() - start_time
                    
                    phase3_results.append({
                        "test": "Messages API",
                        "status": "✅ PASSED" if response.status == 200 else "❌ FAILED",
                        "response_time": f"{response_time:.2f}s",
                        "details": f"HTTP {response.status}"
                    })
                    
        except Exception as e:
            phase3_results.append({
                "test": "Messages API",
                "status": "❌ FAILED",
                "response_time": "timeout",
                "details": f"Error: {str(e)}"
            })
        
        # Calculate Phase 3 success rate
        passed_tests = len([t for t in phase3_results if t["status"] == "✅ PASSED"])
        partial_tests = len([t for t in phase3_results if t["status"] == "⚠️ PARTIAL"])
        total_tests = len(phase3_results)
        
        success_rate = ((passed_tests + partial_tests * 0.5) / total_tests) * 100 if total_tests > 0 else 0
        
        self.test_results["phase3"] = {
            "tests": phase3_results,
            "success_rate": success_rate,
            "passed": passed_tests,
            "partial": partial_tests,
            "failed": total_tests - passed_tests - partial_tests,
            "total": total_tests
        }
        
        return self.test_results["phase3"]
    
    async def test_phase4_integration(self) -> Dict[str, Any]:
        """Test Phase 4: E2E Integration, Performance, Reliability"""
        print("🔍 Testing Phase 4: Integration & Performance...")
        phase4_results = []
        
        # Test 1: Load Testing (Basic)
        try:
            start_time = time.time()
            tasks = []
            
            async with aiohttp.ClientSession() as session:
                # Create 5 concurrent requests
                for i in range(5):
                    task = session.get(f"{self.base_url}/health")
                    tasks.append(task)
                
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                total_time = time.time() - start_time
                
                successful_requests = len([r for r in responses if not isinstance(r, Exception) and r.status == 200])
                
                phase4_results.append({
                    "test": "Load Testing (5 concurrent)",
                    "status": "✅ PASSED" if successful_requests >= 4 else "❌ FAILED",
                    "response_time": f"{total_time:.2f}s",
                    "details": f"{successful_requests}/5 requests successful"
                })
                
        except Exception as e:
            phase4_results.append({
                "test": "Load Testing",
                "status": "❌ FAILED",
                "response_time": "timeout",
                "details": f"Error: {str(e)}"
            })
        
        # Test 2: API Response Consistency
        try:
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                
                # Test multiple endpoints for consistent responses
                endpoints = ["/health", "/api/projects", "/api/agents/status"]
                consistent_responses = 0
                
                for endpoint in endpoints:
                    try:
                        async with session.get(f"{self.base_url}{endpoint}") as response:
                            if response.status in [200, 401]:  # 401 is acceptable for protected endpoints
                                consistent_responses += 1
                    except:
                        pass
                
                response_time = time.time() - start_time
                
                phase4_results.append({
                    "test": "API Response Consistency",
                    "status": "✅ PASSED" if consistent_responses >= 2 else "❌ FAILED",
                    "response_time": f"{response_time:.2f}s",
                    "details": f"{consistent_responses}/{len(endpoints)} endpoints consistent"
                })
                
        except Exception as e:
            phase4_results.append({
                "test": "API Response Consistency",
                "status": "❌ FAILED",
                "response_time": "timeout",
                "details": f"Error: {str(e)}"
            })
        
        # Calculate Phase 4 success rate
        passed_tests = len([t for t in phase4_results if t["status"] == "✅ PASSED"])
        partial_tests = len([t for t in phase4_results if t["status"] == "⚠️ PARTIAL"])
        total_tests = len(phase4_results)
        
        success_rate = ((passed_tests + partial_tests * 0.5) / total_tests) * 100 if total_tests > 0 else 0
        
        self.test_results["phase4"] = {
            "tests": phase4_results,
            "success_rate": success_rate,
            "passed": passed_tests,
            "partial": partial_tests,
            "failed": total_tests - passed_tests - partial_tests,
            "total": total_tests
        }
        
        return self.test_results["phase4"]
    
    def print_detailed_results(self):
        """Print comprehensive test results"""
        print("\n" + "="*80)
        print("🏆 YMERA PLATFORM COMPREHENSIVE TEST RESULTS")
        print("="*80)
        
        total_passed = 0
        total_partial = 0
        total_failed = 0
        total_tests = 0
        
        for phase_name, phase_data in self.test_results.items():
            if phase_name == "overall":
                continue
                
            print(f"\n📊 {phase_name.upper()} RESULTS:")
            print(f"   Success Rate: {phase_data['success_rate']:.1f}%")
            print(f"   Tests: {phase_data.get('passed', 0)} passed, {phase_data.get('partial', 0)} partial, {phase_data.get('failed', 0)} failed")
            
            for test in phase_data["tests"]:
                print(f"   {test['status']} {test['test']} ({test['response_time']}) - {test['details']}")
            
            total_passed += phase_data.get('passed', 0)
            total_partial += phase_data.get('partial', 0) 
            total_failed += phase_data.get('failed', 0)
            total_tests += phase_data.get('total', 0)
        
        # Calculate overall success rate
        overall_success_rate = ((total_passed + total_partial * 0.5) / total_tests) * 100 if total_tests > 0 else 0
        
        print(f"\n🎯 OVERALL PLATFORM STATUS:")
        print(f"   Total Success Rate: {overall_success_rate:.1f}%")
        print(f"   Total Tests: {total_tests} ({total_passed} passed, {total_partial} partial, {total_failed} failed)")
        
        # Determine platform status
        if overall_success_rate >= 85:
            print(f"   Status: ✅ EXCELLENT - Platform ready for production!")
        elif overall_success_rate >= 70:
            print(f"   Status: ⚠️ GOOD - Minor issues to address")
        else:
            print(f"   Status: ❌ NEEDS WORK - Critical issues found")
        
        self.test_results["overall"]["success_rate"] = overall_success_rate
        
        return overall_success_rate
    
    async def run_all_tests(self):
        """Execute complete test suite"""
        print("🚀 Starting YMERA Platform Comprehensive Testing...")
        print(f"📍 Testing URL: {self.base_url}")
        print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Run all test phases
        await self.test_phase1_foundation()
        await self.test_phase2_functionality()
        await self.test_phase3_advanced()
        await self.test_phase4_integration()
        
        # Print results
        overall_rate = self.print_detailed_results()
        
        print(f"\n🏁 Testing completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return overall_rate

async def main():
    """Main test execution"""
    suite = YMERATestSuite()
    overall_success_rate = await suite.run_all_tests()
    
    return overall_success_rate

if __name__ == "__main__":
    try:
        success_rate = asyncio.run(main())
        print(f"\n🎯 Final Result: {success_rate:.1f}% success rate")
        
        if success_rate >= 85:
            exit(0)  # Success
        else:
            exit(1)  # Some issues found
            
    except Exception as e:
        print(f"\n❌ Test suite failed to run: {e}")
        exit(2)  # Critical failure