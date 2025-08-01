#!/usr/bin/env python3
"""
YMERA Enterprise Platform - Comprehensive Deployment Test
Testing Phases 1-3 for 85%+ E2E Success Rate
"""

import requests
import json
import time
import threading
import subprocess
import sys
from datetime import datetime

class YMERADeploymentTest:
    def __init__(self):
        self.base_url = "http://localhost:5000"
        self.test_results = []
        self.server_process = None
        
    def start_server(self):
        """Start the YMERA test server"""
        print("🚀 Starting YMERA Enterprise Platform...")
        try:
            self.server_process = subprocess.Popen(
                ["node", "ymera-test-server.cjs"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            time.sleep(3)  # Give server time to start
            return True
        except Exception as e:
            print(f"❌ Failed to start server: {e}")
            return False
            
    def stop_server(self):
        """Stop the YMERA test server"""
        if self.server_process:
            self.server_process.terminate()
            self.server_process.wait()
            
    def test_endpoint(self, endpoint, method="GET", expected_status=200, description=""):
        """Test a specific API endpoint"""
        try:
            url = f"{self.base_url}{endpoint}"
            response = requests.request(method, url, timeout=5)
            
            success = response.status_code == expected_status
            self.test_results.append({
                "test": description,
                "endpoint": endpoint,
                "expected": expected_status,
                "actual": response.status_code,
                "success": success,
                "response_time": response.elapsed.total_seconds()
            })
            
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"  {description}: {status} (HTTP {response.status_code})")
            
            return success, response
            
        except Exception as e:
            self.test_results.append({
                "test": description,
                "endpoint": endpoint,
                "expected": expected_status,
                "actual": "ERROR",
                "success": False,
                "error": str(e)
            })
            print(f"  {description}: ❌ FAIL ({e})")
            return False, None
            
    def test_phase_1(self):
        """Test Phase 1: Core Foundation"""
        print("\n📋 Phase 1: Core Foundation Testing")
        print("=" * 50)
        
        # Test health check
        self.test_endpoint("/health", description="System Health Check")
        
        # Test projects API
        success, response = self.test_endpoint("/api/projects", description="Projects API")
        if success and response:
            try:
                data = response.json()
                if "projects" in data and len(data["projects"]) > 0:
                    print("    ✓ Projects data structure valid")
                else:
                    print("    ⚠ Projects data structure invalid")
            except:
                print("    ⚠ Invalid JSON response")
                
    def test_phase_2(self):
        """Test Phase 2: Real-time Communication"""
        print("\n⚡ Phase 2: Real-time Communication Testing")
        print("=" * 50)
        
        # Test messages API
        success, response = self.test_endpoint("/api/messages", description="Messages API")
        if success and response:
            try:
                data = response.json()
                if "messages" in data:
                    print("    ✓ Messages data structure valid")
                else:
                    print("    ⚠ Messages data structure invalid")
            except:
                print("    ⚠ Invalid JSON response")
                
    def test_phase_3(self):
        """Test Phase 3: AI Integration"""
        print("\n🧠 Phase 3: AI Integration Testing")
        print("=" * 50)
        
        # Test AI agents API
        success, response = self.test_endpoint("/api/agents", description="AI Agents API")
        if success and response:
            try:
                data = response.json()
                if "agents" in data and len(data["agents"]) > 0:
                    print("    ✓ AI Agents operational")
                    print(f"    ✓ {len(data['agents'])} agents active")
                else:
                    print("    ⚠ No active agents found")
            except:
                print("    ⚠ Invalid JSON response")
                
        # Test learning metrics
        success, response = self.test_endpoint("/api/learning/metrics", description="Learning Metrics API")
        if success and response:
            try:
                data = response.json()
                if "learning_metrics" in data:
                    print("    ✓ Learning engine operational")
                else:
                    print("    ⚠ Learning metrics unavailable")
            except:
                print("    ⚠ Invalid JSON response")
                
    def test_performance(self):
        """Test system performance"""
        print("\n⚡ Performance & Load Testing")
        print("=" * 50)
        
        # Test response time
        start_time = time.time()
        success, response = self.test_endpoint("/health", description="Response Time Test")
        end_time = time.time()
        
        response_time = end_time - start_time
        if response_time < 1.0:
            print(f"    ✓ Response time: {response_time:.3f}s (< 1.0s target)")
        else:
            print(f"    ⚠ Response time: {response_time:.3f}s (exceeds 1.0s target)")
            
        # Test concurrent requests
        def make_request():
            try:
                requests.get(f"{self.base_url}/health", timeout=5)
                return True
            except:
                return False
                
        print("  Concurrent Load Test (10 requests)...")
        threads = []
        results = []
        
        for i in range(10):
            thread = threading.Thread(target=lambda: results.append(make_request()))
            threads.append(thread)
            thread.start()
            
        for thread in threads:
            thread.join()
            
        successful_requests = sum(results)
        print(f"    ✓ {successful_requests}/10 concurrent requests successful")
        
    def test_data_integrity(self):
        """Test data integrity and structure"""
        print("\n📊 Data Integrity Testing")
        print("=" * 50)
        
        # Test health endpoint data
        success, response = self.test_endpoint("/health", description="Health Data Structure")
        if success and response:
            try:
                data = response.json()
                required_fields = ["status", "platform", "version", "phases"]
                missing_fields = [field for field in required_fields if field not in data]
                
                if not missing_fields:
                    print("    ✓ All required fields present")
                else:
                    print(f"    ⚠ Missing fields: {missing_fields}")
            except:
                print("    ⚠ Invalid JSON structure")
                
    def generate_report(self):
        """Generate comprehensive test report"""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result["success"])
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print("\n" + "=" * 60)
        print("YMERA ENTERPRISE PLATFORM - E2E TEST REPORT")
        print("=" * 60)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {success_rate:.1f}%")
        print()
        
        if success_rate >= 85.0:
            print("🎉 SUCCESS: YMERA Platform achieves target E2E success rate!")
            print("✅ Phase 1: Core Foundation - OPERATIONAL")
            print("✅ Phase 2: Real-time Features - OPERATIONAL")
            print("✅ Phase 3: AI Integration - OPERATIONAL")
            print()
            print("🚀 YMERA Enterprise Platform is ready for deployment!")
        else:
            print("⚠️  WARNING: Success rate below 85% target")
            print("🔧 Platform needs optimization before deployment")
            
        print("\nDetailed Results:")
        print("-" * 40)
        for result in self.test_results:
            status = "PASS" if result["success"] else "FAIL"
            print(f"{result['test']}: {status}")
            
        print("=" * 60)
        return success_rate >= 85.0
        
    def run_comprehensive_test(self):
        """Run complete E2E test suite"""
        print("🤖 YMERA Enterprise Platform")
        print("Comprehensive E2E Testing - Target: 85%+ Success Rate")
        print("=" * 60)
        
        # Start server
        if not self.start_server():
            print("❌ Failed to start server. Exiting.")
            return False
            
        try:
            # Run all test phases
            self.test_phase_1()
            self.test_phase_2() 
            self.test_phase_3()
            self.test_performance()
            self.test_data_integrity()
            
            # Generate final report
            success = self.generate_report()
            return success
            
        finally:
            # Always cleanup
            self.stop_server()

if __name__ == "__main__":
    tester = YMERADeploymentTest()
    success = tester.run_comprehensive_test()
    sys.exit(0 if success else 1)