import { useEffect, useRef } from "react";
import * as THREE from "three";

interface Module {
  name: string;
  position: { x: number; y: number; z: number };
  color: number;
}

export function ThreeVisualization() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const sceneRef = useRef<THREE.Scene>();
  const rendererRef = useRef<THREE.WebGLRenderer>();
  const cameraRef = useRef<THREE.PerspectiveCamera>();
  const moduleObjectsRef = useRef<THREE.Mesh[]>([]);
  const animationRef = useRef<number>();

  useEffect(() => {
    if (!canvasRef.current) return;

    // Scene setup
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(
      75,
      canvasRef.current.clientWidth / canvasRef.current.clientHeight,
      0.1,
      1000
    );
    const renderer = new THREE.WebGLRenderer({ 
      canvas: canvasRef.current, 
      alpha: true, 
      antialias: true 
    });

    renderer.setSize(canvasRef.current.clientWidth, canvasRef.current.clientHeight);
    renderer.setClearColor(0x000000, 0);

    // Store references
    sceneRef.current = scene;
    rendererRef.current = renderer;
    cameraRef.current = camera;

    // Module configurations with YMERA branding colors
    const modules: Module[] = [
      { name: 'Auth Module', position: { x: -2, y: 1, z: 0 }, color: 0xFBBF24 },
      { name: 'API Gateway', position: { x: 0, y: 1, z: 0 }, color: 0x2563EB },
      { name: 'Database', position: { x: 2, y: 1, z: 0 }, color: 0x991B1B },
      { name: 'WebSocket', position: { x: -1, y: -1, z: 0 }, color: 0x10B981 },
      { name: 'File System', position: { x: 1, y: -1, z: 0 }, color: 0x8B5CF6 }
    ];

    const moduleObjects: THREE.Mesh[] = [];

    // Create module cubes
    modules.forEach((mod, index) => {
      const geometry = new THREE.BoxGeometry(0.8, 0.8, 0.8);
      const material = new THREE.MeshBasicMaterial({ 
        color: mod.color, 
        wireframe: true,
        transparent: true,
        opacity: 0.8
      });
      
      const cube = new THREE.Mesh(geometry, material);
      cube.position.set(mod.position.x, mod.position.y, mod.position.z);
      
      // Add floating animation offset
      cube.userData = { 
        originalY: mod.position.y,
        floatOffset: index * Math.PI / 3
      };
      
      scene.add(cube);
      moduleObjects.push(cube);
    });

    moduleObjectsRef.current = moduleObjects;

    // Create connection lines between modules
    const lineMaterial = new THREE.LineBasicMaterial({ 
      color: 0xFBBF24, 
      transparent: true, 
      opacity: 0.3 
    });
    
    // Connect modules with lines
    for (let i = 0; i < modules.length - 1; i++) {
      const geometry = new THREE.BufferGeometry();
      const points = [
        new THREE.Vector3(modules[i].position.x, modules[i].position.y, modules[i].position.z),
        new THREE.Vector3(modules[i + 1].position.x, modules[i + 1].position.y, modules[i + 1].position.z)
      ];
      geometry.setFromPoints(points);
      
      const line = new THREE.Line(geometry, lineMaterial);
      scene.add(line);
    }
    
    camera.position.z = 5;
    camera.position.y = 0.5;

    // Animation loop
    function animate() {
      animationRef.current = requestAnimationFrame(animate);
      
      const time = Date.now() * 0.001;
      
      // Animate module floating and rotation
      moduleObjects.forEach((cube) => {
        cube.rotation.x += 0.01;
        cube.rotation.y += 0.01;
        
        // Floating animation
        cube.position.y = cube.userData.originalY + Math.sin(time + cube.userData.floatOffset) * 0.2;
      });
      
      // Gentle camera rotation
      camera.position.x = Math.sin(time * 0.1) * 0.5;
      camera.lookAt(0, 0, 0);
      
      renderer.render(scene, camera);
    }

    // Handle resize
    function handleResize() {
      if (!canvasRef.current || !camera || !renderer) return;
      
      const width = canvasRef.current.clientWidth;
      const height = canvasRef.current.clientHeight;
      
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
    }

    window.addEventListener('resize', handleResize);
    animate();

    return () => {
      window.removeEventListener('resize', handleResize);
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      
      // Cleanup Three.js objects
      moduleObjects.forEach(cube => {
        cube.geometry.dispose();
        (cube.material as THREE.Material).dispose();
      });
      
      scene.clear();
      renderer.dispose();
    };
  }, []);

  return (
    <div className="relative bg-black/30 rounded-lg overflow-hidden" style={{ height: '400px' }}>
      <canvas ref={canvasRef} className="w-full h-full" />
      
      {/* 3D Scene Controls */}
      <div className="absolute top-4 right-4 space-y-2">
        <button className="p-2 bg-black/50 rounded-lg hover:bg-black/70 transition-colors">
          <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
          </svg>
        </button>
        <button className="p-2 bg-black/50 rounded-lg hover:bg-black/70 transition-colors">
          <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
        </button>
        <button className="p-2 bg-black/50 rounded-lg hover:bg-black/70 transition-colors">
          <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 9v6m4-6v6m7-3a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        </button>
      </div>
      
      {/* Module Labels */}
      <div className="absolute bottom-4 left-4 space-y-1 text-xs">
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 bg-yellow-500 rounded"></div>
          <span>Authentication Module</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 bg-blue-500 rounded"></div>
          <span>API Gateway</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 bg-red-800 rounded"></div>
          <span>Database Layer</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 bg-green-500 rounded"></div>
          <span>WebSocket Server</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 bg-purple-500 rounded"></div>
          <span>File System</span>
        </div>
      </div>
    </div>
  );
}
