import * as THREE from "three";

export interface ModuleConfig {
  name: string;
  position: { x: number; y: number; z: number };
  color: number;
  size?: number;
}

export class ThreeSceneManager {
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private renderer: THREE.WebGLRenderer;
  private moduleObjects: THREE.Mesh[] = [];
  private animationId: number | null = null;

  constructor(canvas: HTMLCanvasElement) {
    // Scene setup
    this.scene = new THREE.Scene();
    
    // Camera setup
    this.camera = new THREE.PerspectiveCamera(
      75,
      canvas.clientWidth / canvas.clientHeight,
      0.1,
      1000
    );
    this.camera.position.set(0, 0.5, 5);

    // Renderer setup
    this.renderer = new THREE.WebGLRenderer({ 
      canvas, 
      alpha: true, 
      antialias: true 
    });
    this.renderer.setSize(canvas.clientWidth, canvas.clientHeight);
    this.renderer.setClearColor(0x000000, 0);
  }

  addModules(modules: ModuleConfig[]) {
    modules.forEach((mod, index) => {
      const geometry = new THREE.BoxGeometry(
        mod.size || 0.8, 
        mod.size || 0.8, 
        mod.size || 0.8
      );
      
      const material = new THREE.MeshBasicMaterial({ 
        color: mod.color, 
        wireframe: true,
        transparent: true,
        opacity: 0.8
      });
      
      const cube = new THREE.Mesh(geometry, material);
      cube.position.set(mod.position.x, mod.position.y, mod.position.z);
      
      // Add animation data
      cube.userData = { 
        originalY: mod.position.y,
        floatOffset: index * Math.PI / 3,
        name: mod.name
      };
      
      this.scene.add(cube);
      this.moduleObjects.push(cube);
    });

    this.addConnectionLines(modules);
  }

  private addConnectionLines(modules: ModuleConfig[]) {
    const lineMaterial = new THREE.LineBasicMaterial({ 
      color: 0xFBBF24, 
      transparent: true, 
      opacity: 0.3 
    });
    
    // Connect modules in sequence
    for (let i = 0; i < modules.length - 1; i++) {
      const geometry = new THREE.BufferGeometry();
      const points = [
        new THREE.Vector3(
          modules[i].position.x, 
          modules[i].position.y, 
          modules[i].position.z
        ),
        new THREE.Vector3(
          modules[i + 1].position.x, 
          modules[i + 1].position.y, 
          modules[i + 1].position.z
        )
      ];
      geometry.setFromPoints(points);
      
      const line = new THREE.Line(geometry, lineMaterial);
      this.scene.add(line);
    }
  }

  startAnimation() {
    const animate = () => {
      this.animationId = requestAnimationFrame(animate);
      
      const time = Date.now() * 0.001;
      
      // Animate modules
      this.moduleObjects.forEach((cube) => {
        // Rotation
        cube.rotation.x += 0.01;
        cube.rotation.y += 0.01;
        
        // Floating animation
        cube.position.y = cube.userData.originalY + 
          Math.sin(time + cube.userData.floatOffset) * 0.2;
      });
      
      // Gentle camera movement
      this.camera.position.x = Math.sin(time * 0.1) * 0.5;
      this.camera.lookAt(0, 0, 0);
      
      this.renderer.render(this.scene, this.camera);
    };

    animate();
  }

  stopAnimation() {
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
      this.animationId = null;
    }
  }

  handleResize(width: number, height: number) {
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }

  dispose() {
    this.stopAnimation();
    
    // Dispose of geometries and materials
    this.moduleObjects.forEach(cube => {
      cube.geometry.dispose();
      (cube.material as THREE.Material).dispose();
    });
    
    this.scene.clear();
    this.renderer.dispose();
  }

  getModuleAtPosition(x: number, y: number): string | null {
    // Convert screen coordinates to normalized device coordinates
    const mouse = new THREE.Vector2();
    mouse.x = (x / this.renderer.domElement.clientWidth) * 2 - 1;
    mouse.y = -(y / this.renderer.domElement.clientHeight) * 2 + 1;

    // Create raycaster
    const raycaster = new THREE.Raycaster();
    raycaster.setFromCamera(mouse, this.camera);

    // Check intersections
    const intersects = raycaster.intersectObjects(this.moduleObjects);
    
    if (intersects.length > 0) {
      return intersects[0].object.userData.name;
    }
    
    return null;
  }
}
