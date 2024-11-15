import * as THREE from './js/three.module.js';
//import { Ammo }  from './js/ammo.wasm.js';
import { OBJLoader } from './js/OBJLoader.js';


let scene, camera, renderer, physicsWorld;
let clothGeometry, clothParticles = [], clothConstraints = [];
let TSHIRT_URL = 'tshirt.obj';  // T-shirt OBJ URL




// Initialize Three.js scene
function init() {
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    renderer = new THREE.WebGLRenderer();
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(renderer.domElement);
    
    // Lighting
    let light = new THREE.DirectionalLight(0xffffff, 1);
    light.position.set(5, 10, 7.5).normalize();
    scene.add(light);
    
    // Camera position
    camera.position.z = 5;
    
    // Load T-shirt mesh
    let loader = new OBJLoader();
    loader.load(TSHIRT_URL, function(object) {
        clothGeometry = object.children[0].geometry;
        clothGeometry.computeVertexNormals();
        clothGeometry.dynamic = true;
        scene.add(object);
        createClothPhysics();
    });

    initPhysicsWorld();
    animate();
}

// Initialize Ammo.js physics world
function initPhysicsWorld() {
    let collisionConfiguration = new Ammo.btDefaultCollisionConfiguration();
    let dispatcher = new Ammo.btCollisionDispatcher(collisionConfiguration);
    let broadphase = new Ammo.btDbvtBroadphase();
    let solver = new Ammo.btSequentialImpulseConstraintSolver();
    physicsWorld = new Ammo.btDiscreteDynamicsWorld(dispatcher, broadphase, solver, collisionConfiguration);
    physicsWorld.setGravity(new Ammo.btVector3(0, -9.81, 0));
}

// Create cloth physics simulation
function createClothPhysics() {
    let vertices = clothGeometry.attributes.position.array;
    for (let i = 0; i < vertices.length; i += 3) {
        let x = vertices[i], y = vertices[i + 1], z = vertices[i + 2];
        let particle = createClothParticle(x, y, z);
        clothParticles.push(particle);
    }

    // Create constraints between neighboring particles
    for (let i = 0; i < clothParticles.length; i++) {
        if (i % 10 != 9 && i < clothParticles.length - 10) {  // Example constraint logic
            addClothConstraint(clothParticles[i], clothParticles[i + 1], 0.5);
            addClothConstraint(clothParticles[i], clothParticles[i + 10], 0.5);
        }
    }
}

// Create a particle for the cloth
function createClothParticle(x, y, z) {
    let mass = 0.1;
    let shape = new Ammo.btSphereShape(0.1);
    let transform = new Ammo.btTransform();
    transform.setIdentity();
    transform.setOrigin(new Ammo.btVector3(x, y, z));
    let motionState = new Ammo.btDefaultMotionState(transform);
    let localInertia = new Ammo.btVector3(0, 0, 0);
    shape.calculateLocalInertia(mass, localInertia);
    let bodyInfo = new Ammo.btRigidBodyConstructionInfo(mass, motionState, shape, localInertia);
    let body = new Ammo.btRigidBody(bodyInfo);
    physicsWorld.addRigidBody(body);
    return body;
}

// Add a constraint between two particles
function addClothConstraint(p1, p2, stiffness) {
    let transform1 = new Ammo.btTransform();
    transform1.setIdentity();
    let transform2 = new Ammo.btTransform();
    transform2.setIdentity();
    let constraint = new Ammo.btGeneric6DofSpringConstraint(p1, p2, transform1, transform2, true);
    constraint.setLinearLowerLimit(new Ammo.btVector3(0, 0, 0));
    constraint.setLinearUpperLimit(new Ammo.btVector3(1, 1, 1));
    constraint.enableSpring(0, true);
    constraint.setStiffness(0, stiffness);
    physicsWorld.addConstraint(constraint);
    clothConstraints.push(constraint);
}

// Animation loop
function animate() {
    requestAnimationFrame(animate);
    physicsWorld.stepSimulation(1 / 60, 2);
    
    // Update cloth vertices
    let positions = clothGeometry.attributes.position.array;
    for (let i = 0; i < clothParticles.length; i++) {
        let position = clothParticles[i].getCenterOfMassPosition();
        positions[i * 3] = position.x();
        positions[i * 3 + 1] = position.y();
        positions[i * 3 + 2] = position.z();
    }
    clothGeometry.attributes.position.needsUpdate = true;

    renderer.render(scene, camera);
}

// Resize handler
window.addEventListener('resize', function() {
    renderer.setSize(window.innerWidth, window.innerHeight);
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
});

document.addEventListener('ammoLoaded', (event) => {
    const Ammo = event.detail; // Get the Ammo object from the event detail
    console.log('Using Ammo in cloth_simulation.js:');

    // Call a function to start your cloth simulation
    init();
});

