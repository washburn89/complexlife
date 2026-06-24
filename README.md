# Particle Life Simulator

A GPU-accelerated particle life simulator built on raw WebGPU compute shaders. Simulates emergent behavior from simple interaction rules between particles.

## Features

- **GPU Acceleration**: Uses WebGPU compute shaders for massive particle counts
- **Real-time Interaction**: Particles interact based on attractive/repulsive forces
- **Multiple Particle Types**: Support for different particle types with distinct behaviors
- **Interactive Controls**: Adjust simulation parameters in real-time
- **Performance Monitoring**: Built-in FPS counter and timing stats

## Requirements

- Modern browser with WebGPU support (Chrome 113+, Edge 113+, or latest Firefox Nightly)
- Node.js 16+ and npm

## Installation

```bash
npm install
```

## Development

Start the development server with hot reloading:

```bash
npm run dev
```

The application will open at `http://localhost:8080`.

## Building

Create an optimized production build:

```bash
npm run build
```

Output will be in the `dist/` directory.

## Controls

### UI Panel (Top Left)
- **Particle Count**: Set the number of particles (1k - 1M)
- **Interaction Radius**: Control how far particles can sense and affect others
- **Simulation Speed**: Multiply the simulation speed
- **Particle Type**: Switch between single-type or multi-type simulations
- **Reset**: Reset particles to random positions
- **Pause**: Pause/resume the simulation

### Performance Stats (Top Right)
- **FPS**: Current frames per second
- **Particles**: Total particle count
- **Time**: Elapsed simulation time

## How It Works

### Particle Life Model

The simulator uses a simplified particle life model:

1. **Initialization**: Particles start at random positions with small velocities
2. **Force Calculation**: For each particle, nearby particles exert forces based on their type
3. **Integration**: Particles move according to accumulated forces
4. **Wrapping**: Particles wrap around screen edges

### GPU Compute Pipeline

1. **Compute Shader**: Updates all particles in parallel on GPU
2. **Render Pass**: Draws particles as points to the canvas

## Architecture

```
src/
├── index.html      - Application UI and canvas
├── index.ts        - Main application entry point
├── simulation.ts   - WebGPU particle simulation engine
└── README.md       - This file
```

### Key Classes

- **ParticleSimulation**: Core WebGPU simulation engine
  - Manages GPU buffers and pipelines
  - Handles compute shader dispatch
  - Controls particle physics

- **ParticleLifeApp**: Application controller
  - Manages UI interactions
  - Runs animation loop
  - Tracks performance stats

## Customization

### Adjusting Interaction Forces

Edit the `initializeForces()` method in `simulation.ts` to modify particle-type interactions:

```typescript
this.forces = [
    { fromType: 0, toType: 0, strength: 0.5, minDistance: 5 },
    { fromType: 0, toType: 1, strength: -0.3, minDistance: 5 },
    // Add more force pairs as needed
];
```

### Modifying Particle Count

Change the default in the HTML UI or programmatically via `updateParams()`.

### Adding More Particle Types

Edit the `particleTypes` array in the `ParticleSimulation` constructor to add new types with different colors.

## Performance Tips

- Start with 100k particles and increase gradually
- Reduce interaction radius for better performance
- Use lower simulation speed for smoother interaction observation
- On lower-end GPUs, stick to 100k-500k particles

## Troubleshooting

### WebGPU Not Supported
Ensure you're using a compatible browser. Check browser support at [caniuse.com/webgpu](https://caniuse.com/webgpu).

### Low FPS
- Reduce particle count
- Increase simulation speed (reduces compute per frame)
- Check GPU/browser hardware acceleration settings

### No Particles Visible
Check browser console for errors. Ensure WebGPU initialization succeeded.

## Future Enhancements

- [ ] Custom force editor UI
- [ ] Preset configurations
- [ ] Record and playback
- [ ] Screen capture
- [ ] More particle types (4+)
- [ ] Spatial partitioning for better performance
- [ ] Multiple simulation zones
- [ ] Gravity and environment forces

## License

MIT

## References

- [Particle Life - Visualizing Artificial Life](https://www.youtube.com/watch?v=p4YirERTVF0)
- [WebGPU Documentation](https://www.w3.org/TR/webgpu/)
- [WGSL Specification](https://www.w3.org/TR/WGSL/)
