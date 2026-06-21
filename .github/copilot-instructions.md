# Particle Life Simulator - Development Guidelines

## Project Overview

This is a WebGPU-based particle life simulator built with Babylon.js. It demonstrates GPU-accelerated physics simulation with support for 1M+ particles.

## Technology Stack

- **GPU API**: WebGPU with compute shaders
- **Framework**: Babylon.js 6.40+
- **Language**: TypeScript
- **Build Tool**: Webpack 5
- **Target**: Modern browsers with WebGPU support

## Development Workflow

### Setup
1. Install dependencies: `npm install`
2. Start dev server: `npm run dev`
3. Open browser at `http://localhost:8080`

### Building
- Development: `npm run dev` (with hot reload)
- Production: `npm run build`

## Key Files

- `src/index.html` - UI and canvas
- `src/index.ts` - Application controller
- `src/simulation.ts` - WebGPU simulation engine
- `webpack.config.js` - Build configuration
- `tsconfig.json` - TypeScript settings

## Code Style

- Use TypeScript with strict mode enabled
- Follow Google TypeScript style guide
- Use descriptive variable/function names
- Add JSDoc comments for public APIs
- Organize imports alphabetically

## WebGPU Development

### Key Concepts
- **Buffers**: Store particle data on GPU
- **Compute Shaders**: Update all particles in parallel
- **Render Pass**: Draw particles to canvas
- **Pipeline**: Defines shader entry points and bind groups

### Performance Considerations
- Workgroup size: 256 (tune for your target GPU)
- Particle limit: Browser memory and GPU limits
- Friction factor: Currently 0.98 (tune for desired behavior)
- Force scale: Adjust `strength` in force definitions

## Common Tasks

### Adding a New Particle Type

1. Add color definition in `ParticleSimulation.particleTypes`
2. Add force rules in `initializeForces()`
3. Update UI to allow selection

### Adjusting Physics Parameters

1. Modify `initializeForces()` for interaction strength
2. Change friction (0.98) in compute shader for damping
3. Adjust `minDistance` for force cutoff

### Improving Performance

1. Reduce workgroup size if GPU struggles
2. Implement spatial partitioning (future)
3. Use object pooling for buffers
4. Profile with browser DevTools

## Testing

Currently manual testing. Future: Add unit tests for physics calculations.

## Debugging

1. Enable browser DevTools
2. Check console for WebGPU errors
3. Use `console.log` in JS (not possible in shaders directly)
4. Verify shader compilation with pipeline creation

## Browser Compatibility

- Chrome/Edge 113+
- Firefox Nightly (experimental)
- Safari: Not yet supported

## Future Enhancements

Priority features:
1. Multi-type particle UI editor
2. Save/load configurations
3. Better spatial partitioning
4. Record simulation video
5. Advanced rendering (trails, post-processing)
