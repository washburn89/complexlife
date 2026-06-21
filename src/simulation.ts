export interface ParticleType {
    id: number;
    color: [number, number, number, number];
    radius: number;
}

export interface ForceMatrix {
    [fromType: number]: {
        [toType: number]: {
            strength: number;
            radius: number;
        };
    };
}

export interface SimulationParams {
    particleCount: number;
    simulationSpeed: number;
    particleTypes: ParticleType[];
    worldWidth: number;
    worldHeight: number;
    forceMatrix: ForceMatrix;
}

export interface ViewState {
    cx: number;
    cy: number;
    zoom: number;
}

export interface TransformRule {
    upperEnabled:   boolean;
    upperThreshold: number;
    upperTarget:    number;
    lowerEnabled:   boolean;
    lowerThreshold: number;
    lowerTarget:    number;
}

export const MAX_TYPES = 20;

const OPEN_MULT = 5;

// 20 visually distinct colours used in both JS (UI) and WGSL (render).
// JS hex values are kept in sync with the WGSL vec4f constants below.
export const TYPE_COLORS_HEX: string[] = [
    '#ff2020', '#00ee00', '#2060ff', '#ffee00',
    '#ff20ff', '#00ffff', '#ff8800', '#9900ff',
    '#ff0088', '#00ff88',
    '#eeeeee', '#ff7744', '#aaff00', '#00ccbb',
    '#5533ff', '#ff3399', '#ffcc00', '#cc8833',
    '#228833', '#999999',
];

export class ParticleSimulation {
    private canvas: HTMLCanvasElement;
    private adapter: GPUAdapter | null = null;
    private device:  GPUDevice  | null = null;
    private queue:   GPUQueue   | null = null;
    private context: GPUCanvasContext | null = null;

    private particleBuffer:   GPUBuffer | null = null;
    private paramsBuffer:     GPUBuffer | null = null;
    private forcesBuffer:     GPUBuffer | null = null;
    private viewBuffer:       GPUBuffer | null = null;
    private transformBuffer:  GPUBuffer | null = null;
    private quadVertexBuffer: GPUBuffer | null = null;

    private computePipeline:  GPUComputePipeline | null = null;
    private renderPipeline:   GPURenderPipeline  | null = null;
    private computeBindGroup: GPUBindGroup | null = null;
    private renderBindGroup:  GPUBindGroup | null = null;

    private params: SimulationParams;
    private view:   ViewState = { cx: 0, cy: 0, zoom: 1 };

    private configWidth  = 1600;
    private configHeight = 900;

    private numTypes = 10;
    private simMode  = 0;
    private edgeMode = 0;

    private backgroundColor = { r: 0.05, g: 0.05, b: 0.08 };
    private colorSaturation = 1.0;
    private particleGlow    = 0.6;  // 0 = hard solid circle, 1 = wide gaussian orb

    private transformRules: TransformRule[] = [];
    private poleConfigs = new Uint32Array(MAX_TYPES);
    private poleBuffer: GPUBuffer | null = null;

    private isInitialized  = false;
    private isPaused       = false;
    private simulationTime = 0;

    constructor(canvas: HTMLCanvasElement) {
        this.canvas       = canvas;
        this.configWidth  = canvas.width;
        this.configHeight = canvas.height;
        this.params = {
            particleCount:   3000,
            simulationSpeed: 1,
            particleTypes:   [],
            worldWidth:      canvas.width,
            worldHeight:     canvas.height,
            forceMatrix:     {},
        };
        this.view = { cx: canvas.width / 2, cy: canvas.height / 2, zoom: 1 };
        this.initializeForceMatrix();
        this.initializeTransformRules();
        this.initializePoleConfigs();
    }

    // ── Init helpers ──────────────────────────────────────────────────────────

    private initializeForceMatrix(): void {
        for (let from = 0; from < MAX_TYPES; from++) {
            this.params.forceMatrix[from] = {};
            for (let to = 0; to < MAX_TYPES; to++) {
                this.params.forceMatrix[from][to] = {
                    strength: (Math.random() * 2 - 1) * 0.7,
                    radius:   70 + Math.random() * 40,
                };
            }
        }
    }

    private initializeTransformRules(): void {
        const n = this.numTypes;
        this.transformRules = Array.from({ length: MAX_TYPES * MAX_TYPES }, (_, idx) => {
            const source = Math.floor(idx / MAX_TYPES);
            const randTarget = () => {
                let t = Math.floor(Math.random() * n);
                while (t === source && n > 1) t = Math.floor(Math.random() * n);
                return t;
            };
            return {
                upperEnabled:   Math.random() < 0.25,
                upperThreshold: 0.3 + Math.random() * 0.5,
                upperTarget:    randTarget(),
                lowerEnabled:   Math.random() < 0.25,
                lowerThreshold: -(0.3 + Math.random() * 0.5),
                lowerTarget:    randTarget(),
            };
        });
    }

    private initializePoleConfigs(): void {
        this.poleConfigs.fill(0);  // all monopoles by default
    }

    private generatePoleData(): Float32Array<ArrayBuffer> {
        const ab   = new ArrayBuffer(MAX_TYPES * 4);
        const data = new Float32Array(ab);
        for (let i = 0; i < MAX_TYPES; i++) data[i] = this.poleConfigs[i];
        return data as Float32Array<ArrayBuffer>;
    }

    // ── WebGPU init ───────────────────────────────────────────────────────────

    async initialize(): Promise<void> {
        if (!navigator.gpu) throw new Error('WebGPU is not supported');
        this.adapter = await navigator.gpu.requestAdapter();
        if (!this.adapter) throw new Error('Failed to get GPU adapter');
        this.device = await this.adapter.requestDevice();
        this.queue  = this.device.queue;

        const ctx = this.canvas.getContext('webgpu') as GPUCanvasContext | null;
        if (!ctx) throw new Error('Failed to get WebGPU context');
        this.context = ctx;
        const fmt = navigator.gpu.getPreferredCanvasFormat();
        ctx.configure({ device: this.device, format: fmt, alphaMode: 'opaque' });

        await this.createBuffers();
        await this.createPipelines();
        this.isInitialized = true;
    }

    // ── Data generators ───────────────────────────────────────────────────────

    private generateParticleData(): Float32Array<ArrayBuffer> {
        const n    = this.params.particleCount;
        const ab   = new ArrayBuffer(n * 6 * 4);
        const data = new Float32Array(ab);
        const spawnX0 = this.edgeMode === 1 ? (OPEN_MULT - 1) / 2 * this.configWidth  : 0;
        const spawnY0 = this.edgeMode === 1 ? (OPEN_MULT - 1) / 2 * this.configHeight : 0;
        for (let i = 0; i < n; i++) {
            const b = i * 6;
            data[b + 0] = spawnX0 + Math.random() * this.configWidth;
            data[b + 1] = spawnY0 + Math.random() * this.configHeight;
            data[b + 2] = (Math.random() - 0.5) * 2;
            data[b + 3] = (Math.random() - 0.5) * 2;
            data[b + 4] = i % this.numTypes;
            data[b + 5] = 0;
        }
        return data;
    }

    private generateForcesData(): Float32Array<ArrayBuffer> {
        const ab   = new ArrayBuffer(MAX_TYPES * MAX_TYPES * 2 * 4);
        const data = new Float32Array(ab);
        for (let from = 0; from < MAX_TYPES; from++) {
            for (let to = 0; to < MAX_TYPES; to++) {
                const idx = (from * MAX_TYPES + to) * 2;
                const c   = this.params.forceMatrix[from]?.[to];
                data[idx + 0] = c?.strength ?? 0;
                data[idx + 1] = c?.radius   ?? 100;
            }
        }
        return data;
    }

    private generateTransformData(): Float32Array<ArrayBuffer> {
        const ab   = new ArrayBuffer(MAX_TYPES * MAX_TYPES * 6 * 4);
        const data = new Float32Array(ab);
        for (let i = 0; i < MAX_TYPES * MAX_TYPES; i++) {
            const r = this.transformRules[i];
            const b = i * 6;
            data[b + 0] = r.upperEnabled   ? 1 : 0;
            data[b + 1] = r.upperThreshold;
            data[b + 2] = r.upperTarget;
            data[b + 3] = r.lowerEnabled   ? 1 : 0;
            data[b + 4] = r.lowerThreshold;
            data[b + 5] = r.lowerTarget;
        }
        return data;
    }

    // params.w packs three values: numTypes (bits 16-23), edgeMode (bits 8-15), simMode (bits 0-7).
    // All fit as small unsigned ints stored as a float32 (exact for integers up to 2^24).
    private paramsArray(): Float32Array<ArrayBuffer> {
        const packed = (this.numTypes << 16) | (this.edgeMode << 8) | this.simMode;
        return new Float32Array([
            this.params.simulationSpeed,
            this.params.worldWidth,
            this.params.worldHeight,
            packed,
        ]) as Float32Array<ArrayBuffer>;
    }

    private defaultView(): ViewState {
        if (this.edgeMode === 1) {
            return {
                cx:   OPEN_MULT / 2 * this.configWidth,
                cy:   OPEN_MULT / 2 * this.configHeight,
                zoom: OPEN_MULT,
            };
        }
        return { cx: this.configWidth / 2, cy: this.configHeight / 2, zoom: 1 };
    }

    // ── Buffer helpers ────────────────────────────────────────────────────────

    private makeBuffer(label: string, data: Float32Array<ArrayBuffer>, usage: number): GPUBuffer {
        const buf = this.device!.createBuffer({
            label, size: data.byteLength, mappedAtCreation: true, usage,
        });
        new Float32Array(buf.getMappedRange()).set(data);
        buf.unmap();
        return buf;
    }

    private async createBuffers(): Promise<void> {
        const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
        const U = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;

        this.particleBuffer  = this.makeBuffer('particles', this.generateParticleData(), S);
        this.paramsBuffer    = this.makeBuffer('params',    this.paramsArray(),           U);
        this.forcesBuffer    = this.makeBuffer('forces',    this.generateForcesData(),    S);
        this.transformBuffer = this.makeBuffer('transform', this.generateTransformData(), S);
        this.poleBuffer      = this.makeBuffer('poles',     this.generatePoleData(),      S);
        // 8-float view buffer: cx, cy, zoom, sat, glow, _, _, _
        this.viewBuffer = this.makeBuffer('view', new Float32Array([
            this.view.cx, this.view.cy, this.view.zoom,
            this.colorSaturation, this.particleGlow, 0, 0, 0,
        ]) as Float32Array<ArrayBuffer>, U);

        const quad = new Float32Array([-1,-1, 1,-1, 1,1, -1,-1, 1,1, -1,1]);
        this.quadVertexBuffer = this.makeBuffer('quad', quad as Float32Array<ArrayBuffer>, GPUBufferUsage.VERTEX);
    }

    private async createPipelines(): Promise<void> {
        if (!this.device || !this.context || !this.particleBuffer || !this.paramsBuffer ||
            !this.forcesBuffer || !this.transformBuffer || !this.viewBuffer ||
            !this.quadVertexBuffer || !this.poleBuffer) {
            throw new Error('Buffers not initialized');
        }

        const computeLayout = this.device.createBindGroupLayout({ entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        ]});
        this.computePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [computeLayout] }),
            compute: { module: this.device.createShaderModule({ code: this.getComputeShaderCode() }), entryPoint: 'main' },
        });

        const renderLayout = this.device.createBindGroupLayout({ entries: [
            { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
            { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
            { binding: 2, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        ]});
        const renderModule = this.device.createShaderModule({ code: this.getRenderShaderCode() });
        this.renderPipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [renderLayout] }),
            vertex: {
                module: renderModule, entryPoint: 'vertexMain',
                buffers: [{ arrayStride: 8, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x2' }] }],
            },
            fragment: { module: renderModule, entryPoint: 'fragmentMain',
                targets: [{ format: navigator.gpu.getPreferredCanvasFormat(), blend: {
                    color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
                    alpha: { srcFactor: 'one',       dstFactor: 'zero',                operation: 'add' },
                } }] },
            primitive: { topology: 'triangle-list' },
        });

        this.rebuildBindGroups();
    }

    private rebuildBindGroups(): void {
        if (!this.device || !this.particleBuffer || !this.paramsBuffer || !this.forcesBuffer ||
            !this.transformBuffer || !this.viewBuffer || !this.poleBuffer ||
            !this.computePipeline || !this.renderPipeline) return;

        this.computeBindGroup = this.device.createBindGroup({
            layout: this.computePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer  } },
                { binding: 1, resource: { buffer: this.paramsBuffer    } },
                { binding: 2, resource: { buffer: this.forcesBuffer    } },
                { binding: 3, resource: { buffer: this.transformBuffer } },
                { binding: 4, resource: { buffer: this.poleBuffer      } },
            ],
        });
        this.renderBindGroup = this.device.createBindGroup({
            layout: this.renderPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } },
                { binding: 1, resource: { buffer: this.paramsBuffer   } },
                { binding: 2, resource: { buffer: this.viewBuffer     } },
            ],
        });
    }

    // ── Shaders ───────────────────────────────────────────────────────────────

    private getComputeShaderCode(): string {
        return /* wgsl */`
            struct Particle    { pos: vec2f, vel: vec2f, typeId: f32, _pad: f32 }
            struct ForceEntry  { strength: f32, radius: f32 }
            struct TransformRule {
                upperEnabled: f32, upperThreshold: f32, upperTarget: f32,
                lowerEnabled: f32, lowerThreshold: f32, lowerTarget: f32,
            }

            @group(0) @binding(0) var<storage, read_write> particles:      array<Particle>;
            @group(0) @binding(1) var<uniform>             params:         vec4f;
            @group(0) @binding(2) var<storage, read>       forces:         array<ForceEntry>;
            @group(0) @binding(3) var<storage, read>       transformRules: array<TransformRule>;
            @group(0) @binding(4) var<storage, read>       poleConfigs:    array<f32>;

            // Polar field mask. ux,uy = unit vector from emitter to receiver.
            // poleData bits 0-3 = poleCount, bits 4+ = sign bits for each lobe.
            // Returns a multiplier in [-1, 1]: positive amplifies, negative reverses force.
            fn poleMask(ux: f32, uy: f32, emitVel: vec2f, poleData: u32) -> f32 {
                let poleCount = poleData & 0xFu;
                if (poleCount == 0u) { return 1.0; }
                let vm = length(emitVel);
                if (vm < 0.01) { return 1.0; }  // stationary emitter: monopole fallback
                let ivx = emitVel.x / vm;
                let ivy = emitVel.y / vm;
                if (poleCount == 2u) {
                    // Dipole: +1 in front of emitter, -1 behind.
                    return ivx * ux + ivy * uy;
                }
                // N poles (3-6): equally spaced lobes, each independently signed.
                let signBits = poleData >> 4u;
                var rawMask: f32 = 0.0;
                let angStep = 6.28318530718 / f32(poleCount);
                let cosA = cos(angStep);
                let sinA = sin(angStep);
                var pvx = ivx;
                var pvy = ivy;
                for (var k: u32 = 0u; k < poleCount; k++) {
                    let d    = pvx * ux + pvy * uy;
                    let lobe = select(0.0, d * d, d > 0.0);  // max(0, cos)^2
                    let sign = select(-1.0, 1.0, (signBits & (1u << k)) != 0u);
                    rawMask += sign * lobe;
                    let npx = pvx * cosA - pvy * sinA;
                    let npy = pvx * sinA + pvy * cosA;
                    pvx = npx; pvy = npy;
                }
                return clamp(rawMask, -1.0, 1.0);
            }

            fn uhash(v: u32) -> u32 {
                var x = v ^ (v >> 16u);
                x *= 0x45d9f3bu;
                x ^= x >> 16u;
                return x;
            }
            fn rand01(seed: u32) -> f32 {
                return f32(uhash(seed)) / 4294967295.0;
            }

            // Probability of transformation per tick.
            // x = force / threshold (same-sign values → x > 0 when condition met).
            // Ramps from 0 at x=0.5 up to 50% at x=3, using smoothstep.
            fn transformProb(force: f32, threshold: f32) -> f32 {
                if (abs(threshold) < 0.001) { return 0.0; }
                let x = force / threshold;
                if (x < 0.5) { return 0.0; }
                let t = clamp((x - 0.5) / 2.5, 0.0, 1.0);
                return t * t * (3.0 - 2.0 * t) * 0.5;
            }

            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) id: vec3u) {
                let idx   = id.x;
                let count = arrayLength(&particles);
                if (idx >= count) { return; }

                let speed    = params.x;
                let width    = params.y;
                let height   = params.z;
                let packed   = u32(params.w);
                let simMode  = packed & 0xFFu;
                let edgeMode = (packed >> 8u) & 0xFFu;
                let numTypes = packed >> 16u;

                var p      = particles[idx];
                let myType = u32(p.typeId);
                var accel  = vec2f(0.0);
                var typeForce: array<f32, 20>;

                for (var i: u32 = 0u; i < count; i++) {
                    if (i == idx) { continue; }
                    let other    = particles[i];
                    let otherType = u32(other.typeId);

                    var dx = other.pos.x - p.pos.x;
                    var dy = other.pos.y - p.pos.y;

                    if (edgeMode == 0u) {
                        if (dx >  width  * 0.5) { dx -= width;  }
                        if (dx < -width  * 0.5) { dx += width;  }
                        if (dy >  height * 0.5) { dy -= height; }
                        if (dy < -height * 0.5) { dy += height; }
                    }

                    let dist = sqrt(dx * dx + dy * dy);
                    if (dist < 0.1) { continue; }

                    let f = forces[myType * 20u + otherType];
                    if (dist > f.radius) { continue; }

                    let norm = dist / f.radius;
                    var mag: f32;
                    if (norm < 0.3) {
                        mag = (norm / 0.3 - 1.0);
                    } else {
                        mag = f.strength * (1.0 - abs(1.0 - norm) / 0.7);
                    }

                    let poleData = u32(poleConfigs[otherType]);
                    let mask     = poleMask(-dx / dist, -dy / dist, other.vel, poleData);
                    let contrib  = mag * 0.1 * mask;
                    accel += vec2f(dx, dy) / dist * contrib;
                    typeForce[otherType] += contrib;
                }

                p.vel = p.vel * 0.85 + accel * speed;
                p.pos = p.pos + p.vel;

                if (edgeMode == 0u) {
                    if (p.pos.x < 0.0)   { p.pos.x += width;  }
                    if (p.pos.x > width)  { p.pos.x -= width;  }
                    if (p.pos.y < 0.0)   { p.pos.y += height; }
                    if (p.pos.y > height) { p.pos.y -= height; }
                } else {
                    if (p.pos.x < 0.0)   { p.pos.x = 0.0;    p.vel.x =  abs(p.vel.x) * 0.5; }
                    if (p.pos.x > width)  { p.pos.x = width;  p.vel.x = -abs(p.vel.x) * 0.5; }
                    if (p.pos.y < 0.0)   { p.pos.y = 0.0;    p.vel.y =  abs(p.vel.y) * 0.5; }
                    if (p.pos.y > height) { p.pos.y = height; p.vel.y = -abs(p.vel.y) * 0.5; }
                }

                if (simMode == 1u) {
                    let baseSeed = uhash(idx) ^ uhash(u32(abs(p.pos.x) * 157.0 + 1.0))
                                              ^ uhash(u32(abs(p.pos.y) * 239.0 + 1.0));
                    for (var t: u32 = 0u; t < numTypes; t++) {
                        let rule = transformRules[myType * 20u + t];
                        if (rule.upperEnabled > 0.5) {
                            let prob = transformProb(typeForce[t], rule.upperThreshold);
                            if (prob > 0.0 && rand01(baseSeed ^ uhash(t * 3u + 0u)) < prob) {
                                p.typeId = rule.upperTarget;
                                break;
                            }
                        }
                        if (rule.lowerEnabled > 0.5) {
                            let prob = transformProb(typeForce[t], rule.lowerThreshold);
                            if (prob > 0.0 && rand01(baseSeed ^ uhash(t * 3u + 1u)) < prob) {
                                p.typeId = rule.lowerTarget;
                                break;
                            }
                        }
                    }
                }

                particles[idx] = p;
            }
        `;
    }

    private getRenderShaderCode(): string {
        return /* wgsl */`
            struct Particle { pos: vec2f, vel: vec2f, typeId: f32, _pad: f32 }
            struct VOut {
                @builtin(position) pos: vec4f,
                @location(0) uv: vec2f,
                @location(1) color: vec4f,
                @location(2) @interpolate(flat) glow: f32,
            }
            // view: cx, cy, zoom, saturation, glow, _, _, _  (32 bytes)
            struct View { cx: f32, cy: f32, zoom: f32, sat: f32, glow: f32, _a: f32, _b: f32, _c: f32 }

            @group(0) @binding(0) var<storage, read> particles: array<Particle>;
            @group(0) @binding(1) var<uniform>       params:    vec4f;
            @group(0) @binding(2) var<uniform>       view:      View;

            const COLORS = array<vec4f, 20>(
                vec4f(1.00,0.13,0.13,1.0), vec4f(0.00,0.93,0.00,1.0),
                vec4f(0.13,0.38,1.00,1.0), vec4f(1.00,0.93,0.00,1.0),
                vec4f(1.00,0.13,1.00,1.0), vec4f(0.00,1.00,1.00,1.0),
                vec4f(1.00,0.53,0.00,1.0), vec4f(0.60,0.00,1.00,1.0),
                vec4f(1.00,0.00,0.53,1.0), vec4f(0.00,1.00,0.53,1.0),
                vec4f(0.93,0.93,0.93,1.0), vec4f(1.00,0.47,0.27,1.0),
                vec4f(0.67,1.00,0.00,1.0), vec4f(0.00,0.80,0.73,1.0),
                vec4f(0.33,0.20,1.00,1.0), vec4f(1.00,0.20,0.60,1.0),
                vec4f(1.00,0.80,0.00,1.0), vec4f(0.80,0.53,0.20,1.0),
                vec4f(0.13,0.53,0.20,1.0), vec4f(0.60,0.60,0.60,1.0)
            );

            @vertex
            fn vertexMain(@location(0) quad: vec2f, @builtin(instance_index) inst: u32) -> VOut {
                let p      = particles[inst];
                let worldW = params.y;
                let worldH = params.z;
                let zoom   = view.zoom;
                let nx = (p.pos.x - view.cx) * 2.0 * zoom / worldW;
                let ny = -(p.pos.y - view.cy) * 2.0 * zoom / worldH;
                // Quad grows with glow so the gaussian tail has room to spread.
                let quadScale = 0.006 * (1.0 + view.glow * 3.5);
                var o: VOut;
                o.pos   = vec4f(nx + quad.x * quadScale, ny + quad.y * quadScale, 0.0, 1.0);
                o.uv    = quad;
                o.color = COLORS[min(u32(p.typeId), 19u)];
                o.glow  = view.glow;
                return o;
            }

            @fragment
            fn fragmentMain(i: VOut) -> @location(0) vec4f {
                let d = length(i.uv);  // 0 at centre, 1 at quad's "circle" edge

                // Solid disk: opaque inside, 0 outside.
                let solidBright = max(0.0, 1.0 - d * 0.3);
                let solidAlpha  = select(0.0, solidBright, d < 1.0);

                // Gaussian glow: smooth orb that fades beyond the circle boundary.
                // k controls tightness — lower k → wider, softer glow.
                let k         = mix(12.0, 1.8, i.glow);
                let glowAlpha = exp(-d * d * k);

                // Mix: glow=0 → hard disk, glow=1 → soft orb.
                let alpha = mix(solidAlpha, glowAlpha, i.glow);
                if (alpha < 0.004) { discard; }

                // Saturation adjustment (sat=0 → greyscale, sat=1 → full, sat>1 → vivid)
                let lum = dot(i.color.rgb, vec3f(0.299, 0.587, 0.114));
                let col = mix(vec3f(lum), i.color.rgb, view.sat);

                return vec4f(col * alpha, alpha);
            }
        `;
    }

    // ── Public API ────────────────────────────────────────────────────────────

    update(): void {
        if (!this.isInitialized || !this.device || !this.queue || !this.context || this.isPaused) return;

        this.queue.writeBuffer(this.paramsBuffer!,    0, this.paramsArray());
        this.queue.writeBuffer(this.forcesBuffer!,    0, this.generateForcesData());
        this.queue.writeBuffer(this.transformBuffer!, 0, this.generateTransformData());
        this.queue.writeBuffer(this.poleBuffer!,      0, this.generatePoleData());
        this.queue.writeBuffer(this.viewBuffer!, 0, new Float32Array([
            this.view.cx, this.view.cy, this.view.zoom,
            this.colorSaturation, this.particleGlow, 0, 0, 0,
        ]) as Float32Array<ArrayBuffer>);

        const enc = this.device.createCommandEncoder();

        const cp = enc.beginComputePass();
        cp.setPipeline(this.computePipeline!);
        cp.setBindGroup(0, this.computeBindGroup!);
        cp.dispatchWorkgroups(Math.ceil(this.params.particleCount / 256));
        cp.end();

        const bg = this.backgroundColor;
        const rp = enc.beginRenderPass({
            colorAttachments: [{
                view: this.context.getCurrentTexture().createView(),
                clearValue: { r: bg.r, g: bg.g, b: bg.b, a: 1 },
                loadOp: 'clear', storeOp: 'store',
            }],
        });
        rp.setPipeline(this.renderPipeline!);
        rp.setBindGroup(0, this.renderBindGroup!);
        rp.setVertexBuffer(0, this.quadVertexBuffer!);
        rp.draw(6, this.params.particleCount);
        rp.end();

        this.queue.submit([enc.finish()]);
        this.simulationTime += 0.016;
    }

    updateParams(p: Partial<SimulationParams>): void {
        if (p.simulationSpeed !== undefined) this.params.simulationSpeed = p.simulationSpeed;
        if (p.forceMatrix)                   this.params.forceMatrix = p.forceMatrix;
    }

    setParticleCount(count: number): void {
        if (!this.isInitialized || !this.device || !this.queue) return;
        this.params.particleCount = count;
        this.particleBuffer?.destroy();
        this.particleBuffer = this.makeBuffer(
            'particles', this.generateParticleData(),
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        );
        this.rebuildBindGroups();
    }

    setNumTypes(n: number): void {
        this.numTypes = Math.max(1, Math.min(MAX_TYPES, Math.round(n)));
        // Clamp all transform targets to the new active range
        const maxT = this.numTypes - 1;
        for (const r of this.transformRules) {
            r.upperTarget = Math.min(r.upperTarget, maxT);
            r.lowerTarget = Math.min(r.lowerTarget, maxT);
        }
        if (!this.isInitialized || !this.queue || !this.particleBuffer || !this.paramsBuffer) return;
        this.queue.writeBuffer(this.particleBuffer, 0, this.generateParticleData());
        this.queue.writeBuffer(this.paramsBuffer,   0, this.paramsArray());
    }

    setWorldSize(w: number, h: number): void {
        this.configWidth  = w;
        this.configHeight = h;
        this.params.worldWidth  = this.edgeMode === 1 ? OPEN_MULT * w : w;
        this.params.worldHeight = this.edgeMode === 1 ? OPEN_MULT * h : h;
        this.view = this.defaultView();
        if (!this.isInitialized || !this.queue || !this.particleBuffer || !this.paramsBuffer) return;
        this.queue.writeBuffer(this.particleBuffer, 0, this.generateParticleData());
        this.queue.writeBuffer(this.paramsBuffer,   0, this.paramsArray());
        this.simulationTime = 0;
    }

    setEdgeMode(mode: 0 | 1): void {
        this.edgeMode = mode;
        this.params.worldWidth  = mode === 1 ? OPEN_MULT * this.configWidth  : this.configWidth;
        this.params.worldHeight = mode === 1 ? OPEN_MULT * this.configHeight : this.configHeight;
        this.view = this.defaultView();
        if (!this.isInitialized || !this.queue || !this.particleBuffer || !this.paramsBuffer) return;
        this.queue.writeBuffer(this.particleBuffer, 0, this.generateParticleData());
        this.queue.writeBuffer(this.paramsBuffer,   0, this.paramsArray());
        this.simulationTime = 0;
    }

    setSimMode(mode: 0 | 1): void { this.simMode = mode; }
    getEdgeMode():    number    { return this.edgeMode; }
    getSimMode():     number    { return this.simMode; }
    getNumTypes():    number    { return this.numTypes; }
    getDefaultView(): ViewState { return this.defaultView(); }

    setBackgroundColor(r: number, g: number, b: number): void {
        this.backgroundColor = { r, g, b };
    }
    getBackgroundColor(): { r: number; g: number; b: number } { return { ...this.backgroundColor }; }

    setColorSaturation(s: number): void { this.colorSaturation = Math.max(0, Math.min(2, s)); }
    getColorSaturation(): number { return this.colorSaturation; }

    setParticleGlow(g: number): void { this.particleGlow = Math.max(0, Math.min(1, g)); }
    getParticleGlow(): number { return this.particleGlow; }

    setView(cx: number, cy: number, zoom: number): void { this.view = { cx, cy, zoom }; }
    getView(): ViewState { return { ...this.view }; }

    setTransformRule(source: number, trigger: number, rule: TransformRule): void {
        this.transformRules[source * MAX_TYPES + trigger] = { ...rule };
    }
    getTransformRules(): TransformRule[] { return this.transformRules; }

    randomizeForces(): void { this.initializeForceMatrix(); }
    randomizeTransformRules(): void { this.initializeTransformRules(); }

    reset(): void {
        if (!this.isInitialized || !this.queue || !this.particleBuffer) return;
        this.view = this.defaultView();
        this.queue.writeBuffer(this.particleBuffer, 0, this.generateParticleData());
        this.simulationTime = 0;
    }

    togglePause(): void { this.isPaused = !this.isPaused; }
    getTime(): number   { return this.simulationTime; }
    getParams(): SimulationParams { return this.params; }
    getConfigSize(): { w: number; h: number } { return { w: this.configWidth, h: this.configHeight }; }
    isPaused_(): boolean { return this.isPaused; }

    // ── Poles API ─────────────────────────────────────────────────────────────

    getPoleConfigs(): Array<{ poleCount: number; signBits: number }> {
        return Array.from({ length: this.numTypes }, (_, t) => ({
            poleCount: this.poleConfigs[t] & 0xF,
            signBits:  this.poleConfigs[t] >> 4,
        }));
    }

    setPoleConfig(typeId: number, poleCount: number, signBits: number): void {
        if (typeId < 0 || typeId >= MAX_TYPES) return;
        this.poleConfigs[typeId] = (poleCount & 0xF) | ((signBits & 0x3F) << 4);
        if (this.isInitialized && this.queue && this.poleBuffer) {
            this.queue.writeBuffer(this.poleBuffer, 0, this.generatePoleData());
        }
    }

    randomizePoles(): void {
        const options = [0, 0, 0, 2, 3, 4, 5, 6];  // monopole favoured
        for (let t = 0; t < this.numTypes; t++) {
            const pc = options[Math.floor(Math.random() * options.length)];
            const sb = pc >= 3 ? Math.floor(Math.random() * (1 << pc)) : 0;
            this.poleConfigs[t] = (pc & 0xF) | ((sb & 0x3F) << 4);
        }
        if (this.isInitialized && this.queue && this.poleBuffer) {
            this.queue.writeBuffer(this.poleBuffer, 0, this.generatePoleData());
        }
    }

    // ── Serialization ─────────────────────────────────────────────────────────

    exportState(): object {
        const n  = this.numTypes;
        const cs = this.getConfigSize();
        const fm: object[][] = [];
        const tr: object[] = [];
        for (let from = 0; from < n; from++) {
            fm[from] = [];
            for (let to = 0; to < n; to++) {
                fm[from][to] = { ...this.params.forceMatrix[from]?.[to] ?? { strength: 0, radius: 100 } };
                tr.push({ ...this.transformRules[from * MAX_TYPES + to] });
            }
        }
        const poles = Array.from({ length: n }, (_, t) => ({
            poleCount: this.poleConfigs[t] & 0xF,
            signBits:  this.poleConfigs[t] >> 4,
        }));
        return {
            version:          3,
            numTypes:         n,
            particleCount:    this.params.particleCount,
            simulationSpeed:  this.params.simulationSpeed,
            worldWidth:       cs.w,
            worldHeight:      cs.h,
            simMode:          this.simMode,
            edgeMode:         this.edgeMode,
            backgroundColor:  this.backgroundColor,
            colorSaturation:  this.colorSaturation,
            particleGlow:     this.particleGlow,
            forceMatrix:      fm,
            transformRules:   tr,
            poleConfigs:      poles,
        };
    }

    importState(state: any): void {
        const n = Math.max(1, Math.min(MAX_TYPES, Number(state.numTypes) || 10));
        this.numTypes = n;

        if (state.simulationSpeed != null) this.params.simulationSpeed = Number(state.simulationSpeed);
        if (state.simMode  != null) this.simMode  = Number(state.simMode)  as 0 | 1;
        if (state.edgeMode != null) this.edgeMode = Number(state.edgeMode) as 0 | 1;

        if (state.backgroundColor) {
            const bg = state.backgroundColor;
            this.backgroundColor = { r: Number(bg.r) || 0, g: Number(bg.g) || 0, b: Number(bg.b) || 0 };
        }
        if (state.colorSaturation != null) this.colorSaturation = Number(state.colorSaturation);
        if (state.particleGlow    != null) this.particleGlow    = Number(state.particleGlow);

        if (Array.isArray(state.forceMatrix)) {
            for (let from = 0; from < n; from++) {
                this.params.forceMatrix[from] ??= {};
                for (let to = 0; to < n; to++) {
                    const c = state.forceMatrix[from]?.[to];
                    if (c) this.params.forceMatrix[from][to] = { strength: Number(c.strength), radius: Number(c.radius) };
                }
            }
        }
        if (Array.isArray(state.transformRules)) {
            for (let from = 0; from < n; from++) {
                for (let to = 0; to < n; to++) {
                    const r = state.transformRules[from * n + to];
                    if (r) this.transformRules[from * MAX_TYPES + to] = {
                        upperEnabled:   Boolean(r.upperEnabled),
                        upperThreshold: Number(r.upperThreshold),
                        upperTarget:    Number(r.upperTarget),
                        lowerEnabled:   Boolean(r.lowerEnabled),
                        lowerThreshold: Number(r.lowerThreshold),
                        lowerTarget:    Number(r.lowerTarget),
                    };
                }
            }
        }

        const w = Number(state.worldWidth)  || this.configWidth;
        const h = Number(state.worldHeight) || this.configHeight;
        this.configWidth  = w;
        this.configHeight = h;
        this.params.worldWidth  = this.edgeMode === 1 ? OPEN_MULT * w : w;
        this.params.worldHeight = this.edgeMode === 1 ? OPEN_MULT * h : h;
        this.view = this.defaultView();
        if (state.particleCount) this.params.particleCount = Number(state.particleCount);

        if (Array.isArray(state.poleConfigs)) {
            for (let t = 0; t < n; t++) {
                const pc = state.poleConfigs[t];
                if (pc) this.poleConfigs[t] = (Number(pc.poleCount) & 0xF) | ((Number(pc.signBits) & 0x3F) << 4);
            }
        }

        if (!this.isInitialized || !this.queue) return;
        this.particleBuffer?.destroy();
        this.particleBuffer = this.makeBuffer(
            'particles', this.generateParticleData(),
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        );
        this.rebuildBindGroups();
        this.queue.writeBuffer(this.forcesBuffer!,    0, this.generateForcesData());
        this.queue.writeBuffer(this.transformBuffer!, 0, this.generateTransformData());
        this.queue.writeBuffer(this.poleBuffer!,      0, this.generatePoleData());
        this.queue.writeBuffer(this.paramsBuffer!,    0, this.paramsArray());
        this.simulationTime = 0;
    }
}
