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
            minRadius: number;
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

const OPEN_MULT            = 8;
const MAX_GRID_DIM         = 64;
const MAX_CELLS            = MAX_GRID_DIM * MAX_GRID_DIM;  // 4096
const MAX_PARTICLE_CAPACITY = 300_000;

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

// Triangular distribution sample: min a, peak mode c, max b
function triRand(a: number, c: number, b: number): number {
    const u = Math.random();
    const fc = (c - a) / (b - a);
    return u < fc
        ? a + Math.sqrt(u * (b - a) * (c - a))
        : b - Math.sqrt((1 - u) * (b - a) * (b - c));
}

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
    private transformBuffer:     GPUBuffer | null = null;
    private quadVertexBuffer:    GPUBuffer | null = null;
    private gridCellCountBuffer: GPUBuffer | null = null;
    private gridCellStartBuffer: GPUBuffer | null = null;
    private gridListBuffer:      GPUBuffer | null = null;
    private gridParamsBuffer:    GPUBuffer | null = null;

    private computeBindGroupLayout:  GPUBindGroupLayout   | null = null;
    private computePipeline:         GPUComputePipeline   | null = null;
    private clearGridPipeline:       GPUComputePipeline   | null = null;
    private countParticlesPipeline:  GPUComputePipeline   | null = null;
    private prefixSumPipeline:       GPUComputePipeline   | null = null;
    private scatterPipeline:         GPUComputePipeline   | null = null;
    private renderPipeline:          GPURenderPipeline    | null = null;
    private renderPipelineAdd:       GPURenderPipeline    | null = null;
    private computeBindGroup:        GPUBindGroup         | null = null;
    private renderBindGroup:         GPUBindGroup         | null = null;

    private params: SimulationParams;
    private view:   ViewState = { cx: 0, cy: 0, zoom: 1 };

    private configWidth  = 1600;
    private configHeight = 900;

    private numTypes = 10;
    private simMode  = 0;
    private edgeMode = 0;

    private backgroundColor = { r: 0.05, g: 0.05, b: 0.08 };
    private colorSaturation  = 1.0;
    private particleGlow     = 0.0;  // 0 = hard solid circle, 1 = wide gaussian orb
    private particleAlpha    = 1.0;  // 0 = fully transparent, 1 = fully opaque
    private additiveStrength = 0.7;  // scales per-particle light contribution in additive mode
    private shapeMode        = 1;    // 0 = circles, 1 = procedural polygons
    private blendMode        = 1;    // 0 = standard over, 1 = additive (bloom)

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
            particleCount:   20000,
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
                    strength:  (Math.random() * 2 - 1) * 0.7,
                    radius:    70 + Math.random() * 40,
                    minRadius: 0,
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
        const ab   = new ArrayBuffer(MAX_TYPES * MAX_TYPES * 3 * 4);
        const data = new Float32Array(ab);
        for (let from = 0; from < MAX_TYPES; from++) {
            for (let to = 0; to < MAX_TYPES; to++) {
                const idx = (from * MAX_TYPES + to) * 3;
                const c   = this.params.forceMatrix[from]?.[to];
                data[idx + 0] = c?.strength  ?? 0;
                data[idx + 1] = c?.radius    ?? 100;
                data[idx + 2] = c?.minRadius ?? 0;
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
        // 12-float view buffer (48 bytes, multiple of 16 for WGSL uniform alignment):
        // cx, cy, zoom, sat, glow, alpha, canvasW, canvasH, additiveStr, _p1, _p2, _p3
        this.viewBuffer = this.makeBuffer('view', new Float32Array([
            this.view.cx, this.view.cy, this.view.zoom,
            this.colorSaturation, this.particleGlow, this.particleAlpha,
            this.canvas.width, this.canvas.height,
            this.additiveStrength, this.shapeMode, 0, 0,
        ]) as Float32Array<ArrayBuffer>, U);

        const quad = new Float32Array([-1,-1, 1,-1, 1,1, -1,-1, 1,1, -1,1]);
        this.quadVertexBuffer = this.makeBuffer('quad', quad as Float32Array<ArrayBuffer>, GPUBufferUsage.VERTEX);

        const GS = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
        this.gridCellCountBuffer = this.device!.createBuffer({ label: 'gridCellCount', size: MAX_CELLS * 4, usage: GS });
        this.gridCellStartBuffer = this.device!.createBuffer({ label: 'gridCellStart', size: MAX_CELLS * 4, usage: GS });
        this.gridListBuffer      = this.device!.createBuffer({ label: 'gridList',      size: MAX_PARTICLE_CAPACITY * 4, usage: GS });
        this.gridParamsBuffer    = this.device!.createBuffer({ label: 'gridParams',    size: 16, usage: GS });
    }

    private async createPipelines(): Promise<void> {
        if (!this.device || !this.context || !this.particleBuffer || !this.paramsBuffer ||
            !this.forcesBuffer || !this.transformBuffer || !this.viewBuffer ||
            !this.quadVertexBuffer || !this.poleBuffer ||
            !this.gridCellCountBuffer || !this.gridCellStartBuffer ||
            !this.gridListBuffer || !this.gridParamsBuffer) {
            throw new Error('Buffers not initialized');
        }

        // Single shared layout for all compute pipelines (9 bindings).
        // Each shader only declares the subset it needs; the layout can have extras.
        this.computeBindGroupLayout = this.device.createBindGroupLayout({ entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },            // particles
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },            // params
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // forces
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // transforms
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // poles
            { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // gridParams
            { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // gridCellCount
            { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // gridCellStart
            { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // gridList
        ]});
        const computePipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [this.computeBindGroupLayout] });
        const mkCompute = (code: string) => this.device!.createComputePipeline({
            layout: computePipelineLayout,
            compute: { module: this.device!.createShaderModule({ code }), entryPoint: 'main' },
        });
        this.clearGridPipeline      = mkCompute(this.getClearGridShaderCode());
        this.countParticlesPipeline = mkCompute(this.getCountParticlesShaderCode());
        this.prefixSumPipeline      = mkCompute(this.getPrefixSumShaderCode());
        this.scatterPipeline        = mkCompute(this.getScatterShaderCode());
        this.computePipeline        = mkCompute(this.getComputeShaderCode());

        const renderLayout = this.device.createBindGroupLayout({ entries: [
            { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
            { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
            { binding: 2, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        ]});
        const renderPipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [renderLayout] });
        const renderModule = this.device.createShaderModule({ code: this.getRenderShaderCode() });
        const vertexState: GPUVertexState = {
            module: renderModule, entryPoint: 'vertexMain',
            buffers: [{ arrayStride: 8, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x2' }] }],
        };
        const fmt = navigator.gpu.getPreferredCanvasFormat();
        this.renderPipeline = this.device.createRenderPipeline({
            layout: renderPipelineLayout, vertex: vertexState, primitive: { topology: 'triangle-list' },
            fragment: { module: renderModule, entryPoint: 'fragmentMain',
                targets: [{ format: fmt, blend: {
                    color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
                    alpha: { srcFactor: 'one',       dstFactor: 'zero',                operation: 'add' },
                } }] },
        });
        this.renderPipelineAdd = this.device.createRenderPipeline({
            layout: renderPipelineLayout, vertex: vertexState, primitive: { topology: 'triangle-list' },
            fragment: { module: renderModule, entryPoint: 'fragmentMain',
                targets: [{ format: fmt, blend: {
                    color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
                    alpha: { srcFactor: 'one', dstFactor: 'zero', operation: 'add' },
                } }] },
        });

        this.rebuildBindGroups();
    }

    private rebuildBindGroups(): void {
        if (!this.device || !this.computeBindGroupLayout ||
            !this.particleBuffer || !this.paramsBuffer || !this.forcesBuffer ||
            !this.transformBuffer || !this.viewBuffer || !this.poleBuffer ||
            !this.gridCellCountBuffer || !this.gridCellStartBuffer ||
            !this.gridListBuffer || !this.gridParamsBuffer ||
            !this.computePipeline || !this.renderPipeline || !this.renderPipelineAdd) return;

        this.computeBindGroup = this.device.createBindGroup({
            layout: this.computeBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer       } },
                { binding: 1, resource: { buffer: this.paramsBuffer         } },
                { binding: 2, resource: { buffer: this.forcesBuffer         } },
                { binding: 3, resource: { buffer: this.transformBuffer      } },
                { binding: 4, resource: { buffer: this.poleBuffer           } },
                { binding: 5, resource: { buffer: this.gridParamsBuffer     } },
                { binding: 6, resource: { buffer: this.gridCellCountBuffer  } },
                { binding: 7, resource: { buffer: this.gridCellStartBuffer  } },
                { binding: 8, resource: { buffer: this.gridListBuffer       } },
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

    // ── Grid helpers ──────────────────────────────────────────────────────────

    private computeGridParams(): { gridW: number; gridH: number; cellSize: number; numCells: number } {
        let maxRadius = 1;
        for (let from = 0; from < MAX_TYPES; from++)
            for (let to = 0; to < MAX_TYPES; to++) {
                const r = this.params.forceMatrix[from]?.[to]?.radius ?? 100;
                if (r > maxRadius) maxRadius = r;
            }
        const cellSize = Math.max(maxRadius,
            this.params.worldWidth  / MAX_GRID_DIM,
            this.params.worldHeight / MAX_GRID_DIM);
        // min(3) prevents duplicate-cell visits when wrapping in toroidal mode
        const gridW = Math.max(3, Math.min(MAX_GRID_DIM, Math.ceil(this.params.worldWidth  / cellSize)));
        const gridH = Math.max(3, Math.min(MAX_GRID_DIM, Math.ceil(this.params.worldHeight / cellSize)));
        return { gridW, gridH, cellSize, numCells: gridW * gridH };
    }

    // ── Shaders ───────────────────────────────────────────────────────────────

    private getClearGridShaderCode(): string {
        return /* wgsl */`
            @group(0) @binding(6) var<storage, read_write> cellCount: array<u32>;
            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) id: vec3u) {
                if (id.x < ${MAX_CELLS}u) { cellCount[id.x] = 0u; }
            }
        `;
    }

    private getCountParticlesShaderCode(): string {
        return /* wgsl */`
            struct Particle   { pos: vec2f, vel: vec2f, typeId: f32, _pad: f32 }
            struct GridParams { gridW: u32, gridH: u32, numCells: u32, cellSize: f32 }
            @group(0) @binding(0) var<storage, read_write> particles:  array<Particle>;
            @group(0) @binding(5) var<storage, read>       gridParams: GridParams;
            @group(0) @binding(6) var<storage, read_write> cellCount:  array<atomic<u32>>;
            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) id: vec3u) {
                let idx = id.x;
                if (idx >= arrayLength(&particles)) { return; }
                let p  = particles[idx];
                let gw = gridParams.gridW;
                let gh = gridParams.gridH;
                let cs = gridParams.cellSize;
                let cx = min(u32(max(p.pos.x, 0.0) / cs), gw - 1u);
                let cy = min(u32(max(p.pos.y, 0.0) / cs), gh - 1u);
                atomicAdd(&cellCount[cy * gw + cx], 1u);
            }
        `;
    }

    private getPrefixSumShaderCode(): string {
        return /* wgsl */`
            struct GridParams { gridW: u32, gridH: u32, numCells: u32, cellSize: f32 }
            @group(0) @binding(5) var<storage, read>       gridParams: GridParams;
            @group(0) @binding(6) var<storage, read_write> cellCount:  array<u32>;
            @group(0) @binding(7) var<storage, read_write> cellStart:  array<u32>;
            @compute @workgroup_size(1)
            fn main() {
                var running = 0u;
                let nc = gridParams.numCells;
                for (var i = 0u; i < nc; i++) {
                    let cnt    = cellCount[i];
                    cellStart[i] = running;
                    running   += cnt;
                    cellCount[i] = 0u;
                }
            }
        `;
    }

    private getScatterShaderCode(): string {
        return /* wgsl */`
            struct Particle   { pos: vec2f, vel: vec2f, typeId: f32, _pad: f32 }
            struct GridParams { gridW: u32, gridH: u32, numCells: u32, cellSize: f32 }
            @group(0) @binding(0) var<storage, read_write> particles:  array<Particle>;
            @group(0) @binding(5) var<storage, read>       gridParams: GridParams;
            @group(0) @binding(6) var<storage, read_write> cellCount:  array<atomic<u32>>;
            @group(0) @binding(7) var<storage, read_write> cellStart:  array<u32>;
            @group(0) @binding(8) var<storage, read_write> gridList:   array<u32>;
            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) id: vec3u) {
                let idx = id.x;
                if (idx >= arrayLength(&particles)) { return; }
                let p  = particles[idx];
                let gw = gridParams.gridW;
                let gh = gridParams.gridH;
                let cs = gridParams.cellSize;
                let cx   = min(u32(max(p.pos.x, 0.0) / cs), gw - 1u);
                let cy   = min(u32(max(p.pos.y, 0.0) / cs), gh - 1u);
                let cell = cy * gw + cx;
                let slot = cellStart[cell] + atomicAdd(&cellCount[cell], 1u);
                gridList[slot] = idx;
            }
        `;
    }

    private getComputeShaderCode(): string {
        return /* wgsl */`
            struct Particle    { pos: vec2f, vel: vec2f, typeId: f32, _pad: f32 }
            struct ForceEntry  { strength: f32, radius: f32, minRadius: f32 }
            struct TransformRule {
                upperEnabled: f32, upperThreshold: f32, upperTarget: f32,
                lowerEnabled: f32, lowerThreshold: f32, lowerTarget: f32,
            }
            struct GridParams  { gridW: u32, gridH: u32, numCells: u32, cellSize: f32 }

            @group(0) @binding(0) var<storage, read_write> particles:      array<Particle>;
            @group(0) @binding(1) var<uniform>             params:         vec4f;
            @group(0) @binding(2) var<storage, read>       forces:         array<ForceEntry>;
            @group(0) @binding(3) var<storage, read>       transformRules: array<TransformRule>;
            @group(0) @binding(4) var<storage, read>       poleConfigs:    array<f32>;
            @group(0) @binding(5) var<storage, read>       gridParams:     GridParams;
            @group(0) @binding(6) var<storage, read_write> cellCount:      array<u32>;
            @group(0) @binding(7) var<storage, read_write> cellStart:      array<u32>;
            @group(0) @binding(8) var<storage, read_write> gridList:       array<u32>;

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
                let idx = id.x;
                if (idx >= arrayLength(&particles)) { return; }

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

                let gw   = i32(gridParams.gridW);
                let gh   = i32(gridParams.gridH);
                let cs   = gridParams.cellSize;
                let myGx = clamp(i32(p.pos.x / cs), 0, gw - 1);
                let myGy = clamp(i32(p.pos.y / cs), 0, gh - 1);

                for (var goy = -1; goy <= 1; goy++) {
                    for (var gox = -1; gox <= 1; gox++) {
                        var ngx = myGx + gox;
                        var ngy = myGy + goy;
                        if (edgeMode == 0u) {
                            ngx = ((ngx % gw) + gw) % gw;
                            ngy = ((ngy % gh) + gh) % gh;
                        } else {
                            if (ngx < 0 || ngx >= gw || ngy < 0 || ngy >= gh) { continue; }
                        }
                        let cell  = u32(ngy) * gridParams.gridW + u32(ngx);
                        let start = cellStart[cell];
                        let end   = start + cellCount[cell];
                        for (var k = start; k < end; k++) {
                            let i = gridList[k];
                            if (i == idx) { continue; }
                            let other     = particles[i];
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

                            let f     = forces[myType * 20u + otherType];
                            let range = f.radius - f.minRadius;
                            if (dist < f.minRadius || dist > f.radius || range <= 0.0) { continue; }

                            let norm = (dist - f.minRadius) / range;
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
                    }
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
                @location(0) uv:    vec2f,
                @location(1) color: vec4f,
                @location(2) @interpolate(flat) glow:   f32,
                @location(3) @interpolate(flat) typeId: f32,
            }
            // view: cx,cy,zoom, sat,glow,alpha, canvasW,canvasH, additiveStr,shapeMode,_p2,_p3  (48 B)
            struct View { cx:f32, cy:f32, zoom:f32, sat:f32, glow:f32, alpha:f32,
                          canvasW:f32, canvasH:f32, additiveStr:f32, shapeMode:f32, _p2:f32, _p3:f32 }

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
                // World-space constant size: 20 world-units radius, scales with zoom so
                // zooming in reveals larger particles. aspectX keeps quads square in pixels.
                let quadScale = 20.0 * 2.0 * zoom / worldH * (1.0 + view.glow * 3.5);
                let aspectX   = view.canvasH / view.canvasW;
                var o: VOut;
                o.pos    = vec4f(nx + quad.x * quadScale * aspectX, ny + quad.y * quadScale, 0.0, 1.0);
                o.uv     = quad;
                o.color  = COLORS[min(u32(p.typeId), 19u)];
                o.glow   = view.glow;
                o.typeId = p.typeId;
                return o;
            }

            // ── Shape helpers (hash-driven polygon SDF) ───────────────────────

            fn uhashR(v: u32) -> u32 {
                var x = v ^ (v >> 16u); x *= 0x45d9f3bu; x ^= x >> 16u; return x;
            }

            // Outer-vertex radius for star tips, inner for notches; plain polygon otherwise.
            fn shapeVertexR(tid: u32, vIdx: u32, isStar: bool) -> f32 {
                let h = f32(uhashR(tid * 1013u + vIdx * 179u + 33u)) / 4294967295.0;
                if (isStar) {
                    if ((vIdx & 1u) == 0u) { return 0.55 + h * 0.45; }  // tip  [0.55, 1.0]
                    else                   { return 0.10 + h * 0.25; }  // notch [0.10, 0.35]
                }
                return 0.50 + h * 0.50;   // irregular convex  [0.5, 1.0]
            }

            // Polar outline radius at angle for a given type (linearly interpolated between vertices).
            fn shapeRadius(angle: f32, tid: u32) -> f32 {
                let numSides = 3u + (uhashR(tid * 7919u + 1u) % 5u);         // 3-7 sides
                let isStar   = (uhashR(tid * 3131u + 7u) % 3u) != 0u;        // ~2/3 are stars
                let sectors  = select(numSides, numSides * 2u, isStar);
                let sweep    = 6.28318530718 / f32(sectors);
                let a        = (angle + 3.14159265359) / sweep;
                let secIdx   = u32(a) % sectors;
                let r0       = shapeVertexR(tid, secIdx,                isStar);
                let r1       = shapeVertexR(tid, (secIdx + 1u) % sectors, isStar);
                return mix(r0, r1, fract(a));
            }

            // Normalised distance: 1 = at shape boundary, <1 = inside, >1 = outside.
            fn shapeDist(uv: vec2f, tid: u32) -> f32 {
                let d = length(uv);
                if (d < 0.001) { return 0.0; }
                return d / max(shapeRadius(atan2(uv.y, uv.x), tid), 0.001);
            }

            // ── Fragment ──────────────────────────────────────────────────────

            @fragment
            fn fragmentMain(i: VOut) -> @location(0) vec4f {
                let d = select(length(i.uv), shapeDist(i.uv, u32(i.typeId)), view.shapeMode > 0.5);

                let solidBright = max(0.0, 1.0 - d * 0.3);
                let solidAlpha  = select(0.0, solidBright, d < 1.0);

                let k         = mix(12.0, 1.8, i.glow);
                let glowAlpha = exp(-d * d * k);

                let alpha = mix(solidAlpha, glowAlpha, i.glow) * view.alpha;
                if (alpha < 0.004) { discard; }

                let lum = dot(i.color.rgb, vec3f(0.299, 0.587, 0.114));
                let col = mix(vec3f(lum), i.color.rgb, view.sat);
                return vec4f(col * alpha * view.additiveStr, alpha);
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
            this.colorSaturation, this.particleGlow, this.particleAlpha,
            this.canvas.width, this.canvas.height,
            this.additiveStrength, this.shapeMode, 0, 0,
        ]) as Float32Array<ArrayBuffer>);

        // Upload grid params (cellSize must always ≥ maxRadius so 3×3 neighbourhood is sufficient)
        const gp = this.computeGridParams();
        const gpData = new Uint32Array(4);
        gpData[0] = gp.gridW; gpData[1] = gp.gridH; gpData[2] = gp.numCells;
        new Float32Array(gpData.buffer)[3] = gp.cellSize;
        this.queue.writeBuffer(this.gridParamsBuffer!, 0, gpData);

        const enc = this.device.createCommandEncoder();
        const N   = this.params.particleCount;

        const clearCp = enc.beginComputePass();
        clearCp.setPipeline(this.clearGridPipeline!);
        clearCp.setBindGroup(0, this.computeBindGroup!);
        clearCp.dispatchWorkgroups(Math.ceil(MAX_CELLS / 256));
        clearCp.end();

        const countCp = enc.beginComputePass();
        countCp.setPipeline(this.countParticlesPipeline!);
        countCp.setBindGroup(0, this.computeBindGroup!);
        countCp.dispatchWorkgroups(Math.ceil(N / 256));
        countCp.end();

        const prefixCp = enc.beginComputePass();
        prefixCp.setPipeline(this.prefixSumPipeline!);
        prefixCp.setBindGroup(0, this.computeBindGroup!);
        prefixCp.dispatchWorkgroups(1);
        prefixCp.end();

        const scatterCp = enc.beginComputePass();
        scatterCp.setPipeline(this.scatterPipeline!);
        scatterCp.setBindGroup(0, this.computeBindGroup!);
        scatterCp.dispatchWorkgroups(Math.ceil(N / 256));
        scatterCp.end();

        const forceCp = enc.beginComputePass();
        forceCp.setPipeline(this.computePipeline!);
        forceCp.setBindGroup(0, this.computeBindGroup!);
        forceCp.dispatchWorkgroups(Math.ceil(N / 256));
        forceCp.end();

        const bg = this.backgroundColor;
        const rp = enc.beginRenderPass({
            colorAttachments: [{
                view: this.context.getCurrentTexture().createView(),
                clearValue: { r: bg.r, g: bg.g, b: bg.b, a: 1 },
                loadOp: 'clear', storeOp: 'store',
            }],
        });
        rp.setPipeline(this.blendMode === 1 ? this.renderPipelineAdd! : this.renderPipeline!);
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

    setParticleAlpha(a: number): void { this.particleAlpha = Math.max(0, Math.min(1, a)); }
    getParticleAlpha(): number { return this.particleAlpha; }

    setAdditiveStrength(v: number): void { this.additiveStrength = Math.max(0, Math.min(1, v)); }
    getAdditiveStrength(): number { return this.additiveStrength; }

    setBlendMode(m: 0 | 1): void { this.blendMode = m; }
    getBlendMode(): number { return this.blendMode; }

    setShapeMode(m: 0 | 1): void { this.shapeMode = m; }
    getShapeMode(): number { return this.shapeMode; }

    setView(cx: number, cy: number, zoom: number): void { this.view = { cx, cy, zoom }; }
    getView(): ViewState { return { ...this.view }; }

    setTransformRule(source: number, trigger: number, rule: TransformRule): void {
        this.transformRules[source * MAX_TYPES + trigger] = { ...rule };
    }
    getTransformRules(): TransformRule[] { return this.transformRules; }

    randomizeForces(): void { this.initializeForceMatrix(); }

    randomizeStrengths(): void {
        for (let from = 0; from < MAX_TYPES; from++)
            for (let to = 0; to < MAX_TYPES; to++) {
                const c = this.params.forceMatrix[from]?.[to];
                if (c) c.strength = (Math.random() * 2 - 1) * 0.7;
            }
    }

    randomizeMaxRadii(): void {
        for (let from = 0; from < MAX_TYPES; from++)
            for (let to = 0; to < MAX_TYPES; to++) {
                const c = this.params.forceMatrix[from]?.[to];
                if (c) {
                    c.radius = triRand(10, 100, 250);
                    if (c.minRadius >= c.radius) c.minRadius = 0;
                }
            }
    }

    randomizeMinRadii(): void {
        for (let from = 0; from < MAX_TYPES; from++)
            for (let to = 0; to < MAX_TYPES; to++) {
                const c = this.params.forceMatrix[from]?.[to];
                // Triangular: mode at 15% of max radius, tail up to 80%
                if (c) c.minRadius = Math.min(c.radius - 1, triRand(0, c.radius * 0.15, c.radius * 0.8));
            }
    }

    zeroMinRadii(): void {
        for (let from = 0; from < MAX_TYPES; from++)
            for (let to = 0; to < MAX_TYPES; to++) {
                const c = this.params.forceMatrix[from]?.[to];
                if (c) c.minRadius = 0;
            }
    }

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
            particleAlpha:     this.particleAlpha,
            additiveStrength:  this.additiveStrength,
            blendMode:         this.blendMode,
            shapeMode:         this.shapeMode,
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
        if (state.particleAlpha    != null) this.particleAlpha    = Number(state.particleAlpha);
        if (state.additiveStrength != null) this.additiveStrength = Number(state.additiveStrength);
        if (state.blendMode        != null) this.blendMode        = Number(state.blendMode) as 0 | 1;
        if (state.shapeMode        != null) this.shapeMode        = Number(state.shapeMode) as 0 | 1;

        if (Array.isArray(state.forceMatrix)) {
            for (let from = 0; from < n; from++) {
                this.params.forceMatrix[from] ??= {};
                for (let to = 0; to < n; to++) {
                    const c = state.forceMatrix[from]?.[to];
                    if (c) this.params.forceMatrix[from][to] = { strength: Number(c.strength), radius: Number(c.radius), minRadius: Number(c.minRadius) || 0 };
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
