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
    upperInclusive: boolean;   // false = ramp from zero (>), true = step at threshold (>=)
    upperThreshold: number;
    upperTarget:    number;
    lowerEnabled:   boolean;
    lowerInclusive: boolean;   // false = ramp from zero (<), true = step at threshold (<=)
    lowerThreshold: number;
    lowerTarget:    number;
}

export interface DiagnosticData {
    index:          number;
    typeId:         number;
    pos:            [number, number];
    vel:            [number, number];
    speed:          number;
    directionDeg:   number;
    typeForces:     number[];   // length = numTypes
    transformProbs: number[];   // length = numTypes
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
    private friction         = 0.85; // velocity multiplier per tick (1 = no drag, 0 = full stop)
    private maxTransformRate = 0.5;  // peak probability per tick for type conversion in mode 1

    // ── Entity tracking ────────────────────────────────────────────────────────
    private isTracking        = false;
    private trackComX         = 0;
    private trackComY         = 0;
    private trackRadius       = 200;
    private trackDeathRadius  = 800;
    private trackReadPending  = false;
    private trackingParamBuffer:    GPUBuffer | null = null;
    private trackingStatsBuffer:    GPUBuffer | null = null;
    private trackingStagingBuffer:  GPUBuffer | null = null;
    private trackingBGL:            GPUBindGroupLayout | null = null;
    private trackingBindGroup:      GPUBindGroup | null = null;
    private clearTrackPipeline:     GPUComputePipeline | null = null;
    private accumTrackPipeline:     GPUComputePipeline | null = null;
    onTrackingStop: (() => void) | null = null;

    // ── Particle diagnostic / inspector ───────────────────────────────────────
    private selectedParticleIdx    = -1;
    private snapStagingBuffer:       GPUBuffer | null = null;
    private snapReadPending          = false;
    private diagParamBuffer:         GPUBuffer | null = null;
    private diagOutputBuffer:        GPUBuffer | null = null;
    private diagStagingBuffer:       GPUBuffer | null = null;
    private diagBGL:                 GPUBindGroupLayout | null = null;
    private diagPipeline:            GPUComputePipeline | null = null;
    private diagBindGroup:           GPUBindGroup | null = null;
    private diagReadPending          = false;
    public  diagData:                DiagnosticData | null = null;
    public  onDiagnosticUpdate:      ((data: DiagnosticData | null) => void) | null = null;

    // ── Remap types pipeline (non-destructive type count change) ──────────────
    private remapParamsBuffer: GPUBuffer | null = null;
    private remapBGL:          GPUBindGroupLayout | null = null;
    private remapPipeline:     GPUComputePipeline | null = null;
    private remapBindGroup:    GPUBindGroup | null = null;

    // ── Cursor force pipeline ──────────────────────────────────────────────────
    private cursorParamBuffer: GPUBuffer | null = null;
    private cursorBGL:         GPUBindGroupLayout | null = null;
    private cursorPipeline:    GPUComputePipeline | null = null;
    private cursorBindGroup:   GPUBindGroup | null = null;

    // ── Erase / spawn (paint tool) ─────────────────────────────────────────────
    private eraseParamBuffer:  GPUBuffer | null = null;
    private erasePipeline:     GPUComputePipeline | null = null;
    private eraseBindGroup:    GPUBindGroup | null = null;
    private eraseFrameCounter  = 0;

    // ── Mode 2 (mass) ─────────────────────────────────────────────────────────
    private typeMass: number[] = Array(MAX_TYPES).fill(1);
    private typeMassBuffer:       GPUBuffer | null = null;
    private m2ClaimedBuffer:      GPUBuffer | null = null;
    private m2ActiveCountBuffer:  GPUBuffer | null = null;
    private m2StagingBuffer:      GPUBuffer | null = null;
    private m2FrameCounterBuffer: GPUBuffer | null = null;
    private m2TransformBGL:       GPUBindGroupLayout | null = null;
    private m2TransformPipeline:  GPUComputePipeline | null = null;
    private m2TransformBindGroup: GPUBindGroup | null = null;
    private m2FrameCounter = 1;
    private m2CountPending = false;

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
                upperInclusive: false,
                upperThreshold: 0.3 + Math.random() * 0.5,
                upperTarget:    randTarget(),
                lowerEnabled:   Math.random() < 0.25,
                lowerInclusive: false,
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

    private generateTypeMassData(): Uint32Array<ArrayBuffer> {
        const data = new Uint32Array(MAX_TYPES);
        for (let i = 0; i < MAX_TYPES; i++) data[i] = Math.max(1, Math.min(8, this.typeMass[i] ?? 1));
        return data as Uint32Array<ArrayBuffer>;
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
            data[b + 0] = r.upperEnabled ? (r.upperInclusive ? 2 : 1) : 0;
            data[b + 1] = r.upperThreshold;
            data[b + 2] = r.upperTarget;
            data[b + 3] = r.lowerEnabled ? (r.lowerInclusive ? 2 : 1) : 0;
            data[b + 4] = r.lowerThreshold;
            data[b + 5] = r.lowerTarget;
        }
        return data;
    }

    // params: two vec4f (32 bytes).
    // [0] speed, worldW, worldH, packed(simMode|edgeMode|numTypes as float)
    // [1] friction, maxTransformRate, 0, 0
    private paramsArray(): Float32Array<ArrayBuffer> {
        const packed = (this.numTypes << 16) | (this.edgeMode << 8) | this.simMode;
        return new Float32Array([
            this.params.simulationSpeed,
            this.params.worldWidth,
            this.params.worldHeight,
            packed,
            this.friction,
            this.maxTransformRate,
            0, 0,
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
        const S  = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
        const U  = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;

        // Pre-allocate at max capacity so the paint/spawn tool can append without reallocation
        this.particleBuffer = this.device!.createBuffer({ label: 'particles', size: MAX_PARTICLE_CAPACITY * 6 * 4, usage: S });
        this.queue!.writeBuffer(this.particleBuffer, 0, this.generateParticleData());
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
            this.additiveStrength, this.shapeMode, this.simulationTime, 0,
        ]) as Float32Array<ArrayBuffer>, U);

        const quad = new Float32Array([-1,-1, 1,-1, 1,1, -1,-1, 1,1, -1,1]);
        this.quadVertexBuffer = this.makeBuffer('quad', quad as Float32Array<ArrayBuffer>, GPUBufferUsage.VERTEX);

        const GS = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
        this.gridCellCountBuffer = this.device!.createBuffer({ label: 'gridCellCount', size: MAX_CELLS * 4, usage: GS });
        this.gridCellStartBuffer = this.device!.createBuffer({ label: 'gridCellStart', size: MAX_CELLS * 4, usage: GS });
        this.gridListBuffer      = this.device!.createBuffer({ label: 'gridList',      size: MAX_PARTICLE_CAPACITY * 4, usage: GS });
        this.gridParamsBuffer    = this.device!.createBuffer({ label: 'gridParams',    size: 16, usage: GS });

        // Entity tracking buffers
        this.trackingParamBuffer   = this.device!.createBuffer({ label: 'trackParam',   size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.trackingStatsBuffer   = this.device!.createBuffer({ label: 'trackStats',   size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        this.trackingStagingBuffer = this.device!.createBuffer({ label: 'trackStaging', size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

        // Diagnostic / inspector buffers
        // snapStagingBuffer: used to copy entire particle array to CPU for nearest-particle search
        const snapSize = MAX_PARTICLE_CAPACITY * 6 * 4; // 6 floats per particle
        this.snapStagingBuffer = this.device!.createBuffer({ label: 'snapStaging', size: snapSize, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        this.diagParamBuffer   = this.device!.createBuffer({ label: 'diagParam',   size: 16,       usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.diagOutputBuffer  = this.device!.createBuffer({ label: 'diagOutput',  size: 192,      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        this.diagStagingBuffer = this.device!.createBuffer({ label: 'diagStaging', size: 192,      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

        this.remapParamsBuffer  = this.device!.createBuffer({ label: 'remapParams',  size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.cursorParamBuffer  = this.device!.createBuffer({ label: 'cursorParams', size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

        // Mode 2 (mass) buffers
        this.typeMassBuffer      = this.device!.createBuffer({ label: 'typeMass',       size: 80,  usage: GPUBufferUsage.UNIFORM  | GPUBufferUsage.COPY_DST });
        this.m2ClaimedBuffer     = this.device!.createBuffer({ label: 'm2Claimed',      size: MAX_PARTICLE_CAPACITY * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        this.m2ActiveCountBuffer = this.device!.createBuffer({ label: 'm2ActiveCount',  size: 16,  usage: GPUBufferUsage.STORAGE  | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
        this.m2StagingBuffer     = this.device!.createBuffer({ label: 'm2Staging',      size: 16,  usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        this.m2FrameCounterBuffer = this.device!.createBuffer({ label: 'm2FrameCounter', size: 16, usage: GPUBufferUsage.UNIFORM  | GPUBufferUsage.COPY_DST });
        // Initialize type masses to 1
        this.queue!.writeBuffer(this.typeMassBuffer, 0, this.generateTypeMassData());

        this.eraseParamBuffer = this.device!.createBuffer({ label: 'eraseParams', size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    }

    private async createPipelines(): Promise<void> {
        if (!this.device || !this.context || !this.particleBuffer || !this.paramsBuffer ||
            !this.forcesBuffer || !this.transformBuffer || !this.viewBuffer ||
            !this.quadVertexBuffer || !this.poleBuffer ||
            !this.gridCellCountBuffer || !this.gridCellStartBuffer ||
            !this.gridListBuffer || !this.gridParamsBuffer) {
            throw new Error('Buffers not initialized');
        }

        // Single shared layout for all compute pipelines (10 bindings).
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
            { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },           // typeMasses (mode 2)
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

        // Entity tracking pipeline (separate bind group layout)
        this.trackingBGL = this.device.createBindGroupLayout({ entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // particles
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform'           } }, // trackParams
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage'           } }, // stats (atomics)
        ]});
        const trackPipeLayout = this.device.createPipelineLayout({ bindGroupLayouts: [this.trackingBGL] });
        const trackModule = this.device.createShaderModule({ code: this.getTrackingShaderCode() });
        this.clearTrackPipeline = this.device.createComputePipeline({ layout: trackPipeLayout, compute: { module: trackModule, entryPoint: 'clearStats' } });
        this.accumTrackPipeline = this.device.createComputePipeline({ layout: trackPipeLayout, compute: { module: trackModule, entryPoint: 'accumStats' } });

        // Diagnostic pipeline (uses spatial hash + force rules; poleConfigs omitted to stay within 8 storage limit)
        this.diagBGL = this.device.createBindGroupLayout({ entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // particles
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform'           } }, // simParams
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // forces
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // transformRules
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // gridParams
            { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // cellCount
            { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // cellStart
            { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // gridList
            { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform'           } }, // diagParams
            { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage'           } }, // output
        ]});
        this.diagPipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.diagBGL] }),
            compute: { module: this.device.createShaderModule({ code: this.getDiagnosticShaderCode() }), entryPoint: 'main' },
        });

        // Remap types pipeline
        this.remapBGL = this.device.createBindGroupLayout({ entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        ]});
        this.remapPipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.remapBGL] }),
            compute: { module: this.device.createShaderModule({ code: this.getRemapTypesShaderCode() }), entryPoint: 'main' },
        });

        // Cursor force pipeline
        this.cursorBGL = this.device.createBindGroupLayout({ entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        ]});
        this.cursorPipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.cursorBGL] }),
            compute: { module: this.device.createShaderModule({ code: this.getCursorForceShaderCode() }), entryPoint: 'main' },
        });

        // Erase/paint pipeline (reuses cursorBGL: binding 0=particles, binding 1=uniform)
        this.erasePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.cursorBGL!] }),
            compute: { module: this.device.createShaderModule({ code: this.getEraseShaderCode() }), entryPoint: 'main' },
        });

        // Mode 2 mass-conserving transform pipeline (separate BGL with 7 storage + 2 uniform)
        this.m2TransformBGL = this.device.createBindGroupLayout({ entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage'           } }, // particles
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage'           } }, // claimed (atomic)
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage'           } }, // activeCount (atomic)
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // cellCount
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // cellStart
            { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // gridList
            { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // gridParams
            { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform'           } }, // typeMasses
            { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform'           } }, // frameCounter
        ]});
        this.m2TransformPipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.m2TransformBGL] }),
            compute: { module: this.device.createShaderModule({ code: this.getMode2TransformShaderCode() }), entryPoint: 'main' },
        });

        this.rebuildBindGroups();
    }

    private rebuildBindGroups(): void {
        if (!this.device || !this.computeBindGroupLayout ||
            !this.particleBuffer || !this.paramsBuffer || !this.forcesBuffer ||
            !this.transformBuffer || !this.viewBuffer || !this.poleBuffer ||
            !this.gridCellCountBuffer || !this.gridCellStartBuffer ||
            !this.gridListBuffer || !this.gridParamsBuffer || !this.typeMassBuffer ||
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
                { binding: 9, resource: { buffer: this.typeMassBuffer       } },
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

        if (this.trackingBGL && this.trackingParamBuffer && this.trackingStatsBuffer) {
            this.trackingBindGroup = this.device.createBindGroup({
                layout: this.trackingBGL,
                entries: [
                    { binding: 0, resource: { buffer: this.particleBuffer      } },
                    { binding: 1, resource: { buffer: this.trackingParamBuffer } },
                    { binding: 2, resource: { buffer: this.trackingStatsBuffer } },
                ],
            });
        }

        if (this.diagBGL && this.diagParamBuffer && this.diagOutputBuffer) {
            this.diagBindGroup = this.device.createBindGroup({
                layout: this.diagBGL,
                entries: [
                    { binding: 0, resource: { buffer: this.particleBuffer      } },
                    { binding: 1, resource: { buffer: this.paramsBuffer        } },
                    { binding: 2, resource: { buffer: this.forcesBuffer        } },
                    { binding: 3, resource: { buffer: this.transformBuffer     } },
                    { binding: 4, resource: { buffer: this.gridParamsBuffer    } },
                    { binding: 5, resource: { buffer: this.gridCellCountBuffer } },
                    { binding: 6, resource: { buffer: this.gridCellStartBuffer } },
                    { binding: 7, resource: { buffer: this.gridListBuffer      } },
                    { binding: 8, resource: { buffer: this.diagParamBuffer     } },
                    { binding: 9, resource: { buffer: this.diagOutputBuffer    } },
                ],
            });
        }

        if (this.remapBGL && this.remapParamsBuffer) {
            this.remapBindGroup = this.device.createBindGroup({
                layout: this.remapBGL,
                entries: [
                    { binding: 0, resource: { buffer: this.particleBuffer   } },
                    { binding: 1, resource: { buffer: this.remapParamsBuffer } },
                ],
            });
        }

        if (this.cursorBGL && this.cursorParamBuffer) {
            this.cursorBindGroup = this.device.createBindGroup({
                layout: this.cursorBGL,
                entries: [
                    { binding: 0, resource: { buffer: this.particleBuffer   } },
                    { binding: 1, resource: { buffer: this.cursorParamBuffer } },
                ],
            });
        }

        if (this.cursorBGL && this.eraseParamBuffer) {
            this.eraseBindGroup = this.device.createBindGroup({
                layout: this.cursorBGL,
                entries: [
                    { binding: 0, resource: { buffer: this.particleBuffer!  } },
                    { binding: 1, resource: { buffer: this.eraseParamBuffer } },
                ],
            });
        }

        if (this.m2TransformBGL && this.m2ClaimedBuffer && this.m2ActiveCountBuffer &&
            this.typeMassBuffer && this.m2FrameCounterBuffer) {
            this.m2TransformBindGroup = this.device.createBindGroup({
                layout: this.m2TransformBGL,
                entries: [
                    { binding: 0, resource: { buffer: this.particleBuffer        } },
                    { binding: 1, resource: { buffer: this.m2ClaimedBuffer       } },
                    { binding: 2, resource: { buffer: this.m2ActiveCountBuffer   } },
                    { binding: 3, resource: { buffer: this.gridCellCountBuffer!  } },
                    { binding: 4, resource: { buffer: this.gridCellStartBuffer!  } },
                    { binding: 5, resource: { buffer: this.gridListBuffer!       } },
                    { binding: 6, resource: { buffer: this.gridParamsBuffer!     } },
                    { binding: 7, resource: { buffer: this.typeMassBuffer        } },
                    { binding: 8, resource: { buffer: this.m2FrameCounterBuffer  } },
                ],
            });
        }
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
                if (p.typeId < 0.0) { return; }  // dead particle (mode 2)
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
                if (p.typeId < 0.0) { return; }  // dead particle (mode 2)
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
            struct SimParams  { speed: f32, worldW: f32, worldH: f32, packed: f32,
                                friction: f32, maxRate: f32, _p2: f32, _p3: f32 }

            struct TypeMasses { m: array<vec4u, 5> }

            @group(0) @binding(0) var<storage, read_write> particles:      array<Particle>;
            @group(0) @binding(1) var<uniform>             params:         SimParams;
            @group(0) @binding(2) var<storage, read>       forces:         array<ForceEntry>;
            @group(0) @binding(3) var<storage, read>       transformRules: array<TransformRule>;
            @group(0) @binding(4) var<storage, read>       poleConfigs:    array<f32>;
            @group(0) @binding(5) var<storage, read>       gridParams:     GridParams;
            @group(0) @binding(6) var<storage, read_write> cellCount:      array<u32>;
            @group(0) @binding(7) var<storage, read_write> cellStart:      array<u32>;
            @group(0) @binding(8) var<storage, read_write> gridList:       array<u32>;
            @group(0) @binding(9) var<uniform>             typeMasses:     TypeMasses;

            fn getMass(t: u32) -> u32 { return typeMasses.m[t >> 2u][t & 3u]; }

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

            // enabled: 1 = strict (> / <), 2 = inclusive (>= / <=).
            // isUpper: 1 = upper rule (fire when high), 0 = lower (fire when low).
            fn transformProb(force: f32, threshold: f32, enabled: f32, isUpper: f32) -> f32 {
                if (enabled > 1.5) {
                    // Inclusive: direct comparison, no division needed.
                    if (isUpper > 0.5) { return select(0.0, params.maxRate, force >= threshold); }
                    else               { return select(0.0, params.maxRate, force <= threshold); }
                } else {
                    // Strict: ramp from zero toward threshold.
                    if (abs(threshold) < 0.001) {
                        // threshold = 0: step at zero — fire for any force in the right direction.
                        if (isUpper > 0.5) { return select(0.0, params.maxRate, force > 0.0); }
                        else               { return select(0.0, params.maxRate, force < 0.0); }
                    }
                    let x = force / threshold;
                    if (x <= 0.0) { return 0.0; }
                    let t = min(x, 1.0);
                    return t * t * params.maxRate;
                }
            }

            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) id: vec3u) {
                let idx = id.x;
                if (idx >= arrayLength(&particles)) { return; }

                let speed    = params.speed;
                let width    = params.worldW;
                let height   = params.worldH;
                let packed   = u32(params.packed);
                let simMode  = packed & 0xFFu;
                let edgeMode = (packed >> 8u) & 0xFFu;
                let numTypes = packed >> 16u;

                var p      = particles[idx];
                if (p.typeId < 0.0) { return; }  // dead particle (mode 2 merge victim)
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
                            let other = particles[i];
                            if (other.typeId < 0.0) { continue; }  // skip dead particles
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
                            var outerMag: f32 = 0.0;
                            if (norm < 0.3) {
                                mag = (norm / 0.3 - 1.0);
                            } else {
                                mag = f.strength * (1.0 - abs(1.0 - norm) / 0.7);
                                outerMag = mag;
                            }

                            let poleData = u32(poleConfigs[otherType]);
                            let mask     = poleMask(-dx / dist, -dy / dist, other.vel, poleData);
                            let contrib  = mag * 0.1 * mask;
                            accel += vec2f(dx, dy) / dist * contrib;
                            typeForce[otherType] += outerMag * 0.1 * mask;
                        }
                    }
                }

                // friction^speed: correct discrete-time exponential decay for variable dt
                let fric = pow(params.friction, speed);
                // Mode 2: mass-based inertia — higher mass = less acceleration
                if (simMode == 2u) {
                    let myMass = f32(getMass(myType));
                    p.vel = p.vel * fric + accel * speed / max(myMass, 1.0);
                } else {
                    p.vel = p.vel * fric + accel * speed;
                }
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

                if (simMode == 1u || simMode == 2u) {
                    let baseSeed = uhash(idx) ^ uhash(u32(abs(p.pos.x) * 157.0 + 1.0))
                                              ^ uhash(u32(abs(p.pos.y) * 239.0 + 1.0));
                    var maxProb: f32 = 0.0;
                    var newType: i32 = -1;
                    for (var t: u32 = 0u; t < numTypes; t++) {
                        let rule = transformRules[myType * 20u + t];
                        if (rule.upperEnabled > 0.5) {
                            let prob = transformProb(typeForce[t], rule.upperThreshold, rule.upperEnabled, 1.0);
                            maxProb = max(maxProb, prob);
                            if (newType < 0 && prob > 0.0 && rand01(baseSeed ^ uhash(t * 3u + 0u)) < prob) {
                                newType = i32(rule.upperTarget);
                            }
                        }
                        if (rule.lowerEnabled > 0.5) {
                            let prob = transformProb(typeForce[t], rule.lowerThreshold, rule.lowerEnabled, 0.0);
                            maxProb = max(maxProb, prob);
                            if (newType < 0 && prob > 0.0 && rand01(baseSeed ^ uhash(t * 3u + 1u)) < prob) {
                                newType = i32(rule.lowerTarget);
                            }
                        }
                    }
                    if (simMode == 1u) {
                        // Mode 1: apply transform immediately
                        if (newType >= 0) { p.typeId = f32(newType); }
                        // Smooth stored prob toward current maxProb so whitening transitions gradually
                        let prevProb = clamp(p._pad, 0.0, 0.5);
                        p._pad = mix(prevProb, maxProb, 0.06);
                    } else {
                        // Mode 2: store desired target in _pad; second pass handles mass conservation
                        p._pad = select(-1.0, f32(newType), newType >= 0);
                    }
                } else {
                    p._pad = 0.0;
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
            // view: cx,cy,zoom, sat,glow,alpha, canvasW,canvasH, additiveStr,shapeMode,simTime,_p3  (48 B)
            struct View { cx:f32, cy:f32, zoom:f32, sat:f32, glow:f32, alpha:f32,
                          canvasW:f32, canvasH:f32, additiveStr:f32, shapeMode:f32, simTime:f32, _p3:f32 }

            struct SimParams { speed: f32, worldW: f32, worldH: f32, packed: f32,
                              friction: f32, maxRate: f32, _p2: f32, _p3: f32 }

            @group(0) @binding(0) var<storage, read> particles: array<Particle>;
            @group(0) @binding(1) var<uniform>       params:    SimParams;
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
                // Dead particles (mode 2 merge victims): render off-screen so all 6 verts
                // form zero-area triangles and no fragments are produced.
                if (p.typeId < 0.0) {
                    var o: VOut;
                    o.pos = vec4f(2.0, 2.0, 2.0, 1.0);
                    o.uv = vec2f(0.0); o.color = vec4f(0.0); o.glow = 0.0; o.typeId = 0.0;
                    return o;
                }
                let worldW = params.worldW;
                let worldH = params.worldH;
                let zoom   = view.zoom;
                let nx = (p.pos.x - view.cx) * 2.0 * zoom / worldW;
                let ny = -(p.pos.y - view.cy) * 2.0 * zoom / worldH;
                // World-space constant size: 20 world-units radius, scales with zoom so
                // zooming in reveals larger particles. aspectX keeps quads square in pixels.
                // Instability whitening: _pad stores smoothed max transform probability [0, 0.5].
                // Particle tints toward white in its own hue; at max prob → 75% toward white.
                let normProb  = clamp(p._pad * 2.0, 0.0, 1.0);  // 0→0, 0.5→1
                let whiteness = normProb * 0.75;
                let quadScale = 20.0 * 2.0 * zoom / worldH * (1.0 + view.glow * 3.5);
                let aspectX   = view.canvasH / view.canvasW;
                var o: VOut;
                o.pos    = vec4f(nx + quad.x * quadScale * aspectX, ny + quad.y * quadScale, 0.0, 1.0);
                o.uv     = quad;
                let ownColor = COLORS[min(u32(p.typeId), 19u)];
                o.color  = mix(ownColor, vec4f(1.0, 1.0, 1.0, 1.0), whiteness);
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

    private getTrackingShaderCode(): string {
        return /* wgsl */`
            struct Particle    { pos: vec2f, vel: vec2f, typeId: f32, _pad: f32 }
            struct TrackParams { comX: f32, comY: f32, trackRadius: f32, enabled: u32 }

            @group(0) @binding(0) var<storage, read>       particles: array<Particle>;
            @group(0) @binding(1) var<uniform>             tp: TrackParams;
            @group(0) @binding(2) var<storage, read_write> stats: array<atomic<u32>>;

            // Pass 1: zero the 4 slots (run 1 workgroup, only threads 0-3 do work)
            @compute @workgroup_size(256)
            fn clearStats(@builtin(global_invocation_id) gid: vec3u) {
                if (gid.x < 4u) { atomicStore(&stats[gid.x], 0u); }
            }

            // Pass 2: accumulate (pos - com) offset and spread for particles in radius.
            // Stats layout: [sumDX, sumDY, count, maxDist] all as u32.
            // sumDX/sumDY are signed via bitcast; atomicAdd on u32 is modular = correct i32 arithmetic.
            @compute @workgroup_size(256)
            fn accumStats(@builtin(global_invocation_id) gid: vec3u) {
                if (tp.enabled == 0u) { return; }
                let i = gid.x;
                if (i >= arrayLength(&particles)) { return; }
                let p  = particles[i];
                let dx = p.pos.x - tp.comX;
                let dy = p.pos.y - tp.comY;
                let r  = tp.trackRadius;
                if (dx * dx + dy * dy < r * r) {
                    atomicAdd(&stats[0], bitcast<u32>(i32(dx)));
                    atomicAdd(&stats[1], bitcast<u32>(i32(dy)));
                    atomicAdd(&stats[2], 1u);
                    atomicMax(&stats[3], u32(sqrt(dx * dx + dy * dy)));
                }
            }
        `;
    }

    private getDiagnosticShaderCode(): string {
        return /* wgsl */`
            struct Particle   { pos: vec2f, vel: vec2f, typeId: f32, _pad: f32 }
            struct ForceEntry { strength: f32, radius: f32, minRadius: f32 }
            struct TRule      { uEnabled: f32, uThresh: f32, uTarget: f32,
                                lEnabled: f32, lThresh: f32, lTarget: f32 }
            struct GridParams { gridW: u32, gridH: u32, numCells: u32, cellSize: f32 }
            struct SimParams  { speed: f32, worldW: f32, worldH: f32, packed: f32,
                                friction: f32, maxRate: f32, _p2: f32, _p3: f32 }
            struct DiagParams { selectedIdx: u32, _p1: u32, _p2: u32, _p3: u32 }
            struct DiagOutput {
                pos:           vec2f,
                vel:           vec2f,
                typeId:        f32,
                speed:         f32,
                typeForce:     array<f32, 20>,
                transformProb: array<f32, 20>,
            }

            @group(0) @binding(0) var<storage, read>       particles: array<Particle>;
            @group(0) @binding(1) var<uniform>             sp:        SimParams;
            @group(0) @binding(2) var<storage, read>       forces:    array<ForceEntry>;
            @group(0) @binding(3) var<storage, read>       tRules:    array<TRule>;
            @group(0) @binding(4) var<storage, read>       gridP:     GridParams;
            @group(0) @binding(5) var<storage, read>       cellCount: array<u32>;
            @group(0) @binding(6) var<storage, read>       cellStart: array<u32>;
            @group(0) @binding(7) var<storage, read>       gridList:  array<u32>;
            @group(0) @binding(8) var<uniform>             dp:        DiagParams;
            @group(0) @binding(9) var<storage, read_write> output:    DiagOutput;

            fn dTransformProb(force: f32, threshold: f32, enabled: f32, isUpper: f32) -> f32 {
                if (enabled > 1.5) {
                    if (isUpper > 0.5) { return select(0.0, sp.maxRate, force >= threshold); }
                    else               { return select(0.0, sp.maxRate, force <= threshold); }
                } else {
                    if (abs(threshold) < 0.001) {
                        if (isUpper > 0.5) { return select(0.0, sp.maxRate, force > 0.0); }
                        else               { return select(0.0, sp.maxRate, force < 0.0); }
                    }
                    let x = force / threshold;
                    if (x <= 0.0) { return 0.0; }
                    let t = min(x, 1.0);
                    return t * t * sp.maxRate;
                }
            }

            @compute @workgroup_size(1)
            fn main() {
                let idx = dp.selectedIdx;
                if (idx >= arrayLength(&particles)) { return; }
                let p      = particles[idx];
                let myType = u32(p.typeId);

                let packed   = u32(sp.packed);
                let edgeMode = (packed >> 8u) & 0xFFu;
                let width    = sp.worldW;
                let height   = sp.worldH;

                var typeForce: array<f32, 20>;

                let gw   = i32(gridP.gridW);
                let gh   = i32(gridP.gridH);
                let cs   = gridP.cellSize;
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
                        let cell  = u32(ngy) * gridP.gridW + u32(ngx);
                        let start = cellStart[cell];
                        let end   = start + cellCount[cell];
                        for (var k = start; k < end; k++) {
                            let i         = gridList[k];
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
                            if (norm >= 0.3) {
                                let mag = f.strength * (1.0 - abs(1.0 - norm) / 0.7);
                                typeForce[otherType] += mag * 0.1;
                            }
                        }
                    }
                }

                output.pos    = p.pos;
                output.vel    = p.vel;
                output.typeId = p.typeId;
                output.speed  = length(p.vel);

                for (var t: u32 = 0u; t < 20u; t++) {
                    output.typeForce[t] = typeForce[t];
                    let rule = tRules[myType * 20u + t];
                    var prob: f32 = 0.0;
                    if (rule.uEnabled > 0.5) { prob = max(prob, dTransformProb(typeForce[t], rule.uThresh, rule.uEnabled, 1.0)); }
                    if (rule.lEnabled > 0.5) { prob = max(prob, dTransformProb(typeForce[t], rule.lThresh, rule.lEnabled, 0.0)); }
                    output.transformProb[t] = prob;
                }
            }
        `;
    }

    private getCursorForceShaderCode(): string {
        return /* wgsl */`
            struct Particle    { pos: vec2f, vel: vec2f, typeId: f32, _pad: f32 }
            struct CursorParams { x: f32, y: f32, radius: f32, strength: f32 }

            @group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
            @group(0) @binding(1) var<uniform> cp: CursorParams;

            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) id: vec3u) {
                let i = id.x;
                if (i >= arrayLength(&particles)) { return; }
                var p = particles[i];
                let dx = p.pos.x - cp.x;
                let dy = p.pos.y - cp.y;
                let dist2 = dx * dx + dy * dy;
                if (dist2 > cp.radius * cp.radius || dist2 < 0.0001) { return; }
                let dist    = sqrt(dist2);
                let falloff = 1.0 - dist / cp.radius;
                p.vel += vec2f(dx / dist, dy / dist) * cp.strength * falloff;
                particles[i] = p;
            }
        `;
    }

    private getEraseShaderCode(): string {
        return /* wgsl */`
            struct Particle    { pos: vec2f, vel: vec2f, typeId: f32, _pad: f32 }
            struct EraseParams { x: f32, y: f32, radius: f32, killProb: f32,
                                 typeFilter: i32, seed: u32, _p2: u32, _p3: u32 }

            @group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
            @group(0) @binding(1) var<uniform> ep: EraseParams;

            fn uhash(v: u32) -> u32 { var x = v ^ (v >> 16u); x *= 0x45d9f3bu; x ^= x >> 16u; return x; }

            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) id: vec3u) {
                let i = id.x;
                if (i >= arrayLength(&particles)) { return; }
                var p = particles[i];
                if (p.typeId < 0.0) { return; }
                if (ep.typeFilter >= 0 && i32(p.typeId) != ep.typeFilter) { return; }
                let dx = p.pos.x - ep.x;
                let dy = p.pos.y - ep.y;
                if (dx * dx + dy * dy > ep.radius * ep.radius) { return; }
                if (ep.killProb >= 1.0 || f32(uhash(i ^ uhash(ep.seed))) / 4294967295.0 < ep.killProb) {
                    p.typeId = -1.0;
                    particles[i] = p;
                }
            }
        `;
    }

    private getRemapTypesShaderCode(): string {
        return /* wgsl */`
            struct Particle    { pos: vec2f, vel: vec2f, typeId: f32, _pad: f32 }
            struct RemapParams { newN: u32, _p1: u32, _p2: u32, _p3: u32 }

            @group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
            @group(0) @binding(1) var<uniform> rp: RemapParams;

            fn uhash(v: u32) -> u32 {
                var x = v ^ (v >> 16u);
                x *= 0x45d9f3bu;
                x ^= (x >> 16u);
                return x;
            }

            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) id: vec3u) {
                let i = id.x;
                if (i >= arrayLength(&particles)) { return; }
                let t = u32(particles[i].typeId);
                if (t >= rp.newN) {
                    particles[i].typeId = f32(uhash(i ^ uhash(t)) % rp.newN);
                }
            }
        `;
    }

    private getMode2TransformShaderCode(): string {
        return /* wgsl */`
            struct Particle    { pos: vec2f, vel: vec2f, typeId: f32, _pad: f32 }
            struct GridParams  { gridW: u32, gridH: u32, numCells: u32, cellSize: f32 }
            struct TypeMasses  { m: array<vec4u, 5> }
            struct ActiveCount { n: atomic<u32> }
            struct FrameCount  { n: u32, baseCount: u32, _p2: u32, _p3: u32 }

            @group(0) @binding(0) var<storage, read_write> particles:   array<Particle>;
            @group(0) @binding(1) var<storage, read_write> claimed:     array<atomic<u32>>;
            @group(0) @binding(2) var<storage, read_write> activeCount: ActiveCount;
            @group(0) @binding(3) var<storage, read>       cellCount:   array<u32>;
            @group(0) @binding(4) var<storage, read>       cellStart:   array<u32>;
            @group(0) @binding(5) var<storage, read>       gridList:    array<u32>;
            @group(0) @binding(6) var<storage, read>       gridP:       GridParams;
            @group(0) @binding(7) var<uniform>             tm:          TypeMasses;
            @group(0) @binding(8) var<uniform>             fc:          FrameCount;

            fn getMass(t: u32) -> u32 { return tm.m[t >> 2u][t & 3u]; }

            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) id: vec3u) {
                let idx = id.x;
                if (idx >= fc.baseCount) { return; }

                var p = particles[idx];
                if (p.typeId < 0.0) { return; }  // already dead

                let desiredTarget = i32(p._pad);
                if (desiredTarget < 0) { return; }  // no transform pending

                // Claim self atomically — prevents being consumed while we process
                let ownClaim = atomicExchange(&claimed[idx], fc.n);
                if (ownClaim == fc.n) { return; }  // another particle already claimed us

                let srcType = u32(p.typeId);
                let dstType = u32(desiredTarget);
                if (dstType >= 20u || srcType == dstType) {
                    atomicStore(&claimed[idx], 0u);
                    p._pad = 0.0; particles[idx] = p; return;
                }

                let srcMass = getMass(srcType);
                let dstMass = getMass(dstType);
                let canSplit = (srcMass >= dstMass) && (srcMass % dstMass == 0u);
                let canMerge = (dstMass > srcMass)  && (dstMass % srcMass == 0u);

                if (!canSplit && !canMerge) {
                    // Incompatible masses — cannot conserve mass; block this transform
                    atomicStore(&claimed[idx], 0u);
                    p._pad = 0.0; particles[idx] = p; return;
                }

                if (canSplit) {
                    let n = srcMass / dstMass;
                    p.typeId = f32(dstType);
                    p._pad   = 1.0;  // instability pulse: just transformed
                    particles[idx] = p;
                    // Spawn n-1 additional particles at nearby positions
                    for (var s = 1u; s < n; s++) {
                        let slot = atomicAdd(&activeCount.n, 1u);
                        if (slot < arrayLength(&particles)) {
                            let angle = f32(s) * 2.399963f;  // golden-angle offset
                            var np: Particle;
                            np.pos    = p.pos + vec2f(cos(angle), sin(angle)) * 3.0;
                            np.vel    = p.vel * 0.7;
                            np.typeId = f32(dstType);
                            np._pad   = 0.0;
                            particles[slot] = np;
                        }
                    }
                    return;
                }

                // Merge: need dstMass/srcMass − 1 additional same-type neighbors
                let needed = dstMass / srcMass - 1u;
                var claimedCount = 0u;
                var claimedSlots: array<u32, 8>;  // max needed = 7 (mass-1 → mass-8)

                let gw = i32(gridP.gridW);
                let gh = i32(gridP.gridH);
                let cs = gridP.cellSize;
                let myGx = clamp(i32(p.pos.x / cs), 0, gw - 1);
                let myGy = clamp(i32(p.pos.y / cs), 0, gh - 1);

                for (var goy = -1; goy <= 1 && claimedCount < needed; goy++) {
                    for (var gox = -1; gox <= 1 && claimedCount < needed; gox++) {
                        let ngx = myGx + gox;
                        let ngy = myGy + goy;
                        if (ngx < 0 || ngx >= gw || ngy < 0 || ngy >= gh) { continue; }
                        let cell  = u32(ngy * gw + ngx);
                        let start = cellStart[cell];
                        let cnt   = cellCount[cell];
                        for (var k = 0u; k < cnt && claimedCount < needed; k++) {
                            let j = gridList[start + k];
                            if (j == idx) { continue; }
                            let q = particles[j];
                            if (q.typeId < 0.0 || u32(q.typeId) != srcType) { continue; }
                            // Atomically try to claim neighbor j
                            let old = atomicExchange(&claimed[j], fc.n);
                            if (old != fc.n) {
                                claimedSlots[claimedCount] = j;
                                claimedCount++;
                            }
                        }
                    }
                }

                if (claimedCount >= needed) {
                    // Success: transform self, kill consumed neighbors
                    p.typeId = f32(dstType);
                    p._pad   = 1.0;
                    particles[idx] = p;
                    for (var k = 0u; k < needed; k++) {
                        particles[claimedSlots[k]].typeId = -1.0;
                        particles[claimedSlots[k]]._pad   =  0.0;
                    }
                } else {
                    // Failure: release all partial claims and self
                    for (var k = 0u; k < claimedCount; k++) {
                        atomicStore(&claimed[claimedSlots[k]], 0u);
                    }
                    atomicStore(&claimed[idx], 0u);
                    p._pad = 0.0; particles[idx] = p;
                }
            }
        `;
    }

    // ── Public API ────────────────────────────────────────────────────────────

    update(): void {
        if (!this.isInitialized || !this.device || !this.queue || !this.context) return;

        this.queue.writeBuffer(this.viewBuffer!, 0, new Float32Array([
            this.view.cx, this.view.cy, this.view.zoom,
            this.colorSaturation, this.particleGlow, this.particleAlpha,
            this.canvas.width, this.canvas.height,
            this.additiveStrength, this.shapeMode, this.simulationTime, 0,
        ]) as Float32Array<ArrayBuffer>);

        const enc = this.device.createCommandEncoder();
        const N   = this.params.particleCount;

        if (!this.isPaused) {
            this.queue.writeBuffer(this.paramsBuffer!,    0, this.paramsArray());
            this.queue.writeBuffer(this.forcesBuffer!,    0, this.generateForcesData());
            this.queue.writeBuffer(this.transformBuffer!, 0, this.generateTransformData());
            this.queue.writeBuffer(this.poleBuffer!,      0, this.generatePoleData());

            const gp = this.computeGridParams();
            const gpData = new Uint32Array(4);
            gpData[0] = gp.gridW; gpData[1] = gp.gridH; gpData[2] = gp.numCells;
            new Float32Array(gpData.buffer)[3] = gp.cellSize;
            this.queue.writeBuffer(this.gridParamsBuffer!, 0, gpData);

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

            // Mode 2: mass-conserving transform pass (runs after main physics)
            if (this.simMode === 2 && this.m2TransformPipeline && this.m2TransformBindGroup &&
                this.m2ActiveCountBuffer && this.m2FrameCounterBuffer && this.typeMassBuffer) {
                this.m2FrameCounter = ((this.m2FrameCounter % 0xFFFFFFFE) + 1) >>> 0;
                this.queue.writeBuffer(this.typeMassBuffer, 0, this.generateTypeMassData());
                const m2AcData = new Uint32Array(4); m2AcData[0] = N;
                this.queue.writeBuffer(this.m2ActiveCountBuffer, 0, m2AcData);
                const m2FcData = new Uint32Array(4); m2FcData[0] = this.m2FrameCounter; m2FcData[1] = N;
                this.queue.writeBuffer(this.m2FrameCounterBuffer, 0, m2FcData);
                const m2Cp = enc.beginComputePass();
                m2Cp.setPipeline(this.m2TransformPipeline);
                m2Cp.setBindGroup(0, this.m2TransformBindGroup);
                m2Cp.dispatchWorkgroups(Math.ceil(N / 256));
                m2Cp.end();
            }

            this.simulationTime += 0.016;
        }

        // Entity tracking passes (run every frame so CoM updates even while paused)
        if (this.isTracking && this.trackingBindGroup && this.clearTrackPipeline && this.accumTrackPipeline) {
            this.queue.writeBuffer(this.trackingParamBuffer!, 0, new Float32Array([
                this.trackComX, this.trackComY, this.trackRadius, 1.0,
            ]));
            const clrCp = enc.beginComputePass();
            clrCp.setPipeline(this.clearTrackPipeline);
            clrCp.setBindGroup(0, this.trackingBindGroup);
            clrCp.dispatchWorkgroups(1);
            clrCp.end();
            const accCp = enc.beginComputePass();
            accCp.setPipeline(this.accumTrackPipeline);
            accCp.setBindGroup(0, this.trackingBindGroup);
            accCp.dispatchWorkgroups(Math.ceil(N / 256));
            accCp.end();
        }

        // Diagnostic pass (runs every frame when a particle is selected)
        if (this.selectedParticleIdx >= this.params.particleCount) {
            this.selectedParticleIdx = -1;
            this.diagData = null;
            this.onDiagnosticUpdate?.(null);
        }
        const doDiag = this.selectedParticleIdx >= 0 && !this.diagReadPending &&
                       !!this.diagPipeline && !!this.diagBindGroup;
        if (doDiag) {
            const dp = new Uint32Array(4);
            dp[0] = this.selectedParticleIdx;
            this.queue.writeBuffer(this.diagParamBuffer!, 0, dp);
            const diagCp = enc.beginComputePass();
            diagCp.setPipeline(this.diagPipeline!);
            diagCp.setBindGroup(0, this.diagBindGroup!);
            diagCp.dispatchWorkgroups(1);
            diagCp.end();
        }

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

        const doReadback = this.isTracking && !this.trackReadPending;
        if (doReadback) {
            enc.copyBufferToBuffer(this.trackingStatsBuffer!, 0, this.trackingStagingBuffer!, 0, 16);
        }
        if (doDiag) {
            enc.copyBufferToBuffer(this.diagOutputBuffer!, 0, this.diagStagingBuffer!, 0, 192);
        }

        // Mode 2: copy active count to staging so CPU can track spawns/deaths
        const doM2Count = this.simMode === 2 && !this.isPaused && !this.m2CountPending &&
                          !!this.m2ActiveCountBuffer && !!this.m2StagingBuffer;
        if (doM2Count) {
            enc.copyBufferToBuffer(this.m2ActiveCountBuffer!, 0, this.m2StagingBuffer!, 0, 4);
        }

        this.queue.submit([enc.finish()]);

        if (doDiag) {
            this.diagReadPending = true;
            this.diagStagingBuffer!.mapAsync(GPUMapMode.READ).then(() => {
                const f32  = new Float32Array(this.diagStagingBuffer!.getMappedRange());
                const n    = this.numTypes;
                const data: DiagnosticData = {
                    index:        this.selectedParticleIdx,
                    typeId:       Math.round(f32[4]),
                    pos:          [f32[0], f32[1]],
                    vel:          [f32[2], f32[3]],
                    speed:        f32[5],
                    directionDeg: Math.atan2(f32[3], f32[2]) * 180 / Math.PI,
                    typeForces:   Array.from({ length: n }, (_, t) => f32[6  + t]),
                    transformProbs: Array.from({ length: n }, (_, t) => f32[26 + t]),
                };
                this.diagStagingBuffer!.unmap();
                this.diagReadPending = false;
                this.diagData = data;
                this.onDiagnosticUpdate?.(data);
            });
        }

        if (doReadback) {
            this.trackReadPending = true;
            this.trackingStagingBuffer!.mapAsync(GPUMapMode.READ).then(() => {
                const buf = this.trackingStagingBuffer!.getMappedRange();
                const u32 = new Uint32Array(buf);
                const i32 = new Int32Array(buf);
                const count     = u32[2];
                const maxSpread = u32[3];
                if (count >= 3) {
                    this.trackComX += i32[0] / count;
                    this.trackComY += i32[1] / count;
                    if (maxSpread > this.trackDeathRadius) {
                        this.isTracking = false;
                        this.onTrackingStop?.();
                    } else {
                        this.view = { ...this.view, cx: this.trackComX, cy: this.trackComY };
                    }
                } else if (count === 0) {
                    this.isTracking = false;
                    this.onTrackingStop?.();
                }
                this.trackingStagingBuffer!.unmap();
                this.trackReadPending = false;
            });
        }

        if (doM2Count) {
            this.m2CountPending = true;
            this.m2StagingBuffer!.mapAsync(GPUMapMode.READ, 0, 4).then(() => {
                const u32 = new Uint32Array(this.m2StagingBuffer!.getMappedRange(0, 4));
                const newCount = Math.min(u32[0], MAX_PARTICLE_CAPACITY);
                this.m2StagingBuffer!.unmap();
                this.m2CountPending = false;
                if (newCount !== this.params.particleCount) {
                    this.params.particleCount = newCount;
                    this.queue?.writeBuffer(this.paramsBuffer!, 0, this.paramsArray());
                }
            });
        }
    }

    updateParams(p: Partial<SimulationParams>): void {
        if (p.simulationSpeed !== undefined) this.params.simulationSpeed = p.simulationSpeed;
        if (p.forceMatrix)                   this.params.forceMatrix = p.forceMatrix;
    }

    applyCursorForce(worldX: number, worldY: number, worldRadius: number, strength: number): void {
        if (!this.isInitialized || !this.device || !this.queue ||
            !this.cursorPipeline || !this.cursorBindGroup || !this.cursorParamBuffer) return;
        this.queue.writeBuffer(this.cursorParamBuffer, 0, new Float32Array([worldX, worldY, worldRadius, strength]));
        const enc = this.device.createCommandEncoder();
        const cp  = enc.beginComputePass();
        cp.setPipeline(this.cursorPipeline);
        cp.setBindGroup(0, this.cursorBindGroup);
        cp.dispatchWorkgroups(Math.ceil(this.params.particleCount / 256));
        cp.end();
        this.queue.submit([enc.finish()]);
    }

    eraseParticlesInBrush(wx: number, wy: number, radius: number, killProb: number, typeFilter: number): void {
        if (!this.isInitialized || !this.device || !this.queue ||
            !this.erasePipeline || !this.eraseBindGroup || !this.eraseParamBuffer) return;
        const ab  = new ArrayBuffer(32);
        const f32 = new Float32Array(ab);
        const i32 = new Int32Array(ab);
        const u32 = new Uint32Array(ab);
        f32[0] = wx; f32[1] = wy; f32[2] = radius; f32[3] = Math.min(1, Math.max(0, killProb));
        i32[4] = typeFilter;
        u32[5] = this.eraseFrameCounter++ >>> 0;
        this.queue.writeBuffer(this.eraseParamBuffer, 0, ab);
        const enc = this.device.createCommandEncoder();
        const cp  = enc.beginComputePass();
        cp.setPipeline(this.erasePipeline);
        cp.setBindGroup(0, this.eraseBindGroup);
        cp.dispatchWorkgroups(Math.ceil(this.params.particleCount / 256));
        cp.end();
        this.queue.submit([enc.finish()]);
    }

    spawnParticlesInBrush(wx: number, wy: number, radius: number, typeId: number, count: number): void {
        if (!this.isInitialized || !this.queue || !this.particleBuffer || !this.paramsBuffer) return;
        const N        = this.params.particleCount;
        const spawnable = Math.min(count, MAX_PARTICLE_CAPACITY - N);
        if (spawnable <= 0) return;
        const ab   = new ArrayBuffer(spawnable * 6 * 4);
        const data = new Float32Array(ab);
        for (let i = 0; i < spawnable; i++) {
            const r = radius * Math.sqrt(Math.random());
            const a = Math.random() * 2 * Math.PI;
            const b = i * 6;
            data[b]     = wx + r * Math.cos(a);
            data[b + 1] = wy + r * Math.sin(a);
            data[b + 4] = typeId;
        }
        this.queue.writeBuffer(this.particleBuffer, N * 6 * 4, data);
        this.params.particleCount = N + spawnable;
        this.queue.writeBuffer(this.paramsBuffer, 0, this.paramsArray());
    }

    selectParticleAt(worldX: number, worldY: number): void {
        if (!this.isInitialized || !this.device || !this.queue ||
            !this.particleBuffer || !this.snapStagingBuffer || this.snapReadPending) return;

        const n        = this.params.particleCount;
        const byteSize = n * 6 * 4; // 6 floats per particle
        const enc = this.device.createCommandEncoder();
        enc.copyBufferToBuffer(this.particleBuffer, 0, this.snapStagingBuffer, 0, byteSize);
        this.queue.submit([enc.finish()]);

        this.snapReadPending = true;
        this.snapStagingBuffer.mapAsync(GPUMapMode.READ, 0, byteSize).then(() => {
            const f32 = new Float32Array(this.snapStagingBuffer!.getMappedRange(0, byteSize));
            let bestDist = Infinity, bestIdx = -1;
            for (let i = 0; i < n; i++) {
                const dx = f32[i * 6] - worldX;
                const dy = f32[i * 6 + 1] - worldY;
                const d  = dx * dx + dy * dy;
                if (d < bestDist) { bestDist = d; bestIdx = i; }
            }
            this.snapStagingBuffer!.unmap();
            this.snapReadPending = false;
            if (bestIdx >= 0) { this.selectedParticleIdx = bestIdx; }
        });
    }

    clearParticleSelection(): void {
        this.selectedParticleIdx = -1;
        this.diagData = null;
        this.onDiagnosticUpdate?.(null);
    }

    getSelectedParticleIndex(): number { return this.selectedParticleIdx; }

    setParticleCount(count: number): void {
        if (!this.isInitialized || !this.queue || !this.particleBuffer || !this.paramsBuffer) return;
        this.params.particleCount = count;
        this.queue.writeBuffer(this.particleBuffer, 0, this.generateParticleData());
        this.queue.writeBuffer(this.paramsBuffer, 0, this.paramsArray());
    }

    setNumTypes(n: number): void {
        const newN = Math.max(1, Math.min(MAX_TYPES, Math.round(n)));
        const oldN = this.numTypes;
        if (newN === oldN) return;

        if (newN > oldN) {
            // Add force matrix entries for the new types with random values
            for (let t = oldN; t < newN; t++) {
                if (!this.params.forceMatrix[t]) this.params.forceMatrix[t] = {};
                for (let other = 0; other < newN; other++) {
                    if (this.params.forceMatrix[t][other] == null) {
                        this.params.forceMatrix[t][other] = {
                            strength:  (Math.random() * 2 - 1) * 0.7,
                            radius:    triRand(10, 100, 250),
                            minRadius: 0,
                        };
                    }
                    if (!this.params.forceMatrix[other]) this.params.forceMatrix[other] = {};
                    if (other !== t && this.params.forceMatrix[other][t] == null) {
                        this.params.forceMatrix[other][t] = {
                            strength:  (Math.random() * 2 - 1) * 0.7,
                            radius:    triRand(10, 100, 250),
                            minRadius: 0,
                        };
                    }
                }
            }
            // New types start with all transform rules disabled
            for (let t = oldN; t < newN; t++) {
                for (let other = 0; other < MAX_TYPES; other++) {
                    this.transformRules[t * MAX_TYPES + other].upperEnabled   = false;
                    this.transformRules[t * MAX_TYPES + other].upperInclusive = false;
                    this.transformRules[t * MAX_TYPES + other].lowerEnabled   = false;
                    this.transformRules[t * MAX_TYPES + other].lowerInclusive = false;
                    if (other !== t) {
                        this.transformRules[other * MAX_TYPES + t].upperEnabled   = false;
                        this.transformRules[other * MAX_TYPES + t].upperInclusive = false;
                        this.transformRules[other * MAX_TYPES + t].lowerEnabled   = false;
                        this.transformRules[other * MAX_TYPES + t].lowerInclusive = false;
                    }
                }
            }
            this.numTypes = newN;
        } else {
            // Clamp transform targets to the new active range
            const maxT = newN - 1;
            for (const r of this.transformRules) {
                r.upperTarget = Math.min(r.upperTarget, maxT);
                r.lowerTarget = Math.min(r.lowerTarget, maxT);
            }
            this.numTypes = newN;
            // GPU pass: particles with typeId >= newN get remapped to random [0, newN)
            if (this.isInitialized && this.device && this.queue &&
                this.remapPipeline && this.remapBindGroup && this.remapParamsBuffer) {
                const rp = new Uint32Array(4);
                rp[0] = newN;
                this.queue.writeBuffer(this.remapParamsBuffer, 0, rp);
                const enc = this.device.createCommandEncoder();
                const cp  = enc.beginComputePass();
                cp.setPipeline(this.remapPipeline);
                cp.setBindGroup(0, this.remapBindGroup);
                cp.dispatchWorkgroups(Math.ceil(this.params.particleCount / 256));
                cp.end();
                this.queue.submit([enc.finish()]);
            }
        }

        if (!this.isInitialized || !this.queue || !this.paramsBuffer) return;
        this.queue.writeBuffer(this.paramsBuffer, 0, this.paramsArray());
        if (this.forcesBuffer)    this.queue.writeBuffer(this.forcesBuffer,    0, this.generateForcesData());
        if (this.transformBuffer) this.queue.writeBuffer(this.transformBuffer, 0, this.generateTransformData());
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

    setSimMode(mode: 0 | 1 | 2): void { this.simMode = mode; }
    getEdgeMode():    number    { return this.edgeMode; }
    getSimMode():     number    { return this.simMode; }

    setTypeMass(typeIdx: number, mass: number): void {
        if (typeIdx < 0 || typeIdx >= MAX_TYPES) return;
        this.typeMass[typeIdx] = Math.max(1, Math.min(8, Math.round(mass)));
        if (this.queue && this.typeMassBuffer) {
            this.queue.writeBuffer(this.typeMassBuffer, 0, this.generateTypeMassData());
        }
    }
    getTypeMass(): number[] { return this.typeMass.slice(0, this.numTypes); }

    randomizeMasses(): void {
        const n = this.numTypes;
        if (n <= 1) {
            this.typeMass[0] = 1 + Math.floor(Math.random() * 8);
            if (this.queue && this.typeMassBuffer)
                this.queue.writeBuffer(this.typeMassBuffer, 0, this.generateTypeMassData());
            return;
        }

        const compat = (a: number, b: number) => a % b === 0 || b % a === 0;

        // All masses in [1,8] that divide or are divisible by m
        const compatibleWith = (m: number): number[] => {
            const out: number[] = [];
            for (let c = 1; c <= 8; c++) if (compat(c, m)) out.push(c);
            return out;
        };

        const masses = Array.from({ length: n }, () => 1 + Math.floor(Math.random() * 8));

        // Up to 3 passes: fix any stranded type (no compatible partner in the set)
        for (let pass = 0; pass < 3; pass++) {
            for (let i = 0; i < n; i++) {
                const hasPartner = masses.some((m, j) => j !== i && compat(masses[i], m));
                if (!hasPartner) {
                    // Pick a random OTHER type's current mass as the anchor,
                    // then replace mass[i] with a random value compatible with that anchor.
                    const others = masses.filter((_, j) => j !== i);
                    const ref    = others[Math.floor(Math.random() * others.length)];
                    const opts   = compatibleWith(ref);
                    masses[i]    = opts[Math.floor(Math.random() * opts.length)];
                }
            }
        }

        for (let i = 0; i < n; i++) this.typeMass[i] = masses[i];
        if (this.queue && this.typeMassBuffer)
            this.queue.writeBuffer(this.typeMassBuffer, 0, this.generateTypeMassData());
    }
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

    setFriction(v: number): void { this.friction = Math.max(0, Math.min(0.99,v)); }
    getFriction(): number { return this.friction; }

    setMaxTransformRate(v: number): void { this.maxTransformRate = Math.max(0.01, Math.min(1.0, v)); }
    getMaxTransformRate(): number { return this.maxTransformRate; }

    setView(cx: number, cy: number, zoom: number): void { this.view = { cx, cy, zoom }; }
    getView(): ViewState { return { ...this.view }; }

    startTracking(comX: number, comY: number, radius: number): void {
        this.isTracking       = true;
        this.trackComX        = comX;
        this.trackComY        = comY;
        this.trackRadius      = Math.max(10, radius);
        this.trackDeathRadius = radius * 4;
        this.trackReadPending = false;
    }
    stopTracking(): void { this.isTracking = false; }
    isEntityTracking(): boolean { return this.isTracking; }
    getTrackingInfo(): { comX: number; comY: number; radius: number } | null {
        if (!this.isTracking) return null;
        return { comX: this.trackComX, comY: this.trackComY, radius: this.trackRadius };
    }

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
        const masses = Array.from({ length: n }, (_, t) => this.typeMass[t] ?? 1);
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
            friction:          this.friction,
            maxTransformRate:  this.maxTransformRate,
            forceMatrix:      fm,
            transformRules:   tr,
            poleConfigs:      poles,
            typeMass:         masses,
        };
    }

    importState(state: any): void {
        const n = Math.max(1, Math.min(MAX_TYPES, Number(state.numTypes) || 10));
        this.numTypes = n;

        if (state.simulationSpeed != null) this.params.simulationSpeed = Number(state.simulationSpeed);
        if (state.simMode  != null) this.simMode  = Number(state.simMode)  as 0 | 1 | 2;
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
        if (state.friction         != null) this.friction         = Math.max(0, Math.min(0.99,Number(state.friction)));
        if (state.maxTransformRate != null) this.maxTransformRate = Math.max(0.01, Math.min(1.0, Number(state.maxTransformRate)));

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
                        upperInclusive: Boolean(r.upperInclusive),
                        upperThreshold: Number(r.upperThreshold),
                        upperTarget:    Number(r.upperTarget),
                        lowerEnabled:   Boolean(r.lowerEnabled),
                        lowerInclusive: Boolean(r.lowerInclusive),
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

        if (Array.isArray(state.typeMass)) {
            for (let t = 0; t < n; t++) {
                const m = state.typeMass[t];
                if (m != null) this.typeMass[t] = Math.max(1, Math.min(8, Math.round(Number(m))));
            }
        }
        if (Array.isArray(state.poleConfigs)) {
            for (let t = 0; t < n; t++) {
                const pc = state.poleConfigs[t];
                if (pc) this.poleConfigs[t] = (Number(pc.poleCount) & 0xF) | ((Number(pc.signBits) & 0x3F) << 4);
            }
        }

        if (!this.isInitialized || !this.queue) return;
        this.queue.writeBuffer(this.particleBuffer!,  0, this.generateParticleData());
        this.queue.writeBuffer(this.forcesBuffer!,    0, this.generateForcesData());
        this.queue.writeBuffer(this.transformBuffer!, 0, this.generateTransformData());
        this.queue.writeBuffer(this.poleBuffer!,      0, this.generatePoleData());
        this.queue.writeBuffer(this.paramsBuffer!,    0, this.paramsArray());
        if (this.typeMassBuffer) this.queue.writeBuffer(this.typeMassBuffer, 0, this.generateTypeMassData());
        this.simulationTime = 0;
    }
}
