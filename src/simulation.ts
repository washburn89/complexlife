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

// Transform #2: per-type force-based boolean rules. Each source type owns a set
// of conditions (C1, C2, …); a condition compares the accumulated force from one
// trigger type against a threshold with an operator (>, >=, <, <=). On top of the
// conditions sit transform rules: each rule is a target type plus a boolean
// expression over the conditions (and / or / nand / nor / not / parentheses), e.g.
// "(C1 and C2) or C3". The first rule whose expression is true transforms the
// particle (probabilistically, gated by Max Transform per tick).
export interface DnfCondition { trigger: number; op: 0 | 1 | 2 | 3; threshold: number; }  // op 0:> 1:>= 2:< 3:<=
export interface DnfRule      { target: number; expr: string; rpn: number[]; }            // rpn = compiled tokens
export interface DnfType      { conditions: DnfCondition[]; rules: DnfRule[]; }

// Boolean-expression compiler for DNF rules. Parses an expression over condition
// refs C1, C2, … combined with and/or/nand/nor/not and parentheses into
// reverse-Polish tokens for the GPU stack machine. Token u32 layout:
//   operand:  (0 << 16) | conditionIndex
//   operator: (1 << 16) | opcode      (0=NOT, 1=AND, 2=OR, 3=NAND, 4=NOR)
export const DNF_MAX_TOKENS = 24;
export function compileBoolExpr(expr: string, numConditions: number): { rpn: number[]; error: string | null } {
    const s = (expr ?? '').trim();
    if (!s) return { rpn: [], error: null };  // empty = rule never fires
    type Tk = { t: 'cond' | 'op' | 'lp' | 'rp'; v?: number };
    const toks: Tk[] = [];
    const re = /\s*(\(|\)|[Cc](\d+)|&&|\|\||nand|nor|and|or|not|[&|!~])\s*/y;
    let i = 0;
    while (i < s.length) {
        re.lastIndex = i;
        const m = re.exec(s);
        if (!m || m.index !== i) return { rpn: [], error: `Unexpected "${s.slice(i, i + 8)}"` };
        i = re.lastIndex;
        const w = m[1].toLowerCase();
        if (w === '(') toks.push({ t: 'lp' });
        else if (w === ')') toks.push({ t: 'rp' });
        else if (m[2] !== undefined) {
            const idx = parseInt(m[2], 10) - 1;
            if (idx < 0 || idx >= numConditions) return { rpn: [], error: `C${m[2]} is out of range` };
            toks.push({ t: 'cond', v: idx });
        }
        else if (w === 'not' || w === '!' || w === '~') toks.push({ t: 'op', v: 0 });
        else if (w === 'and' || w === '&' || w === '&&') toks.push({ t: 'op', v: 1 });
        else if (w === 'or'  || w === '|' || w === '||') toks.push({ t: 'op', v: 2 });
        else if (w === 'nand') toks.push({ t: 'op', v: 3 });
        else if (w === 'nor')  toks.push({ t: 'op', v: 4 });
    }
    // Shunting-yard. Precedence: NOT 4 (unary, right-assoc), AND/NAND 3, OR/NOR 2.
    const prec = (op: number) => op === 0 ? 4 : (op === 1 || op === 3 ? 3 : 2);
    const out: number[] = [];
    const ops: Tk[] = [];
    const emit = (op: number) => out.push((1 << 16) | op);
    for (const tk of toks) {
        if (tk.t === 'cond') out.push((0 << 16) | tk.v!);
        else if (tk.t === 'op') {
            const p = prec(tk.v!);
            while (ops.length) {
                const top = ops[ops.length - 1];
                if (top.t !== 'op') break;
                const tp = prec(top.v!);
                if (tk.v === 0 ? tp > p : tp >= p) emit((ops.pop() as Tk).v!);  // unary right-assoc
                else break;
            }
            ops.push(tk);
        }
        else if (tk.t === 'lp') ops.push(tk);
        else {  // rp
            let found = false;
            while (ops.length) { const top = ops.pop()!; if (top.t === 'lp') { found = true; break; } emit(top.v!); }
            if (!found) return { rpn: [], error: 'Unbalanced )' };
        }
    }
    while (ops.length) { const top = ops.pop()!; if (top.t === 'lp') return { rpn: [], error: 'Unbalanced (' }; emit(top.v!); }
    if (out.length > DNF_MAX_TOKENS) return { rpn: [], error: 'Expression too long' };
    // Stack-simulate to confirm it reduces to exactly one value.
    let sp = 0;
    for (const tok of out) {
        const kind = tok >> 16, val = tok & 0xffff;
        if (kind === 0) sp++;
        else if (val === 0) { if (sp < 1) return { rpn: [], error: 'Malformed expression' }; }
        else { if (sp < 2) return { rpn: [], error: 'Malformed expression' }; sp--; }
    }
    if (sp !== 1) return { rpn: [], error: 'Malformed expression' };
    return { rpn: out, error: null };
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

// Transform #2 (DNF) bounds. Per source type: up to MAX_DNF_CONDITIONS conditions
// and MAX_DNF_RULES rules. GPU layout (uniform, u32 via vec4 packing) per type:
//   [0]=numConditions [1]=numRules [2,3]=pad                            (4 u32)
//   conditions: MAX_DNF_CONDITIONS × [trigger, op, threshold(f32 bits), pad]
//   rules:      MAX_DNF_RULES × [target, numTokens, pad, pad, token0..23]
// → 4 + 6*4 + 4*(4+24) = 4 + 24 + 112 = 140 u32 (35 vec4u) per type.
export const MAX_DNF_CONDITIONS = 6;
export const MAX_DNF_RULES      = 8;
const DNF_TYPE_STRIDE_U32 = 4 + MAX_DNF_CONDITIONS * 4 + MAX_DNF_RULES * (4 + DNF_MAX_TOKENS);  // 140

// Each particle re-evaluates its (Mode 4 / Mode 5) transform rules once every
// TFORM_STRIDE frames instead of every frame; the kernel staggers this by
// workgroup so ~1/TFORM_STRIDE of particles evaluate per frame (warp-uniform, so
// it is a real cost reduction). Per-tick transform probability is scaled up to
// compensate, keeping the long-run transform rate roughly unchanged.
const TFORM_STRIDE = 3;

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
    private particlesSortedBuffer: GPUBuffer | null = null;  // cell-sorted copy for coherent neighbor reads
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
    private reorderPipeline:         GPUComputePipeline   | null = null;
    private renderPipeline:          GPURenderPipeline    | null = null;
    private renderPipelineAdd:       GPURenderPipeline    | null = null;
    private computeBindGroup:        GPUBindGroup         | null = null;
    private forceBindGroup:          GPUBindGroup         | null = null;  // like computeBindGroup but binding 8 = sortedParticles
    private reorderBGL:              GPUBindGroupLayout   | null = null;
    private reorderBindGroup:        GPUBindGroup         | null = null;
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
    private poleWorldFrame = false;  // false = lobes anchored to velocity; true = anchored to world +x axis (3+ poles)

    // ── Mode 3 (patchy particles / directional bonding) ─────────────────────────
    // Per-particle orientation lives in a separate buffer (θ, ω) so the Particle
    // struct stays 6 floats and every other mode is untouched. Each type exposes a
    // valence: 0 = isotropic, 2-6 = that many bond patches spaced evenly around the
    // particle. Two particles bond when a patch on each points at the other, which
    // applies a radial spring (toward bondDist) plus a torque that aligns the patch.
    private patchCount: number[] = Array(MAX_TYPES).fill(0);
    private patchBondRange    = 60;    // max centre-to-centre distance a bond can act over
    private patchAngStiffness = 0.3;   // torque strength aligning a patch to the bond axis
    private patchAngFriction  = 0.8;   // angular-velocity damping per tick (lower = more damped)
    private patchWidth        = 6;     // angular selectivity (higher = narrower patches)
    private patchIsoScale     = 1.0;   // full asymmetric force matrix (the "ship" engine) by default
    private patchCoreStrength = 0.6;   // excluded-volume repulsion that keeps structures open
    // Per-type bond params: each particle applies its OWN strength/rest-length to the
    // pull it feels, so mismatched partners simply produce non-reciprocal (motile) bonds.
    private patchTypeBondStr:  number[] = Array(MAX_TYPES).fill(0.2);
    private patchTypeBondDist: number[] = Array(MAX_TYPES).fill(26);
    // Bond-affinity matrix [from*MAX_TYPES + to]: 0 = no bond, >0 scales bond strength.
    // Asymmetric is allowed (predator/prey bonds). Default 1 = everything can bond.
    private patchAffinity: number[] = Array(MAX_TYPES * MAX_TYPES).fill(1);
    private orientationBuffer:       GPUBuffer | null = null;
    private sortedOrientationBuffer: GPUBuffer | null = null;
    private patchConfigBuffer:       GPUBuffer | null = null;
    private patchTablesBuffer:       GPUBuffer | null = null;
    private mode3BGL:                GPUBindGroupLayout | null = null;
    private mode3Pipeline:           GPUComputePipeline | null = null;
    private mode3BindGroup:          GPUBindGroup       | null = null;

    // ── Mode 5 (DNF transforms / Transform #2) ──────────────────────────────────
    // Reuses the Mode 3 patchy kernel for physics; on top, it evaluates per-source-
    // type force-based conditions and boolean transform rules (see DnfType above).
    private dnfTypes: DnfType[] = Array.from({ length: MAX_TYPES }, () => ({ conditions: [], rules: [] }));
    private dnfRulesBuffer: GPUBuffer | null = null;
    // Mode 5 directional bonding: off by default — Mode 5 is DNF transforms on a
    // plain isotropic-force substrate. Toggle on to bring patchy bonds back. Passed
    // to the kernel via params._p2 so it gates all patch passes (and patch render).
    private dnfBonding = false;

    private isInitialized  = false;
    private isPaused       = false;
    private simulationTime = 0;
    // Frame counter (mod a multiple of TFORM_STRIDE) so the kernel can spread the
    // per-particle transform/DNF evaluation across frames for performance.
    private frameCounter   = 0;

    // Dirty flags: config buffers are only re-uploaded to the GPU when their
    // CPU-side data actually changes, instead of every frame.
    private forcesDirty    = true;
    private transformDirty = true;
    private patchDirty     = true;
    private dnfDirty       = true;

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
        this.forcesDirty = true;
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
        this.transformDirty = true;
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

    // Mode 3 orientation: 2 floats per particle — angle θ (random) and angular
    // velocity ω (0). Initialised once at full capacity so reset/spawn paths that
    // only rewrite positions inherit valid random orientations for free.
    private generateOrientationData(): Float32Array<ArrayBuffer> {
        const data = new Float32Array(MAX_PARTICLE_CAPACITY * 2);
        for (let i = 0; i < MAX_PARTICLE_CAPACITY; i++) {
            data[i * 2]     = Math.random() * Math.PI * 2;
            data[i * 2 + 1] = 0;
        }
        return data as Float32Array<ArrayBuffer>;
    }

    // Mode 3 patch config uniform: 8 leading floats of global params followed by
    // the 20 per-type valence counts packed 4-per-vec4u (matches WGSL layout).
    // (Slots 0 and 2 once held global bond strength/distance — now per-type, see
    // generatePatchTablesData — so they are left as unused zeros.)
    private generatePatchConfigData(): ArrayBuffer {
        const ab  = new ArrayBuffer(8 * 4 + MAX_TYPES * 4);  // 32 + 80 = 112 bytes
        const f32 = new Float32Array(ab);
        const u32 = new Uint32Array(ab);
        f32[0] = 0;
        f32[1] = this.patchBondRange;
        f32[2] = 0;
        f32[3] = this.patchAngStiffness;
        f32[4] = this.patchAngFriction;
        f32[5] = this.patchWidth;
        f32[6] = this.patchIsoScale;
        f32[7] = this.patchCoreStrength;
        for (let i = 0; i < MAX_TYPES; i++) {
            u32[8 + i] = Math.max(0, Math.min(6, Math.round(this.patchCount[i] ?? 0)));
        }
        return ab;
    }

    // Mode 3 per-type/affinity tables uniform. Layout (all f32, tight vec4 packing):
    //   [0..399]   affinity[from*20 + to]   (array<vec4f,100>)
    //   [400..419] per-type bond strength   (array<vec4f,5>)
    //   [420..439] per-type bond rest dist  (array<vec4f,5>)
    private generatePatchTablesData(): ArrayBuffer {
        const ab  = new ArrayBuffer(440 * 4);  // 1760 bytes (multiple of 16)
        const f32 = new Float32Array(ab);
        for (let i = 0; i < MAX_TYPES * MAX_TYPES; i++) f32[i] = this.patchAffinity[i] ?? 0;
        for (let t = 0; t < MAX_TYPES; t++) {
            f32[400 + t] = this.patchTypeBondStr[t]  ?? 0;
            f32[420 + t] = this.patchTypeBondDist[t] ?? 26;
        }
        return ab;
    }

    // Mode 5 DNF uniform. Flat u32 array, DNF_TYPE_STRIDE_U32 (140) per source type;
    // see the constant's comment for the per-type layout. Decoded by matching
    // vec4 offsets in the WGSL kernel's Mode 5 transform block.
    private generateDnfData(): Uint32Array<ArrayBuffer> {
        const ab  = new ArrayBuffer(MAX_TYPES * DNF_TYPE_STRIDE_U32 * 4);
        const u32 = new Uint32Array(ab);
        const f32 = new Float32Array(ab);
        const maxTarget = Math.max(0, this.numTypes - 1);  // never transform to an inactive type
        for (let s = 0; s < MAX_TYPES; s++) {
            const dt   = this.dnfTypes[s] ?? { conditions: [], rules: [] };
            const base = s * DNF_TYPE_STRIDE_U32;
            const conds = dt.conditions ?? [];
            const rules = dt.rules ?? [];
            const nC = Math.min(conds.length, MAX_DNF_CONDITIONS);
            const nR = Math.min(rules.length, MAX_DNF_RULES);
            u32[base]     = nC;
            u32[base + 1] = nR;
            for (let c = 0; c < nC; c++) {
                const cd  = conds[c];
                const off = base + 4 + c * 4;
                u32[off]     = Math.max(0, Math.min(MAX_TYPES - 1, cd.trigger | 0));
                u32[off + 1] = Math.max(0, Math.min(3, cd.op | 0));
                f32[off + 2] = cd.threshold ?? 0;
            }
            for (let r = 0; r < nR; r++) {
                const ru   = rules[r];
                const roff = base + 4 + MAX_DNF_CONDITIONS * 4 + r * (4 + DNF_MAX_TOKENS);
                const rpn  = (ru.rpn ?? []).slice(0, DNF_MAX_TOKENS);
                u32[roff]     = Math.max(0, Math.min(maxTarget, ru.target | 0));
                u32[roff + 1] = rpn.length;
                for (let t = 0; t < rpn.length; t++) u32[roff + 4 + t] = rpn[t] >>> 0;
            }
        }
        return u32 as Uint32Array<ArrayBuffer>;
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
    // [0] speed, worldW, worldH, packed(simMode|edgeMode|numTypes|poleWorldFrame as float)
    // [1] friction, maxTransformRate, dnfBonding(0/1), 0
    private paramsArray(): Float32Array<ArrayBuffer> {
        const packed = (this.poleWorldFrame ? (1 << 24) : 0)
                     | (this.numTypes << 16) | (this.edgeMode << 8) | this.simMode;
        return new Float32Array([
            this.params.simulationSpeed,
            this.params.worldWidth,
            this.params.worldHeight,
            packed,
            this.friction,
            this.maxTransformRate,
            this.dnfBonding ? 1 : 0,
            this.frameCounter,
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
        // Cell-sorted copy of the particles, rebuilt each frame by the reorder pass.
        this.particlesSortedBuffer = this.device!.createBuffer({ label: 'particlesSorted', size: MAX_PARTICLE_CAPACITY * 6 * 4, usage: GPUBufferUsage.STORAGE });
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
        // gridParams is read as storage by the classic passes and as a uniform by
        // the Mode 3/4 kernel, so it carries both usage flags.
        this.gridParamsBuffer    = this.device!.createBuffer({ label: 'gridParams',    size: 16, usage: GS | GPUBufferUsage.UNIFORM });

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

        // Mode 3 (patchy) buffers
        this.orientationBuffer = this.device!.createBuffer({ label: 'orientation', size: MAX_PARTICLE_CAPACITY * 2 * 4, usage: S });
        this.queue!.writeBuffer(this.orientationBuffer, 0, this.generateOrientationData());
        this.sortedOrientationBuffer = this.device!.createBuffer({ label: 'orientationSorted', size: MAX_PARTICLE_CAPACITY * 2 * 4, usage: GPUBufferUsage.STORAGE });
        this.patchConfigBuffer = this.device!.createBuffer({ label: 'patchConfig', size: 112, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.queue!.writeBuffer(this.patchConfigBuffer, 0, this.generatePatchConfigData());
        this.patchTablesBuffer = this.device!.createBuffer({ label: 'patchTables', size: 1760, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.queue!.writeBuffer(this.patchTablesBuffer, 0, this.generatePatchTablesData());
        // Mode 5 DNF rules (uniform). MAX_TYPES * 68 u32 = 5440 bytes.
        this.dnfRulesBuffer = this.device!.createBuffer({ label: 'dnfRules', size: MAX_TYPES * DNF_TYPE_STRIDE_U32 * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.queue!.writeBuffer(this.dnfRulesBuffer, 0, this.generateDnfData());
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

        // Reorder pass uses its own layout. It cell-sorts the particle copy and, in
        // lockstep, the per-particle orientation so Mode 3 can read both coherently.
        this.reorderBGL = this.device.createBindGroupLayout({ entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // particles
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // gridList
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage'           } }, // sortedParticles
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // orientation
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage'           } }, // sortedOrientation
        ]});
        this.reorderPipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.reorderBGL] }),
            compute: { module: this.device.createShaderModule({ code: this.getReorderShaderCode() }), entryPoint: 'main' },
        });

        const renderLayout = this.device.createBindGroupLayout({ entries: [
            { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
            { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
            { binding: 2, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
            { binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }, // orientation (Mode 3)
            { binding: 4, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },           // patchConfig (Mode 3)
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

        // Mode 3 (patchy) physics pipeline — its own layout so it can carry the two
        // orientation buffers without pushing the shared compute layout past the
        // 8-storage-buffer limit. Reuses the grid + sorted-particle buffers.
        this.mode3BGL = this.device.createBindGroupLayout({ entries: [
            { binding: 0,  visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage'           } }, // particles (rw)
            { binding: 1,  visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // sortedParticles
            { binding: 2,  visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage'           } }, // orientation (rw)
            { binding: 3,  visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // sortedOrientation
            { binding: 4,  visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // forces
            { binding: 5,  visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform'           } }, // gridParams (uniform here)
            { binding: 6,  visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // cellCount
            { binding: 7,  visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // cellStart
            { binding: 8,  visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform'           } }, // params
            { binding: 9,  visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform'           } }, // patchConfig
            { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // transformRules (mode 4)
            { binding: 11, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform'           } }, // patchTables (affinity + per-type)
            { binding: 12, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform'           } }, // dnfRules (mode 5)
        ]});
        this.mode3Pipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.mode3BGL] }),
            compute: { module: this.device.createShaderModule({ code: this.getMode3ShaderCode() }), entryPoint: 'main' },
        });

        this.rebuildBindGroups();
    }

    private rebuildBindGroups(): void {
        if (!this.device || !this.computeBindGroupLayout ||
            !this.particleBuffer || !this.particlesSortedBuffer || !this.paramsBuffer || !this.forcesBuffer ||
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
        // Identical to computeBindGroup except binding 8 is the sorted particle copy
        // (the force pass reads neighbours from it; it no longer needs gridList).
        this.forceBindGroup = this.device.createBindGroup({
            layout: this.computeBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer        } },
                { binding: 1, resource: { buffer: this.paramsBuffer          } },
                { binding: 2, resource: { buffer: this.forcesBuffer          } },
                { binding: 3, resource: { buffer: this.transformBuffer       } },
                { binding: 4, resource: { buffer: this.poleBuffer            } },
                { binding: 5, resource: { buffer: this.gridParamsBuffer      } },
                { binding: 6, resource: { buffer: this.gridCellCountBuffer   } },
                { binding: 7, resource: { buffer: this.gridCellStartBuffer   } },
                { binding: 8, resource: { buffer: this.particlesSortedBuffer } },
                { binding: 9, resource: { buffer: this.typeMassBuffer        } },
            ],
        });
        if (this.reorderBGL && this.orientationBuffer && this.sortedOrientationBuffer) {
            this.reorderBindGroup = this.device.createBindGroup({
                layout: this.reorderBGL,
                entries: [
                    { binding: 0, resource: { buffer: this.particleBuffer          } },
                    { binding: 1, resource: { buffer: this.gridListBuffer          } },
                    { binding: 2, resource: { buffer: this.particlesSortedBuffer   } },
                    { binding: 3, resource: { buffer: this.orientationBuffer       } },
                    { binding: 4, resource: { buffer: this.sortedOrientationBuffer } },
                ],
            });
        }
        this.renderBindGroup = this.device.createBindGroup({
            layout: this.renderPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer    } },
                { binding: 1, resource: { buffer: this.paramsBuffer      } },
                { binding: 2, resource: { buffer: this.viewBuffer        } },
                { binding: 3, resource: { buffer: this.orientationBuffer! } },
                { binding: 4, resource: { buffer: this.patchConfigBuffer! } },
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

        if (this.mode3BGL && this.orientationBuffer && this.sortedOrientationBuffer && this.patchConfigBuffer && this.patchTablesBuffer && this.dnfRulesBuffer) {
            this.mode3BindGroup = this.device.createBindGroup({
                layout: this.mode3BGL,
                entries: [
                    { binding: 0,  resource: { buffer: this.particleBuffer          } },
                    { binding: 1,  resource: { buffer: this.particlesSortedBuffer   } },
                    { binding: 2,  resource: { buffer: this.orientationBuffer       } },
                    { binding: 3,  resource: { buffer: this.sortedOrientationBuffer } },
                    { binding: 4,  resource: { buffer: this.forcesBuffer            } },
                    { binding: 5,  resource: { buffer: this.gridParamsBuffer        } },
                    { binding: 6,  resource: { buffer: this.gridCellCountBuffer     } },
                    { binding: 7,  resource: { buffer: this.gridCellStartBuffer     } },
                    { binding: 8,  resource: { buffer: this.paramsBuffer            } },
                    { binding: 9,  resource: { buffer: this.patchConfigBuffer       } },
                    { binding: 10, resource: { buffer: this.transformBuffer         } },
                    { binding: 11, resource: { buffer: this.patchTablesBuffer!      } },
                    { binding: 12, resource: { buffer: this.dnfRulesBuffer!         } },
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

    // Exclusive prefix sum over all grid cells, computed by a single workgroup of
    // 256 threads instead of one serial thread. Each thread scans a contiguous
    // chunk of cells; the per-chunk totals are scanned, then the chunk offsets are
    // folded back in. cellCount is reset to 0 so the scatter pass can reuse it as
    // an atomic write cursor. Cells beyond numCells are already 0 (cleared by the
    // clear pass), so scanning the full fixed range is equivalent and uniform.
    private getPrefixSumShaderCode(): string {
        const CHUNK = MAX_CELLS / 256;  // 4096 / 256 = 16
        return /* wgsl */`
            @group(0) @binding(6) var<storage, read_write> cellCount:  array<u32>;
            @group(0) @binding(7) var<storage, read_write> cellStart:  array<u32>;

            var<workgroup> chunkTotal: array<u32, 256>;

            const CHUNK = ${CHUNK}u;

            @compute @workgroup_size(256)
            fn main(@builtin(local_invocation_id) lid: vec3u) {
                let t    = lid.x;
                let base = t * CHUNK;

                // Phase 1: sequential sum of this thread's chunk.
                var sum = 0u;
                for (var i = 0u; i < CHUNK; i++) { sum += cellCount[base + i]; }
                chunkTotal[t] = sum;
                workgroupBarrier();

                // Phase 2: exclusive scan of the 256 chunk totals (one thread).
                if (t == 0u) {
                    var running = 0u;
                    for (var k = 0u; k < 256u; k++) {
                        let c = chunkTotal[k];
                        chunkTotal[k] = running;
                        running += c;
                    }
                }
                workgroupBarrier();

                // Phase 3: write exclusive prefix for each cell, reset count to 0.
                var running = chunkTotal[t];
                for (var i = 0u; i < CHUNK; i++) {
                    let idx = base + i;
                    let cnt = cellCount[idx];
                    cellStart[idx] = running;
                    running += cnt;
                    cellCount[idx] = 0u;
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

    // Gather pass: copy each particle into its cell-sorted slot so the force pass
    // reads neighbours from contiguous memory instead of gathering through gridList.
    private getReorderShaderCode(): string {
        return /* wgsl */`
            struct Particle { pos: vec2f, vel: vec2f, typeId: f32, _pad: f32 }
            @group(0) @binding(0) var<storage, read>       particles:        array<Particle>;
            @group(0) @binding(1) var<storage, read>       gridList:         array<u32>;
            @group(0) @binding(2) var<storage, read_write> sortedParticles:  array<Particle>;
            @group(0) @binding(3) var<storage, read>       orientation:       array<vec2f>;
            @group(0) @binding(4) var<storage, read_write> sortedOrientation: array<vec2f>;
            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) id: vec3u) {
                let k = id.x;
                if (k >= arrayLength(&gridList)) { return; }
                let src = gridList[k];
                sortedParticles[k]   = particles[src];
                sortedOrientation[k] = orientation[src];
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
            // binding 8 holds the cell-sorted particle copy (see reorder pass), so
            // neighbours within a cell are contiguous in memory. Declared read_write
            // to match the shared layout's 'storage' binding type; only read here.
            @group(0) @binding(8) var<storage, read_write> sortedParticles: array<Particle>;
            @group(0) @binding(9) var<uniform>             typeMasses:     TypeMasses;

            fn getMass(t: u32) -> u32 { return typeMasses.m[t >> 2u][t & 3u]; }

            // Polar field mask. ux,uy = unit vector from emitter to receiver.
            // poleData bits 0-3 = poleCount, bits 4+ = sign bits for each lobe.
            // Returns a multiplier in [-1, 1]: positive amplifies, negative reverses force.
            fn poleMask(ux: f32, uy: f32, emitVel: vec2f, poleData: u32, worldFrame: u32) -> f32 {
                let poleCount = poleData & 0xFu;
                if (poleCount == 0u) { return 1.0; }
                let vm = length(emitVel);

                // Dipole is always velocity-aligned: +1 in front of emitter, -1 behind.
                if (poleCount == 2u) {
                    if (vm < 0.01) { return 1.0; }  // stationary: monopole fallback
                    return (emitVel.x / vm) * ux + (emitVel.y / vm) * uy;
                }

                // N poles (3-6): equally spaced lobes, each independently signed.
                // Base axis is the velocity direction, or the fixed world +x axis when
                // worldFrame is set (so which lobe is positive only globally rotates the
                // pattern instead of changing its angle relative to motion).
                var bx: f32; var by: f32;
                if (worldFrame == 1u) {
                    bx = 1.0; by = 0.0;
                } else {
                    if (vm < 0.01) { return 1.0; }  // stationary: monopole fallback
                    bx = emitVel.x / vm; by = emitVel.y / vm;
                }
                let signBits = poleData >> 4u;
                var rawMask: f32 = 0.0;
                let angStep = 6.28318530718 / f32(poleCount);
                let cosA = cos(angStep);
                let sinA = sin(angStep);
                var pvx = bx;
                var pvy = by;
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
                let numTypes = (packed >> 16u) & 0xFFu;
                let poleWorldFrame = (packed >> 24u) & 1u;

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
                            let other = sortedParticles[k];
                            // self is skipped by the dist < 0.1 guard below (identical position)
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
                            let mask     = poleMask(-dx / dist, -dy / dist, other.vel, poleData, poleWorldFrame);
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
                @location(4) @interpolate(flat) spin:   f32,
                @location(5) @interpolate(flat) patchN: f32,
            }
            // view: cx,cy,zoom, sat,glow,alpha, canvasW,canvasH, additiveStr,shapeMode,simTime,_p3  (48 B)
            struct View { cx:f32, cy:f32, zoom:f32, sat:f32, glow:f32, alpha:f32,
                          canvasW:f32, canvasH:f32, additiveStr:f32, shapeMode:f32, simTime:f32, _p3:f32 }

            struct SimParams { speed: f32, worldW: f32, worldH: f32, packed: f32,
                              friction: f32, maxRate: f32, _p2: f32, _p3: f32 }

            struct PatchConfig {
                bondStrength: f32, bondRange: f32, bondDist: f32, angStiffness: f32,
                angFriction: f32, patchWidth: f32, isoScale: f32, coreStrength: f32,
                patchCount: array<vec4u, 5>,
            }

            @group(0) @binding(0) var<storage, read> particles: array<Particle>;
            @group(0) @binding(1) var<uniform>       params:    SimParams;
            @group(0) @binding(2) var<uniform>       view:      View;
            @group(0) @binding(3) var<storage, read> orientation: array<vec2f>;
            @group(0) @binding(4) var<uniform>       patchCfg:  PatchConfig;

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
                    o.spin = 0.0; o.patchN = 0.0;
                    return o;
                }
                let pm = u32(params.packed) & 0xFFu;
                // Patchy modes draw orientation/patches; Mode 5 only when bonding is on.
                let simMode3 = pm == 3u || pm == 4u || (pm == 5u && params._p2 > 0.5);
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
                let tt   = u32(p.typeId);
                o.spin   = select(0.0, orientation[inst].x, simMode3);
                o.patchN = select(0.0, f32(patchCfg.patchCount[tt >> 2u][tt & 3u]), simMode3);
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
                // Mode 3: spin the sampling frame by the particle's orientation so the
                // procedural shape (and its patch nubs) visibly rotate with θ.
                var uv = i.uv;
                if (i.patchN > 0.5) {
                    let c = cos(i.spin); let s = sin(i.spin);
                    uv = vec2f(c * i.uv.x - s * i.uv.y, s * i.uv.x + c * i.uv.y);
                }
                let d = select(length(uv), shapeDist(uv, u32(i.typeId)), view.shapeMode > 0.5);

                let solidBright = max(0.0, 1.0 - d * 0.3);
                let solidAlpha  = select(0.0, solidBright, d < 1.0);

                let k         = mix(12.0, 1.8, i.glow);
                let glowAlpha = exp(-d * d * k);

                var alpha = mix(solidAlpha, glowAlpha, i.glow) * view.alpha;

                // Patch nubs: bright lobes at each valence direction near the rim.
                var patchBoost = 0.0;
                if (i.patchN > 0.5) {
                    let ang  = atan2(uv.y, uv.x);
                    let step = 6.28318530718 / i.patchN;
                    let pa   = round(ang / step) * step;
                    let nub  = pow(max(cos(ang - pa), 0.0), 16.0);
                    let ring = smoothstep(0.45, 0.85, d) * (1.0 - smoothstep(1.0, 1.25, d));
                    patchBoost = nub * ring;
                    alpha = max(alpha, patchBoost * 0.95 * view.alpha);
                }
                if (alpha < 0.004) { discard; }

                let lum = dot(i.color.rgb, vec3f(0.299, 0.587, 0.114));
                let col = mix(vec3f(lum), i.color.rgb, view.sat) + vec3f(patchBoost) * 0.7;
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

    // Mode 3/4: patchy particles. On top of the usual isotropic type-pair force this
    // adds directional "patch" bonds. Each type has a valence (patchCount) of evenly
    // spaced patches fixed to its orientation. Two particles bond when a patch on
    // each points at the other; the bond is a radial spring toward bondDist plus a
    // torque that rotates each particle so its patch locks onto the bond axis. The
    // result is angle-selective coordination — chains, rings and lattices instead of
    // amorphous blobs. Orientation lives in its own buffer (θ, ω); the Particle
    // struct is untouched. In Mode 4 the same kernel also applies Mode-1-style type
    // transforms, so bonded structures can react and change type.
    private getMode3ShaderCode(): string {
        return /* wgsl */`
            struct Particle   { pos: vec2f, vel: vec2f, typeId: f32, _pad: f32 }
            struct ForceEntry { strength: f32, radius: f32, minRadius: f32 }
            struct TransformRule {
                upperEnabled: f32, upperThreshold: f32, upperTarget: f32,
                lowerEnabled: f32, lowerThreshold: f32, lowerTarget: f32,
            }
            struct GridParams { gridW: u32, gridH: u32, numCells: u32, cellSize: f32 }
            struct SimParams  { speed: f32, worldW: f32, worldH: f32, packed: f32,
                                friction: f32, maxRate: f32, _p2: f32, _p3: f32 }
            struct PatchConfig {
                _bs: f32, bondRange: f32, _bd: f32, angStiffness: f32,
                angFriction: f32, patchWidth: f32, isoScale: f32, coreStrength: f32,
                patchCount: array<vec4u, 5>,
            }
            struct PatchTables {
                affinity: array<vec4f, 100>,  // [from*20 + to]
                bondStr:  array<vec4f, 5>,    // per-type bond strength
                bondDist: array<vec4f, 5>,    // per-type bond rest length
            }
            // Mode 5 DNF data. ${DNF_TYPE_STRIDE_U32 / 4} vec4u per source type, read a vec4 at a
            // time in the Mode 5 transform block.
            struct DnfRules { d: array<vec4u, ${MAX_TYPES * DNF_TYPE_STRIDE_U32 / 4}> }

            @group(0) @binding(0)  var<storage, read_write> particles:         array<Particle>;
            @group(0) @binding(1)  var<storage, read>       sortedParticles:   array<Particle>;
            @group(0) @binding(2)  var<storage, read_write> orientation:        array<vec2f>;
            @group(0) @binding(3)  var<storage, read>       sortedOrientation:  array<vec2f>;
            @group(0) @binding(4)  var<storage, read>       forces:            array<ForceEntry>;
            @group(0) @binding(5)  var<uniform>             gridParams:        GridParams;
            @group(0) @binding(6)  var<storage, read>       cellCount:         array<u32>;
            @group(0) @binding(7)  var<storage, read>       cellStart:         array<u32>;
            @group(0) @binding(8)  var<uniform>             params:            SimParams;
            @group(0) @binding(9)  var<uniform>             cfg:               PatchConfig;
            @group(0) @binding(10) var<storage, read>       transformRules:    array<TransformRule>;
            @group(0) @binding(11) var<uniform>             tab:               PatchTables;
            @group(0) @binding(12) var<uniform>             dnf:               DnfRules;

            fn patchN(t: u32) -> u32 { return cfg.patchCount[t >> 2u][t & 3u]; }
            fn bondStrOf(t: u32)  -> f32 { return tab.bondStr[t >> 2u][t & 3u]; }
            fn bondDistOf(t: u32) -> f32 { return tab.bondDist[t >> 2u][t & 3u]; }
            fn affinityOf(a: u32, b: u32) -> f32 { let i = a * 20u + b; return tab.affinity[i >> 2u][i & 3u]; }

            const TAU = 6.28318530718;

            fn uhash(v: u32) -> u32 {
                var x = v ^ (v >> 16u); x *= 0x45d9f3bu; x ^= x >> 16u; return x;
            }
            fn rand01(seed: u32) -> f32 { return f32(uhash(seed)) / 4294967295.0; }

            fn transformProb(force: f32, threshold: f32, enabled: f32, isUpper: f32) -> f32 {
                if (enabled > 1.5) {
                    if (isUpper > 0.5) { return select(0.0, params.maxRate, force >= threshold); }
                    else               { return select(0.0, params.maxRate, force <= threshold); }
                } else {
                    if (abs(threshold) < 0.001) {
                        if (isUpper > 0.5) { return select(0.0, params.maxRate, force > 0.0); }
                        else               { return select(0.0, params.maxRate, force < 0.0); }
                    }
                    let x = force / threshold;
                    if (x <= 0.0) { return 0.0; }
                    let t = min(x, 1.0);
                    return t * t * params.maxRate;
                }
            }

            // Best alignment of any of n evenly-spaced patches (first at angle base)
            // with the target direction aim. Returns the largest cos(patch - aim) in
            // .x, the angle of that best patch in .y, and its index in .z.
            fn bestPatch(base: f32, n: u32, aim: f32) -> vec3f {
                let step = TAU / f32(n);
                var bestAl  = -2.0;
                var bestAng = base;
                var bestK   = 0u;
                for (var i: u32 = 0u; i < n; i++) {
                    let ang = base + f32(i) * step;
                    let al  = cos(ang - aim);
                    if (al > bestAl) { bestAl = al; bestAng = ang; bestK = i; }
                }
                return vec3f(bestAl, bestAng, f32(bestK));
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
                let numTypes = (packed >> 16u) & 0xFFu;

                var p = particles[idx];
                if (p.typeId < 0.0) { return; }
                let myType   = u32(p.typeId);
                // Mode 5 runs on a plain force substrate unless directional bonding
                // is toggled on (params._p2). For modes 3/4 patches are always active.
                let patchesActive = simMode != 5u || params._p2 > 0.5;
                let myPatch  = select(0u, patchN(myType), patchesActive);
                let ori      = orientation[idx];
                let myTheta  = ori.x;
                var myOmega  = ori.y;

                var accel  = vec2f(0.0);
                var torque = 0.0;
                var typeForce: array<f32, 20>;  // per-type accumulated force (Mode 4 & 5 transforms)
                var bestBond: array<f32, 6>;    // strongest bond demand seen on each patch
                var bestIdx:  array<u32, 6>;    // which neighbour won that patch

                let gw   = i32(gridParams.gridW);
                let gh   = i32(gridParams.gridH);
                let cs   = gridParams.cellSize;
                let myGx = clamp(i32(p.pos.x / cs), 0, gw - 1);
                let myGy = clamp(i32(p.pos.y / cs), 0, gh - 1);
                let myBondStr  = bondStrOf(myType);
                let myBondDist = bondDistOf(myType);
                let core = myBondDist * 0.9;     // excluded-volume radius (per-type size)

                // ── Pass A: winner-take-all per patch. Each neighbour is assigned to my
                // best-aligned patch; we keep only the single strongest candidate per
                // patch. Pass B then bonds the winner and *repels* the losers, so a
                // particle truly saturates at its valence (no loose rosette shell).
                if (myPatch > 0u) {
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
                                let other = sortedParticles[k];
                                if (other.typeId < 0.0) { continue; }
                                let otherPatch = patchN(u32(other.typeId));
                                if (otherPatch == 0u) { continue; }
                                var dx = other.pos.x - p.pos.x;
                                var dy = other.pos.y - p.pos.y;
                                if (edgeMode == 0u) {
                                    if (dx >  width  * 0.5) { dx -= width;  }
                                    if (dx < -width  * 0.5) { dx += width;  }
                                    if (dy >  height * 0.5) { dy -= height; }
                                    if (dy < -height * 0.5) { dy += height; }
                                }
                                let dist = sqrt(dx * dx + dy * dy);
                                if (dist < 0.1 || dist > cfg.bondRange) { continue; }
                                let aff = affinityOf(myType, u32(other.typeId));
                                if (aff <= 0.0) { continue; }
                                let axis   = atan2(dy, dx);
                                let mine   = bestPatch(myTheta, myPatch, axis);
                                let theirs = bestPatch(sortedOrientation[k].x, otherPatch, axis + 3.14159265359);
                                let bond   = pow(max(mine.x, 0.0), cfg.patchWidth) * pow(max(theirs.x, 0.0), cfg.patchWidth) * aff;
                                let pk     = u32(mine.z);
                                if (bond > bestBond[pk]) { bestBond[pk] = bond; bestIdx[pk] = k; }
                            }
                        }
                    }
                }

                // ── Pass B: apply isotropic force, excluded volume, and saturated bonds.
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
                            let other = sortedParticles[k];
                            if (other.typeId < 0.0) { continue; }
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
                            let dir = vec2f(dx, dy) / dist;

                            // Isotropic background force. Inner repulsion stays at full
                            // strength (keeps a hard core); the outer attractive/repulsive
                            // shell is scaled by isoScale so directional bonds — not the
                            // particle-life soup — drive cohesion in patchy mode.
                            let f     = forces[myType * 20u + otherType];
                            let range = f.radius - f.minRadius;
                            if (dist >= f.minRadius && dist <= f.radius && range > 0.0) {
                                let norm = (dist - f.minRadius) / range;
                                if (norm < 0.3) {
                                    accel += dir * (norm / 0.3 - 1.0) * 0.1;
                                } else {
                                    let outerMag = f.strength * (1.0 - abs(1.0 - norm) / 0.7);
                                    accel += dir * outerMag * 0.1 * cfg.isoScale;
                                    typeForce[otherType] += outerMag * 0.1;
                                }
                            }

                            // Excluded volume: firm repulsion inside the core radius for
                            // every neighbour, so saturated structures stay open instead
                            // of collapsing to close-packed balls. (Patch-specific, so
                            // skipped in Mode 5 when directional bonding is off.)
                            if (patchesActive && dist < core) {
                                let q = 1.0 - dist / core;
                                accel -= dir * cfg.coreStrength * q * q;
                            }

                            // Directional patch bond, winner-take-all per patch.
                            // Each particle applies its OWN strength + rest length, gated by
                            // its affinity to the partner's type (asymmetric = non-reciprocal).
                            let otherPatch = patchN(otherType);
                            let aff = affinityOf(myType, otherType);
                            if (myPatch > 0u && otherPatch > 0u && aff > 0.0 && dist <= cfg.bondRange) {
                                let axis   = atan2(dy, dx);          // me -> other
                                let mine   = bestPatch(myTheta, myPatch, axis);
                                let theirs = bestPatch(sortedOrientation[k].x, otherPatch, axis + 3.14159265359);
                                let bond   = pow(max(mine.x, 0.0), cfg.patchWidth) * pow(max(theirs.x, 0.0), cfg.patchWidth) * aff;
                                let pk     = u32(mine.z);
                                if (bond > 0.0001) {
                                    if (k == bestIdx[pk]) {
                                        // Winner: a real bond — spring toward this type's rest
                                        // length + torque to lock the patch onto the axis.
                                        let disp = (dist - myBondDist) / cfg.bondRange;
                                        accel  += dir * myBondStr * bond * disp;
                                        torque += cfg.angStiffness * bond * sin(axis - mine.y);
                                    } else {
                                        // Loser competing for an occupied patch: push it away
                                        // so coordination stays at the valence and the
                                        // structure can keep tiling outward.
                                        accel -= dir * myBondStr * bond * 0.6;
                                    }
                                }
                            }
                        }
                    }
                }

                let fric = pow(params.friction, speed);
                p.vel = p.vel * fric + accel * speed;
                p.pos = p.pos + p.vel;

                if (edgeMode == 0u) {
                    if (p.pos.x < 0.0)    { p.pos.x += width;  }
                    if (p.pos.x > width)  { p.pos.x -= width;  }
                    if (p.pos.y < 0.0)    { p.pos.y += height; }
                    if (p.pos.y > height) { p.pos.y -= height; }
                } else {
                    if (p.pos.x < 0.0)    { p.pos.x = 0.0;    p.vel.x =  abs(p.vel.x) * 0.5; }
                    if (p.pos.x > width)  { p.pos.x = width;  p.vel.x = -abs(p.vel.x) * 0.5; }
                    if (p.pos.y < 0.0)    { p.pos.y = 0.0;    p.vel.y =  abs(p.vel.y) * 0.5; }
                    if (p.pos.y > height) { p.pos.y = height; p.vel.y = -abs(p.vel.y) * 0.5; }
                }

                // Integrate orientation; keep θ wrapped to avoid float drift.
                let angFric = pow(cfg.angFriction, speed);
                myOmega = myOmega * angFric + torque * speed;
                var theta = myTheta + myOmega * speed;
                theta = theta - TAU * round(theta / TAU);

                // Transforms (Mode 4 & 5) are evaluated once per TFORM_STRIDE frames,
                // staggered by workgroup so only ~1/stride of particles evaluate each
                // frame. Warp-uniform, so it is a genuine cost reduction; the dice are
                // scaled by the stride to keep the long-run transform rate similar.
                let doTransform = ((u32(params._p3) + (idx >> 8u)) % ${TFORM_STRIDE}u) == 0u;

                // Mode 4: Mode-1-style type transforms driven by accumulated force.
                if (simMode == 4u && doTransform) {
                    let baseSeed = uhash(idx) ^ uhash(u32(abs(p.pos.x) * 157.0 + 1.0))
                                              ^ uhash(u32(abs(p.pos.y) * 239.0 + 1.0));
                    var maxProb: f32 = 0.0;
                    var newType: i32 = -1;
                    for (var t: u32 = 0u; t < numTypes; t++) {
                        let rule = transformRules[myType * 20u + t];
                        if (rule.upperEnabled > 0.5) {
                            let prob = transformProb(typeForce[t], rule.upperThreshold, rule.upperEnabled, 1.0);
                            maxProb = max(maxProb, prob);
                            if (newType < 0 && prob > 0.0 && rand01(baseSeed ^ uhash(t * 3u + 0u)) < min(1.0, prob * ${TFORM_STRIDE}.0)) {
                                newType = i32(rule.upperTarget);
                            }
                        }
                        if (rule.lowerEnabled > 0.5) {
                            let prob = transformProb(typeForce[t], rule.lowerThreshold, rule.lowerEnabled, 0.0);
                            maxProb = max(maxProb, prob);
                            if (newType < 0 && prob > 0.0 && rand01(baseSeed ^ uhash(t * 3u + 1u)) < min(1.0, prob * ${TFORM_STRIDE}.0)) {
                                newType = i32(rule.lowerTarget);
                            }
                        }
                    }
                    if (newType >= 0) { p.typeId = f32(newType); }
                    let prevProb = clamp(p._pad, 0.0, 0.5);
                    p._pad = mix(prevProb, maxProb, 0.06);
                } else if (simMode == 5u && doTransform) {
                    // Mode 5 / Transform #2: force-based conditions combined by a
                    // boolean expression per rule. Conditions compare accumulated
                    // per-type force against a threshold; the first rule whose RPN
                    // expression evaluates true wins its dice roll and transforms.
                    // Each per-type block is ${DNF_TYPE_STRIDE_U32 / 4} vec4u; read a whole vec4 at a time.
                    let baseSeed = uhash(idx) ^ uhash(u32(abs(p.pos.x) * 157.0 + 1.0))
                                              ^ uhash(u32(abs(p.pos.y) * 239.0 + 1.0));
                    let tbase   = myType * ${DNF_TYPE_STRIDE_U32 / 4}u;   // vec4 index of this type's block
                    let hdr     = dnf.d[tbase];
                    let numCond = hdr.x;
                    let numRule = hdr.y;
                    var cond: array<bool, ${MAX_DNF_CONDITIONS}>;
                    for (var ci: u32 = 0u; ci < numCond; ci++) {
                        let cv  = dnf.d[tbase + 1u + ci];     // [trigger, op, threshold, _]
                        let fv  = typeForce[cv.x];
                        let thr = bitcast<f32>(cv.z);
                        var r: bool;
                        if      (cv.y == 0u) { r = fv >  thr; }
                        else if (cv.y == 1u) { r = fv >= thr; }
                        else if (cv.y == 2u) { r = fv <  thr; }
                        else                 { r = fv <= thr; }
                        cond[ci] = r;
                    }
                    var newType: i32 = -1;
                    var fired: f32 = 0.0;
                    let rate = min(1.0, params.maxRate * ${TFORM_STRIDE}.0);
                    for (var ri: u32 = 0u; ri < numRule; ri++) {
                        let rbase = tbase + ${(4 + MAX_DNF_CONDITIONS * 4) / 4}u + ri * ${(4 + DNF_MAX_TOKENS) / 4}u;
                        let rh    = dnf.d[rbase];             // [target, numTokens, _, _]
                        let tgt   = rh.x;
                        let nTok  = rh.y;
                        if (nTok == 0u) { continue; }
                        var st: array<bool, 16>;
                        var sp: u32 = 0u;
                        for (var ti: u32 = 0u; ti < nTok; ti++) {
                            let tv   = dnf.d[rbase + 1u + (ti >> 2u)];  // 4 tokens per vec4
                            let tok  = tv[ti & 3u];
                            let kind = tok >> 16u;
                            let val  = tok & 0xFFFFu;
                            if (kind == 0u) {
                                if (sp < 16u) { st[sp] = cond[min(val, ${MAX_DNF_CONDITIONS - 1}u)]; sp = sp + 1u; }
                            } else if (val == 0u) {            // NOT
                                if (sp >= 1u) { st[sp - 1u] = !st[sp - 1u]; }
                            } else if (sp >= 2u) {             // AND/OR/NAND/NOR
                                let b = st[sp - 1u];
                                let a = st[sp - 2u];
                                sp = sp - 1u;
                                if      (val == 1u) { st[sp - 1u] = a && b; }
                                else if (val == 2u) { st[sp - 1u] = a || b; }
                                else if (val == 3u) { st[sp - 1u] = !(a && b); }
                                else                { st[sp - 1u] = !(a || b); }
                            }
                        }
                        if (sp >= 1u && st[0]) {
                            fired = 1.0;
                            if (newType < 0 && rand01(baseSeed ^ uhash(ri * 7u + 13u)) < rate) {
                                newType = i32(tgt);
                            }
                        }
                    }
                    if (newType >= 0) { p.typeId = f32(newType); }
                    let prevProb = clamp(p._pad, 0.0, 0.5);
                    p._pad = mix(prevProb, fired * params.maxRate, 0.06);
                } else if (simMode != 4u && simMode != 5u) {
                    p._pad = 0.0;
                }

                particles[idx]   = p;
                orientation[idx] = vec2f(theta, myOmega);
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

        // Config buffers are uploaded only when their CPU-side data changed.
        // (poles are written immediately by their setters, so no flag is needed.)
        if (this.forcesDirty) {
            this.queue.writeBuffer(this.forcesBuffer!, 0, this.generateForcesData());
            this.forcesDirty = false;
        }
        if (this.transformDirty) {
            this.queue.writeBuffer(this.transformBuffer!, 0, this.generateTransformData());
            this.transformDirty = false;
        }
        if (this.patchDirty) {
            this.queue.writeBuffer(this.patchConfigBuffer!, 0, this.generatePatchConfigData());
            this.queue.writeBuffer(this.patchTablesBuffer!, 0, this.generatePatchTablesData());
            this.patchDirty = false;
        }
        if (this.dnfDirty) {
            this.queue.writeBuffer(this.dnfRulesBuffer!, 0, this.generateDnfData());
            this.dnfDirty = false;
        }

        if (!this.isPaused) {
            // Advance the transform-stagger phase. Kept under 2^24 so it stays an
            // exact f32, and a multiple of TFORM_STRIDE so the phase never jumps.
            this.frameCounter = (this.frameCounter + 1) % (TFORM_STRIDE * 1_000_000);
            this.queue.writeBuffer(this.paramsBuffer!,    0, this.paramsArray());

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

            // Reorder particles into cell-sorted order for coherent neighbour reads.
            const reorderCp = enc.beginComputePass();
            reorderCp.setPipeline(this.reorderPipeline!);
            reorderCp.setBindGroup(0, this.reorderBindGroup!);
            reorderCp.dispatchWorkgroups(Math.ceil(N / 256));
            reorderCp.end();

            const forceCp = enc.beginComputePass();
            if ((this.simMode === 3 || this.simMode === 4 || this.simMode === 5) && this.mode3Pipeline && this.mode3BindGroup) {
                // Mode 3/4: patchy physics replaces the classic force pass. It reads the
                // same cell-sorted neighbours plus their orientation and writes back
                // position, velocity and orientation in one kernel (Mode 4 also transforms).
                forceCp.setPipeline(this.mode3Pipeline);
                forceCp.setBindGroup(0, this.mode3BindGroup);
            } else {
                forceCp.setPipeline(this.computePipeline!);
                forceCp.setBindGroup(0, this.forceBindGroup!);
            }
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

        // Entity tracking passes. Skipped while paused so a paused frame is truly
        // frozen — otherwise the camera keeps re-centring on the tracked entity
        // (e.g. when composing a photo-export selection) and the scene looks live.
        if (this.isTracking && !this.isPaused && this.trackingBindGroup && this.clearTrackPipeline && this.accumTrackPipeline) {
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

        const doReadback = this.isTracking && !this.trackReadPending && !this.isPaused;
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
        if (p.forceMatrix)                 { this.params.forceMatrix = p.forceMatrix; this.forcesDirty = true; }
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

        // Force/transform matrices changed: defer upload to the next frame.
        this.forcesDirty    = true;
        this.transformDirty = true;
        this.dnfDirty       = true;  // re-clamp DNF targets to the new active range

        if (!this.isInitialized || !this.queue || !this.paramsBuffer) return;
        this.queue.writeBuffer(this.paramsBuffer, 0, this.paramsArray());
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

    setSimMode(mode: 0 | 1 | 2 | 3 | 4 | 5): void { this.simMode = mode; }
    getEdgeMode():    number    { return this.edgeMode; }
    getSimMode():     number    { return this.simMode; }

    // ── Mode 3 (patchy) controls ────────────────────────────────────────────────
    setPatchCount(typeIdx: number, count: number): void {
        if (typeIdx < 0 || typeIdx >= MAX_TYPES) return;
        const c = Math.max(0, Math.min(6, Math.round(count)));
        this.patchCount[typeIdx] = c === 1 ? 2 : c;  // a single patch is degenerate; bump to 2
        this.patchDirty = true;
    }
    getPatchCount(): number[] { return this.patchCount.slice(0, this.numTypes); }

    randomizePatches(): void {
        // Bias toward small chemistry-like valences: mostly 2 (chains) and 3-4
        // (sheets/lattices), with the occasional isotropic 0.
        const choices = [0, 2, 2, 3, 3, 4, 4, 6];
        for (let t = 0; t < this.numTypes; t++) {
            this.patchCount[t] = choices[Math.floor(Math.random() * choices.length)];
        }
        this.patchDirty = true;
    }

    // Randomize all bonding knobs: globals, per-type strength/rest-length, and the
    // affinity matrix (made sparse so types bond selectively → richer assemblies).
    randomizePatchParams(): void {
        const rand = (a: number, b: number) => a + Math.random() * (b - a);
        this.patchBondRange    = rand(40, 100);
        this.patchWidth        = rand(3, 12);
        this.patchAngStiffness = rand(0.15, 0.6);
        this.patchAngFriction  = rand(0.6, 0.95);  // multiplier; lower = more damping
        this.patchIsoScale     = rand(0.6, 1.0);   // keep the asymmetric force matrix strong
        for (let t = 0; t < this.numTypes; t++) {
            this.patchTypeBondStr[t]  = rand(0.1, 0.5);                       // light: bonds nudge
            this.patchTypeBondDist[t] = this.patchBondRange * rand(0.3, 0.55);
        }
        // Sparse affinity: ~35% of ordered type pairs can bond, giving selective
        // (often asymmetric) relationships rather than universal stickiness.
        for (let from = 0; from < this.numTypes; from++)
            for (let to = 0; to < this.numTypes; to++) {
                this.patchAffinity[from * MAX_TYPES + to] = Math.random() < 0.35 ? rand(0.4, 1.0) : 0;
            }
        this.patchDirty = true;
    }

    setPatchParams(p: Partial<{
        bondRange: number; angStiffness: number; angFriction: number;
        patchWidth: number; isoScale: number; coreStrength: number;
    }>): void {
        if (p.bondRange    != null) this.patchBondRange    = Math.max(5,   Math.min(200, p.bondRange));
        if (p.angStiffness != null) this.patchAngStiffness = Math.max(0,   Math.min(2,   p.angStiffness));
        if (p.angFriction  != null) this.patchAngFriction  = Math.max(0,   Math.min(1,   p.angFriction));
        if (p.patchWidth   != null) this.patchWidth        = Math.max(1,   Math.min(20,  p.patchWidth));
        if (p.isoScale     != null) this.patchIsoScale     = Math.max(0,   Math.min(1,   p.isoScale));
        if (p.coreStrength != null) this.patchCoreStrength = Math.max(0,   Math.min(5,   p.coreStrength));
        this.patchDirty = true;
    }
    getPatchParams() {
        return {
            bondRange:   this.patchBondRange,   angStiffness: this.patchAngStiffness,
            angFriction: this.patchAngFriction, patchWidth:   this.patchWidth,
            isoScale:    this.patchIsoScale,    coreStrength: this.patchCoreStrength,
        };
    }

    // Per-type bond strength + rest length.
    setPatchTypeBond(typeIdx: number, strength?: number, dist?: number): void {
        if (typeIdx < 0 || typeIdx >= MAX_TYPES) return;
        if (strength != null) this.patchTypeBondStr[typeIdx]  = Math.max(0, Math.min(2,   strength));
        if (dist     != null) this.patchTypeBondDist[typeIdx] = Math.max(2, Math.min(150, dist));
        this.patchDirty = true;
    }
    getPatchTypeBondStr():  number[] { return this.patchTypeBondStr.slice(0, this.numTypes); }
    getPatchTypeBondDist(): number[] { return this.patchTypeBondDist.slice(0, this.numTypes); }

    // Bond-affinity matrix (who-bonds-whom, asymmetric).
    setAffinity(from: number, to: number, v: number): void {
        if (from < 0 || from >= MAX_TYPES || to < 0 || to >= MAX_TYPES) return;
        this.patchAffinity[from * MAX_TYPES + to] = Math.max(0, Math.min(2, v));
        this.patchDirty = true;
    }
    getAffinity(from: number, to: number): number {
        return this.patchAffinity[from * MAX_TYPES + to] ?? 0;
    }

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

    // ── Photo capture ──────────────────────────────────────────────────────────

    // Render the current frame into an offscreen texture at full canvas resolution
    // and read the pixels back as tightly-packed, top-to-bottom RGBA (alpha forced
    // opaque to match the on-screen 'opaque' canvas). Returns null if not ready.
    // opts.fullWorld: ignore the live camera and render the entire physics-simmed
    // area (the toroidal world rect) at its native resolution — used by full export
    // in loop edge mode so the image is exactly the simulated region, not the
    // current zoom/pan framing.
    async captureRGBA(opts?: { fullWorld?: boolean }): Promise<{ data: Uint8ClampedArray; width: number; height: number } | null> {
        if (!this.isInitialized || !this.device || !this.queue || !this.viewBuffer ||
            !this.renderBindGroup || !this.renderPipeline || !this.renderPipelineAdd || !this.quadVertexBuffer) {
            return null;
        }
        let w: number, h: number, viewCx: number, viewCy: number, viewZoom: number;
        if (opts?.fullWorld) {
            // Render the whole world rect [0,worldW]×[0,worldH] at zoom 1, centred.
            const maxDim = this.device.limits.maxTextureDimension2D;
            let tw = Math.round(this.params.worldWidth);
            let th = Math.round(this.params.worldHeight);
            const over = Math.max(tw, th) / maxDim;
            if (over > 1) { tw = Math.round(tw / over); th = Math.round(th / over); }  // clamp to device limit
            w = Math.max(1, tw); h = Math.max(1, th);
            viewCx = this.params.worldWidth / 2; viewCy = this.params.worldHeight / 2; viewZoom = 1;
        } else {
            w = this.canvas.width; h = this.canvas.height;
            viewCx = this.view.cx; viewCy = this.view.cy; viewZoom = this.view.zoom;
        }
        if (w <= 0 || h <= 0) return null;
        const fmt = navigator.gpu.getPreferredCanvasFormat();

        // Make sure the view uniform reflects current state (also valid while paused).
        this.queue.writeBuffer(this.viewBuffer, 0, new Float32Array([
            viewCx, viewCy, viewZoom,
            this.colorSaturation, this.particleGlow, this.particleAlpha,
            w, h, this.additiveStrength, this.shapeMode, this.simulationTime, 0,
        ]) as Float32Array<ArrayBuffer>);

        const tex = this.device.createTexture({
            size: [w, h], format: fmt,
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
        });
        const bytesPerRow = Math.ceil((w * 4) / 256) * 256;
        const readBuf = this.device.createBuffer({
            size: bytesPerRow * h,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });

        const bg  = this.backgroundColor;
        const enc = this.device.createCommandEncoder();
        const rp  = enc.beginRenderPass({
            colorAttachments: [{
                view: tex.createView(),
                clearValue: { r: bg.r, g: bg.g, b: bg.b, a: 1 },
                loadOp: 'clear', storeOp: 'store',
            }],
        });
        rp.setPipeline(this.blendMode === 1 ? this.renderPipelineAdd! : this.renderPipeline!);
        rp.setBindGroup(0, this.renderBindGroup!);
        rp.setVertexBuffer(0, this.quadVertexBuffer!);
        rp.draw(6, this.params.particleCount);
        rp.end();
        enc.copyTextureToBuffer(
            { texture: tex },
            { buffer: readBuf, bytesPerRow, rowsPerImage: h },
            { width: w, height: h, depthOrArrayLayers: 1 },
        );
        this.queue.submit([enc.finish()]);

        try {
            await readBuf.mapAsync(GPUMapMode.READ);
        } catch (e) {
            readBuf.destroy(); tex.destroy();
            return null;
        }
        const src  = new Uint8Array(readBuf.getMappedRange());
        const out  = new Uint8ClampedArray(w * h * 4);
        const swap = fmt === 'bgra8unorm';
        const tightRow = bytesPerRow === w * 4;
        for (let y = 0; y < h; y++) {
            const rowOff = y * bytesPerRow;
            const dstOff = y * w * 4;
            if (!swap && tightRow) {
                out.set(src.subarray(rowOff, rowOff + w * 4), dstOff);
            } else {
                for (let x = 0; x < w; x++) {
                    const s = rowOff + x * 4;
                    const d = dstOff + x * 4;
                    out[d]     = swap ? src[s + 2] : src[s];
                    out[d + 1] = src[s + 1];
                    out[d + 2] = swap ? src[s]     : src[s + 2];
                    out[d + 3] = src[s + 3];
                }
            }
            for (let x = 0; x < w; x++) out[dstOff + x * 4 + 3] = 255;  // force opaque
        }
        readBuf.unmap();
        readBuf.destroy();
        tex.destroy();
        return { data: out, width: w, height: h };
    }

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
        this.transformDirty = true;
    }
    getTransformRules(): TransformRule[] { return this.transformRules; }

    randomizeForces(): void { this.initializeForceMatrix(); }

    randomizeStrengths(): void {
        for (let from = 0; from < MAX_TYPES; from++)
            for (let to = 0; to < MAX_TYPES; to++) {
                const c = this.params.forceMatrix[from]?.[to];
                if (c) c.strength = (Math.random() * 2 - 1) * 0.7;
            }
        this.forcesDirty = true;
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
        this.forcesDirty = true;
    }

    randomizeMinRadii(): void {
        for (let from = 0; from < MAX_TYPES; from++)
            for (let to = 0; to < MAX_TYPES; to++) {
                const c = this.params.forceMatrix[from]?.[to];
                // Triangular: mode at 15% of max radius, tail up to 80%
                if (c) c.minRadius = Math.min(c.radius - 1, triRand(0, c.radius * 0.15, c.radius * 0.8));
            }
        this.forcesDirty = true;
    }

    zeroMinRadii(): void {
        for (let from = 0; from < MAX_TYPES; from++)
            for (let to = 0; to < MAX_TYPES; to++) {
                const c = this.params.forceMatrix[from]?.[to];
                if (c) c.minRadius = 0;
            }
        this.forcesDirty = true;
    }

    randomizeTransformRules(): void { this.initializeTransformRules(); }

    // ── Mode 5 (DNF / Transform #2) controls ────────────────────────────────────
    // Deep copy of the active-type rules for the UI to read/edit.
    getDnfTypes(): DnfType[] {
        return Array.from({ length: this.numTypes }, (_, s) => {
            const dt = this.dnfTypes[s] ?? { conditions: [], rules: [] };
            return {
                conditions: dt.conditions.map(c => ({ ...c })),
                rules:      dt.rules.map(r => ({ target: r.target, expr: r.expr, rpn: r.rpn.slice() })),
            };
        });
    }

    setDnfType(sourceType: number, conditions: DnfCondition[], rules: DnfRule[]): void {
        if (sourceType < 0 || sourceType >= MAX_TYPES) return;
        const conds = (conditions ?? []).slice(0, MAX_DNF_CONDITIONS).map(c => ({
            trigger:   Math.max(0, Math.min(MAX_TYPES - 1, Math.round(c.trigger))),
            op:        Math.max(0, Math.min(3, Math.round(c.op))) as 0 | 1 | 2 | 3,
            threshold: Number(c.threshold) || 0,
        }));
        const ruleList = (rules ?? []).slice(0, MAX_DNF_RULES).map(r => {
            // Recompile defensively so stored rpn always matches the expression text.
            const { rpn } = compileBoolExpr(r.expr ?? '', conds.length);
            return {
                target: Math.max(0, Math.min(MAX_TYPES - 1, Math.round(r.target))),
                expr:   r.expr ?? '',
                rpn:    (r.rpn && r.rpn.length ? r.rpn : rpn).slice(0, DNF_MAX_TOKENS),
            };
        });
        this.dnfTypes[sourceType] = { conditions: conds, rules: ruleList };
        this.dnfDirty = true;
    }

    clearDnfRules(): void {
        for (let s = 0; s < MAX_TYPES; s++) this.dnfTypes[s] = { conditions: [], rules: [] };
        this.dnfDirty = true;
    }

    getDnfBonding(): boolean { return this.dnfBonding; }
    setDnfBonding(on: boolean): void {
        this.dnfBonding = on;
        // Push immediately so the change applies even while paused.
        if (this.isInitialized && this.queue && this.paramsBuffer) {
            this.queue.writeBuffer(this.paramsBuffer, 0, this.paramsArray());
        }
    }

    // Generate plausible force-based rules. Each type gets several force conditions
    // and a handful of transform rules (mirroring the classic transform's density,
    // so particles rarely dead-end). Rules combine conditions with random boolean
    // expressions and aim at varied target types.
    randomizeDnfRules(): void {
        const n = this.numTypes;
        const randType = (exclude = -1) => {
            let t = Math.floor(Math.random() * n);
            while (t === exclude && n > 1) t = Math.floor(Math.random() * n);
            return t;
        };
        const randThresh = () => Math.round((Math.random() * 1.4 - 0.4) * 100) / 100;  // ~ -0.4 .. 1.0
        const pick = <T,>(a: T[]) => a[Math.floor(Math.random() * a.length)];

        // A random boolean expression over conditions C1..C{nCond}. Only positive
        // and/or logic — negation (not/nor/nand) is true in the common low-force
        // state, which makes rules fire almost every tick and particles never settle.
        const randExpr = (nCond: number): string => {
            const k = 1 + Math.floor(Math.random() * Math.min(3, nCond));  // 1-3 terms
            const pool = Array.from({ length: nCond }, (_, i) => i + 1);
            // shuffle
            for (let i = pool.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [pool[i], pool[j]] = [pool[j], pool[i]]; }
            const term = (i: number) => 'C' + pool[i];
            let expr = term(0);
            for (let i = 1; i < k; i++) expr = `${expr} ${pick(['and', 'or'])} ${term(i)}`;
            // Occasionally parenthesise the head pair for variety: (A op B) op C.
            if (k === 3 && Math.random() < 0.4) {
                expr = `(${term(0)} ${pick(['and', 'or'])} ${term(1)}) ${pick(['and', 'or'])} ${term(2)}`;
            }
            return expr;
        };

        for (let s = 0; s < MAX_TYPES; s++) {
            if (s >= n) { this.dnfTypes[s] = { conditions: [], rules: [] }; continue; }
            const nCond = 2 + Math.floor(Math.random() * 3);  // 2-4 conditions
            const conditions: DnfCondition[] = [];
            for (let c = 0; c < nCond; c++) {
                conditions.push({
                    trigger: randType(),
                    op: (Math.random() < 0.6 ? 0 : 2) as 0 | 2,  // mostly ">", sometimes "<"
                    threshold: randThresh(),
                });
            }
            const nRules = 2 + Math.floor(Math.random() * (MAX_DNF_RULES - 1));  // 2..MAX_DNF_RULES
            const rules: DnfRule[] = [];
            for (let r = 0; r < nRules; r++) {
                const expr = randExpr(nCond);
                const { rpn } = compileBoolExpr(expr, nCond);
                rules.push({ target: randType(s), expr, rpn });
            }
            this.dnfTypes[s] = { conditions, rules };
        }
        this.dnfDirty = true;
    }

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

    getPoleFrame(): boolean { return this.poleWorldFrame; }
    setPoleFrame(world: boolean): void {
        this.poleWorldFrame = world;
        // Push immediately so the change applies even while paused.
        if (this.isInitialized && this.queue && this.paramsBuffer) {
            this.queue.writeBuffer(this.paramsBuffer, 0, this.paramsArray());
        }
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
        const patches = Array.from({ length: n }, (_, t) => this.patchCount[t] ?? 0);
        return {
            version:          3,
            numTypes:         n,
            particleCount:    this.params.particleCount,
            simulationSpeed:  this.params.simulationSpeed,
            worldWidth:       cs.w,
            worldHeight:      cs.h,
            simMode:          this.simMode,
            edgeMode:         this.edgeMode,
            poleWorldFrame:   this.poleWorldFrame,
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
            patchCount:       patches,
            patchParams:      this.getPatchParams(),
            patchBondStr:     this.patchTypeBondStr.slice(0, n),
            patchBondDist:    this.patchTypeBondDist.slice(0, n),
            patchAffinity:    Array.from({ length: n }, (_, from) =>
                                  Array.from({ length: n }, (_, to) => this.patchAffinity[from * MAX_TYPES + to])),
            dnfTypes:         this.getDnfTypes(),
            dnfBonding:       this.dnfBonding,
        };
    }

    importState(state: any): void {
        const n = Math.max(1, Math.min(MAX_TYPES, Number(state.numTypes) || 10));
        this.numTypes = n;

        if (state.simulationSpeed != null) this.params.simulationSpeed = Number(state.simulationSpeed);
        if (state.simMode  != null) this.simMode  = Number(state.simMode)  as 0 | 1 | 2 | 3 | 4 | 5;
        if (state.edgeMode != null) this.edgeMode = Number(state.edgeMode) as 0 | 1;
        if (state.poleWorldFrame != null) this.poleWorldFrame = Boolean(state.poleWorldFrame);

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
        if (Array.isArray(state.patchCount)) {
            for (let t = 0; t < n; t++) {
                const c = state.patchCount[t];
                if (c != null) this.patchCount[t] = Math.max(0, Math.min(6, Math.round(Number(c))));
            }
        }
        if (state.patchParams) this.setPatchParams(state.patchParams);
        if (Array.isArray(state.patchBondStr)) {
            for (let t = 0; t < n; t++) if (state.patchBondStr[t] != null)
                this.patchTypeBondStr[t] = Math.max(0, Math.min(2, Number(state.patchBondStr[t])));
        }
        if (Array.isArray(state.patchBondDist)) {
            for (let t = 0; t < n; t++) if (state.patchBondDist[t] != null)
                this.patchTypeBondDist[t] = Math.max(2, Math.min(150, Number(state.patchBondDist[t])));
        }
        if (Array.isArray(state.patchAffinity)) {
            for (let from = 0; from < n; from++)
                for (let to = 0; to < n; to++) {
                    const v = state.patchAffinity[from]?.[to];
                    if (v != null) this.patchAffinity[from * MAX_TYPES + to] = Math.max(0, Math.min(2, Number(v)));
                }
        }
        this.patchDirty = true;
        if (Array.isArray(state.dnfTypes)) {
            this.clearDnfRules();
            for (let s = 0; s < n; s++) {
                const dt = state.dnfTypes[s];
                if (dt && Array.isArray(dt.conditions) && Array.isArray(dt.rules)) {
                    this.setDnfType(s, dt.conditions, dt.rules);
                }
            }
        }
        this.dnfDirty = true;
        if (state.dnfBonding != null) this.dnfBonding = Boolean(state.dnfBonding);

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
