import { ParticleSimulation, TransformRule, DnfCondition, DnfRule, MAX_TYPES, MAX_DNF_CONDITIONS, MAX_DNF_RULES, compileBoolExpr, TYPE_COLORS_HEX, DiagnosticData } from './simulation';
// Vendored (MIT) so the repo is self-contained — no npm install needed. See
// src/vendor/webm-muxer.LICENSE.txt.
import { Muxer, ArrayBufferTarget } from './vendor/webm-muxer';

// Short 3-letter type names, decoupled from colours (we have up to 50 types).
const TYPE_LABELS = [
    'Axo', 'Bex', 'Cyl', 'Dax', 'Eon', 'Fyn', 'Gad', 'Hex', 'Ivo', 'Jax',
    'Kor', 'Lum', 'Mox', 'Nyx', 'Orb', 'Pyx', 'Qua', 'Rho', 'Syl', 'Tau',
    'Uxo', 'Vex', 'Wyn', 'Xan', 'Yvo', 'Zed', 'Arc', 'Bly', 'Cri', 'Dro',
    'Elu', 'Fro', 'Gly', 'Hru', 'Ixi', 'Jek', 'Kip', 'Lof', 'Mun', 'Nim',
    'Oss', 'Pel', 'Qib', 'Rax', 'Sol', 'Tiv', 'Umo', 'Vit', 'Wox', 'Zor',
];
const TYPE_HEX = TYPE_COLORS_HEX;
// DNF condition operators, indexed by op code (0:> 1:>= 2:< 3:<=).
const DNF_OPSYM = ['>', '≥', '<', '≤'];

// ── Color helpers ─────────────────────────────────────────────────────────────

function strengthToColor(v: number): string {
    // Perceptual ramp so the full ±5 range stays distinguishable (sqrt keeps small
    // values visible; magnitudes past ±5 saturate).
    const mag = Math.sqrt(Math.min(Math.abs(v), 5) / 5);
    const t = Math.round(40 + mag * 190);
    return v < 0 ? `rgb(${t},0,0)` : `rgb(0,${t},0)`;
}

function radiusToColor(v: number): string {
    const t = Math.max(0, Math.min(1, (v - 10) / 190));
    return `rgb(8,${Math.round(8 + t * 72)},${Math.round(40 + t * 210)})`;
}

function minRadiusToColor(v: number): string {
    const t = Math.max(0, Math.min(1, v / 200));
    return `rgb(${Math.round(30 + t * 160)},${Math.round(16 + t * 56)},8)`;
}

function transformCellHTML(rule: TransformRule): { bg: string; html: string } {
    const u = rule.upperEnabled;
    const l = rule.lowerEnabled;
    if (!u && !l) return { bg: '#111', html: '<span style="color:#333">·</span>' };
    const pip = (t: number, arrow: string) =>
        `${arrow}<span style="color:${TYPE_HEX[t] ?? '#fff'};font-weight:bold">${TYPE_LABELS[t] ?? '?'}</span>`;
    if (u && l) return { bg: '#0a1520', html: pip(rule.upperTarget, '↑') + pip(rule.lowerTarget, '↓') };
    if (u) return { bg: '#0a2010', html: pip(rule.upperTarget, '↑') };
    return         { bg: '#200a0a', html: pip(rule.lowerTarget, '↓') };
}

// Hex "#rrggbb" → {r,g,b} floats [0,1]
function hexToRgb(hex: string): { r: number; g: number; b: number } {
    const n = parseInt(hex.replace('#', ''), 16);
    return { r: ((n >> 16) & 255) / 255, g: ((n >> 8) & 255) / 255, b: (n & 255) / 255 };
}

// {r,g,b} floats [0,1] → hex "#rrggbb"
function rgbToHex(r: number, g: number, b: number): string {
    const h = (v: number) => Math.round(v * 255).toString(16).padStart(2, '0');
    return `#${h(r)}${h(g)}${h(b)}`;
}

// ── App ───────────────────────────────────────────────────────────────────────

class ParticleLifeApp {
    private canvas: HTMLCanvasElement;
    private sim: ParticleSimulation | null = null;
    private animId: number | null = null;

    private editorCell:    HTMLTableCellElement | null = null;
    private editorDismiss: ((e: MouseEvent) => void) | null = null;
    private teSource  = -1;
    private teTrigger = -1;
    private teDismiss: ((e: MouseEvent) => void) | null = null;

    private frameCount      = 0;
    private lastTime        = performance.now();
    private autoPause       = true;
    private autoPauseMinFps = 45;
    private lowFpsFrames    = 0;

    // ── Panel + overlay ───────────────────────────────────────────────────────
    private panelCollapsed  = false;
    private overlayCanvas:  HTMLCanvasElement | null = null;
    private overlayCtx:     CanvasRenderingContext2D | null = null;

    // ── Entity tracking ───────────────────────────────────────────────────────
    private trackMode: 'idle' | 'selecting' | 'tracking' = 'idle';
    private selBoxActive = false;
    private selBox = { sx0: 0, sy0: 0, sx1: 0, sy1: 0 };

    // ── Particle inspector ────────────────────────────────────────────────────
    private diagPanelEl:  HTMLElement | null = null;
    private inspectMode = false;

    // ── Cursor tools ──────────────────────────────────────────────────────────
    private activeCursorTool:     'none' | 'grab' | 'force' | 'paint' | 'erase' = 'none';
    private brushWorldRadius      = 150;
    private forceStrength         = 2.0;
    private paintTypeId           = 0;
    private paintRate             = 10;   // particles/frame for paint; /1000 = kill prob for erase
    private cursorMouseX          = -9999;
    private cursorMouseY          = -9999;
    private cursorMouseButtons    = 0;   // bitmask: bit 0 = left, bit 2 = right

    // ── Master randomize ──────────────────────────────────────────────────────
    // Which categories the master Randomize button affects. `modes: null` = always
    // shown (forces); otherwise only when the current sim mode is in the list.
    private readonly randomizeCats: {
        key: string; label: string; modes: number[] | null;
        run: () => void; refresh: 'forces' | 'transform' | 'poles' | 'masses' | 'valences' | 'bonding' | 'dnf' | 'charges';
    }[] = [
        { key: 'strength',  label: 'Force strength', modes: null,      run: () => this.sim!.randomizeStrengths(),     refresh: 'forces' },
        { key: 'maxRadius', label: 'Max radius',     modes: null,      run: () => this.sim!.randomizeMaxRadii(),      refresh: 'forces' },
        { key: 'minRadius', label: 'Min radius',     modes: null,      run: () => this.sim!.randomizeMinRadii(),      refresh: 'forces' },
        { key: 'transform', label: 'Transform rules', modes: [1, 2, 4], run: () => this.sim!.randomizeTransformRules(), refresh: 'transform' },
        { key: 'poles',     label: 'Poles',          modes: [1, 2],    run: () => this.sim!.randomizePoles(),         refresh: 'poles' },
        { key: 'masses',    label: 'Masses',         modes: [2],       run: () => this.sim!.randomizeMasses(),        refresh: 'masses' },
        { key: 'valences',  label: 'Valences',       modes: [3, 4, 5], run: () => this.sim!.randomizePatches(),       refresh: 'valences' },
        { key: 'bonding',   label: 'Bonding',        modes: [3, 4, 5], run: () => this.sim!.randomizePatchParams(),   refresh: 'bonding' },
        { key: 'dnf',       label: 'DNF rules',      modes: [5],       run: () => this.sim!.randomizeDnfRules(),      refresh: 'dnf' },
        { key: 'charges',   label: 'Charges',        modes: [6],       run: () => this.sim!.randomizeCharges(),       refresh: 'charges' },
    ];
    // Default selection — minRadius and poles off, as those are rarely wanted.
    private randomizeSel: Record<string, boolean> = {
        strength: true, maxRadius: true, minRadius: false,
        transform: true, poles: false, masses: true, valences: true, bonding: true, dnf: true,
    };

    // ── DNF editor (mode 5) ───────────────────────────────────────────────────
    private dnfEditType: number | null = null;   // which source type's editor is open

    // ── Photo export selection ────────────────────────────────────────────────
    private photoSelMode     = false;   // armed: dragging a region to export
    private photoSelDragging = false;
    private photoSelDidPause  = false;  // true if entering select mode paused the sim (so release can resume)
    private photoSelBox: { sx0: number; sy0: number; sx1: number; sy1: number } | null = null;
    private photoSelTarget: 'png' | 'video' = 'png';   // what a completed selection does

    // ── Video recording ───────────────────────────────────────────────────────
    private videoDurationSec = 10;
    private isRecording      = false;
    private recorder: MediaRecorder | null = null;
    private recordCopy: (() => void) | null = null;   // per-frame blit into the record canvas
    private recordStopTimer = 0;
    private recordCountdownTimer = 0;
    // Offline (deterministic) render: steps the sim frame-by-frame and encodes a
    // fixed-60fps video regardless of how long each frame takes to compute.
    private isRenderingVideo = false;
    private renderCancel     = false;

    constructor() {
        this.canvas = document.getElementById('sim-canvas') as HTMLCanvasElement;
        this.overlayCanvas = document.getElementById('overlay') as HTMLCanvasElement;
        this.overlayCtx = this.overlayCanvas.getContext('2d');
        this.diagPanelEl = document.getElementById('diag-panel');
        this.resizeOverlay();
        this.setupUI();
        this.setupCanvasEvents();
        this.setupCursorPanel();
        window.addEventListener('resize', () => { this.fitCanvas(); this.resizeOverlay(); });
        window.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') { this.stopInspect(); }
        });
        document.getElementById('diag-close')!.addEventListener('click', () => {
            this.stopInspect();
        });
    }

    private resizeOverlay(): void {
        if (!this.overlayCanvas) return;
        this.overlayCanvas.width  = window.innerWidth;
        this.overlayCanvas.height = window.innerHeight;
    }

    // ── Canvas sizing ─────────────────────────────────────────────────────────

    // The canvas (viewport) fills the window; the simulation world is a separate
    // size projected into it (centred, with background margins) by the renderer —
    // so there are no letterbox bars and the world needn't be screen-shaped.
    private fitCanvas(): void {
        const w = Math.max(1, window.innerWidth);
        const h = Math.max(1, window.innerHeight);
        this.canvas.width  = w;
        this.canvas.height = h;
        this.canvas.style.width  = `${w}px`;
        this.canvas.style.height = `${h}px`;
    }

    private setCanvasSize(w: number, h: number): void {
        this.canvas.width  = w;
        this.canvas.height = h;
        this.fitCanvas();
    }

    // ── UI setup ─────────────────────────────────────────────────────────────

    private setupUI(): void {
        // Tabs
        document.querySelectorAll<HTMLButtonElement>('.tab-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
                document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
                btn.classList.add('active');
                document.getElementById(`pane-${btn.dataset.tab}`)?.classList.add('active');
                this.closeForceEditor();
                this.closeTransformEditor();
            });
        });

        // Speed
        const speedEl = document.getElementById('speed') as HTMLInputElement;
        speedEl.addEventListener('input', () => {
            const v = parseFloat(speedEl.value);
            document.getElementById('speedValue')!.textContent = `${v.toFixed(1)}×`;
            this.sim?.updateParams({ simulationSpeed: v });
        });

        // Particle count
        const countEl = document.getElementById('particleCount') as HTMLInputElement;
        const applyCount = () => {
            const n = parseInt(countEl.value);
            if (n > 0 && this.sim) {
                this.sim.setParticleCount(n);
                document.getElementById('particleCountDisplay')!.textContent = String(n);
            }
        };
        countEl.addEventListener('change', applyCount);
        countEl.addEventListener('keydown', (e) => { if (e.key === 'Enter') applyCount(); });

        // Type count
        const typeEl = document.getElementById('typeCount') as HTMLInputElement;
        const applyTypes = () => {
            const n = Math.max(1, Math.min(MAX_TYPES, parseInt(typeEl.value) || 10));
            typeEl.value = String(n);
            if (this.sim) {
                this.closeForceEditor();
                this.closeTransformEditor();
                this.sim.setNumTypes(n);
                this.refreshForceMatrices();
                this.refreshTransformMatrix();
                this.refreshMassTable();
                this.refreshDnfPanel();
                this.refreshPaintTypePicker();
            }
        };
        typeEl.addEventListener('change', applyTypes);
        typeEl.addEventListener('keydown', (e) => { if (e.key === 'Enter') applyTypes(); });

        // Buttons
        document.getElementById('resetBtn')!.addEventListener('click', () => this.sim?.reset());
        document.getElementById('pauseBtn')!.addEventListener('click', () => {
            if (!this.sim) return;
            this.sim.togglePause();
            this.syncPauseButton();
        });

        // Auto-pause checkbox + min-FPS input
        const apChk = document.getElementById('autoPauseChk') as HTMLInputElement;
        apChk.addEventListener('change', () => { this.autoPause = apChk.checked; });
        const apFpsEl = document.getElementById('autoPauseMinFps') as HTMLInputElement;
        apFpsEl.value = String(this.autoPauseMinFps);
        apFpsEl.addEventListener('change', () => {
            this.autoPauseMinFps = Math.max(1, Math.min(120, Math.round(Number(apFpsEl.value))));
            apFpsEl.value = String(this.autoPauseMinFps);
        });

        // Sim mode
        document.getElementById('mode0-btn')!.addEventListener('click', () => this.setSimMode(0));
        document.getElementById('mode1-btn')!.addEventListener('click', () => this.setSimMode(1));
        document.getElementById('mode2-btn')!.addEventListener('click', () => this.setSimMode(2));
        document.getElementById('mode3-btn')!.addEventListener('click', () => this.setSimMode(3));
        document.getElementById('mode4-btn')!.addEventListener('click', () => this.setSimMode(4));
        document.getElementById('mode5-btn')!.addEventListener('click', () => this.setSimMode(5));
        document.getElementById('mode6-btn')!.addEventListener('click', () => this.setSimMode(6));

        // Per-matrix randomize buttons
        document.getElementById('randomizeStrengthBtn')!.addEventListener('click', () => {
            this.sim?.randomizeStrengths();
            this.refreshForceMatrices();
        });
        document.getElementById('randomizeMaxRadiusBtn')!.addEventListener('click', () => {
            this.sim?.randomizeMaxRadii();
            this.refreshForceMatrices();
        });
        document.getElementById('randomizeMinRadiusBtn')!.addEventListener('click', () => {
            this.sim?.randomizeMinRadii();
            this.refreshForceMatrices();
        });
        document.getElementById('zeroMinRadiusBtn')!.addEventListener('click', () => {
            this.sim?.zeroMinRadii();
            this.refreshForceMatrices();
        });

        // Randomize masses (mode 2)
        document.getElementById('randomizeMassBtn')!.addEventListener('click', () => {
            this.sim?.randomizeMasses();
            this.refreshMassTable();
        });

        // Patchy mode (mode 3) controls
        document.getElementById('randomizePatchesBtn')!.addEventListener('click', () => {
            this.sim?.randomizePatches();
            this.refreshPatchTable();
        });
        document.getElementById('randomizeBondingBtn')!.addEventListener('click', () => {
            this.sim?.randomizePatchParams();
            this.refreshPatchUI();
        });

        // Master randomize: action button + caret that opens the checklist.
        document.getElementById('randomizeNowBtn')!.addEventListener('click', () => this.runMasterRandomize());
        document.getElementById('randomizeMenuBtn')!.addEventListener('click', (e) => {
            e.stopPropagation();
            this.toggleRandomizeMenu();
        });
        // Click anywhere else closes the menu.
        document.addEventListener('click', (e) => {
            const pop = document.getElementById('randomize-popover');
            const caret = document.getElementById('randomizeMenuBtn');
            if (!pop || !pop.classList.contains('visible')) return;
            if (pop.contains(e.target as Node) || caret?.contains(e.target as Node)) return;
            this.toggleRandomizeMenu(false);
        });
        const patchSlider = (id: string, valId: string, key:
            'bondRange' | 'angStiffness' | 'angFriction' | 'patchWidth' | 'isoScale',
            fmt: (v: number) => string) => {
            const el  = document.getElementById(id) as HTMLInputElement;
            const out = document.getElementById(valId)!;
            el.addEventListener('input', () => {
                const v = Number(el.value);
                out.textContent = fmt(v);
                this.sim?.setPatchParams({ [key]: v });
            });
        };
        patchSlider('patchRangeSlider',    'patchRangeValue',    'bondRange',    v => String(Math.round(v)));
        patchSlider('patchWidthSlider',    'patchWidthValue',    'patchWidth',   v => String(v));
        patchSlider('patchIsoSlider',      'patchIsoValue',      'isoScale',     v => v.toFixed(2));
        patchSlider('patchAngSlider',      'patchAngValue',      'angStiffness', v => v.toFixed(2));
        // Spin damping reads 0 = none, 1 = full; internally angFriction is the
        // per-tick angular-velocity multiplier, i.e. the inverse of the slider.
        const spinDampEl  = document.getElementById('patchAngFricSlider') as HTMLInputElement;
        const spinDampOut = document.getElementById('patchAngFricValue')!;
        spinDampEl.addEventListener('input', () => {
            const v = Number(spinDampEl.value);
            spinDampOut.textContent = v.toFixed(2);
            this.sim?.setPatchParams({ angFriction: 1 - v });
        });

        // Randomize transform rules
        document.getElementById('randomizeTransformBtn')!.addEventListener('click', () => {
            this.sim?.randomizeTransformRules();
            this.refreshTransformMatrix();
        });

        // DNF (mode 5 / Transform #2) controls
        document.getElementById('randomizeDnfBtn')!.addEventListener('click', () => {
            this.sim?.randomizeDnfRules();
            this.refreshDnfPanel();
        });
        document.getElementById('clearDnfBtn')!.addEventListener('click', () => {
            this.sim?.clearDnfRules();
            this.refreshDnfPanel();
        });
        document.getElementById('dnfBondOffBtn')!.addEventListener('click', () => this.setDnfBonding(false));
        document.getElementById('dnfBondOnBtn')!.addEventListener('click', () => this.setDnfBonding(true));

        // QFT (mode 6) controls
        document.getElementById('randomizeChargesBtn')!.addEventListener('click', () => {
            this.sim?.randomizeCharges();
            this.refreshQftPanel();
        });
        const qftFieldCount = document.getElementById('qftFieldCount') as HTMLInputElement;
        qftFieldCount.addEventListener('change', () => {
            const v = Math.max(1, Math.min(12, Math.round(Number(qftFieldCount.value) || 3)));
            qftFieldCount.value = String(v);
            this.sim?.setNumFields(v);
            this.refreshQftPanel();
        });
        const dnfMaxSlider = document.getElementById('dnfMaxSlider') as HTMLInputElement;
        dnfMaxSlider.addEventListener('input', () => {
            const v = parseFloat(dnfMaxSlider.value);
            document.getElementById('dnfMaxValue')!.textContent = v.toFixed(2);
            this.sim?.setMaxTransformRate(v);
            // Keep the Transform-mode slider in sync (same global rate).
            const ot = document.getElementById('maxTransformSlider') as HTMLInputElement | null;
            if (ot) { ot.value = String(v); document.getElementById('maxTransformValue')!.textContent = v.toFixed(2); }
        });

        // Randomize poles
        document.getElementById('randomizePolesBtn')!.addEventListener('click', () => {
            this.sim?.randomizePoles();
            this.buildPolePanel();
        });

        // Pole reference frame (velocity vs world) for 3+ poles
        document.getElementById('poleFrameVelBtn')!.addEventListener('click', () => {
            this.sim?.setPoleFrame(false);
            this.syncPoleFrameButtons();
        });
        document.getElementById('poleFrameWorldBtn')!.addEventListener('click', () => {
            this.sim?.setPoleFrame(true);
            this.syncPoleFrameButtons();
        });

        // World size (physics space only — canvas pixel resolution stays fixed)
        document.getElementById('applyWorldSize')!.addEventListener('click', () => {
            const w = parseInt((document.getElementById('worldW') as HTMLInputElement).value) || 1600;
            const h = parseInt((document.getElementById('worldH') as HTMLInputElement).value) || 900;
            this.sim?.setWorldSize(w, h);
            this.updateZoomDisplay();
        });

        // Edge mode
        document.getElementById('edgeLoopBtn')!.addEventListener('click', () => this.setEdgeMode(0));
        document.getElementById('edgeOpenBtn')!.addEventListener('click', () => this.setEdgeMode(1));

        // Blend mode
        document.getElementById('blendStandardBtn')!.addEventListener('click', () => this.setBlendMode(0));
        document.getElementById('blendAdditiveBtn')!.addEventListener('click', () => this.setBlendMode(1));

        // Shape mode
        document.getElementById('shapeCircleBtn')!.addEventListener('click', () => this.setShapeMode(0));
        document.getElementById('shapePolyBtn')!.addEventListener('click', () => this.setShapeMode(1));

        // Background color
        const bgPicker = document.getElementById('bgColorPicker') as HTMLInputElement;
        bgPicker.addEventListener('input', () => {
            const { r, g, b } = hexToRgb(bgPicker.value);
            this.sim?.setBackgroundColor(r, g, b);
        });

        // Saturation
        const satSlider = document.getElementById('satSlider') as HTMLInputElement;
        satSlider.addEventListener('input', () => {
            const v = parseFloat(satSlider.value);
            document.getElementById('satValue')!.textContent = v.toFixed(2);
            this.sim?.setColorSaturation(v);
        });

        // Particle glow
        const glowSlider = document.getElementById('glowSlider') as HTMLInputElement;
        glowSlider.addEventListener('input', () => {
            const v = parseFloat(glowSlider.value);
            document.getElementById('glowValue')!.textContent = v.toFixed(2);
            this.sim?.setParticleGlow(v);
        });

        // Particle opacity
        const alphaSlider = document.getElementById('alphaSlider') as HTMLInputElement;
        alphaSlider.addEventListener('input', () => {
            const v = parseFloat(alphaSlider.value);
            document.getElementById('alphaValue')!.textContent = v.toFixed(2);
            this.sim?.setParticleAlpha(v);
        });

        // Additive blend strength
        const addStrSlider = document.getElementById('addStrSlider') as HTMLInputElement;
        addStrSlider.addEventListener('input', () => {
            const v = parseFloat(addStrSlider.value);
            document.getElementById('addStrValue')!.textContent = v.toFixed(2);
            this.sim?.setAdditiveStrength(v);
        });

        // Friction. The slider reads as drag intuition (0 = frictionless,
        // 1 = stops fast); internally friction is a per-tick velocity multiplier,
        // so the stored value is the inverse of the slider.
        const frictionSlider = document.getElementById('frictionSlider') as HTMLInputElement;
        frictionSlider.addEventListener('input', () => {
            const v = parseFloat(frictionSlider.value);
            document.getElementById('frictionValue')!.textContent = v.toFixed(2);
            this.sim?.setFriction(1 - v);
        });

        // Max transform rate
        const maxTransformSlider = document.getElementById('maxTransformSlider') as HTMLInputElement;
        maxTransformSlider.addEventListener('input', () => {
            const v = parseFloat(maxTransformSlider.value);
            document.getElementById('maxTransformValue')!.textContent = v.toFixed(2);
            this.sim?.setMaxTransformRate(v);
        });

        // Photo export (PNG)
        document.getElementById('exportFullBtn')!.addEventListener('click', () => {
            if (this.sim?.getEdgeMode() === 1) return;  // disabled in open mode
            this.exportFullCanvas();
        });
        document.getElementById('exportSelBtn')!.addEventListener('click', () => this.togglePhotoSelect('png'));

        // Video export (WebM). While busy, the buttons cancel the current job.
        document.getElementById('recordFullBtn')!.addEventListener('click', () => {
            if (this.isRenderingVideo) { this.renderCancel = true; }
            else if (this.isRecording) { this.stopRecording(); }
            else this.recordFullVideo();
        });
        document.getElementById('recordSelBtn')!.addEventListener('click', () => {
            if (this.isRenderingVideo) { this.renderCancel = true; return; }
            if (this.isRecording) return;
            this.togglePhotoSelect('video');
        });
        const durEl = document.getElementById('videoDurSlider') as HTMLInputElement;
        durEl.addEventListener('input', () => {
            this.videoDurationSec = Math.max(5, Math.min(30, Math.round(Number(durEl.value) / 5) * 5));
            document.getElementById('videoDurValue')!.textContent = `${this.videoDurationSec}s`;
        });

        // Export / Import
        document.getElementById('exportBtn')!.addEventListener('click', () => this.exportConfig());
        document.getElementById('importBtn')!.addEventListener('click', () => {
            (document.getElementById('importFile') as HTMLInputElement).click();
        });
        document.getElementById('importFile')!.addEventListener('change', (e) => {
            const file = (e.target as HTMLInputElement).files?.[0];
            if (file) this.importConfig(file);
            (e.target as HTMLInputElement).value = '';  // allow re-selecting same file
        });

        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.closeForceEditor(); this.closeTransformEditor();
                if (this.trackMode === 'selecting') { this.trackMode = 'idle'; this.syncTrackButton(); }
                if (this.photoSelMode) this.exitPhotoSelect();
                if (this.isRenderingVideo) this.renderCancel = true;   // cancel & discard offline render
                else if (this.isRecording) this.stopRecording();       // end real-time capture (saves what's captured)
            }
            if (e.key === ' ' && !(e.target instanceof HTMLInputElement) && !(e.target instanceof HTMLTextAreaElement)) {
                e.preventDefault();
                if (!this.sim) return;
                this.sim.togglePause();
                this.syncPauseButton();
            }
        });

        // Panel collapse
        document.getElementById('collapseBtn')!.addEventListener('click', () => {
            this.panelCollapsed = !this.panelCollapsed;
            document.getElementById('ui')!.classList.toggle('collapsed', this.panelCollapsed);
            const btn = document.getElementById('collapseBtn')!;
            btn.style.left = this.panelCollapsed ? '0' : '340px';
            btn.innerHTML  = this.panelCollapsed ? '&#9654;' : '&#9664;';
        });

        // Bottom dock
        document.getElementById('btmTrackBtn')!.addEventListener('click', () => this.handleTrackButton());
        document.getElementById('btmInspectBtn')!.addEventListener('click', () => {
            if (this.photoSelMode) this.exitPhotoSelect();
            this.inspectMode = !this.inspectMode;
            if (!this.inspectMode) this.stopInspect();
            this.syncInspectButton();
        });

        // Tracking-stop callback (fires when entity dies)
        // wired once sim is created — done in the sim init callback
    }

    private syncPauseButton(): void {
        const paused = this.sim?.isPaused_() ?? false;
        const btn = document.getElementById('pauseBtn')!;
        btn.innerHTML = paused ? '&#9654; Resume' : '&#9646;&#9646; Pause';
        btn.classList.toggle('active', paused);
    }

    private setSimMode(mode: 0 | 1 | 2 | 3 | 4 | 5 | 6): void {
        this.sim?.setSimMode(mode);
        for (let m = 0; m <= 6; m++) {
            document.getElementById(`mode${m}-btn`)!.classList.toggle('selected', mode === m);
        }
        // Transform rules apply in modes 1 (Transform), 2 (Mass) and 4 (Patchy+T).
        const hasTransform = mode === 1 || mode === 2 || mode === 4;
        // DNF rules apply only in mode 5.
        const hasDnf = mode === 5;
        // QFT charge fields apply only in mode 6.
        const hasQft = mode === 6;
        document.getElementById('qft-panel')!.classList.toggle('visible', hasQft);
        if (hasQft) this.refreshQftPanel();
        const dnfBond = this.sim?.getDnfBonding() ?? false;
        // Patch controls: always in modes 3/4; in mode 5 only when bonding is on.
        const showPatch = mode === 3 || mode === 4 || (mode === 5 && dnfBond);
        document.getElementById('transform-panel')!.classList.toggle('visible', hasTransform);
        document.getElementById('mode2-panel')!.classList.toggle('visible', mode === 2);
        document.getElementById('mode3-panel')!.classList.toggle('visible', showPatch);
        document.getElementById('dnf-panel')!.classList.toggle('visible', hasDnf);
        if (mode === 2) this.refreshMassTable();
        if (mode === 3 || mode === 4) {
            // First time in a patchy mode with no valences set: seed some so the
            // directional behaviour is visible immediately.
            if (this.sim && this.sim.getPatchCount().every(v => v === 0)) {
                this.sim.randomizePatches();
            }
            this.refreshPatchUI();
        }
        if (hasDnf) {
            // First entry with no rules: seed some so transforms are visible at once.
            if (this.sim && this.sim.getDnfTypes().every(dt => dt.conditions.length === 0 && dt.rules.length === 0)) {
                this.sim.randomizeDnfRules();
            }
            this.syncDnfBondingButtons();
            if (showPatch) this.refreshPatchUI();
            this.refreshDnfPanel();
        }
    }

    // Reflect the Mode-5 directional-bonding toggle and show/hide the patch panel.
    private syncDnfBondingButtons(): void {
        const on = this.sim?.getDnfBonding() ?? false;
        document.getElementById('dnfBondOnBtn')!.classList.toggle('selected', on);
        document.getElementById('dnfBondOffBtn')!.classList.toggle('selected', !on);
    }

    private setDnfBonding(on: boolean): void {
        if (!this.sim) return;
        this.sim.setDnfBonding(on);
        // Turning bonding on with no valences set: seed some so bonds are visible.
        if (on && this.sim.getPatchCount().every(v => v === 0)) this.sim.randomizePatches();
        document.getElementById('mode3-panel')!.classList.toggle('visible', on);
        if (on) this.refreshPatchUI();
        this.syncDnfBondingButtons();
    }

    private setEdgeMode(mode: 0 | 1): void {
        if (!this.sim) return;
        this.sim.setEdgeMode(mode);
        document.getElementById('edgeLoopBtn')!.classList.toggle('selected', mode === 0);
        document.getElementById('edgeOpenBtn')!.classList.toggle('selected', mode === 1);
        this.updateZoomDisplay();
        this.syncExportButtons();
    }

    private setBlendMode(mode: 0 | 1): void {
        this.sim?.setBlendMode(mode);
        document.getElementById('blendStandardBtn')!.classList.toggle('selected', mode === 0);
        document.getElementById('blendAdditiveBtn')!.classList.toggle('selected', mode === 1);
    }

    private setShapeMode(mode: 0 | 1): void {
        this.sim?.setShapeMode(mode);
        document.getElementById('shapeCircleBtn')!.classList.toggle('selected', mode === 0);
        document.getElementById('shapePolyBtn')!.classList.toggle('selected', mode === 1);
    }

    // ── Entity tracking helpers ───────────────────────────────────────────────

    // Half-extents (world units) visible from the view centre to each viewport edge.
    // Mirrors the isotropic "contain" projection in the render vertex shader.
    private viewSpans(): { spanX: number; spanY: number } {
        const p = this.sim!.getParams();
        const v = this.sim!.getView();
        const aspect = this.canvas.width / this.canvas.height;
        const worldAspect = p.worldWidth / p.worldHeight;
        if (aspect > worldAspect) {
            const spanY = (p.worldHeight * 0.5) / v.zoom;
            return { spanX: spanY * aspect, spanY };
        }
        const spanX = (p.worldWidth * 0.5) / v.zoom;
        return { spanX, spanY: spanX / aspect };
    }

    private screenToWorld(sx: number, sy: number): { wx: number; wy: number } {
        if (!this.sim) return { wx: 0, wy: 0 };
        const rect = this.canvas.getBoundingClientRect();
        const view = this.sim.getView();
        const { spanX, spanY } = this.viewSpans();
        const fx = (sx - rect.left - rect.width  / 2) / (rect.width  / 2);  // -1..1
        const fy = (sy - rect.top  - rect.height / 2) / (rect.height / 2);  // -1..1 (down +)
        return { wx: view.cx + fx * spanX, wy: view.cy + fy * spanY };
    }

    // Screen (client) pixels per world unit — isotropic under the contain camera.
    private pxPerWorld(): number {
        const rect = this.canvas.getBoundingClientRect();
        return (rect.width / 2) / this.viewSpans().spanX;
    }

    private worldToScreen(wx: number, wy: number): { sx: number; sy: number } {
        const rect = this.canvas.getBoundingClientRect();
        const v = this.sim!.getView();
        const s = this.pxPerWorld();
        return {
            sx: rect.left + rect.width  / 2 + (wx - v.cx) * s,
            sy: rect.top  + rect.height / 2 + (wy - v.cy) * s,
        };
    }

    private handleTrackButton(): void {
        if (this.photoSelMode) this.exitPhotoSelect();
        if (this.trackMode === 'tracking') {
            this.sim?.stopTracking();
            this.trackMode = 'idle';
        } else if (this.trackMode === 'selecting') {
            this.trackMode = 'idle';
            document.body.classList.remove('track-selecting');
        } else {
            this.trackMode = 'selecting';
            document.body.classList.add('track-selecting');
        }
        this.syncTrackButton();
    }

    private syncTrackButton(): void {
        // If the entity died externally, snap back to idle
        if (this.trackMode === 'tracking' && !this.sim?.isEntityTracking()) {
            this.trackMode = 'idle';
        }
        const btn = document.getElementById('btmTrackBtn')!;
        if (this.trackMode === 'tracking') {
            btn.innerHTML = '&#9633; Stop Track';
            btn.classList.add('danger');
            btn.classList.remove('active');
        } else if (this.trackMode === 'selecting') {
            btn.innerHTML = '&#10005; Cancel';
            btn.classList.add('active');
            btn.classList.remove('danger');
        } else {
            btn.innerHTML = '&#8982; Track Entity';
            btn.classList.remove('active', 'danger');
        }
        this.syncCanvasCursor();
    }

    private drawOverlay(): void {
        if (!this.overlayCtx || !this.overlayCanvas) return;
        const ctx = this.overlayCtx;
        ctx.clearRect(0, 0, this.overlayCanvas.width, this.overlayCanvas.height);

        // Brush circle for cursor tools (suppressed during inspect/track override)
        if (this.activeCursorTool !== 'none' && !this.inspectMode && this.trackMode === 'idle'
                && this.cursorMouseX > -9000 && this.sim) {
            const sr = this.brushWorldRadius * this.pxPerWorld();
            const pressing = this.cursorMouseButtons !== 0;
            const tool = this.activeCursorTool;
            const idleColor  = tool === 'paint' ? 'rgba(0,255,100,0.4)'  : tool === 'erase' ? 'rgba(255,80,80,0.4)'  : 'rgba(255,255,255,0.4)';
            const pressColor = tool === 'paint' ? 'rgba(0,255,100,0.75)' : tool === 'erase' ? 'rgba(255,80,80,0.75)' : 'rgba(0,170,255,0.7)';
            ctx.save();
            // Filled red overlay when right-click erasing (instant-all mode)
            if (tool === 'erase' && (this.cursorMouseButtons & 4)) {
                ctx.fillStyle = 'rgba(255,60,60,0.15)';
                ctx.beginPath();
                ctx.arc(this.cursorMouseX, this.cursorMouseY, Math.max(2, sr), 0, Math.PI * 2);
                ctx.fill();
            }
            ctx.strokeStyle = pressing ? pressColor : idleColor;
            ctx.lineWidth = 1.5;
            ctx.setLineDash([4, 4]);
            ctx.beginPath();
            ctx.arc(this.cursorMouseX, this.cursorMouseY, Math.max(2, sr), 0, Math.PI * 2);
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.restore();
        }

        if (this.trackMode === 'selecting' && this.selBoxActive) {
            const { sx0, sy0, sx1, sy1 } = this.selBox;
            ctx.strokeStyle = '#0af';
            ctx.lineWidth = 1.5;
            ctx.setLineDash([5, 4]);
            ctx.strokeRect(Math.min(sx0, sx1), Math.min(sy0, sy1),
                           Math.abs(sx1 - sx0), Math.abs(sy1 - sy0));
            ctx.setLineDash([]);
        }

        if (this.trackMode === 'tracking' && this.sim) {
            const info = this.sim.getTrackingInfo();
            if (!info) { this.trackMode = 'idle'; this.syncTrackButton(); return; }
            const { sx, sy } = this.worldToScreen(info.comX, info.comY);
            const sr = info.radius * this.pxPerWorld();

            const drift = (performance.now() * 0.02) % 20;
            ctx.save();
            ctx.strokeStyle = 'rgba(255,200,0,0.55)';
            ctx.lineWidth = 1.5;
            ctx.setLineDash([6, 6]);
            ctx.lineDashOffset = -drift;
            ctx.beginPath();
            ctx.arc(sx, sy, Math.max(4, sr), 0, Math.PI * 2);
            ctx.stroke();
            ctx.setLineDash([]);

            ctx.strokeStyle = 'rgba(255,200,0,0.8)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(sx - 10, sy); ctx.lineTo(sx + 10, sy);
            ctx.moveTo(sx, sy - 10); ctx.lineTo(sx, sy + 10);
            ctx.stroke();
            ctx.restore();
        }

        // Selected particle: highlight ring + velocity arrow
        const diagData = this.sim?.diagData;
        if (diagData && this.sim) {
            const { sx, sy } = this.worldToScreen(diagData.pos[0], diagData.pos[1]);
            const scaleY = this.pxPerWorld();

            // Highlight ring around selected particle
            ctx.save();
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.arc(sx, sy, 8, 0, Math.PI * 2);
            ctx.stroke();

            // Velocity arrow
            const speed = diagData.speed;
            if (speed > 0.001) {
                const arrowPx = Math.min(speed * scaleY * 8, 120);
                const nx = diagData.vel[0] / speed;
                const ny = diagData.vel[1] / speed;
                const ex = sx + nx * arrowPx;
                const ey = sy + ny * arrowPx;
                const hw = 5, hl = 10;
                const px = -ny, py = nx;

                ctx.strokeStyle = '#0f8';
                ctx.fillStyle   = '#0f8';
                ctx.lineWidth = 1.5;
                ctx.beginPath();
                ctx.moveTo(sx, sy);
                ctx.lineTo(ex - nx * hl, ey - ny * hl);
                ctx.stroke();
                ctx.beginPath();
                ctx.moveTo(ex, ey);
                ctx.lineTo(ex - nx * hl + px * hw, ey - ny * hl + py * hw);
                ctx.lineTo(ex - nx * hl - px * hw, ey - ny * hl - py * hw);
                ctx.closePath();
                ctx.fill();
            }
            ctx.restore();
        }

        // Photo export selection rectangle + output pixel dimensions
        if (this.photoSelMode && this.photoSelBox) {
            const b  = this.photoSelBox;
            const x  = Math.min(b.sx0, b.sx1), y = Math.min(b.sy0, b.sy1);
            const bw = Math.abs(b.sx1 - b.sx0), bh = Math.abs(b.sy1 - b.sy0);
            ctx.save();
            ctx.strokeStyle = '#ff3df0';
            ctx.lineWidth = 1.5;
            ctx.setLineDash([6, 4]);
            ctx.strokeRect(x, y, bw, bh);
            ctx.setLineDash([]);

            const dim   = this.clientBoxToCanvasRect(b);
            const label = `${dim.w} × ${dim.h} px`;
            ctx.font = '12px monospace';
            const tw = ctx.measureText(label).width;
            // Place label just outside the rectangle (above-left; below if no room above)
            const lx = x;
            let   ly = y - 7;
            if (ly < 14) ly = y + bh + 16;
            ctx.fillStyle = 'rgba(0,0,0,0.65)';
            ctx.fillRect(lx - 3, ly - 12, tw + 6, 16);
            ctx.fillStyle = '#ff3df0';
            ctx.fillText(label, lx, ly);
            ctx.restore();
        }
    }

    private updateDiagPanel(data: DiagnosticData | null): void {
        if (!this.diagPanelEl) return;
        if (!data) {
            this.diagPanelEl.classList.remove('visible');
            this.inspectMode = false;
            this.syncInspectButton();
            return;
        }
        this.diagPanelEl.classList.add('visible');
        const n = data.typeForces.length;

        document.getElementById('diag-index')!.textContent = String(data.index);

        const typeEl = document.getElementById('diag-type')!;
        typeEl.innerHTML = `<span class="diag-type-swatch" style="background:${TYPE_HEX[data.typeId] ?? '#888'}"></span>${TYPE_LABELS[data.typeId] ?? '?'} (${data.typeId})`;

        document.getElementById('diag-speed')!.textContent = `${data.speed.toFixed(3)} u/t`;

        const deg = data.directionDeg;
        document.getElementById('diag-dir')!.textContent = `${deg.toFixed(1)}°`;

        document.getElementById('diag-pos')!.textContent = `${data.pos[0].toFixed(0)}, ${data.pos[1].toFixed(0)}`;

        // Forces
        const forcesEl = document.getElementById('diag-forces')!;
        forcesEl.innerHTML = '';
        const maxForce = Math.max(...data.typeForces.map(Math.abs), 0.001);
        for (let t = 0; t < n; t++) {
            const f = data.typeForces[t];
            const row = document.createElement('div');
            row.className = 'diag-force-row';
            const pct = (f / maxForce) * 50;
            const color = f >= 0 ? '#0d6' : '#e44';
            row.innerHTML = `
                <span class="diag-force-pip" style="background:${TYPE_HEX[t] ?? '#888'}"></span>
                <span style="color:#888;width:12px;text-align:center">${TYPE_LABELS[t] ?? t}</span>
                <div class="diag-force-bar-wrap">
                    <div class="diag-force-bar" style="background:${color};width:${Math.abs(pct)}%;${f >= 0 ? 'left:50%' : `left:${50 + pct}%`}"></div>
                </div>
                <span class="diag-force-val" style="color:${color}">${f.toFixed(3)}</span>`;
            forcesEl.appendChild(row);
        }

        // Transform probabilities
        const transEl = document.getElementById('diag-transforms')!;
        transEl.innerHTML = '';
        for (let t = 0; t < n; t++) {
            const p = data.transformProbs[t];
            if (p <= 0) continue;
            const row = document.createElement('div');
            row.className = 'diag-prob-row';
            row.innerHTML = `
                <span class="diag-force-pip" style="background:${TYPE_HEX[t] ?? '#888'}"></span>
                <span style="color:#888;width:12px;text-align:center">${TYPE_LABELS[t] ?? t}</span>
                <div class="diag-prob-bar-wrap">
                    <div class="diag-prob-bar" style="width:${(p * 100).toFixed(1)}%"></div>
                </div>
                <span class="diag-prob-val">${(p * 100).toFixed(1)}%</span>`;
            transEl.appendChild(row);
        }
        if (transEl.children.length === 0) {
            transEl.innerHTML = '<span style="color:#444;font-size:10px">none active</span>';
        }
    }

    // ── Pan / zoom ────────────────────────────────────────────────────────────

    private stopTracking(): void {
        if (this.trackMode === 'tracking') {
            this.sim?.stopTracking();
            this.trackMode = 'idle';
            this.syncTrackButton();
        }
    }

    private stopInspect(): void {
        this.inspectMode = false;
        this.sim?.clearParticleSelection();
        this.syncInspectButton();
    }

    private syncInspectButton(): void {
        const btn = document.getElementById('btmInspectBtn')!;
        btn.classList.toggle('active', this.inspectMode);
        btn.textContent = this.inspectMode ? '⦳ Click a particle' : '🔍 Inspect';
        document.body.classList.toggle('inspect-mode', this.inspectMode);
        this.syncCanvasCursor();
    }

    private setupCanvasEvents(): void {
        let panning = false, lastX = 0, lastY = 0, panDist = 0;
        const PAN_BREAK_THRESHOLD = 30; // pixels before tracking breaks

        this.canvas.addEventListener('mousedown', (e) => {
            // Middle-click → always pans regardless of tool
            if (e.button === 1) {
                e.preventDefault();
                panning = true; lastX = e.clientX; lastY = e.clientY; panDist = 0;
                this.canvas.style.cursor = 'grabbing';
                return;
            }
            // Photo export selection overrides everything else
            if (e.button === 0 && this.photoSelMode) {
                e.preventDefault();
                this.photoSelDragging = true;
                this.photoSelBox = { sx0: e.clientX, sy0: e.clientY, sx1: e.clientX, sy1: e.clientY };
                return;
            }
            // Track / Inspect override cursor tools — check first
            if (e.button === 0 && this.trackMode === 'selecting') {
                e.preventDefault();
                this.selBoxActive = true;
                this.selBox = { sx0: e.clientX, sy0: e.clientY, sx1: e.clientX, sy1: e.clientY };
                return;
            }
            if (e.button === 0 && this.inspectMode && this.sim) {
                const { wx, wy } = this.screenToWorld(e.clientX, e.clientY);
                this.sim.selectParticleAt(wx, wy);
                this.inspectMode = false;
                this.syncInspectButton();
                return;
            }
            // Cursor tools (only when no track/inspect override is active)
            if (e.button === 0 && this.activeCursorTool === 'grab') {
                e.preventDefault();
                panning = true; lastX = e.clientX; lastY = e.clientY; panDist = 0;
                this.cursorMouseButtons |= 1;
                this.syncCanvasCursor();
                return;
            }
            if (this.activeCursorTool === 'force' || this.activeCursorTool === 'paint' || this.activeCursorTool === 'erase') {
                e.preventDefault();
                if (e.button === 0) this.cursorMouseButtons |= 1;
                if (e.button === 2) this.cursorMouseButtons |= 4;
                this.syncCanvasCursor();
                return;
            }
        });

        window.addEventListener('mousemove', (e) => {
            if (panning && this.sim) {
                const dx = e.clientX - lastX, dy = e.clientY - lastY;
                const view = this.sim.getView();
                const wpp  = 1 / this.pxPerWorld();   // world units per screen pixel
                this.sim.setView(
                    view.cx - dx * wpp,
                    view.cy - dy * wpp,
                    view.zoom
                );
                lastX = e.clientX; lastY = e.clientY;
                panDist += Math.sqrt(dx * dx + dy * dy);
                if (panDist > PAN_BREAK_THRESHOLD) this.stopTracking();
            }
            if (this.selBoxActive && this.trackMode === 'selecting') {
                this.selBox.sx1 = e.clientX;
                this.selBox.sy1 = e.clientY;
            }
            if (this.photoSelDragging && this.photoSelBox) {
                this.photoSelBox.sx1 = e.clientX;
                this.photoSelBox.sy1 = e.clientY;
            }
        });

        window.addEventListener('mouseup', (e) => {
            if (e.button === 1 || (e.button === 0 && this.activeCursorTool === 'grab')) {
                panning = false;
            }
            if (e.button === 0 && this.photoSelDragging) {
                this.photoSelDragging = false;
                const box = this.photoSelBox;
                const target = this.photoSelTarget;
                let rect: { x: number; y: number; w: number; h: number } | null = null;
                if (box && this.sim) {
                    const r = this.clientBoxToCanvasRect(box);
                    if (r.w >= 1 && r.h >= 1) {
                        if (target === 'png') this.exportSelection(box);
                        else rect = r;
                    }
                }
                // Selecting auto-ends the selection (and resumes the sim unless it was
                // already paused before entering select mode).
                this.exitPhotoSelect();
                // Video records the live sim, so start only after resuming.
                if (target === 'video' && rect) this.startVideoExport(rect);
            }
            if (e.button === 0 && this.selBoxActive && this.trackMode === 'selecting') {
                this.selBoxActive = false;
                document.body.classList.remove('track-selecting');
                const { sx0, sy0, sx1, sy1 } = this.selBox;
                const w = Math.abs(sx1 - sx0), h = Math.abs(sy1 - sy0);
                if (w > 8 && h > 8 && this.sim) {
                    const { wx: wx0, wy: wy0 } = this.screenToWorld(Math.min(sx0, sx1), Math.min(sy0, sy1));
                    const { wx: wx1, wy: wy1 } = this.screenToWorld(Math.max(sx0, sx1), Math.max(sy0, sy1));
                    const comX   = (wx0 + wx1) / 2;
                    const comY   = (wy0 + wy1) / 2;
                    const radius = Math.max(wx1 - wx0, wy1 - wy0) * 0.75;
                    this.sim.startTracking(comX, comY, radius);
                    this.sim.onTrackingStop = () => { this.trackMode = 'idle'; this.syncTrackButton(); };
                    this.trackMode = 'tracking';
                } else {
                    this.trackMode = 'idle';
                }
                this.syncTrackButton();
            }
        });

        this.canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            if (!this.sim) return;
            const rect   = this.canvas.getBoundingClientRect();
            const view   = this.sim.getView();
            const oldZ   = view.zoom;
            const newZ   = Math.max(0.05, Math.min(200, oldZ * (e.deltaY < 0 ? 1.15 : 1 / 1.15)));
            // Keep the world point under the cursor fixed (spans scale as 1/zoom).
            const fx = (e.clientX - rect.left - rect.width  / 2) / (rect.width  / 2);
            const fy = (e.clientY - rect.top  - rect.height / 2) / (rect.height / 2);
            const s  = this.viewSpans();   // spans at oldZ
            this.sim.setView(
                view.cx + fx * s.spanX * (1 - oldZ / newZ),
                view.cy + fy * s.spanY * (1 - oldZ / newZ),
                newZ
            );
            this.updateZoomDisplay();
        }, { passive: false });

        this.canvas.addEventListener('dblclick', () => {
            if (!this.sim) return;
            const v = this.sim.getDefaultView();
            this.sim.setView(v.cx, v.cy, v.zoom);
            this.updateZoomDisplay();
        });
    }

    private updateZoomDisplay(): void {
        const view = this.sim?.getView();
        if (view) document.getElementById('zoomDisplay')!.textContent = `${view.zoom.toFixed(2)}×`;
    }

    // ── Cursor tools ──────────────────────────────────────────────────────────

    private setupCursorPanel(): void {
        document.getElementById('ctool-grab')!.addEventListener('click',  () => this.setActiveCursorTool('grab'));
        document.getElementById('ctool-force')!.addEventListener('click', () => this.setActiveCursorTool('force'));
        document.getElementById('ctool-paint')!.addEventListener('click', () => this.setActiveCursorTool('paint'));
        document.getElementById('ctool-erase')!.addEventListener('click', () => this.setActiveCursorTool('erase'));

        const brushSlider = document.getElementById('brushSizeSlider') as HTMLInputElement;
        const brushVal    = document.getElementById('brushSizeVal')!;
        brushSlider.addEventListener('input', () => {
            this.brushWorldRadius = parseInt(brushSlider.value);
            brushVal.textContent  = String(this.brushWorldRadius);
        });

        const forceSlider = document.getElementById('forceStrSlider') as HTMLInputElement;
        const forceVal    = document.getElementById('forceStrVal')!;
        forceSlider.addEventListener('input', () => {
            this.forceStrength   = parseFloat(forceSlider.value);
            forceVal.textContent = this.forceStrength.toFixed(1);
        });

        const paintRateSlider = document.getElementById('paintRateSlider') as HTMLInputElement;
        const paintRateVal    = document.getElementById('paintRateVal')!;
        paintRateSlider.addEventListener('input', () => {
            this.paintRate       = parseInt(paintRateSlider.value);
            paintRateVal.textContent = String(this.paintRate);
        });

        // Reposition the contextual tool popover above its dock button on resize
        window.addEventListener('resize', () => this.updateToolPopover());

        window.addEventListener('mousemove', (e) => {
            this.cursorMouseX = e.clientX;
            this.cursorMouseY = e.clientY;
        });
        window.addEventListener('mouseup', (e) => {
            if (e.button === 0) this.cursorMouseButtons &= ~1;
            if (e.button === 2) this.cursorMouseButtons &= ~4;
            this.syncCanvasCursor();
        });
        window.addEventListener('blur', () => { this.cursorMouseButtons = 0; });
    }

    private setActiveCursorTool(tool: 'grab' | 'force' | 'paint' | 'erase'): void {
        if (this.photoSelMode) this.exitPhotoSelect();
        this.activeCursorTool = (this.activeCursorTool === tool) ? 'none' : tool;
        document.getElementById('ctool-grab')!.classList.toggle('active',  this.activeCursorTool === 'grab');
        document.getElementById('ctool-force')!.classList.toggle('active', this.activeCursorTool === 'force');
        document.getElementById('ctool-paint')!.classList.toggle('active', this.activeCursorTool === 'paint');
        document.getElementById('ctool-erase')!.classList.toggle('active', this.activeCursorTool === 'erase');
        document.getElementById('force-section')!.style.display      = this.activeCursorTool === 'force' ? '' : 'none';
        document.getElementById('paint-section')!.style.display      = this.activeCursorTool === 'paint' ? '' : 'none';
        document.getElementById('erase-section')!.style.display      = this.activeCursorTool === 'erase' ? '' : 'none';
        const showRate = this.activeCursorTool === 'paint' || this.activeCursorTool === 'erase';
        document.getElementById('paint-rate-section')!.style.display = showRate ? '' : 'none';
        if (this.activeCursorTool === 'paint') this.refreshPaintTypePicker();
        this.updateToolPopover();
        this.syncCanvasCursor();
    }

    // Show the contextual tool settings popover above the active tool's dock button.
    // Only the brush-using tools (force/paint/erase) have settings; grab has none.
    private updateToolPopover(): void {
        const pop = document.getElementById('tool-popover');
        if (!pop) return;
        const tool = this.activeCursorTool;
        const show = tool === 'force' || tool === 'paint' || tool === 'erase';
        pop.classList.toggle('visible', show);
        if (!show) return;
        const anchor = document.getElementById(`ctool-${tool}`);
        if (!anchor) return;
        const a = anchor.getBoundingClientRect();
        const popW = pop.offsetWidth || 196;
        let left = a.left + a.width / 2 - popW / 2;
        left = Math.max(8, Math.min(window.innerWidth - popW - 8, left));
        pop.style.left   = `${left}px`;
        pop.style.bottom = `${window.innerHeight - a.top + 8}px`;
    }

    // ── Master randomize ──────────────────────────────────────────────────────

    private toggleRandomizeMenu(show?: boolean): void {
        const pop = document.getElementById('randomize-popover');
        const caret = document.getElementById('randomizeMenuBtn');
        if (!pop) return;
        const visible = show ?? !pop.classList.contains('visible');
        if (visible) this.buildRandomizeMenu();
        pop.classList.toggle('visible', visible);
        caret?.classList.toggle('active', visible);
        if (visible) {
            const anchor = document.getElementById('randomizeNowBtn');
            if (anchor) {
                const a = anchor.getBoundingClientRect();
                const popW = pop.offsetWidth || 180;
                let left = a.left + a.width / 2 - popW / 2;
                left = Math.max(8, Math.min(window.innerWidth - popW - 8, left));
                pop.style.left   = `${left}px`;
                pop.style.bottom = `${window.innerHeight - a.top + 8}px`;
            }
        }
    }

    // Rebuild the checklist showing only categories relevant to the current mode.
    private buildRandomizeMenu(): void {
        const list = document.getElementById('randomize-list');
        if (!list) return;
        list.innerHTML = '';
        const mode = this.sim?.getSimMode() ?? 0;
        for (const cat of this.randomizeCats) {
            if (cat.modes !== null && !cat.modes.includes(mode)) continue;
            const row = document.createElement('label');
            row.className = 'rnd-row';
            const cb = document.createElement('input');
            cb.type = 'checkbox';
            cb.checked = this.randomizeSel[cat.key];
            cb.addEventListener('change', () => { this.randomizeSel[cat.key] = cb.checked; });
            row.appendChild(cb);
            row.appendChild(document.createTextNode(cat.label));
            list.appendChild(row);
        }
    }

    // Run every selected category that applies to the current mode, then refresh
    // the panels that changed.
    private runMasterRandomize(): void {
        if (!this.sim) return;
        const mode = this.sim.getSimMode();
        const refreshes = new Set<string>();
        for (const cat of this.randomizeCats) {
            if (cat.modes !== null && !cat.modes.includes(mode)) continue;
            if (!this.randomizeSel[cat.key]) continue;
            cat.run();
            refreshes.add(cat.refresh);
        }
        if (refreshes.has('forces'))    this.refreshForceMatrices();
        if (refreshes.has('transform')) this.refreshTransformMatrix();  // also rebuilds poles
        if (refreshes.has('poles'))     this.buildPolePanel();
        if (refreshes.has('masses'))    this.refreshMassTable();
        if (refreshes.has('valences'))  this.refreshPatchTable();
        if (refreshes.has('bonding'))   this.refreshPatchUI();
        if (refreshes.has('dnf'))       this.refreshDnfPanel();
        if (refreshes.has('charges'))   this.refreshQftPanel();
    }

    private refreshPaintTypePicker(): void {
        const container = document.getElementById('paint-type-picker');
        if (!container || !this.sim) return;
        container.innerHTML = '';
        const n = this.sim.getNumTypes();
        for (let t = 0; t < n; t++) {
            const btn = document.createElement('button');
            btn.style.cssText = `width:16px;height:16px;border-radius:50%;background:${TYPE_HEX[t]};` +
                `border:2px solid ${t === this.paintTypeId ? '#fff' : 'transparent'};` +
                `cursor:pointer;flex-shrink:0;padding:0;min-width:0;`;
            btn.title = TYPE_LABELS[t];
            btn.addEventListener('click', () => {
                this.paintTypeId = t;
                this.refreshPaintTypePicker();
            });
            container.appendChild(btn);
        }
    }

    private syncCanvasCursor(): void {
        if (this.photoSelMode) { this.canvas.style.cursor = 'crosshair'; return; }
        // Inspect/track override: clear inline style so CSS body-class cursor takes effect
        if (this.inspectMode || this.trackMode !== 'idle') {
            this.canvas.style.cursor = '';
            return;
        }
        if (this.activeCursorTool === 'grab') {
            this.canvas.style.cursor = (this.cursorMouseButtons & 1) ? 'grabbing' : 'grab';
        } else if (this.activeCursorTool === 'force' || this.activeCursorTool === 'paint' || this.activeCursorTool === 'erase') {
            this.canvas.style.cursor = 'crosshair';
        } else {
            this.canvas.style.cursor = '';
        }
    }

    // ── Photo export ───────────────────────────────────────────────────────────

    private togglePhotoSelect(target: 'png' | 'video' = 'png'): void {
        // Clicking the same tool again cancels; switching target re-arms.
        if (this.photoSelMode && this.photoSelTarget === target) { this.exitPhotoSelect(); return; }
        if (this.isRecording) return;
        this.photoSelTarget = target;
        if (this.photoSelMode) { this.syncPhotoSelButton(); return; }
        // Turn off any conflicting interaction modes
        if (this.activeCursorTool !== 'none') this.setActiveCursorTool(this.activeCursorTool);  // toggles off
        if (this.trackMode === 'selecting') {
            this.trackMode = 'idle';
            document.body.classList.remove('track-selecting');
            this.syncTrackButton();
        }
        if (this.inspectMode) this.stopInspect();
        this.photoSelMode = true;
        this.photoSelBox  = null;
        // Freeze the frame so the user can compose a selection; remember if it was
        // already paused so we don't resume something they paused themselves.
        this.photoSelDidPause = false;
        if (this.sim && !this.sim.isPaused_()) {
            this.sim.togglePause();
            this.photoSelDidPause = true;
            this.syncPauseButton();
        }
        this.syncPhotoSelButton();
        this.syncCanvasCursor();
    }

    private exitPhotoSelect(): void {
        this.photoSelMode     = false;
        this.photoSelDragging = false;
        this.photoSelBox      = null;
        this.resumeAfterPhotoPause();
        this.syncPhotoSelButton();
        this.syncCanvasCursor();
    }

    // Resume the sim only if entering select mode is what paused it.
    private resumeAfterPhotoPause(): void {
        if (this.photoSelDidPause && this.sim?.isPaused_()) {
            this.sim.togglePause();
            this.syncPauseButton();
        }
        this.photoSelDidPause = false;
    }

    private syncPhotoSelButton(): void {
        const png = document.getElementById('exportSelBtn');
        const vid = document.getElementById('recordSelBtn');
        const pngActive = this.photoSelMode && this.photoSelTarget === 'png';
        const vidActive = this.photoSelMode && this.photoSelTarget === 'video';
        if (png) {
            png.classList.toggle('active', pngActive);
            png.innerHTML = pngActive ? '&#9645; Drag a box · Esc to cancel' : '&#9645; Select &amp; Export PNG';
        }
        if (vid) {
            vid.classList.toggle('active', vidActive);
            vid.innerHTML = vidActive ? '&#9645; Drag a box · Esc to cancel' : '&#9210; Select &amp; Record Video';
        }
    }

    // Grey out full-canvas export in open edge mode (its framing isn't a complete image)
    private syncExportButtons(): void {
        const btn = document.getElementById('exportFullBtn') as HTMLButtonElement | null;
        if (!btn) return;
        const open = this.sim?.getEdgeMode() === 1;
        btn.classList.toggle('disabled-look', open);
        btn.title = open
            ? 'Full-canvas export is only available in closed (Loop) edge mode — use Select & Export instead.'
            : 'Export the entire canvas as a lossless PNG';
    }

    // Map a client-space drag box to integer canvas-pixel coordinates (the export resolution)
    private clientBoxToCanvasRect(box: { sx0: number; sy0: number; sx1: number; sy1: number }):
            { x: number; y: number; w: number; h: number } {
        const rect = this.canvas.getBoundingClientRect();
        const cw = this.canvas.width, ch = this.canvas.height;
        const toX = (cx: number) => Math.round((cx - rect.left) / rect.width  * cw);
        const toY = (cy: number) => Math.round((cy - rect.top)  / rect.height * ch);
        const x0 = Math.max(0, Math.min(cw, toX(Math.min(box.sx0, box.sx1))));
        const y0 = Math.max(0, Math.min(ch, toY(Math.min(box.sy0, box.sy1))));
        const x1 = Math.max(0, Math.min(cw, toX(Math.max(box.sx0, box.sx1))));
        const y1 = Math.max(0, Math.min(ch, toY(Math.max(box.sy0, box.sy1))));
        return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
    }

    private async exportFullCanvas(): Promise<void> {
        if (!this.sim) return;
        // Export exactly the physics-simmed world area at full resolution, ignoring
        // the live camera framing (full export is only offered in loop edge mode).
        const cap = await this.sim.captureRGBA({ fullWorld: true });
        if (!cap) return;
        const blob = await this.rgbaToPngBlob(cap.data, cap.width, cap.height);
        if (blob) this.downloadBlob(blob, this.exportFilename(`${cap.width}x${cap.height}`));
    }

    private async exportSelection(box: { sx0: number; sy0: number; sx1: number; sy1: number }): Promise<void> {
        if (!this.sim) return;
        const r = this.clientBoxToCanvasRect(box);
        if (r.w < 1 || r.h < 1) return;
        const cap = await this.sim.captureRGBA();
        if (!cap) return;
        // Crop the requested region out of the full-resolution capture
        const cropped = new Uint8ClampedArray(r.w * r.h * 4);
        for (let row = 0; row < r.h; row++) {
            const srcStart = ((r.y + row) * cap.width + r.x) * 4;
            cropped.set(cap.data.subarray(srcStart, srcStart + r.w * 4), row * r.w * 4);
        }
        const blob = await this.rgbaToPngBlob(cropped, r.w, r.h);
        if (blob) this.downloadBlob(blob, this.exportFilename(`${r.w}x${r.h}`));
    }

    private rgbaToPngBlob(data: Uint8ClampedArray, w: number, h: number): Promise<Blob | null> {
        const c = document.createElement('canvas');
        c.width = w; c.height = h;
        const ctx = c.getContext('2d');
        if (!ctx) return Promise.resolve(null);
        ctx.putImageData(new ImageData(data as Uint8ClampedArray<ArrayBuffer>, w, h), 0, 0);
        return new Promise(resolve => c.toBlob(b => resolve(b), 'image/png'));
    }

    private downloadBlob(blob: Blob, name: string): void {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url; a.download = name;
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
    }

    private exportFilename(tag: string, ext = 'png'): string {
        const d = new Date();
        const p = (n: number) => String(n).padStart(2, '0');
        const stamp = `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
        return `particle-life-${stamp}-${tag}.${ext}`;
    }

    // ── Video export ─────────────────────────────────────────────────────────
    private recordFullVideo(): void {
        if (this.isRecording || this.isRenderingVideo || !this.sim) return;
        this.startVideoExport({ x: 0, y: 0, w: this.canvas.width, h: this.canvas.height });
    }

    // Prefer the deterministic offline renderer (WebCodecs): it steps the sim
    // frame-by-frame and encodes a fixed-60fps clip however long the compute takes,
    // so heavy sims don't slow/stutter the output. Falls back to real-time capture.
    private startVideoExport(rect: { x: number; y: number; w: number; h: number }): void {
        const hasWebCodecs = typeof (window as any).VideoEncoder !== 'undefined'
            && typeof (window as any).VideoFrame !== 'undefined';
        if (hasWebCodecs) this.renderVideoOffline(rect);
        else this.startRecording(rect);
    }

    private async renderVideoOffline(rect: { x: number; y: number; w: number; h: number }): Promise<void> {
        if (this.isRecording || this.isRenderingVideo || !this.sim) return;
        const VE: any = (window as any).VideoEncoder;
        const fps = 60;
        const total = Math.round(this.videoDurationSec * fps);
        // Downscale the output so encoding stays fast (full window res is overkill
        // for a clip). Longest side is capped; aspect preserved; dims forced even.
        const MAX_DIM = 1280;
        const scale = Math.min(1, MAX_DIM / Math.max(rect.w, rect.h));
        let w = Math.max(2, Math.round(rect.w * scale)); w -= w % 2;
        let h = Math.max(2, Math.round(rect.h * scale)); h -= h % 2;
        const bitrate = Math.min(40_000_000, Math.max(6_000_000, Math.round(w * h * fps * 0.12)));

        // Pick a supported codec (VP9 preferred, then VP8).
        let codecStr = '', muxCodec = 'V_VP9';
        for (const [c, m] of [['vp09.00.10.08', 'V_VP9'], ['vp8', 'V_VP8']] as const) {
            try {
                const sup = await VE.isConfigSupported({ codec: c, width: w, height: h, bitrate, framerate: fps });
                if (sup && sup.supported) { codecStr = c; muxCodec = m; break; }
            } catch { /* try next */ }
        }
        if (!codecStr) { this.startRecording(rect); return; }   // fall back to real-time

        const target = new ArrayBufferTarget();
        const muxer = new Muxer({ target, video: { codec: muxCodec, width: w, height: h, frameRate: fps } });
        let encErr: any = null;
        const encoder = new VE({
            output: (chunk: any, meta: any) => muxer.addVideoChunk(chunk, meta),
            error:  (e: any) => { encErr = e; },
        });
        encoder.configure({ codec: codecStr, width: w, height: h, bitrate, framerate: fps, latencyMode: 'realtime' });

        // Capture target: downscaled (and cropped) copy of the live canvas. Drawing
        // from the canvas on the GPU avoids a per-frame readback + CPU pixel copy.
        const tcanvas = document.createElement('canvas');
        tcanvas.width = w; tcanvas.height = h;
        const tctx = tcanvas.getContext('2d');
        if (!tctx) { try { encoder.close(); } catch {} this.startRecording(rect); return; }

        const VF: any = (window as any).VideoFrame;
        this.isRecording = true;
        this.isRenderingVideo = true;
        this.renderCancel = false;
        const savedAutoPause = this.autoPause;
        this.autoPause = false;

        let ok = false;
        try {
            for (let i = 0; i < total; i++) {
                if (this.renderCancel || encErr) break;
                await this.sim.stepAndAwait();   // deterministic physics step + on-screen render
                // Blit (downscale + crop) the freshly-rendered canvas into the target,
                // then hand it straight to the encoder — no readback, no CPU copy.
                tctx.drawImage(this.canvas, rect.x, rect.y, rect.w, rect.h, 0, 0, w, h);
                const frame = new VF(tcanvas, {
                    timestamp: Math.round(i * 1e6 / fps), duration: Math.round(1e6 / fps),
                });
                encoder.encode(frame, { keyFrame: i % 60 === 0 });
                frame.close();
                this.syncRenderProgress(i + 1, total);
                // Drain encoder backpressure and yield so the UI stays responsive
                // (progress repaint + cancel clicks) and GPU memory doesn't pile up.
                while (encoder.encodeQueueSize > 6 && !this.renderCancel) {
                    await new Promise(r => setTimeout(r, 0));
                }
                // Yield to the UI occasionally (repaint progress + allow cancel)
                // without paying a full frame-wait every step.
                if (i % 8 === 0) await new Promise(r => requestAnimationFrame(() => r(null)));
            }
            if (!this.renderCancel && !encErr) {
                await encoder.flush();
                muxer.finalize();
                const blob = new Blob([target.buffer], { type: 'video/webm' });
                if (blob.size) this.downloadBlob(blob, this.exportFilename(`${w}x${h}-${this.videoDurationSec}s`, 'webm'));
                ok = true;
            }
        } catch (e) {
            console.error('Video render failed:', e);
        } finally {
            try { encoder.close(); } catch { /* already closed */ }
            this.autoPause = savedAutoPause;
            this.isRecording = false;
            this.isRenderingVideo = false;
            this.renderCancel = false;
            this.syncRecordButtons(0);
            if (encErr && !ok) alert('Video render failed (encoder error). See console.');
        }
    }

    private syncRenderProgress(done: number, total: number): void {
        const pct = Math.round(done / total * 100);
        const full = document.getElementById('recordFullBtn');
        if (full) full.innerHTML = `&#9209; Rendering ${pct}% · cancel`;
        const sel = document.getElementById('recordSelBtn');
        if (sel) { sel.classList.add('disabled-look'); sel.innerHTML = '&#9209; cancel'; }
    }

    private pickVideoMime(): string {
        const cands = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm', 'video/mp4'];
        for (const c of cands) {
            if (typeof MediaRecorder !== 'undefined' && MediaRecorder.isTypeSupported(c)) return c;
        }
        return '';
    }

    // Record the live simulation over the given canvas-pixel rect for the chosen
    // duration. A 2D record canvas is blitted from the WebGPU canvas each frame
    // (so we can crop to a sub-region) and captured via MediaRecorder.
    private startRecording(rect: { x: number; y: number; w: number; h: number }): void {
        if (this.isRecording || !this.sim) return;
        if (typeof MediaRecorder === 'undefined') { alert('Video recording is not supported in this browser.'); return; }
        const mime = this.pickVideoMime();
        const w = Math.max(2, Math.round(rect.w)), h = Math.max(2, Math.round(rect.h));
        const fps = 30;

        // Whole canvas → capture its stream directly (most reliable). A sub-region →
        // blit the crop into a 2D record canvas each frame and capture that.
        const fullCanvas = rect.x === 0 && rect.y === 0 && w === this.canvas.width && h === this.canvas.height;
        let stream: MediaStream;
        let perFrame: (() => void) | null = null;
        if (fullCanvas) {
            stream = this.canvas.captureStream(fps);
        } else {
            const rc = document.createElement('canvas');
            rc.width = w; rc.height = h;
            const rctx = rc.getContext('2d');
            if (!rctx) return;
            const bg = this.sim.getBackgroundColor();
            rctx.fillStyle = `rgb(${Math.round(bg.r * 255)},${Math.round(bg.g * 255)},${Math.round(bg.b * 255)})`;
            rctx.fillRect(0, 0, w, h);
            stream = rc.captureStream(fps);
            perFrame = () => { try { rctx.drawImage(this.canvas, rect.x, rect.y, w, h, 0, 0, w, h); } catch { /* frame not ready */ } };
        }
        // Bitrate scaled to area, capped — high enough to keep particle detail crisp.
        const bitrate = Math.min(40_000_000, Math.max(6_000_000, Math.round(w * h * fps * 0.15)));
        let rec: MediaRecorder;
        try {
            rec = new MediaRecorder(stream, mime ? { mimeType: mime, videoBitsPerSecond: bitrate } : { videoBitsPerSecond: bitrate });
        } catch {
            alert('Could not start video recording.');
            return;
        }
        const chunks: BlobPart[] = [];
        rec.ondataavailable = (e) => { if (e.data && e.data.size) chunks.push(e.data); };
        rec.onstop = () => {
            const type = rec.mimeType || mime || 'video/webm';
            const ext  = type.includes('mp4') ? 'mp4' : 'webm';
            const blob = new Blob(chunks, { type });
            if (blob.size) this.downloadBlob(blob, this.exportFilename(`${w}x${h}-${this.videoDurationSec}s`, ext));
            stream.getTracks().forEach(t => t.stop());
        };

        // Per-frame blit (sub-region only): runs in the animation loop after render.
        this.recordCopy = perFrame;

        rec.start();
        this.isRecording = true;
        this.recorder = rec;
        const endAt = performance.now() + this.videoDurationSec * 1000;
        this.recordStopTimer = window.setTimeout(() => this.stopRecording(), this.videoDurationSec * 1000);
        this.recordCountdownTimer = window.setInterval(() => {
            const left = Math.max(0, Math.ceil((endAt - performance.now()) / 1000));
            this.syncRecordButtons(left);
        }, 250);
        this.syncRecordButtons(this.videoDurationSec);
    }

    private stopRecording(): void {
        if (!this.isRecording) return;
        window.clearTimeout(this.recordStopTimer);
        window.clearInterval(this.recordCountdownTimer);
        this.recordCopy = null;
        try { this.recorder?.stop(); } catch { /* already stopped */ }
        this.recorder = null;
        this.isRecording = false;
        this.syncRecordButtons(0);
    }

    private syncRecordButtons(secondsLeft: number): void {
        const full = document.getElementById('recordFullBtn');
        const sel  = document.getElementById('recordSelBtn');
        const rec  = this.isRecording;
        if (full) {
            full.classList.toggle('active', rec);
            full.innerHTML = rec ? `&#9209; Recording… ${secondsLeft}s` : '&#9210; Record Full Video';
        }
        if (sel && !(this.photoSelMode && this.photoSelTarget === 'video')) {
            sel.classList.toggle('disabled-look', rec);
            sel.innerHTML = rec ? '&#9210; Recording…' : '&#9210; Select &amp; Record Video';
        }
    }

    // ── Force matrices ────────────────────────────────────────────────────────

    private buildMatrixTable(tableId: string, kind: 'strength' | 'radius' | 'minRadius'): void {
        const table  = document.getElementById(tableId) as HTMLTableElement;
        const params = this.sim!.getParams();
        const n      = this.sim!.getNumTypes();

        const hdr = document.createElement('tr');
        hdr.innerHTML = '<th></th>' + TYPE_LABELS.slice(0, n).map((lbl, i) =>
            `<th><span class="type-pip" style="background:${TYPE_HEX[i]}"></span>${lbl}</th>`).join('');
        table.appendChild(hdr);

        for (let from = 0; from < n; from++) {
            const row = document.createElement('tr');
            row.innerHTML = `<th class="row-header"><span class="type-pip" style="background:${TYPE_HEX[from]}"></span>${TYPE_LABELS[from]}</th>`;
            for (let to = 0; to < n; to++) {
                const c  = params.forceMatrix[from]?.[to];
                const v  = kind === 'strength' ? (c?.strength ?? 0)
                         : kind === 'radius'   ? (c?.radius   ?? 100)
                         :                       (c?.minRadius ?? 0);
                const td = document.createElement('td');
                td.className = 'mcell';
                td.dataset.from = String(from); td.dataset.to = String(to); td.dataset.kind = kind;
                this.styleForceCellValue(td, kind, v);
                td.addEventListener('click', (e) => { e.stopPropagation(); this.openForceEditor(td); });
                row.appendChild(td);
            }
            table.appendChild(row);
        }
    }

    private styleForceCellValue(td: HTMLTableCellElement, kind: 'strength' | 'radius' | 'minRadius', v: number): void {
        td.style.background = kind === 'strength' ? strengthToColor(v)
                            : kind === 'radius'   ? radiusToColor(v)
                            :                       minRadiusToColor(v);
        td.textContent = kind === 'strength' ? v.toFixed(2) : String(Math.round(v));
        td.title       = String(v.toFixed(3));
    }

    private refreshForceMatrices(): void {
        (['strength-table', 'radius-table', 'min-radius-table'] as const).forEach(id => {
            (document.getElementById(id) as HTMLTableElement).innerHTML = '';
        });
        this.buildMatrixTable('strength-table',    'strength');
        this.buildMatrixTable('radius-table',      'radius');
        this.buildMatrixTable('min-radius-table',  'minRadius');
    }

    // ── Transform matrix ──────────────────────────────────────────────────────

    private buildTransformMatrix(): void {
        const table = document.getElementById('transform-table') as HTMLTableElement;
        table.innerHTML = '';
        const rules = this.sim!.getTransformRules();
        const n     = this.sim!.getNumTypes();

        const hdr = document.createElement('tr');
        hdr.innerHTML = '<th></th>' + TYPE_LABELS.slice(0, n).map((lbl, i) =>
            `<th><span class="type-pip" style="background:${TYPE_HEX[i]}"></span>${lbl}</th>`).join('');
        table.appendChild(hdr);

        for (let source = 0; source < n; source++) {
            const row = document.createElement('tr');
            row.innerHTML = `<th class="row-header"><span class="type-pip" style="background:${TYPE_HEX[source]}"></span>${TYPE_LABELS[source]}</th>`;
            for (let trigger = 0; trigger < n; trigger++) {
                const rule = rules[source * MAX_TYPES + trigger];
                const td   = document.createElement('td');
                td.className = 'tcell';
                td.dataset.source  = String(source);
                td.dataset.trigger = String(trigger);
                this.styleTransformCell(td, rule);
                td.addEventListener('click', (e) => { e.stopPropagation(); this.openTransformEditor(td); });
                row.appendChild(td);
            }
            table.appendChild(row);
        }
    }

    private styleTransformCell(td: HTMLTableCellElement, rule: TransformRule): void {
        const { bg, html } = transformCellHTML(rule);
        td.style.background = bg;
        td.innerHTML = html;
    }

    private refreshTransformMatrix(): void {
        this.buildTransformMatrix();
        this.buildPolePanel();
        this.syncPoleFrameButtons();
    }

    private syncPoleFrameButtons(): void {
        const world = this.sim?.getPoleFrame() ?? false;
        document.getElementById('poleFrameVelBtn')?.classList.toggle('selected', !world);
        document.getElementById('poleFrameWorldBtn')?.classList.toggle('selected', world);
    }

    private refreshMassTable(): void {
        const tbl = document.getElementById('mass-table') as HTMLTableElement;
        if (!tbl || !this.sim) return;
        tbl.innerHTML = '';
        const n      = this.sim.getNumTypes();
        const masses = this.sim.getTypeMass();
        for (let t = 0; t < n; t++) {
            const row   = tbl.insertRow();
            const swCell = row.insertCell();
            const swatch = document.createElement('span');
            swatch.className = 'mass-swatch';
            swatch.style.background = TYPE_HEX[t];
            swCell.appendChild(swatch);

            const labelCell = row.insertCell();
            labelCell.textContent = TYPE_LABELS[t];
            labelCell.style.color = TYPE_HEX[t];

            const inputCell = row.insertCell();
            const inp = document.createElement('input');
            inp.type  = 'number';
            inp.min   = '1'; inp.max = '8'; inp.step = '1';
            inp.value = String(masses[t] ?? 1);
            inp.addEventListener('change', () => {
                const v = Math.max(1, Math.min(8, Math.round(Number(inp.value))));
                inp.value = String(v);
                this.sim?.setTypeMass(t, v);
            });
            inputCell.appendChild(inp);

            const hintCell = row.insertCell();
            hintCell.style.fontSize = '9px';
            hintCell.style.color    = '#555';
            hintCell.textContent    = `mass ${masses[t] ?? 1}`;
            inp.addEventListener('change', () => { hintCell.textContent = `mass ${inp.value}`; });
        }
    }

    private patchHint(v: number): string {
        switch (v) {
            case 0:  return 'isotropic';
            case 2:  return 'chains';
            case 3:  return 'sheets';
            case 4:  return 'lattice';
            case 5:  return 'cluster';
            case 6:  return 'hex';
            default: return '';
        }
    }

    // Push the simulation's current bonding params back into the patchy sliders
    // (after import or a Randomize-bonding action).
    private syncPatchSliders(): void {
        if (!this.sim) return;
        const pp = this.sim.getPatchParams();
        const setSlider = (id: string, valId: string, v: number, txt: string) => {
            (document.getElementById(id) as HTMLInputElement).value = String(v);
            document.getElementById(valId)!.textContent = txt;
        };
        setSlider('patchRangeSlider',    'patchRangeValue',    pp.bondRange,    String(Math.round(pp.bondRange)));
        setSlider('patchWidthSlider',    'patchWidthValue',    pp.patchWidth,   String(Math.round(pp.patchWidth)));
        setSlider('patchIsoSlider',      'patchIsoValue',      pp.isoScale,     pp.isoScale.toFixed(2));
        setSlider('patchAngSlider',      'patchAngValue',      pp.angStiffness, pp.angStiffness.toFixed(2));
        const spinDamp = 1 - pp.angFriction;  // slider = 1 - stored multiplier
        setSlider('patchAngFricSlider',  'patchAngFricValue',  spinDamp, spinDamp.toFixed(2));
    }

    // Refresh everything in the patchy panel (sliders, per-type table, affinity grid).
    private refreshPatchUI(): void {
        this.syncPatchSliders();
        this.refreshPatchTable();
        this.buildAffinityMatrix();
    }

    private refreshPatchTable(): void {
        const tbl = document.getElementById('patch-table') as HTMLTableElement;
        if (!tbl || !this.sim) return;
        tbl.innerHTML = '';
        const n       = this.sim.getNumTypes();
        const patches = this.sim.getPatchCount();
        const strs    = this.sim.getPatchTypeBondStr();
        const dists   = this.sim.getPatchTypeBondDist();

        const hdr = tbl.insertRow();
        ['', 'type', 'valence', 'strength', 'rest'].forEach(h => {
            const th = document.createElement('th');
            th.textContent = h;
            th.style.cssText = 'font-size:9px;color:#666;text-align:left;font-weight:normal;padding:0 4px 2px';
            hdr.appendChild(th);
        });

        const numInput = (val: number, min: number, max: number, step: number, onset: (v: number) => void) => {
            const inp = document.createElement('input');
            inp.type = 'number';
            inp.min = String(min); inp.max = String(max); inp.step = String(step);
            inp.value = String(val);
            inp.addEventListener('change', () => {
                const v = Math.max(min, Math.min(max, Number(inp.value)));
                inp.value = String(v);
                onset(v);
            });
            return inp;
        };

        for (let t = 0; t < n; t++) {
            const row    = tbl.insertRow();
            const swCell = row.insertCell();
            const swatch = document.createElement('span');
            swatch.className = 'mass-swatch';
            swatch.style.background = TYPE_HEX[t];
            swCell.appendChild(swatch);

            const labelCell = row.insertCell();
            labelCell.textContent = TYPE_LABELS[t];
            labelCell.style.color = TYPE_HEX[t];

            // Valence
            const vInp = numInput(patches[t] ?? 0, 0, 6, 1, v => {
                let nv = Math.round(v);
                if (nv === 1) nv = 2;  // single patch is degenerate
                vInp.value = String(nv);
                this.sim?.setPatchCount(t, nv);
            });
            row.insertCell().appendChild(vInp);

            // Per-type bond strength
            row.insertCell().appendChild(
                numInput(Math.round((strs[t] ?? 0) * 100) / 100, 0, 2, 0.05, v => this.sim?.setPatchTypeBond(t, v, undefined)));

            // Per-type rest length
            row.insertCell().appendChild(
                numInput(Math.round(dists[t] ?? 26), 2, 150, 1, v => this.sim?.setPatchTypeBond(t, undefined, v)));
        }
    }

    // Bond-affinity matrix: who bonds whom. Click a cell to cycle 0 -> 0.5 -> 1.
    private buildAffinityMatrix(): void {
        const table = document.getElementById('affinity-table') as HTMLTableElement;
        if (!table || !this.sim) return;
        table.innerHTML = '';
        const n = this.sim.getNumTypes();

        const hdr = document.createElement('tr');
        hdr.innerHTML = '<th></th>' + TYPE_LABELS.slice(0, n).map((lbl, i) =>
            `<th><span class="type-pip" style="background:${TYPE_HEX[i]}"></span>${lbl}</th>`).join('');
        table.appendChild(hdr);

        const paint = (td: HTMLTableCellElement, v: number) => {
            td.textContent = v > 0 ? v.toFixed(1) : '';
            const a = Math.min(1, v);
            td.style.background = v > 0 ? `rgba(120,200,140,${0.15 + 0.5 * a})` : 'rgba(255,255,255,0.03)';
        };

        for (let from = 0; from < n; from++) {
            const row = document.createElement('tr');
            row.innerHTML = `<th class="row-header"><span class="type-pip" style="background:${TYPE_HEX[from]}"></span>${TYPE_LABELS[from]}</th>`;
            for (let to = 0; to < n; to++) {
                const td = document.createElement('td');
                td.className = 'mcell';
                paint(td, this.sim.getAffinity(from, to));
                td.addEventListener('click', (e) => {
                    e.stopPropagation();
                    const cur = this.sim!.getAffinity(from, to);
                    const next = cur <= 0 ? 0.5 : (cur < 1 ? 1 : 0);  // cycle 0 -> .5 -> 1 -> 0
                    this.sim!.setAffinity(from, to, next);
                    paint(td, next);
                });
                row.appendChild(td);
            }
            table.appendChild(row);
        }
    }

    // ── QFT panel (mode 6) ──────────────────────────────────────────────────────
    private refreshQftPanel(): void {
        if (!this.sim) return;
        (document.getElementById('qftFieldCount') as HTMLInputElement).value = String(this.sim.getNumFields());
        this.buildQftFieldTable();
        this.buildQftChargeMatrix();
    }

    private buildQftFieldTable(): void {
        const tbl = document.getElementById('qft-field-table') as HTMLTableElement;
        if (!tbl || !this.sim) return;
        tbl.innerHTML = '';
        const fields = this.sim.getFields();
        const hdr = tbl.insertRow();
        ['field', 'range', 'strength'].forEach(h => {
            const th = document.createElement('th');
            th.textContent = h;
            th.style.cssText = 'font-size:9px;color:#666;text-align:left;font-weight:normal;padding:0 4px 2px';
            hdr.appendChild(th);
        });
        const numInput = (val: number, step: number, onset: (v: number) => void) => {
            const inp = document.createElement('input');
            inp.type = 'number'; inp.step = String(step); inp.value = String(val);
            inp.addEventListener('change', () => onset(Number(inp.value) || 0));
            return inp;
        };
        fields.forEach((fd, f) => {
            const row = tbl.insertRow();
            const nameCell = row.insertCell();
            nameCell.textContent = fd.name;
            nameCell.style.cssText = f === 0 ? 'color:#e88;font-weight:bold' : 'color:#9bd';
            row.insertCell().appendChild(numInput(Math.round(fd.range), 1, v => this.sim?.setFieldParam(f, Math.max(0, v), undefined)));
            row.insertCell().appendChild(numInput(Math.round(fd.strength * 100) / 100, 0.1, v => this.sim?.setFieldParam(f, undefined, v)));
        });
    }

    private buildQftChargeMatrix(): void {
        const table = document.getElementById('qft-charge-table') as HTMLTableElement;
        if (!table || !this.sim) return;
        table.innerHTML = '';
        const n = this.sim.getNumTypes();
        const fields = this.sim.getFields();
        const charges = this.sim.getCharges();

        const hdr = document.createElement('tr');
        hdr.innerHTML = '<th></th>' + fields.map(fd => `<th>${fd.name}</th>`).join('');
        table.appendChild(hdr);

        for (let t = 0; t < n; t++) {
            const row = document.createElement('tr');
            const th = document.createElement('th');
            th.className = 'row-header';
            th.innerHTML = `<span class="type-pip" style="background:${TYPE_HEX[t]}"></span>${TYPE_LABELS[t]}`;
            row.appendChild(th);
            for (let f = 0; f < fields.length; f++) {
                const td = document.createElement('td');
                const inp = document.createElement('input');
                inp.type = 'number'; inp.step = '0.1';
                inp.value = String(Math.round((charges[t]?.[f] ?? 0) * 100) / 100);
                inp.style.cssText = 'width:42px;font-size:9px;background:#1a1a1a;color:#ddd;border:1px solid #333;border-radius:3px;';
                inp.addEventListener('change', () => this.sim?.setCharge(t, f, Number(inp.value) || 0));
                td.appendChild(inp);
                row.appendChild(td);
            }
            table.appendChild(row);
        }
    }

    // ── DNF rules panel (mode 5 / Transform #2) ─────────────────────────────────
    private refreshDnfPanel(): void {
        if (!this.sim) return;
        const rate = this.sim.getMaxTransformRate();
        const slider = document.getElementById('dnfMaxSlider') as HTMLInputElement | null;
        if (slider) { slider.value = String(rate); document.getElementById('dnfMaxValue')!.textContent = rate.toFixed(2); }
        this.buildDnfList();
    }

    // Human-readable "C1: G>0.25  C2: B>1  →  P if (C1 and C2)" for one type.
    private dnfSummary(dt: { conditions: DnfCondition[]; rules: DnfRule[] }): string {
        if (!dt.conditions.length && !dt.rules.length) return '<span class="empty">(no rules — never transforms)</span>';
        const conds = dt.conditions.map((c, i) =>
            `<span class="dnf-cref">C${i + 1}</span>:<span style="color:${TYPE_HEX[c.trigger]}">${TYPE_LABELS[c.trigger]}</span>${DNF_OPSYM[c.op]}${c.threshold}`).join('  ');
        const rules = dt.rules.length
            ? dt.rules.map(r => `→<span style="color:${TYPE_HEX[r.target]}">${TYPE_LABELS[r.target]}</span> if <code>${r.expr || '∅'}</code>`).join('  ')
            : '<span class="empty">(no transform)</span>';
        return `${conds}${conds ? '  ' : ''}${rules}`;
    }

    private buildDnfList(): void {
        const list = document.getElementById('dnf-list');
        if (!list || !this.sim) return;
        list.innerHTML = '';
        const n = this.sim.getNumTypes();
        const types = this.sim.getDnfTypes();

        for (let t = 0; t < n; t++) {
            const row = document.createElement('div');
            row.className = 'dnf-row';
            row.innerHTML =
                `<span class="dnf-src" style="color:${TYPE_HEX[t]}">${TYPE_LABELS[t]}</span>` +
                `<span class="dnf-summary">${this.dnfSummary(types[t] ?? { conditions: [], rules: [] })}</span>`;
            row.addEventListener('click', () => {
                this.dnfEditType = this.dnfEditType === t ? null : t;
                this.buildDnfList();
            });
            list.appendChild(row);

            if (this.dnfEditType === t) {
                const editor = this.buildDnfEditor(t);
                editor.addEventListener('click', (e) => e.stopPropagation());
                list.appendChild(editor);
            }
        }
    }

    private dnfTypeSelect(val: number, onset: (v: number) => void): HTMLSelectElement {
        const n = this.sim!.getNumTypes();
        const sel = document.createElement('select');
        for (let i = 0; i < n; i++) {
            const o = document.createElement('option');
            o.value = String(i); o.textContent = TYPE_LABELS[i];
            if (i === val) o.selected = true;
            sel.appendChild(o);
        }
        sel.addEventListener('change', () => onset(Number(sel.value)));
        return sel;
    }

    // Inline editor for one source type: a list of force conditions plus a list of
    // transform rules (target + boolean expression over the conditions). Mutates a
    // working copy and commits to the sim on every change.
    private buildDnfEditor(t: number): HTMLElement {
        const dt = this.sim!.getDnfTypes()[t] ?? { conditions: [], rules: [] };
        const conditions: DnfCondition[] = dt.conditions.map(c => ({ ...c }));
        const rules: DnfRule[] = dt.rules.map(r => ({ target: r.target, expr: r.expr, rpn: r.rpn.slice() }));
        const commit = () => this.sim!.setDnfType(t, conditions, rules);
        const rebuild = () => { commit(); const fresh = this.buildDnfEditor(t); wrap.replaceWith(fresh); fresh.addEventListener('click', e => e.stopPropagation()); };

        const wrap = document.createElement('div');
        wrap.className = 'dnf-edit';

        // ── Conditions ───────────────────────────────────────────────────────
        const cHead = document.createElement('div');
        cHead.className = 'dnf-section'; cHead.textContent = 'Conditions (force from a type vs threshold)';
        wrap.appendChild(cHead);

        conditions.forEach((cd, ci) => {
            const line = document.createElement('div');
            line.className = 'dnf-line';
            const tag = document.createElement('span');
            tag.className = 'dnf-cref'; tag.textContent = `C${ci + 1}`;
            line.appendChild(tag);
            line.appendChild(document.createTextNode('force of'));
            line.appendChild(this.dnfTypeSelect(cd.trigger, v => { cd.trigger = v; commit(); }));

            const opSel = document.createElement('select');
            DNF_OPSYM.forEach((sym, v) => {
                const o = document.createElement('option');
                o.value = String(v); o.textContent = sym;
                if (v === cd.op) o.selected = true;
                opSel.appendChild(o);
            });
            opSel.addEventListener('change', () => { cd.op = Number(opSel.value) as 0 | 1 | 2 | 3; commit(); });
            line.appendChild(opSel);

            const thr = document.createElement('input');
            thr.type = 'number'; thr.step = '0.05'; thr.min = '-50'; thr.max = '50'; thr.value = String(cd.threshold);
            thr.addEventListener('change', () => { cd.threshold = Number(thr.value) || 0; thr.value = String(cd.threshold); commit(); });
            line.appendChild(thr);

            const del = document.createElement('button');
            del.textContent = '✕'; del.title = 'remove condition';
            del.addEventListener('click', () => { conditions.splice(ci, 1); rebuild(); });
            line.appendChild(del);
            wrap.appendChild(line);
        });

        if (conditions.length < MAX_DNF_CONDITIONS) {
            const add = document.createElement('button');
            add.textContent = '+ condition';
            add.style.alignSelf = 'flex-start';
            add.addEventListener('click', () => { conditions.push({ trigger: 0, op: 0, threshold: 0.25 }); rebuild(); });
            wrap.appendChild(add);
        }

        // ── Transform rules ──────────────────────────────────────────────────
        const rHead = document.createElement('div');
        rHead.className = 'dnf-section'; rHead.textContent = 'Transforms (first true rule wins)';
        wrap.appendChild(rHead);

        rules.forEach((ru, ri) => {
            const line = document.createElement('div');
            line.className = 'dnf-line';
            line.appendChild(document.createTextNode('become'));
            line.appendChild(this.dnfTypeSelect(ru.target, v => { ru.target = v; commit(); }));
            line.appendChild(document.createTextNode('if'));

            const expr = document.createElement('input');
            expr.type = 'text'; expr.className = 'dnf-expr';
            expr.placeholder = 'e.g. (C1 and C2) or C3';
            expr.value = ru.expr;
            const status = document.createElement('span');
            status.className = 'dnf-status';
            const validate = () => {
                const res = compileBoolExpr(expr.value, conditions.length);
                ru.expr = expr.value;
                ru.rpn = res.rpn;
                if (res.error) { status.textContent = '✗ ' + res.error; status.style.color = '#e66'; }
                else if (!expr.value.trim()) { status.textContent = ''; }
                else { status.textContent = '✓'; status.style.color = '#6c6'; }
                commit();
            };
            expr.addEventListener('input', validate);
            line.appendChild(expr);

            const del = document.createElement('button');
            del.textContent = '✕'; del.title = 'remove transform';
            del.addEventListener('click', () => { rules.splice(ri, 1); rebuild(); });
            line.appendChild(del);
            wrap.appendChild(line);

            const statusLine = document.createElement('div');
            statusLine.className = 'dnf-line';
            statusLine.style.minHeight = '12px';
            statusLine.appendChild(status);
            wrap.appendChild(statusLine);
            validate();
        });

        if (rules.length < MAX_DNF_RULES) {
            const n = this.sim!.getNumTypes();
            const add = document.createElement('button');
            add.textContent = '+ transform';
            add.style.alignSelf = 'flex-start';
            add.addEventListener('click', () => {
                rules.push({ target: (t + 1) % Math.max(1, n), expr: conditions.length ? 'C1' : '', rpn: [] });
                rebuild();
            });
            wrap.appendChild(add);
        }

        return wrap;
    }

    private buildPolePanel(): void {
        const list = document.getElementById('pole-list')!;
        list.innerHTML = '';
        if (!this.sim) return;
        const configs = this.sim.getPoleConfigs();
        const n       = this.sim.getNumTypes();

        for (let t = 0; t < n; t++) {
            const { poleCount, signBits } = configs[t];
            const row = document.createElement('div');
            row.className = 'pole-row';

            const pip = document.createElement('span');
            pip.style.cssText = `color:${TYPE_HEX[t]};font-weight:bold;width:14px;flex-shrink:0;font-size:11px;`;
            pip.textContent = TYPE_LABELS[t];
            row.appendChild(pip);

            const countBtns = document.createElement('div');
            countBtns.className = 'pole-count-btns';
            const poleOpts   = [0, 2, 3, 4, 5, 6];
            const poleLabels = ['M', '2', '3', '4', '5', '6'];

            poleOpts.forEach((pc, i) => {
                const btn = document.createElement('button');
                btn.className = `pole-count-btn${poleCount === pc ? ' selected' : ''}`;
                btn.textContent = poleLabels[i];
                btn.title = pc === 0 ? 'Monopole' : pc === 2 ? 'Dipole' : `${pc}-pole`;
                btn.addEventListener('click', () => {
                    countBtns.querySelectorAll('.pole-count-btn').forEach(b => b.classList.remove('selected'));
                    btn.classList.add('selected');
                    const sb = pc >= 3 ? Math.floor(Math.random() * (1 << pc)) : 0;
                    this.sim!.setPoleConfig(t, pc, sb);
                    this.rebuildPoleSignBtns(row, t, pc, sb);
                });
                countBtns.appendChild(btn);
            });
            row.appendChild(countBtns);

            const signContainer = document.createElement('div');
            signContainer.className = 'pole-signs';
            row.appendChild(signContainer);
            this.rebuildPoleSignBtns(row, t, poleCount, signBits);

            list.appendChild(row);
        }
    }

    private rebuildPoleSignBtns(row: HTMLElement, typeId: number, poleCount: number, signBits: number): void {
        const signContainer = row.querySelector<HTMLElement>('.pole-signs')!;
        signContainer.innerHTML = '';
        if (poleCount < 3) return;
        for (let k = 0; k < poleCount; k++) {
            const isPos = (signBits & (1 << k)) !== 0;
            const btn   = document.createElement('button');
            btn.className = `pole-sign-btn ${isPos ? 'pole-sign-pos' : 'pole-sign-neg'}`;
            btn.textContent = isPos ? '+' : '−';
            const capturedBits = signBits;
            btn.addEventListener('click', () => {
                const newBits = capturedBits ^ (1 << k);
                this.sim!.setPoleConfig(typeId, poleCount, newBits);
                this.rebuildPoleSignBtns(row, typeId, poleCount, newBits);
            });
            signContainer.appendChild(btn);
        }
    }

    // ── Force cell editor ─────────────────────────────────────────────────────

    private openForceEditor(td: HTMLTableCellElement): void {
        this.closeForceEditor();
        this.closeTransformEditor();
        const from = parseInt(td.dataset.from!);
        const to   = parseInt(td.dataset.to!);
        const kind = td.dataset.kind as 'strength' | 'radius' | 'minRadius';
        const c    = this.sim!.getParams().forceMatrix[from]?.[to];
        const cur  = kind === 'strength' ? (c?.strength  ?? 0)
                   : kind === 'radius'   ? (c?.radius    ?? 100)
                   :                       (c?.minRadius ?? 0);

        const editor   = document.getElementById('cell-editor')!;
        const title    = document.getElementById('cell-editor-title')!;
        const slider   = document.getElementById('cell-slider')  as HTMLInputElement;
        const display  = document.getElementById('cell-value-display')!;
        const row2     = document.getElementById('cell-editor-row2')!;
        const slider2  = document.getElementById('cell-slider2') as HTMLInputElement;
        const display2 = document.getElementById('cell-value-display2')!;

        const pip = (i: number) =>
            `<span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:${TYPE_HEX[i]};margin-right:2px;vertical-align:middle"></span>`;
        const kindLabel = kind === 'minRadius' ? 'min radius' : kind;
        title.innerHTML = `${pip(from)}${TYPE_LABELS[from]} → ${pip(to)}${TYPE_LABELS[to]} <span style="color:#888">${kindLabel}</span>`;

        if (kind === 'strength') {
            Object.assign(slider, { min: '-5', max: '5', step: '0.05' });
            display.textContent = cur.toFixed(2);
            row2.style.display = 'none';
        } else if (kind === 'radius') {
            Object.assign(slider, { min: '10', max: '250', step: '1' });
            display.textContent = String(Math.round(cur));
            const curMin = c?.minRadius ?? 0;
            slider2.min   = '0';
            slider2.max   = String(Math.max(0, Math.round(cur) - 1));
            slider2.step  = '1';
            slider2.value = String(Math.min(curMin, Math.max(0, Math.round(cur) - 1)));
            display2.textContent = String(Math.round(parseFloat(slider2.value)));
            row2.style.display = '';
        } else {
            // minRadius: single slider, range [0, maxRadius - 1]
            const maxR = c?.radius ?? 100;
            Object.assign(slider, { min: '0', max: String(Math.max(0, Math.round(maxR) - 1)), step: '1' });
            display.textContent = String(Math.round(cur));
            row2.style.display = 'none';
        }
        slider.value = String(cur);

        const rect = td.getBoundingClientRect();
        let left = rect.left, top = rect.bottom + 4;
        if (left + 180 > window.innerWidth)  left = window.innerWidth - 184;
        if (top  + 100 > window.innerHeight) top  = rect.top - (kind === 'radius' ? 104 : 74);
        editor.style.left = `${left}px`; editor.style.top = `${top}px`;
        editor.classList.add('visible');
        this.editorCell = td;

        slider.oninput = () => {
            const v = parseFloat(slider.value);
            display.textContent = kind === 'strength' ? v.toFixed(2) : String(Math.round(v));
            this.styleForceCellValue(td, kind, v);
            this.applyForceCell(from, to, kind, v);
            if (kind === 'radius') {
                // Keep min slider max in sync with max radius
                slider2.max = String(Math.max(0, Math.round(v) - 1));
                if (parseFloat(slider2.value) >= v) {
                    const clamped = Math.max(0, Math.round(v) - 1);
                    slider2.value = String(clamped);
                    display2.textContent = String(clamped);
                    this.applyForceCell(from, to, 'minRadius', clamped);
                    const minCell = document.querySelector(`#min-radius-table td[data-from="${from}"][data-to="${to}"]`) as HTMLTableCellElement | null;
                    if (minCell) this.styleForceCellValue(minCell, 'minRadius', clamped);
                }
            }
        };

        slider2.oninput = () => {
            const v = parseFloat(slider2.value);
            display2.textContent = String(Math.round(v));
            this.applyForceCell(from, to, 'minRadius', v);
            const minCell = document.querySelector(`#min-radius-table td[data-from="${from}"][data-to="${to}"]`) as HTMLTableCellElement | null;
            if (minCell) this.styleForceCellValue(minCell, 'minRadius', v);
        };

        const dismiss = (e: MouseEvent) => {
            if (!editor.contains(e.target as Node) && e.target !== td) this.closeForceEditor();
        };
        this.editorDismiss = dismiss;
        setTimeout(() => document.addEventListener('click', dismiss), 0);
    }

    private closeForceEditor(): void {
        document.getElementById('cell-editor')!.classList.remove('visible');
        if (this.editorDismiss) { document.removeEventListener('click', this.editorDismiss); this.editorDismiss = null; }
        this.editorCell = null;
    }

    private applyForceCell(from: number, to: number, kind: 'strength' | 'radius' | 'minRadius', value: number): void {
        if (!this.sim) return;
        const fm = this.sim.getParams().forceMatrix;
        if (!fm[from]) fm[from] = {};
        if (!fm[from][to]) fm[from][to] = { strength: 0, radius: 100, minRadius: 0 };
        if (kind === 'strength')  fm[from][to].strength  = value;
        else if (kind === 'radius') fm[from][to].radius  = value;
        else                      fm[from][to].minRadius = value;
        this.sim.updateParams({ forceMatrix: fm });
    }

    // ── Transform rule editor ─────────────────────────────────────────────────

    private buildTypePicker(containerId: string, selectedType: number, n: number, onChange: (t: number) => void): void {
        const container = document.getElementById(containerId)!;
        container.innerHTML = '';
        for (let t = 0; t < n; t++) {
            const btn = document.createElement('button');
            btn.className = `type-dot-btn${t === selectedType ? ' selected' : ''}`;
            btn.style.background = TYPE_HEX[t];
            btn.title = TYPE_LABELS[t];
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                container.querySelectorAll('.type-dot-btn').forEach(b => b.classList.remove('selected'));
                btn.classList.add('selected');
                onChange(t);
            });
            container.appendChild(btn);
        }
    }

    private openTransformEditor(td: HTMLTableCellElement): void {
        this.closeTransformEditor();
        this.closeForceEditor();

        const source  = parseInt(td.dataset.source!);
        const trigger = parseInt(td.dataset.trigger!);
        const n       = this.sim!.getNumTypes();
        this.teSource  = source;
        this.teTrigger = trigger;

        const rules = this.sim!.getTransformRules();
        const rule: TransformRule = { ...rules[source * MAX_TYPES + trigger] };

        const editor = document.getElementById('transform-editor')!;
        const title  = document.getElementById('te-title')!;

        const pip = (i: number) =>
            `<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${TYPE_HEX[i]};margin-right:3px;vertical-align:middle"></span>`;
        title.innerHTML = `${pip(source)}${TYPE_LABELS[source]} reacts to ${pip(trigger)}${TYPE_LABELS[trigger]}`;

        const upperEnabled   = document.getElementById('te-upper-enabled')   as HTMLInputElement;
        const upperThreshold = document.getElementById('te-upper-threshold') as HTMLInputElement;
        const upperGt        = document.getElementById('te-upper-gt')        as HTMLButtonElement;
        const upperGte       = document.getElementById('te-upper-gte')       as HTMLButtonElement;
        const lowerEnabled   = document.getElementById('te-lower-enabled')   as HTMLInputElement;
        const lowerThreshold = document.getElementById('te-lower-threshold') as HTMLInputElement;
        const lowerLt        = document.getElementById('te-lower-lt')        as HTMLButtonElement;
        const lowerLte       = document.getElementById('te-lower-lte')       as HTMLButtonElement;

        upperEnabled.checked = rule.upperEnabled;
        upperThreshold.value = rule.upperThreshold.toFixed(3);
        upperGt.classList.toggle('selected',  !rule.upperInclusive);
        upperGte.classList.toggle('selected',  rule.upperInclusive);
        lowerEnabled.checked = rule.lowerEnabled;
        lowerThreshold.value = rule.lowerThreshold.toFixed(3);
        lowerLt.classList.toggle('selected',  !rule.lowerInclusive);
        lowerLte.classList.toggle('selected',  rule.lowerInclusive);

        const applyRule = () => {
            rule.upperEnabled   = upperEnabled.checked;
            rule.upperThreshold = parseFloat(upperThreshold.value) || 0;
            rule.lowerEnabled   = lowerEnabled.checked;
            rule.lowerThreshold = parseFloat(lowerThreshold.value) || 0;
            this.sim!.setTransformRule(source, trigger, rule);
            this.styleTransformCell(td, rule);
        };

        upperEnabled.onchange   = applyRule;
        lowerEnabled.onchange   = applyRule;
        upperThreshold.oninput  = applyRule;
        upperThreshold.onchange = applyRule;
        lowerThreshold.oninput  = applyRule;
        lowerThreshold.onchange = applyRule;

        upperGt.onclick  = () => { rule.upperInclusive = false; upperGt.classList.add('selected');  upperGte.classList.remove('selected'); applyRule(); };
        upperGte.onclick = () => { rule.upperInclusive = true;  upperGte.classList.add('selected'); upperGt.classList.remove('selected');  applyRule(); };
        lowerLt.onclick  = () => { rule.lowerInclusive = false; lowerLt.classList.add('selected');  lowerLte.classList.remove('selected'); applyRule(); };
        lowerLte.onclick = () => { rule.lowerInclusive = true;  lowerLte.classList.add('selected'); lowerLt.classList.remove('selected');  applyRule(); };

        // Clamp existing targets to valid range before showing picker
        rule.upperTarget = Math.min(rule.upperTarget, n - 1);
        rule.lowerTarget = Math.min(rule.lowerTarget, n - 1);

        this.buildTypePicker('te-upper-picker', rule.upperTarget, n, (t) => {
            rule.upperTarget = t; applyRule();
        });
        this.buildTypePicker('te-lower-picker', rule.lowerTarget, n, (t) => {
            rule.lowerTarget = t; applyRule();
        });

        const rect = td.getBoundingClientRect();
        let left = rect.left, top = rect.bottom + 4;
        if (left + 230 > window.innerWidth)  left = window.innerWidth - 234;
        if (top  + 220 > window.innerHeight) top  = rect.top - 224;
        editor.style.left = `${left}px`; editor.style.top = `${top}px`;
        editor.classList.add('visible');

        const dismiss = (e: MouseEvent) => {
            if (!editor.contains(e.target as Node) && e.target !== td) this.closeTransformEditor();
        };
        this.teDismiss = dismiss;
        setTimeout(() => document.addEventListener('click', dismiss), 0);
    }

    private closeTransformEditor(): void {
        document.getElementById('transform-editor')!.classList.remove('visible');
        if (this.teDismiss) { document.removeEventListener('click', this.teDismiss); this.teDismiss = null; }
        this.teSource = this.teTrigger = -1;
    }

    // ── Export / Import ───────────────────────────────────────────────────────

    private exportConfig(): void {
        if (!this.sim) return;
        const state = this.sim.exportState();
        const blob  = new Blob([JSON.stringify(state, null, 2)], { type: 'application/json' });
        const url   = URL.createObjectURL(blob);
        const a     = document.createElement('a');
        a.href = url; a.download = `particle-life-${Date.now()}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    private importConfig(file: File): void {
        const reader = new FileReader();
        reader.onload = (e) => {
            try {
                const state = JSON.parse(e.target!.result as string);
                this.applyImportedState(state);
            } catch {
                alert('Failed to parse config file — make sure it is a valid JSON export from this app.');
            }
        };
        reader.readAsText(file);
    }

    private applyImportedState(state: any): void {
        if (!this.sim) return;
        this.closeForceEditor();
        this.closeTransformEditor();

        // The world size is applied to the sim (below); the canvas just fills the window.
        const w = Number(state.worldWidth)  || 1600;
        const h = Number(state.worldHeight) || 900;
        this.fitCanvas();

        this.sim.importState(state);

        // Sync all UI controls to imported values
        const n = this.sim.getNumTypes();
        (document.getElementById('typeCount')    as HTMLInputElement).value = String(n);
        (document.getElementById('particleCount') as HTMLInputElement).value = String(this.sim.getParams().particleCount);
        (document.getElementById('worldW')        as HTMLInputElement).value = String(w);
        (document.getElementById('worldH')        as HTMLInputElement).value = String(h);
        (document.getElementById('speed')         as HTMLInputElement).value = String(this.sim.getParams().simulationSpeed);
        document.getElementById('speedValue')!.textContent = `${this.sim.getParams().simulationSpeed.toFixed(1)}×`;
        document.getElementById('particleCountDisplay')!.textContent = String(this.sim.getParams().particleCount);

        const simMode  = this.sim.getSimMode()  as 0 | 1 | 2 | 3 | 4 | 5 | 6;
        const edgeMode = this.sim.getEdgeMode() as 0 | 1;
        this.setSimMode(simMode);
        document.getElementById('edgeLoopBtn')!.classList.toggle('selected', edgeMode === 0);
        document.getElementById('edgeOpenBtn')!.classList.toggle('selected', edgeMode === 1);

        const bg = this.sim.getBackgroundColor();
        const bgHex = rgbToHex(bg.r, bg.g, bg.b);
        (document.getElementById('bgColorPicker') as HTMLInputElement).value = bgHex;

        const sat = this.sim.getColorSaturation();
        (document.getElementById('satSlider') as HTMLInputElement).value = String(sat);
        document.getElementById('satValue')!.textContent = sat.toFixed(2);

        const glow = this.sim.getParticleGlow();
        (document.getElementById('glowSlider') as HTMLInputElement).value = String(glow);
        document.getElementById('glowValue')!.textContent = glow.toFixed(2);

        const alpha = this.sim.getParticleAlpha();
        (document.getElementById('alphaSlider') as HTMLInputElement).value = String(alpha);
        document.getElementById('alphaValue')!.textContent = alpha.toFixed(2);

        const addStr = this.sim.getAdditiveStrength();
        (document.getElementById('addStrSlider') as HTMLInputElement).value = String(addStr);
        document.getElementById('addStrValue')!.textContent = addStr.toFixed(2);

        this.setBlendMode(this.sim.getBlendMode() as 0 | 1);
        this.setShapeMode(this.sim.getShapeMode() as 0 | 1);

        // Slider shows drag = 1 - stored multiplier (see input handler).
        const drag = 1 - this.sim.getFriction();
        (document.getElementById('frictionSlider') as HTMLInputElement).value = String(drag);
        document.getElementById('frictionValue')!.textContent = drag.toFixed(2);

        const maxTransform = this.sim.getMaxTransformRate();
        (document.getElementById('maxTransformSlider') as HTMLInputElement).value = String(maxTransform);
        document.getElementById('maxTransformValue')!.textContent = maxTransform.toFixed(2);

        this.refreshPatchUI();
        this.refreshDnfPanel();
        this.refreshQftPanel();

        this.refreshForceMatrices();
        this.refreshTransformMatrix();  // also calls buildPolePanel()
        this.updateZoomDisplay();
    }

    // ── Animation loop ────────────────────────────────────────────────────────

    private updateStats(): void {
        this.frameCount++;
        const now = performance.now(), dt = now - this.lastTime;
        if (dt >= 1000) {
            const fps = Math.round(this.frameCount * 1000 / dt);
            document.getElementById('fps')!.textContent = String(fps);
            this.frameCount = 0; this.lastTime = now;

            const warnEl = document.getElementById('fps-warn')!;
            if (fps < this.autoPauseMinFps) {
                warnEl.style.display = 'inline';
                this.lowFpsFrames++;
                if (this.autoPause && this.lowFpsFrames >= 2 && !this.sim?.isPaused_()) {
                    this.sim?.togglePause();
                    this.syncPauseButton();
                }
            } else {
                warnEl.style.display = 'none';
                this.lowFpsFrames = 0;
            }
        }
        if (this.sim) {
            document.getElementById('time')!.textContent = `${this.sim.getTime().toFixed(1)}s`;
            document.getElementById('particleCountDisplay')!.textContent = String(this.sim.getParams().particleCount);
        }
    }

    private startLoop(): void {
        const tick = () => {
            if (this.activeCursorTool === 'force' && this.cursorMouseButtons !== 0 && this.sim) {
                const { wx, wy } = this.screenToWorld(this.cursorMouseX, this.cursorMouseY);
                const dir = (this.cursorMouseButtons & 1) ? 1 : -1;
                this.sim.applyCursorForce(wx, wy, this.brushWorldRadius, this.forceStrength * dir);
            }
            if (this.activeCursorTool === 'paint' && this.cursorMouseButtons !== 0 && this.sim) {
                const { wx, wy } = this.screenToWorld(this.cursorMouseX, this.cursorMouseY);
                if (this.cursorMouseButtons & 1) {
                    const r = this.brushWorldRadius;
                    const count = Math.max(1, Math.round(this.paintRate * r * r / 10000));
                    this.sim.spawnParticlesInBrush(wx, wy, r, this.paintTypeId, count);
                } else if (this.cursorMouseButtons & 4) {
                    this.sim.eraseParticlesInBrush(wx, wy, this.brushWorldRadius, this.paintRate / 1000, this.paintTypeId);
                }
            }
            if (this.activeCursorTool === 'erase' && this.cursorMouseButtons !== 0 && this.sim) {
                const { wx, wy } = this.screenToWorld(this.cursorMouseX, this.cursorMouseY);
                if (this.cursorMouseButtons & 1) {
                    this.sim.eraseParticlesInBrush(wx, wy, this.brushWorldRadius, this.paintRate / 1000, -1);
                } else if (this.cursorMouseButtons & 4) {
                    this.sim.eraseParticlesInBrush(wx, wy, this.brushWorldRadius, 1.0, -1);
                }
            }
            // During an offline render the render loop drives the sim itself; skip
            // the normal step so we don't advance/capture frames twice.
            if (!this.isRenderingVideo) this.sim?.update();
            this.updateStats();
            this.drawOverlay();
            // Blit the freshly-rendered frame into the video record canvas, if active.
            this.recordCopy?.();
            this.animId = requestAnimationFrame(tick);
        };
        this.animId = requestAnimationFrame(tick);
    }

    // ── Entry point ───────────────────────────────────────────────────────────

    async initialize(): Promise<void> {
        const worldW = parseInt((document.getElementById('worldW') as HTMLInputElement).value) || 4800;
        const worldH = parseInt((document.getElementById('worldH') as HTMLInputElement).value) || 2700;
        this.fitCanvas();   // canvas (viewport) fills the window

        this.sim = new ParticleSimulation(this.canvas);
        try {
            await this.sim.initialize();
            this.sim.setWorldSize(worldW, worldH);   // world is independent of the viewport
        } catch (err) {
            const msg = err instanceof Error ? err.message : String(err);
            document.body.innerHTML = `<div style="color:#f44;padding:40px;font-family:monospace;font-size:14px;">
                WebGPU init failed:<br><br>${msg}<br><br>
                Make sure you're using Chrome/Edge 113+ with WebGPU enabled.</div>`;
            return;
        }

        document.getElementById('particleCountDisplay')!.textContent = String(this.sim.getParams().particleCount);
        this.updateZoomDisplay();

        this.sim.onDiagnosticUpdate = (data) => this.updateDiagPanel(data);

        this.buildMatrixTable('strength-table',   'strength');
        this.buildMatrixTable('radius-table',     'radius');
        this.buildMatrixTable('min-radius-table', 'minRadius');
        this.buildTransformMatrix();
        this.buildPolePanel();
        this.refreshPaintTypePicker();
        this.syncExportButtons();
        this.syncPoleFrameButtons();

        this.startLoop();
    }

    destroy(): void { if (this.animId !== null) cancelAnimationFrame(this.animId); }
}

const app = new ParticleLifeApp();
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => app.initialize());
} else {
    app.initialize();
}
