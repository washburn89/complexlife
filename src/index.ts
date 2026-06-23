import { ParticleSimulation, TransformRule, MAX_TYPES, TYPE_COLORS_HEX, DiagnosticData } from './simulation';

const TYPE_LABELS = [
    'R', 'G', 'B', 'Y', 'M', 'C', 'O', 'P', 'K', 'S',
    'W', 'A', 'L', 'T', 'I', 'N', 'Q', 'H', 'F', 'Z',
];
const TYPE_HEX = TYPE_COLORS_HEX;

// ── Color helpers ─────────────────────────────────────────────────────────────

function strengthToColor(v: number): string {
    const c = Math.max(-1, Math.min(1, v));
    if (c < 0) { const t = Math.round(-c * 180); return `rgb(${t},0,0)`; }
    const t = Math.round(c * 180); return `rgb(0,${t},0)`;
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
    private activeCursorTool:     'none' | 'grab' | 'force' = 'none';
    private brushWorldRadius      = 150;
    private forceStrength         = 2.0;
    private cursorMouseX          = -9999;
    private cursorMouseY          = -9999;
    private cursorMouseButtons    = 0;   // bitmask: bit 0 = left, bit 2 = right
    private cursorPanelCollapsed  = false;

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

    private fitCanvas(): void {
        const w = this.canvas.width, h = this.canvas.height;
        if (!w || !h) return;
        const scale = Math.min(window.innerWidth / w, window.innerHeight / h);
        this.canvas.style.width  = `${w * scale}px`;
        this.canvas.style.height = `${h * scale}px`;
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

        // Randomize transform rules
        document.getElementById('randomizeTransformBtn')!.addEventListener('click', () => {
            this.sim?.randomizeTransformRules();
            this.refreshTransformMatrix();
        });

        // Randomize poles
        document.getElementById('randomizePolesBtn')!.addEventListener('click', () => {
            this.sim?.randomizePoles();
            this.buildPolePanel();
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

        // Friction
        const frictionSlider = document.getElementById('frictionSlider') as HTMLInputElement;
        frictionSlider.addEventListener('input', () => {
            const v = parseFloat(frictionSlider.value);
            document.getElementById('frictionValue')!.textContent = v.toFixed(2);
            this.sim?.setFriction(v);
        });

        // Max transform rate
        const maxTransformSlider = document.getElementById('maxTransformSlider') as HTMLInputElement;
        maxTransformSlider.addEventListener('input', () => {
            const v = parseFloat(maxTransformSlider.value);
            document.getElementById('maxTransformValue')!.textContent = v.toFixed(2);
            this.sim?.setMaxTransformRate(v);
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

        // Bottom bar
        document.getElementById('btmPauseBtn')!.addEventListener('click', () => {
            if (!this.sim) return;
            this.sim.togglePause();
            this.syncPauseButton();
        });
        document.getElementById('btmTrackBtn')!.addEventListener('click', () => this.handleTrackButton());
        document.getElementById('btmInspectBtn')!.addEventListener('click', () => {
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
        btn.textContent = paused ? 'Resume' : 'Pause';
        btn.classList.toggle('active', paused);
        const btm = document.getElementById('btmPauseBtn')!;
        btm.textContent = paused ? '&#9654; Resume' : '&#9646;&#9646; Pause';
        btm.innerHTML   = paused ? '&#9654; Resume' : '&#9646;&#9646; Pause';
        btm.classList.toggle('active', paused);
    }

    private setSimMode(mode: 0 | 1 | 2): void {
        this.sim?.setSimMode(mode);
        document.getElementById('mode0-btn')!.classList.toggle('selected', mode === 0);
        document.getElementById('mode1-btn')!.classList.toggle('selected', mode === 1);
        document.getElementById('mode2-btn')!.classList.toggle('selected', mode === 2);
        document.getElementById('transform-panel')!.classList.toggle('visible', mode === 1 || mode === 2);
        document.getElementById('mode2-panel')!.classList.toggle('visible', mode === 2);
        if (mode === 2) this.refreshMassTable();
    }

    private setEdgeMode(mode: 0 | 1): void {
        if (!this.sim) return;
        this.sim.setEdgeMode(mode);
        document.getElementById('edgeLoopBtn')!.classList.toggle('selected', mode === 0);
        document.getElementById('edgeOpenBtn')!.classList.toggle('selected', mode === 1);
        this.updateZoomDisplay();
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

    private screenToWorld(sx: number, sy: number): { wx: number; wy: number } {
        if (!this.sim) return { wx: 0, wy: 0 };
        const rect   = this.canvas.getBoundingClientRect();
        const params = this.sim.getParams();
        const view   = this.sim.getView();
        return {
            wx: view.cx + (sx - rect.left - rect.width  / 2) * params.worldWidth  / (rect.width  * view.zoom),
            wy: view.cy + (sy - rect.top  - rect.height / 2) * params.worldHeight / (rect.height * view.zoom),
        };
    }

    private handleTrackButton(): void {
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
            const rect   = this.canvas.getBoundingClientRect();
            const params = this.sim.getParams();
            const view   = this.sim.getView();
            const sr = this.brushWorldRadius * view.zoom * rect.width / params.worldWidth;
            const pressing = this.cursorMouseButtons !== 0;
            ctx.save();
            ctx.strokeStyle = pressing ? 'rgba(0,170,255,0.7)' : 'rgba(255,255,255,0.4)';
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
            const rect   = this.canvas.getBoundingClientRect();
            const params = this.sim.getParams();
            const view   = this.sim.getView();
            const sx = rect.left + rect.width  / 2 + (info.comX - view.cx) * view.zoom * rect.width  / params.worldWidth;
            const sy = rect.top  + rect.height / 2 + (info.comY - view.cy) * view.zoom * rect.height / params.worldHeight;
            const sr = info.radius * view.zoom * rect.width / params.worldWidth;

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
            const rect   = this.canvas.getBoundingClientRect();
            const params = this.sim.getParams();
            const view   = this.sim.getView();
            const scaleX = view.zoom * rect.width  / params.worldWidth;
            const scaleY = view.zoom * rect.height / params.worldHeight;
            const sx = rect.left + rect.width  / 2 + (diagData.pos[0] - view.cx) * scaleX;
            const sy = rect.top  + rect.height / 2 + (diagData.pos[1] - view.cy) * scaleY;

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
            if (this.activeCursorTool === 'force') {
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
                const rect   = this.canvas.getBoundingClientRect();
                const params = this.sim.getParams();
                const view   = this.sim.getView();
                this.sim.setView(
                    view.cx - dx * params.worldWidth  / (rect.width  * view.zoom),
                    view.cy - dy * params.worldHeight / (rect.height * view.zoom),
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
        });

        window.addEventListener('mouseup', (e) => {
            if (e.button === 1 || (e.button === 0 && this.activeCursorTool === 'grab')) {
                panning = false;
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
            const params = this.sim.getParams();
            const view   = this.sim.getView();
            const oldZ   = view.zoom;
            const newZ   = Math.max(0.05, Math.min(200, oldZ * (e.deltaY < 0 ? 1.15 : 1 / 1.15)));
            const sx     = (e.clientX - rect.left)  - rect.width  / 2;
            const sy     = (e.clientY - rect.top)   - rect.height / 2;
            this.sim.setView(
                view.cx + sx * params.worldWidth  / rect.width  * (1 / oldZ - 1 / newZ),
                view.cy + sy * params.worldHeight / rect.height * (1 / oldZ - 1 / newZ),
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

        document.getElementById('cursorCollapseBtn')!.addEventListener('click', () => {
            this.cursorPanelCollapsed = !this.cursorPanelCollapsed;
            document.getElementById('cursor-panel')!.classList.toggle('collapsed', this.cursorPanelCollapsed);
            document.body.classList.toggle('cursor-panel-collapsed', this.cursorPanelCollapsed);
            const btn = document.getElementById('cursorCollapseBtn')!;
            btn.innerHTML = this.cursorPanelCollapsed ? '&#9664;' : '&#9654;';
        });

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

    private setActiveCursorTool(tool: 'grab' | 'force'): void {
        this.activeCursorTool = (this.activeCursorTool === tool) ? 'none' : tool;
        document.getElementById('ctool-grab')!.classList.toggle('active',  this.activeCursorTool === 'grab');
        document.getElementById('ctool-force')!.classList.toggle('active', this.activeCursorTool === 'force');
        document.getElementById('force-section')!.style.display = this.activeCursorTool === 'force' ? '' : 'none';
        this.syncCanvasCursor();
    }

    private syncCanvasCursor(): void {
        // Inspect/track override: clear inline style so CSS body-class cursor takes effect
        if (this.inspectMode || this.trackMode !== 'idle') {
            this.canvas.style.cursor = '';
            return;
        }
        if (this.activeCursorTool === 'grab') {
            this.canvas.style.cursor = (this.cursorMouseButtons & 1) ? 'grabbing' : 'grab';
        } else if (this.activeCursorTool === 'force') {
            this.canvas.style.cursor = 'crosshair';
        } else {
            this.canvas.style.cursor = '';
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
            Object.assign(slider, { min: '-1', max: '1', step: '0.01' });
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

        // Apply world size to canvas before handing off to sim
        const w = Number(state.worldWidth)  || 1600;
        const h = Number(state.worldHeight) || 900;
        this.setCanvasSize(w, h);

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

        const simMode  = this.sim.getSimMode()  as 0 | 1 | 2;
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

        const friction = this.sim.getFriction();
        (document.getElementById('frictionSlider') as HTMLInputElement).value = String(friction);
        document.getElementById('frictionValue')!.textContent = friction.toFixed(2);

        const maxTransform = this.sim.getMaxTransformRate();
        (document.getElementById('maxTransformSlider') as HTMLInputElement).value = String(maxTransform);
        document.getElementById('maxTransformValue')!.textContent = maxTransform.toFixed(2);

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
        if (this.sim) document.getElementById('time')!.textContent = `${this.sim.getTime().toFixed(1)}s`;
    }

    private startLoop(): void {
        const tick = () => {
            if (this.activeCursorTool === 'force' && this.cursorMouseButtons !== 0 && this.sim) {
                const { wx, wy } = this.screenToWorld(this.cursorMouseX, this.cursorMouseY);
                const dir = (this.cursorMouseButtons & 1) ? 1 : -1;
                this.sim.applyCursorForce(wx, wy, this.brushWorldRadius, this.forceStrength * dir);
            }
            this.sim?.update();
            this.updateStats();
            this.drawOverlay();
            this.animId = requestAnimationFrame(tick);
        };
        this.animId = requestAnimationFrame(tick);
    }

    // ── Entry point ───────────────────────────────────────────────────────────

    async initialize(): Promise<void> {
        const worldW = parseInt((document.getElementById('worldW') as HTMLInputElement).value) || 4800;
        const worldH = parseInt((document.getElementById('worldH') as HTMLInputElement).value) || 2700;
        this.setCanvasSize(worldW, worldH);

        this.sim = new ParticleSimulation(this.canvas);
        try {
            await this.sim.initialize();
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
