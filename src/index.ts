import { ParticleSimulation, TransformRule, MAX_TYPES, TYPE_COLORS_HEX } from './simulation';

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

    private frameCount    = 0;
    private lastTime      = performance.now();
    private autoPause     = true;
    private lowFpsFrames  = 0;  // counts consecutive 1-second windows below threshold

    constructor() {
        this.canvas = document.getElementById('sim-canvas') as HTMLCanvasElement;
        this.setupUI();
        this.setupCanvasEvents();
        window.addEventListener('resize', () => this.fitCanvas());
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
            }
        };
        typeEl.addEventListener('change', applyTypes);
        typeEl.addEventListener('keydown', (e) => { if (e.key === 'Enter') applyTypes(); });

        // Buttons
        document.getElementById('resetBtn')!.addEventListener('click', () => this.sim?.reset());
        document.getElementById('randomizeBtn')!.addEventListener('click', () => {
            this.sim?.randomizeForces();
            this.refreshForceMatrices();
        });
        document.getElementById('pauseBtn')!.addEventListener('click', () => {
            if (!this.sim) return;
            this.sim.togglePause();
            this.syncPauseButton();
        });

        // Auto-pause checkbox
        const apChk = document.getElementById('autoPauseChk') as HTMLInputElement;
        apChk.addEventListener('change', () => { this.autoPause = apChk.checked; });

        // Sim mode
        document.getElementById('mode0-btn')!.addEventListener('click', () => this.setSimMode(0));
        document.getElementById('mode1-btn')!.addEventListener('click', () => this.setSimMode(1));

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

        // World size
        document.getElementById('applyWorldSize')!.addEventListener('click', () => {
            const w = parseInt((document.getElementById('worldW') as HTMLInputElement).value) || 1600;
            const h = parseInt((document.getElementById('worldH') as HTMLInputElement).value) || 900;
            this.setCanvasSize(w, h);
            this.sim?.setWorldSize(w, h);
            this.updateZoomDisplay();
        });

        // Edge mode
        document.getElementById('edgeLoopBtn')!.addEventListener('click', () => this.setEdgeMode(0));
        document.getElementById('edgeOpenBtn')!.addEventListener('click', () => this.setEdgeMode(1));

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
            if (e.key === 'Escape') { this.closeForceEditor(); this.closeTransformEditor(); }
        });
    }

    private syncPauseButton(): void {
        const paused = this.sim?.isPaused_() ?? false;
        const btn = document.getElementById('pauseBtn')!;
        btn.textContent = paused ? 'Resume' : 'Pause';
        btn.classList.toggle('active', paused);
    }

    private setSimMode(mode: 0 | 1): void {
        this.sim?.setSimMode(mode);
        document.getElementById('mode0-btn')!.classList.toggle('selected', mode === 0);
        document.getElementById('mode1-btn')!.classList.toggle('selected', mode === 1);
        document.getElementById('transform-panel')!.classList.toggle('visible', mode === 1);
    }

    private setEdgeMode(mode: 0 | 1): void {
        if (!this.sim) return;
        this.sim.setEdgeMode(mode);
        document.getElementById('edgeLoopBtn')!.classList.toggle('selected', mode === 0);
        document.getElementById('edgeOpenBtn')!.classList.toggle('selected', mode === 1);
        this.updateZoomDisplay();
    }

    // ── Pan / zoom ────────────────────────────────────────────────────────────

    private setupCanvasEvents(): void {
        let panning = false, lastX = 0, lastY = 0;

        this.canvas.addEventListener('mousedown', (e) => {
            if (e.button !== 1) return;
            e.preventDefault();
            panning = true; lastX = e.clientX; lastY = e.clientY;
            this.canvas.style.cursor = 'grabbing';
        });

        window.addEventListener('mousemove', (e) => {
            if (!panning || !this.sim) return;
            const rect   = this.canvas.getBoundingClientRect();
            const params = this.sim.getParams();
            const view   = this.sim.getView();
            this.sim.setView(
                view.cx - (e.clientX - lastX) * params.worldWidth  / (rect.width  * view.zoom),
                view.cy - (e.clientY - lastY) * params.worldHeight / (rect.height * view.zoom),
                view.zoom
            );
            lastX = e.clientX; lastY = e.clientY;
        });

        window.addEventListener('mouseup', (e) => {
            if (e.button !== 1) return;
            panning = false; this.canvas.style.cursor = '';
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

    // ── Force matrices ────────────────────────────────────────────────────────

    private buildMatrixTable(tableId: string, kind: 'strength' | 'radius'): void {
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
                const v  = kind === 'strength' ? (c?.strength ?? 0) : (c?.radius ?? 100);
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

    private styleForceCellValue(td: HTMLTableCellElement, kind: 'strength' | 'radius', v: number): void {
        td.style.background = kind === 'strength' ? strengthToColor(v) : radiusToColor(v);
        td.textContent      = kind === 'strength' ? v.toFixed(2) : String(Math.round(v));
        td.title            = String(v.toFixed(3));
    }

    private refreshForceMatrices(): void {
        (['strength-table', 'radius-table'] as const).forEach(id => {
            (document.getElementById(id) as HTMLTableElement).innerHTML = '';
        });
        this.buildMatrixTable('strength-table', 'strength');
        this.buildMatrixTable('radius-table',   'radius');
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
        const kind = td.dataset.kind as 'strength' | 'radius';
        const c    = this.sim!.getParams().forceMatrix[from]?.[to];
        const cur  = kind === 'strength' ? (c?.strength ?? 0) : (c?.radius ?? 100);

        const editor  = document.getElementById('cell-editor')!;
        const title   = document.getElementById('cell-editor-title')!;
        const slider  = document.getElementById('cell-slider') as HTMLInputElement;
        const display = document.getElementById('cell-value-display')!;

        const pip = (i: number) =>
            `<span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:${TYPE_HEX[i]};margin-right:2px;vertical-align:middle"></span>`;
        title.innerHTML = `${pip(from)}${TYPE_LABELS[from]} → ${pip(to)}${TYPE_LABELS[to]} <span style="color:#888">${kind}</span>`;

        Object.assign(slider, kind === 'strength' ? { min:'-1', max:'1', step:'0.01' } : { min:'10', max:'250', step:'1' });
        slider.value = String(cur);
        display.textContent = kind === 'strength' ? cur.toFixed(2) : String(Math.round(cur));

        const rect = td.getBoundingClientRect();
        let left = rect.left, top = rect.bottom + 4;
        if (left + 180 > window.innerWidth)  left = window.innerWidth - 184;
        if (top  + 70  > window.innerHeight) top  = rect.top - 74;
        editor.style.left = `${left}px`; editor.style.top = `${top}px`;
        editor.classList.add('visible');
        this.editorCell = td;

        slider.oninput = () => {
            const v = parseFloat(slider.value);
            display.textContent = kind === 'strength' ? v.toFixed(2) : String(Math.round(v));
            this.styleForceCellValue(td, kind, v);
            this.applyForceCell(from, to, kind, v);
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

    private applyForceCell(from: number, to: number, kind: 'strength' | 'radius', value: number): void {
        if (!this.sim) return;
        const fm = this.sim.getParams().forceMatrix;
        if (!fm[from]) fm[from] = {};
        if (!fm[from][to]) fm[from][to] = { strength: 0, radius: 100 };
        if (kind === 'strength') fm[from][to].strength = value; else fm[from][to].radius = value;
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
        const lowerEnabled   = document.getElementById('te-lower-enabled')   as HTMLInputElement;
        const lowerThreshold = document.getElementById('te-lower-threshold') as HTMLInputElement;

        upperEnabled.checked   = rule.upperEnabled;
        upperThreshold.value   = rule.upperThreshold.toFixed(3);
        lowerEnabled.checked   = rule.lowerEnabled;
        lowerThreshold.value   = rule.lowerThreshold.toFixed(3);

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

        const simMode  = this.sim.getSimMode()  as 0 | 1;
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
            if (fps < 5) {
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
            this.sim?.update();
            this.updateStats();
            this.animId = requestAnimationFrame(tick);
        };
        this.animId = requestAnimationFrame(tick);
    }

    // ── Entry point ───────────────────────────────────────────────────────────

    async initialize(): Promise<void> {
        const worldW = parseInt((document.getElementById('worldW') as HTMLInputElement).value) || 1600;
        const worldH = parseInt((document.getElementById('worldH') as HTMLInputElement).value) || 900;
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

        this.buildMatrixTable('strength-table', 'strength');
        this.buildMatrixTable('radius-table',   'radius');
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
