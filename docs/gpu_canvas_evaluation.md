# GPU canvas evaluation

## Why a GPU-backed canvas now?

Dense multichannel windows still rely on `pyqtgraph`'s QGraphicsView plots. Each pan repaints up to `n_channels` × `PlotDataItem` objects on the CPU before we decimate envelopes, so high-rate montages can stutter when the overscan cache misses.【F:ui/main_window.py†L528-L560】 The overscan helper already trims slices to the viewport and bins to pixel budgets, but the final polyline assembly and painting stay single-threaded on the GUI core.【F:core/overscan.py†L13-L31】 A GPU-backed scene can push those rasterization costs to the graphics stack and keep the UI responsive while background decimators warm up.

## Success criteria

1. **Drop-in swap:** Reuse the existing overscan/prefetch pipeline. Loader threads should continue delivering `(t, x)` arrays without backend-specific branching.【F:core/overscan.py†L13-L31】【F:ui/main_window.py†L555-L560】
2. **Shared interactions:** Hover indicators, crosshairs, wheel zoom, and pan buttons stay wired through Qt signals so the control rail and shortcuts remain identical.【F:ui/main_window.py†L537-L560】
3. **Envelope parity:** GPU curves must respect the min/max binning guarantees so visual integrity matches the CPU plots across zoom levels.【F:core/overscan.py†L13-L31】
4. **Fallback ready:** Operators can disable GPU mode per config or automatically when the driver stack is missing.

## Candidate: VisPy (PySide6 backend)

VisPy ships a `SceneCanvas` that embeds cleanly into Qt and routes draw calls to OpenGL while retaining high-level line plot APIs. Embedding a `SceneCanvas` as the viewport widget keeps the Qt layout logic intact and lets us delegate pan/zoom handling to VisPy's camera stack.

### Integration notes

* **Widget swap:** Replace `pg.GraphicsLayoutWidget` with a lightweight proxy that chooses between the existing pyqtgraph plot grid and a VisPy canvas at runtime. The proxy exposes `ensure_row(i)`, `set_data(i, t, x)`, and `set_theme(background, colors, label_active, label_hidden, axis_color)` to map onto either backend.【F:ui/main_window.py†L528-L560】
* **Glyph reuse:** Pre-bake per-channel `LineVisual` objects (one per VisPy sub-viewport) and call `.set_data()` with numpy views supplied by the overscan cache. This mirrors how we mutate `PlotDataItem` today, minimizing churn.
* **Event bridge:** Connect VisPy's `mouse_move`, `mouse_press`, and `mouse_wheel` events back into the existing `_update_hover_indicator`, pan buttons, and zoom controls so shared shortcuts continue to operate.【F:ui/main_window.py†L537-L560】
* **Theming:** Translate our theme palette to VisPy `Color` instances and update stroke widths to match current aesthetics.

### Prototype spike

1. Add `vispy>=0.14` to an optional extras group (commented in `requirements.txt`).
2. Create `ui/canvas_backends/vispy_canvas.py` with a `VispyStackedCanvas` implementing the proxy interface above.
3. Write a spike harness under `scripts/vispy_canvas_spike.py` that instantiates the canvas with synthetic 64×500 Hz data and measures frame times during simulated pans. Use the existing `scripts/profile_draw.py` pattern for consistent timing prints.【F:scripts/profile_draw.py†L1-L44】

## Alternative: Qt Graphical Effects (`QQuick`/`QtCharts`)

Qt's `QQuick` scene graph can deliver GPU acceleration, but it requires moving the viewer into QML (or embedding a `QQuickWidget`) and reimplementing hover/selection logic. That is a heavier migration and risks duplicating the existing layout logic. The `QtCharts` module also lacks the fine-grained control needed for overscan envelopes. VisPy remains the quickest path to GPU acceleration without rewriting the UI shell.

## Rollout plan

1. **Backend abstraction (1 PR):**
   * Introduce `PlotBackendProtocol` (dataclass or ABC) with `install(parent_layout)`, `set_row_visibility()`, `update_curve()`, and `set_hover_marker()`. Default implementation wraps the current pyqtgraph widgets to avoid regressions.
   * Extend `ViewerConfig`/`config.ini` with `canvas_backend = "pyqtgraph"` (default) so operators can opt into experimental GPU mode once it stabilizes.【F:config.py†L10-L92】
2. **VisPy backend (1–2 PRs):**
   * Land the optional dependency and new backend module.
   * Port hover/selection cues and confirm min/max envelopes align with CPU output.
   * Add pytest spikes that render offscreen and assert no exceptions when painting synthetic data.
3. **Performance validation:**
   * Extend `scripts/profile_draw.py` or add a sibling CLI to benchmark both backends against identical overscan tiles.【F:scripts/profile_draw.py†L1-L44】
   * Capture FPS/latency numbers for 32- and 64-channel montages to confirm GPU mode smooths pans before the overscan cache is warm.
4. **Feature flag & telemetry:**
   * Surface a status badge in the UI (e.g., “Renderer: GPU (VisPy)”) and log fallback reasons when the backend fails to initialize.

## Open questions

* **Shared context:** Should we create a single VisPy canvas with sub-ViewBoxes, or allocate one canvas per channel row? A single canvas reduces Qt plumbing but requires custom layout math.
* **Input latency:** Does GPU upload overhead offset the benefit for shorter windows (<10 s)? We should profile both 60 Hz and 120 Hz monitors.
* **Packaging:** Windows deployments may need angle/GL bindings packaged with the app. Investigate pyinstaller/briefcase recipes once the spike proves worthwhile.

With this plan we can validate a GPU canvas without destabilizing the current viewer and gather hard numbers before betting on a full migration.
