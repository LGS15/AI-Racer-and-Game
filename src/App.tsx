import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Activity,
  ArrowDown,
  ArrowLeft,
  ArrowRight,
  ArrowUp,
  Bot,
  CircleDot,
  Download,
  Flag,
  Gauge,
  Keyboard,
  LineChart,
  Network,
  Pause,
  Pencil,
  Play,
  Plus,
  RotateCcw,
  Route,
  Ruler,
  Save,
  Trash2,
  Upload,
  ZoomIn,
  ZoomOut
} from 'lucide-react';
import RacingCanvas, { deleteSelectedCheckpoint, deleteSelectedPoint } from './components/RacingCanvas';
import { ReplayCharts, TrainingProgressCharts } from './components/Charts';
import { bundledTracks } from './data/tracks';
import { ReferenceLineAgent, getAgentLabel, manualActionFromInput } from './domain/agents';
import { ExternalAgent, type BridgeStatus } from './domain/ai/environmentBridge';
import { validateLapTraceJson } from './domain/lapTrace';
import { DEFAULT_SIM_CONFIG, buildObservation, createInitialState, stepSimulation } from './domain/simulation';
import {
  clearTrackPointWidth,
  compileTrack,
  createDefaultTrack,
  getCheckpointTargets,
  getTrackPointBarriers,
  hashTrack,
  updateTrackPoint,
  updateTrackPointWidth,
  validateTrackJson
} from './domain/track';
import { createReplayBuilder, finalizeReplay, recordReplayFrame, sampleReplayFrame, validateReplayJson } from './domain/replay';
import { parseTrainingProgressCsv, type TrainingProgressRow } from './domain/trainingProgress';
import type { AgentAction, AgentObservation, LapTraceJson, Mode, ReplayJson, SimulationState, TrackJson, Vec2 } from './domain/types';

type AgentKind = 'manual' | 'reference' | 'external';
type EditorTool = 'select' | 'insert' | 'checkpoint' | 'barrier';

const seed = 1;
const MIN_TRACK_ZOOM = 0.25;
const MAX_TRACK_ZOOM = 2;
const TRACK_ZOOM_STEP = 0.05;

export default function App() {
  const [track, setTrack] = useState<TrackJson>(() => createDefaultTrack());
  const [mode, setMode] = useState<Mode>('drive');
  const [agentKind, setAgentKind] = useState<AgentKind>('manual');
  const [running, setRunning] = useState(false);
  const [recording, setRecording] = useState(false);
  const [replay, setReplay] = useState<ReplayJson | undefined>();
  const [lapTrace, setLapTrace] = useState<LapTraceJson | undefined>();
  const [trainingProgress, setTrainingProgress] = useState<TrainingProgressRow[] | undefined>();
  const [replayTime, setReplayTime] = useState(0);
  const [replayPlaying, setReplayPlaying] = useState(false);
  const [editorTool, setEditorTool] = useState<EditorTool>('select');
  const [selectedPointId, setSelectedPointId] = useState<string | undefined>();
  const [selectedCheckpointId, setSelectedCheckpointId] = useState<string | undefined>();
  const [trackZoom, setTrackZoom] = useState(1);
  const [notice, setNotice] = useState('');
  const compiled = useMemo(() => {
    try {
      return compileTrack(track);
    } catch {
      return undefined;
    }
  }, [track]);
  const checkpointTargets = useMemo(() => (compiled ? getCheckpointTargets(compiled) : []), [compiled]);
  const barriers = useMemo(() => getTrackPointBarriers(track), [track]);
  const selectedBarrier = useMemo(
    () => barriers.find((barrier) => barrier.point.id === selectedPointId) ?? barriers[0],
    [barriers, selectedPointId]
  );

  const initialState = useMemo(() => (compiled ? createInitialState(compiled, DEFAULT_SIM_CONFIG, seed) : undefined), [compiled]);
  const [snapshot, setSnapshot] = useState<SimulationState | undefined>(initialState);
  const [observation, setObservation] = useState<AgentObservation | undefined>(
    initialState && compiled ? buildObservation(initialState.car, compiled, DEFAULT_SIM_CONFIG) : undefined
  );

  const simRef = useRef<SimulationState | undefined>(initialState);
  const latestObservationRef = useRef<AgentObservation | undefined>(observation);
  const latestActionRef = useRef<AgentAction>({ throttle: 0, brake: 0, steer: 0 });
  const keysRef = useRef<Set<string>>(new Set());
  const heldControlsRef = useRef<Set<string>>(new Set());
  const [heldControlsView, setHeldControlsView] = useState<string[]>([]);
  const referenceAgentRef = useRef(new ReferenceLineAgent());
  const externalAgentRef = useRef(new ExternalAgent());
  const replayBuilderRef = useRef<ReturnType<typeof createReplayBuilder> | null>(null);
  const [wsUrl, setWsUrl] = useState('ws://localhost:8765');
  const [wsStatus, setWsStatus] = useState<BridgeStatus>('disconnected');
  const trackImportRef = useRef<HTMLInputElement | null>(null);
  const replayImportRef = useRef<HTMLInputElement | null>(null);
  const lapTraceImportRef = useRef<HTMLInputElement | null>(null);
  const trainingProgressImportRef = useRef<HTMLInputElement | null>(null);

  const replayTrackMismatch = replay ? replay.trackHash !== hashTrack(track) : false;
  const lapTraceTrackMismatch = lapTrace ? lapTrace.trackHash !== hashTrack(track) : false;
  const bundledTrackId = useMemo(() => {
    const currentHash = hashTrack(track);
    return bundledTracks.find((candidate) => {
      const result = validateTrackJson(candidate.data);
      return result.ok && result.value ? hashTrack(result.value) === currentHash : false;
    })?.id;
  }, [track]);

  const resetRun = useCallback(
    (stopReplay = true) => {
      if (!compiled) return;
      const next = createInitialState(compiled, DEFAULT_SIM_CONFIG, seed);
      simRef.current = next;
      const nextObservation = buildObservation(next.car, compiled, DEFAULT_SIM_CONFIG);
      latestObservationRef.current = nextObservation;
      setSnapshot(next);
      setObservation(nextObservation);
      setRunning(false);
      setRecording(false);
      replayBuilderRef.current = null;
      const agentContext = {
        track,
        seed,
        checkpointCount: track.checkpointCount,
        rayAngles: DEFAULT_SIM_CONFIG.rayAngles
      };
      referenceAgentRef.current.reset(agentContext);
      externalAgentRef.current.setCompiledTrack(compiled);
      externalAgentRef.current.reset(agentContext);
      if (stopReplay) {
        setReplayPlaying(false);
      }
    },
    [compiled, track]
  );

  useEffect(() => {
    resetRun(false);
  }, [resetRun, agentKind]);

  useEffect(() => {
    const agent = externalAgentRef.current;
    agent.onStatusChange = setWsStatus;
    agent.onResetRequest = () => resetRun();
  }, [resetRun]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (isTypingTarget(event.target)) return;
      if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'KeyW', 'KeyA', 'KeyS', 'KeyD'].includes(event.code)) {
        event.preventDefault();
        keysRef.current.add(event.code);
      }
      if (event.code === 'Space') {
        event.preventDefault();
        setRunning((value) => !value);
      }
    };
    const onKeyUp = (event: KeyboardEvent) => {
      keysRef.current.delete(event.code);
    };
    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);
    return () => {
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
    };
  }, []);

  useEffect(() => {
    let frame = 0;
    let last = performance.now();
    let accumulator = 0;
    let lastUi = 0;

    const loop = (now: number) => {
      const dt = Math.min(0.12, (now - last) / 1000);
      last = now;
      accumulator += dt;

      if (running && mode === 'drive' && compiled && simRef.current) {
        while (accumulator >= DEFAULT_SIM_CONFIG.fixedDt) {
          const state = simRef.current;
          const beforeObservation = buildObservation(state.car, compiled, DEFAULT_SIM_CONFIG);
          const action =
            agentKind === 'manual'
              ? manualActionFromInput(keysRef.current, heldControlsRef.current)
              : agentKind === 'external'
              ? externalAgentRef.current.step(beforeObservation, DEFAULT_SIM_CONFIG.fixedDt)
              : referenceAgentRef.current.step(beforeObservation, DEFAULT_SIM_CONFIG.fixedDt);
          const next = stepSimulation(state, action, compiled, DEFAULT_SIM_CONFIG.fixedDt, DEFAULT_SIM_CONFIG);
          const didCollide = next.events.some((event) => event.type === 'collision');
          const afterObservation = buildObservation(next.car, compiled, DEFAULT_SIM_CONFIG, didCollide);
          if (agentKind === 'external') {
            externalAgentRef.current.onTransition(state, next, afterObservation, compiled);
          }
          simRef.current = next;
          latestActionRef.current = action;
          latestObservationRef.current = afterObservation;
          if (replayBuilderRef.current) {
            recordReplayFrame(replayBuilderRef.current, next, action, afterObservation);
          }
          accumulator -= DEFAULT_SIM_CONFIG.fixedDt;
        }
      } else {
        accumulator = 0;
      }

      if (now - lastUi > 34) {
        if (simRef.current) setSnapshot({ ...simRef.current, car: { ...simRef.current.car } });
        if (latestObservationRef.current) setObservation({ ...latestObservationRef.current });
        lastUi = now;
      }
      frame = requestAnimationFrame(loop);
    };

    frame = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(frame);
  }, [agentKind, compiled, mode, running]);

  useEffect(() => {
    if (!replayPlaying || !replay) return undefined;
    let frame = 0;
    let last = performance.now();
    const loop = (now: number) => {
      const dt = (now - last) / 1000;
      last = now;
      setReplayTime((value) => {
        const next = value + dt;
        return next > replay.duration ? 0 : next;
      });
      frame = requestAnimationFrame(loop);
    };
    frame = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(frame);
  }, [replayPlaying, replay]);

  const activeReplayFrame = replay ? sampleReplayFrame(replay, replayTime) : undefined;

  function toggleRecording() {
    if (!compiled || !simRef.current) return;
    if (recording && replayBuilderRef.current) {
      const finished = finalizeReplay(replayBuilderRef.current, simRef.current);
      setReplay(finished);
      setReplayTime(0);
      setMode('analyze');
      setNotice('Replay recorded.');
      replayBuilderRef.current = null;
      setRecording(false);
      return;
    }

    const agentLabel = agentKind === 'manual' ? 'Manual' : agentKind === 'external' ? 'External AI' : 'Reference';
    replayBuilderRef.current = createReplayBuilder(track, { id: agentKind, label: agentLabel }, seed, DEFAULT_SIM_CONFIG);
    setRecording(true);
    setNotice('Recording started.');
  }

  function exportTrack() {
    downloadJson(`${safeFileName(track.metadata.name)}.track.json`, track);
  }

  function exportReplay() {
    if (!replay) return;
    downloadJson(`${safeFileName(track.metadata.name)}.${replay.agent.id}.replay.json`, replay);
  }

  async function importTrack(file?: File) {
    if (!file) return;
    try {
      const parsed = JSON.parse(await file.text()) as unknown;
      const result = validateTrackJson(parsed);
      if (!result.ok || !result.value) {
        setNotice(result.errors.join(' '));
        return;
      }
      setTrack(result.value);
      setSelectedPointId(undefined);
      setSelectedCheckpointId(undefined);
      setMode('edit');
      setNotice(result.warnings.length ? result.warnings.join(' ') : 'Track imported.');
    } catch (error) {
      setNotice(error instanceof Error ? error.message : 'Track import failed.');
    } finally {
      if (trackImportRef.current) trackImportRef.current.value = '';
    }
  }

  async function importReplay(file?: File) {
    if (!file) return;
    try {
      const parsed = JSON.parse(await file.text()) as unknown;
      const result = validateReplayJson(parsed);
      if (!result.ok || !result.value) {
        setNotice(result.errors.join(' '));
        return;
      }
      setReplay(result.value);
      setReplayTime(0);
      setMode('analyze');
      setNotice('Replay imported.');
    } catch (error) {
      setNotice(error instanceof Error ? error.message : 'Replay import failed.');
    } finally {
      if (replayImportRef.current) replayImportRef.current.value = '';
    }
  }

  async function importLapTrace(file?: File) {
    if (!file) return;
    try {
      const parsed = JSON.parse(await file.text()) as unknown;
      const result = validateLapTraceJson(parsed);
      if (!result.ok || !result.value) {
        setNotice(result.errors.join(' '));
        return;
      }
      setLapTrace(result.value);
      setMode('analyze');
      setNotice('Lap trace imported.');
    } catch (error) {
      setNotice(error instanceof Error ? error.message : 'Lap trace import failed.');
    } finally {
      if (lapTraceImportRef.current) lapTraceImportRef.current.value = '';
    }
  }

  async function importTrainingProgress(file?: File) {
    if (!file) return;
    try {
      const result = parseTrainingProgressCsv(await file.text());
      if (!result.ok || !result.value) {
        setNotice(result.errors.join(' '));
        return;
      }
      setTrainingProgress(result.value);
      setMode('analyze');
      setNotice(`Training progress imported (${result.value.length} rows).`);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : 'Training progress import failed.');
    } finally {
      if (trainingProgressImportRef.current) trainingProgressImportRef.current.value = '';
    }
  }

  function holdControl(code: string, isHeld: boolean) {
    const next = new Set(heldControlsRef.current);
    if (isHeld) next.add(code);
    else next.delete(code);
    heldControlsRef.current = next;
    setHeldControlsView([...next]);
  }

  function patchTrack(patch: Partial<TrackJson>) {
    setTrack((current) => ({ ...current, ...patch }));
  }

  function loadBundledTrack(trackId: string) {
    const bundled = bundledTracks.find((candidate) => candidate.id === trackId);
    if (!bundled) return;
    const result = validateTrackJson(JSON.parse(JSON.stringify(bundled.data)) as unknown);
    if (!result.ok || !result.value) {
      setNotice(result.errors.join(' '));
      return;
    }
    setTrack(result.value);
    setSelectedPointId(undefined);
    setSelectedCheckpointId(undefined);
    setEditorTool('barrier');
    setMode('edit');
    setNotice(`${result.value.metadata.name} loaded.`);
  }

  function setStartAtSelectedPoint() {
    if (!selectedPointId) return;
    const pointIndex = track.centerline.findIndex((point) => point.id === selectedPointId);
    if (pointIndex < 0) return;
    setTrack((current) => ({ ...current, start: { ...current.start, pointIndex } }));
  }

  function selectBarrierPoint(pointId: string) {
    setSelectedPointId(pointId);
    setSelectedCheckpointId(undefined);
    setEditorTool('barrier');
    setMode('edit');
  }

  function updateBarrierPoint(pointId: string, patch: { x?: number; y?: number; width?: number }) {
    setTrack((current) => {
      const point = current.centerline.find((candidate) => candidate.id === pointId);
      if (!point) return current;
      let next = current;
      const x = patch.x ?? point.x;
      const y = patch.y ?? point.y;
      if ((patch.x !== undefined || patch.y !== undefined) && Number.isFinite(x) && Number.isFinite(y)) {
        next = updateTrackPoint(next, pointId, {
          x,
          y
        });
      }
      if (patch.width !== undefined && Number.isFinite(patch.width)) {
        next = updateTrackPointWidth(next, pointId, patch.width);
      }
      return next;
    });
  }

  function inheritBarrierWidth(pointId: string) {
    setTrack((current) => clearTrackPointWidth(current, pointId));
  }

  function setClampedTrackZoom(value: number) {
    setTrackZoom(Math.min(MAX_TRACK_ZOOM, Math.max(MIN_TRACK_ZOOM, value)));
  }

  return (
    <main className="appShell">
      <header className="topBar">
        <div className="brandBlock">
          <Gauge size={26} />
          <div>
            <h1>AI Racing Simulator</h1>
            <span>{track.metadata.name}</span>
          </div>
        </div>
        <nav className="modeTabs" aria-label="Mode">
          <button className={mode === 'drive' ? 'active' : ''} onClick={() => setMode('drive')}>
            <Play size={17} />
            Drive
          </button>
          <button className={mode === 'edit' ? 'active' : ''} onClick={() => setMode('edit')}>
            <Pencil size={17} />
            Edit
          </button>
          <button className={mode === 'analyze' ? 'active' : ''} onClick={() => setMode('analyze')}>
            <LineChart size={17} />
            Analyze
          </button>
        </nav>
      </header>

      <section className="workbench">
        <div className="canvasColumn">
          <RacingCanvas
            mode={mode}
            track={track}
            compiled={compiled}
            car={snapshot?.car}
            observation={observation}
            lapTrace={mode === 'analyze' ? lapTrace : undefined}
            replay={mode === 'analyze' ? replay : undefined}
            replayTime={replayTime}
            displayZoom={trackZoom}
            selectedPointId={selectedPointId}
            selectedCheckpointId={selectedCheckpointId}
            editorTool={editorTool}
            onTrackChange={setTrack}
            onSelectPoint={setSelectedPointId}
            onSelectCheckpoint={setSelectedCheckpointId}
          />
          <div className="canvasControls" aria-label="Track display zoom">
            <button
              className="iconButton compact"
              onClick={() => setClampedTrackZoom(trackZoom - TRACK_ZOOM_STEP)}
              disabled={trackZoom <= MIN_TRACK_ZOOM}
              title="Zoom out"
              aria-label="Zoom out"
            >
              <ZoomOut size={16} />
            </button>
            <input
              type="range"
              min={MIN_TRACK_ZOOM}
              max={MAX_TRACK_ZOOM}
              step={TRACK_ZOOM_STEP}
              value={trackZoom}
              onChange={(event) => setClampedTrackZoom(Number(event.target.value))}
              aria-label="Track zoom"
            />
            <strong>{Math.round(trackZoom * 100)}%</strong>
            <button
              className="iconButton compact"
              onClick={() => setClampedTrackZoom(trackZoom + TRACK_ZOOM_STEP)}
              disabled={trackZoom >= MAX_TRACK_ZOOM}
              title="Zoom in"
              aria-label="Zoom in"
            >
              <ZoomIn size={16} />
            </button>
            <button
              className="iconButton compact"
              onClick={() => setClampedTrackZoom(1)}
              disabled={trackZoom === 1}
              title="Reset zoom"
              aria-label="Reset zoom"
            >
              <RotateCcw size={16} />
            </button>
          </div>
          <div className="timelinePanel">
            {mode === 'analyze' && replay ? (
              <>
                <button className="iconButton" onClick={() => setReplayPlaying((value) => !value)} title={replayPlaying ? 'Pause replay' : 'Play replay'}>
                  {replayPlaying ? <Pause size={18} /> : <Play size={18} />}
                </button>
                <input
                  type="range"
                  min={0}
                  max={Math.max(0.01, replay.duration)}
                  step={0.016}
                  value={Math.min(replayTime, replay.duration)}
                  onChange={(event) => setReplayTime(Number(event.target.value))}
                  aria-label="Replay timeline"
                />
                <strong>{replayTime.toFixed(1)}s</strong>
              </>
            ) : mode === 'analyze' && lapTrace ? (
              <>
                <span>Trace lap {lapTrace.lap}</span>
                <span>{lapTrace.lapTime.toFixed(2)}s</span>
                <span>{lapTrace.points.length} points</span>
                <span>{lapTrace.agent.label}</span>
              </>
            ) : (
              <>
                <span>Lap {snapshot?.car.lap ?? 0}</span>
                <span>{snapshot?.car.lapTime.toFixed(2) ?? '0.00'}s</span>
                <span>{observation ? `${Math.round(observation.speed)} px/s` : '0 px/s'}</span>
                <span>{snapshot?.car.collisions ?? 0} impacts</span>
              </>
            )}
          </div>
        </div>

        <aside className="sidePanel">
          <section className="panelSection">
            <div className="sectionHeader">
              <h2>Run</h2>
              <div className="buttonRow">
                <button className="iconButton" onClick={() => setRunning((value) => !value)} title={running ? 'Pause' : 'Start'}>
                  {running ? <Pause size={18} /> : <Play size={18} />}
                </button>
                <button className="iconButton" onClick={() => resetRun()} title="Reset run">
                  <RotateCcw size={18} />
                </button>
                <button className={recording ? 'iconButton danger active' : 'iconButton'} onClick={toggleRecording} title={recording ? 'Stop recording' : 'Record replay'}>
                  <Activity size={18} />
                </button>
              </div>
            </div>
            <div className="agentToggle">
              <button className={agentKind === 'manual' ? 'active' : ''} onClick={() => setAgentKind('manual')}>
                <Keyboard size={17} />
                Manual
              </button>
              <button className={agentKind === 'reference' ? 'active' : ''} onClick={() => setAgentKind('reference')}>
                <Bot size={17} />
                Reference
              </button>
              <button className={agentKind === 'external' ? 'active' : ''} onClick={() => setAgentKind('external')}>
                <Network size={17} />
                External
              </button>
            </div>
            {agentKind === 'external' ? (
              <div className="externalPanel">
                <input
                  className="wsUrlInput"
                  type="text"
                  value={wsUrl}
                  onChange={(e) => setWsUrl(e.target.value)}
                  placeholder="ws://localhost:8765"
                  disabled={wsStatus !== 'disconnected'}
                  spellCheck={false}
                />
                <div className="externalPanelRow">
                  <span className={`wsStatusDot wsStatus-${wsStatus}`} title={wsStatus} />
                  <span className="wsStatusLabel">{wsStatus}</span>
                  {wsStatus === 'disconnected' ? (
                    <button className="wsConnectBtn" onClick={() => externalAgentRef.current.connect(wsUrl)}>Connect</button>
                  ) : (
                    <button className="wsConnectBtn" onClick={() => externalAgentRef.current.disconnect()}>Disconnect</button>
                  )}
                </div>
              </div>
            ) : null}
            {agentKind === 'manual' ? (
              <div className="drivePad" aria-label="Manual controls">
                {[
                  { code: 'ArrowLeft', Icon: ArrowLeft, label: 'Steer left' },
                  { code: 'ArrowUp', Icon: ArrowUp, label: 'Throttle' },
                  { code: 'ArrowDown', Icon: ArrowDown, label: 'Brake' },
                  { code: 'ArrowRight', Icon: ArrowRight, label: 'Steer right' }
                ].map(({ code, Icon, label }) => (
                  <button
                    key={code}
                    className={heldControlsView.includes(code) ? 'active' : ''}
                    onPointerDown={() => holdControl(code, true)}
                    onPointerUp={() => holdControl(code, false)}
                    onPointerLeave={() => holdControl(code, false)}
                    title={label}
                    aria-label={label}
                  >
                    <Icon size={18} />
                  </button>
                ))}
              </div>
            ) : null}
          </section>

          <section className="panelSection">
            <div className="sectionHeader">
              <h2>Track</h2>
              <div className="buttonRow">
                <button className="iconButton" onClick={() => trackImportRef.current?.click()} title="Import track">
                  <Upload size={18} />
                </button>
                <button className="iconButton" onClick={exportTrack} title="Export track">
                  <Download size={18} />
                </button>
              </div>
            </div>
            <label className="field">
              <span>Name</span>
              <input
                value={track.metadata.name}
                onChange={(event) =>
                  setTrack((current) => ({ ...current, metadata: { ...current.metadata, name: event.target.value || 'Untitled Track' } }))
                }
              />
            </label>
            <label className="field">
              <span>Saved track</span>
              <select value={bundledTrackId ?? ''} onChange={(event) => loadBundledTrack(event.target.value)}>
                <option value="">Custom / current</option>
                {bundledTracks.map((bundled) => (
                  <option key={bundled.id} value={bundled.id}>
                    {bundled.label}
                  </option>
                ))}
              </select>
            </label>
            <label className="field">
              <span>Width</span>
              <input
                type="range"
                min={50}
                max={220}
                value={track.globalWidth}
                onChange={(event) => patchTrack({ globalWidth: Number(event.target.value) })}
              />
            </label>
            <label className="field">
              <span>Checkpoints {track.checkpoints ? checkpointTargets.length : track.checkpointCount}</span>
              <input
                type="range"
                min={4}
                max={32}
                value={track.checkpointCount}
                onChange={(event) => {
                  setSelectedCheckpointId(undefined);
                  patchTrack({ checkpointCount: Number(event.target.value), checkpoints: undefined });
                }}
              />
            </label>
            {mode === 'edit' ? (
              <div className="editTools">
                <button className={editorTool === 'select' ? 'active' : ''} onClick={() => setEditorTool('select')} title="Select and drag points">
                  <Pencil size={17} />
                </button>
                <button className={editorTool === 'insert' ? 'active' : ''} onClick={() => setEditorTool('insert')} title="Insert centerline point">
                  <Plus size={17} />
                </button>
                <button
                  className={editorTool === 'checkpoint' ? 'active' : ''}
                  onClick={() => {
                    setEditorTool('checkpoint');
                    setSelectedPointId(undefined);
                  }}
                  title="Add and drag checkpoints"
                >
                  <CircleDot size={17} />
                </button>
                <button
                  className={editorTool === 'barrier' ? 'active' : ''}
                  onClick={() => {
                    setEditorTool('barrier');
                    setSelectedCheckpointId(undefined);
                  }}
                  title="Drag track limits"
                >
                  <Ruler size={17} />
                </button>
                <button onClick={setStartAtSelectedPoint} disabled={!selectedPointId} title="Set start line">
                  <Flag size={17} />
                </button>
                <button
                  onClick={() => {
                    const next =
                      editorTool === 'checkpoint'
                        ? deleteSelectedCheckpoint(track, compiled, selectedCheckpointId)
                        : deleteSelectedPoint(track, selectedPointId);
                    setTrack(next);
                    setSelectedPointId(undefined);
                    setSelectedCheckpointId(undefined);
                  }}
                  disabled={
                    editorTool === 'checkpoint'
                      ? !selectedCheckpointId || checkpointTargets.length <= 2
                      : !selectedPointId || track.centerline.length <= 3
                  }
                  title={editorTool === 'checkpoint' ? 'Delete selected checkpoint' : 'Delete selected point'}
                >
                  <Trash2 size={17} />
                </button>
              </div>
            ) : null}
          </section>

          {mode === 'edit' ? (
            <section className="panelSection barrierPanel">
              <div className="sectionHeader">
                <h2>Barriers</h2>
                <strong className="sectionMeta">{barriers.length} points</strong>
              </div>
              {selectedBarrier ? (
                <>
                  <div className="barrierDetailHeader">
                    <span>Selected</span>
                    <strong>#{selectedBarrier.index + 1}</strong>
                    <button
                      className="miniButton"
                      onClick={() => inheritBarrierWidth(selectedBarrier.point.id)}
                      disabled={selectedBarrier.usesGlobalWidth}
                      title="Use global width"
                    >
                      Global
                    </button>
                  </div>
                  <div className="barrierGrid">
                    <label>
                      <span>X</span>
                      <input
                        type="number"
                        value={formatInputNumber(selectedBarrier.point.x)}
                        onFocus={() => selectBarrierPoint(selectedBarrier.point.id)}
                        onChange={(event) => updateBarrierPoint(selectedBarrier.point.id, { x: Number(event.target.value) })}
                      />
                    </label>
                    <label>
                      <span>Y</span>
                      <input
                        type="number"
                        value={formatInputNumber(selectedBarrier.point.y)}
                        onFocus={() => selectBarrierPoint(selectedBarrier.point.id)}
                        onChange={(event) => updateBarrierPoint(selectedBarrier.point.id, { y: Number(event.target.value) })}
                      />
                    </label>
                    <label>
                      <span>Width</span>
                      <input
                        type="number"
                        min={40}
                        max={260}
                        step={1}
                        value={formatInputNumber(selectedBarrier.width)}
                        onFocus={() => selectBarrierPoint(selectedBarrier.point.id)}
                        onChange={(event) => updateBarrierPoint(selectedBarrier.point.id, { width: Number(event.target.value) })}
                      />
                    </label>
                  </div>
                  <div className="edgeReadout">
                    <div>
                      <span>Left</span>
                      <strong>{formatPoint(selectedBarrier.left)}</strong>
                    </div>
                    <div>
                      <span>Right</span>
                      <strong>{formatPoint(selectedBarrier.right)}</strong>
                    </div>
                  </div>
                </>
              ) : null}
              <div className="barrierList" aria-label="Barrier control points">
                {barriers.map((barrier) => (
                  <button
                    key={barrier.point.id}
                    className={barrier.point.id === selectedBarrier?.point.id ? 'barrierRow active' : 'barrierRow'}
                    onClick={() => selectBarrierPoint(barrier.point.id)}
                  >
                    <span>#{barrier.index + 1}</span>
                    <strong>{Math.round(barrier.width)}px</strong>
                    <small>L {formatPoint(barrier.left)}</small>
                    <small>R {formatPoint(barrier.right)}</small>
                  </button>
                ))}
              </div>
            </section>
          ) : null}

          <section className="panelSection">
            <div className="sectionHeader">
              <h2>Replay</h2>
              <div className="buttonRow">
                <button className="iconButton" onClick={() => replayImportRef.current?.click()} title="Import replay">
                  <Upload size={18} />
                </button>
                <button className="iconButton" onClick={exportReplay} disabled={!replay} title="Export replay">
                  <Save size={18} />
                </button>
              </div>
            </div>
            {replay ? (
              <>
                <div className={replayTrackMismatch ? 'statusLine warning' : 'statusLine'}>
                  <span>{replay.agent.label}</span>
                  <strong>{replay.frames.length} frames</strong>
                </div>
                {activeReplayFrame ? (
                  <div className="statusLine">
                    <span>Ghost</span>
                    <strong>{Math.round(activeReplayFrame.speed)} px/s</strong>
                  </div>
                ) : null}
              </>
            ) : (
              <div className="statusLine">
                <span>Empty</span>
                <strong>-</strong>
              </div>
            )}
          </section>

          <section className="panelSection">
            <div className="sectionHeader">
              <h2>Lap Trace</h2>
              <div className="buttonRow">
                <button className="iconButton" onClick={() => lapTraceImportRef.current?.click()} title="Import lap trace">
                  <Route size={18} />
                </button>
              </div>
            </div>
            {lapTrace ? (
              <>
                <div className={lapTraceTrackMismatch ? 'statusLine warning' : 'statusLine'}>
                  <span>{lapTrace.agent.label}</span>
                  <strong>{lapTrace.lapTime.toFixed(2)}s</strong>
                </div>
                <div className="statusLine">
                  <span>Path</span>
                  <strong>{lapTrace.points.length} points</strong>
                </div>
              </>
            ) : (
              <div className="statusLine">
                <span>Empty</span>
                <strong>-</strong>
              </div>
            )}
          </section>

          <section className="panelSection analysisPanel">
            <div className="sectionHeader">
              <h2>Analysis</h2>
              <div className="buttonRow">
                <button className="iconButton" onClick={() => trainingProgressImportRef.current?.click()} title="Import training progress CSV">
                  <Upload size={18} />
                </button>
              </div>
            </div>
            <ReplayCharts replay={replay} />
            <TrainingProgressCharts rows={trainingProgress} />
          </section>
        </aside>
      </section>

      <input ref={trackImportRef} type="file" accept="application/json,.json" hidden onChange={(event) => void importTrack(event.target.files?.[0])} />
      <input ref={replayImportRef} type="file" accept="application/json,.json" hidden onChange={(event) => void importReplay(event.target.files?.[0])} />
      <input ref={lapTraceImportRef} type="file" accept="application/json,.json" hidden onChange={(event) => void importLapTrace(event.target.files?.[0])} />
      <input ref={trainingProgressImportRef} type="file" accept="text/csv,.csv" hidden onChange={(event) => void importTrainingProgress(event.target.files?.[0])} />

      <div className={notice ? 'toast show' : 'toast'}>{notice}</div>
    </main>
  );
}

function isTypingTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  return ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName) || target.isContentEditable;
}

function downloadJson(filename: string, data: unknown) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

function safeFileName(value: string): string {
  return value.trim().toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/(^-|-$)+/g, '') || 'track';
}

function formatInputNumber(value: number): string {
  return Number.isFinite(value) ? String(Math.round(value * 10) / 10) : '0';
}

function formatPoint(point: Vec2): string {
  return `${formatInputNumber(point.x)}, ${formatInputNumber(point.y)}`;
}
