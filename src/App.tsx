import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Activity,
  Bot,
  CircleDot,
  Download,
  Flag,
  Gauge,
  Keyboard,
  LineChart,
  Pause,
  Pencil,
  Play,
  Plus,
  RotateCcw,
  Ruler,
  Save,
  Trash2,
  Upload
} from 'lucide-react';
import RacingCanvas, { deleteSelectedCheckpoint, deleteSelectedPoint } from './components/RacingCanvas';
import { ReplayCharts } from './components/Charts';
import { ReferenceLineAgent, getAgentLabel, manualActionFromInput } from './domain/agents';
import { DEFAULT_SIM_CONFIG, buildObservation, createInitialState, stepSimulation } from './domain/simulation';
import { compileTrack, createDefaultTrack, getCheckpointTargets, hashTrack, validateTrackJson } from './domain/track';
import { createReplayBuilder, finalizeReplay, recordReplayFrame, sampleReplayFrame, validateReplayJson } from './domain/replay';
import type { AgentAction, AgentObservation, Mode, ReplayJson, SimulationState, TrackJson } from './domain/types';

type AgentKind = 'manual' | 'reference';
type EditorTool = 'select' | 'insert' | 'checkpoint' | 'barrier';

const seed = 1;

export default function App() {
  const [track, setTrack] = useState<TrackJson>(() => createDefaultTrack());
  const [mode, setMode] = useState<Mode>('drive');
  const [agentKind, setAgentKind] = useState<AgentKind>('manual');
  const [running, setRunning] = useState(false);
  const [recording, setRecording] = useState(false);
  const [replay, setReplay] = useState<ReplayJson | undefined>();
  const [replayTime, setReplayTime] = useState(0);
  const [replayPlaying, setReplayPlaying] = useState(false);
  const [editorTool, setEditorTool] = useState<EditorTool>('select');
  const [selectedPointId, setSelectedPointId] = useState<string | undefined>();
  const [selectedCheckpointId, setSelectedCheckpointId] = useState<string | undefined>();
  const [notice, setNotice] = useState('');
  const compiled = useMemo(() => {
    try {
      return compileTrack(track);
    } catch {
      return undefined;
    }
  }, [track]);
  const checkpointTargets = useMemo(() => (compiled ? getCheckpointTargets(compiled) : []), [compiled]);

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
  const replayBuilderRef = useRef<ReturnType<typeof createReplayBuilder> | null>(null);
  const trackImportRef = useRef<HTMLInputElement | null>(null);
  const replayImportRef = useRef<HTMLInputElement | null>(null);

  const replayTrackMismatch = replay ? replay.trackHash !== hashTrack(track) : false;

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
      referenceAgentRef.current.reset({
        track,
        seed,
        checkpointCount: track.checkpointCount,
        rayAngles: DEFAULT_SIM_CONFIG.rayAngles
      });
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
              : referenceAgentRef.current.step(beforeObservation, DEFAULT_SIM_CONFIG.fixedDt);
          const next = stepSimulation(state, action, compiled, DEFAULT_SIM_CONFIG.fixedDt, DEFAULT_SIM_CONFIG);
          const didCollide = next.events.some((event) => event.type === 'collision');
          const afterObservation = buildObservation(next.car, compiled, DEFAULT_SIM_CONFIG, didCollide);
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

    replayBuilderRef.current = createReplayBuilder(track, { id: agentKind, label: getAgentLabel(agentKind) }, seed, DEFAULT_SIM_CONFIG);
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

  function setStartAtSelectedPoint() {
    if (!selectedPointId) return;
    const pointIndex = track.centerline.findIndex((point) => point.id === selectedPointId);
    if (pointIndex < 0) return;
    setTrack((current) => ({ ...current, start: { ...current.start, pointIndex } }));
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
            replay={mode === 'analyze' ? replay : undefined}
            replayTime={replayTime}
            selectedPointId={selectedPointId}
            selectedCheckpointId={selectedCheckpointId}
            editorTool={editorTool}
            onTrackChange={setTrack}
            onSelectPoint={setSelectedPointId}
            onSelectCheckpoint={setSelectedCheckpointId}
          />
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
            </div>
            {agentKind === 'manual' ? (
              <div className="drivePad" aria-label="Manual controls">
                {[
                  ['ArrowLeft', '←'],
                  ['ArrowUp', '↑'],
                  ['ArrowDown', '↓'],
                  ['ArrowRight', '→']
                ].map(([code, label]) => (
                  <button
                    key={code}
                    className={heldControlsView.includes(code) ? 'active' : ''}
                    onPointerDown={() => holdControl(code, true)}
                    onPointerUp={() => holdControl(code, false)}
                    onPointerLeave={() => holdControl(code, false)}
                  >
                    {label}
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

          <section className="panelSection analysisPanel">
            <div className="sectionHeader">
              <h2>Analysis</h2>
            </div>
            <ReplayCharts replay={replay} />
          </section>
        </aside>
      </section>

      <input ref={trackImportRef} type="file" accept="application/json,.json" hidden onChange={(event) => void importTrack(event.target.files?.[0])} />
      <input ref={replayImportRef} type="file" accept="application/json,.json" hidden onChange={(event) => void importReplay(event.target.files?.[0])} />

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
