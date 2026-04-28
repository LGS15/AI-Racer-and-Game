import { useEffect, useRef } from 'react';
import { CirclePlus, Plus } from 'lucide-react';
import { DEFAULT_SIM_CONFIG } from '../domain/simulation';
import {
  addCheckpoint,
  compileTrack,
  getCheckpointTargets,
  getCheckpoints,
  getTrackPointBarriers,
  getTrackPointNormal,
  insertPointAfterNearestSegment,
  moveCheckpoint,
  removeCheckpoint,
  removeTrackPoint,
  updateTrackPoint,
  updateTrackPointWidth
} from '../domain/track';
import { distance, dot, fromAngle, normalize, rightNormal, sub } from '../domain/geometry';
import { sampleReplayFrame } from '../domain/replay';
import type { AgentObservation, CarState, CompiledTrack, Mode, ReplayJson, TrackJson, TrackPoint, Vec2 } from '../domain/types';

const WORLD_WIDTH = 1200;
const WORLD_HEIGHT = 800;

type EditorTool = 'select' | 'insert' | 'checkpoint' | 'barrier';
type DragTarget =
  | { type: 'point'; id: string }
  | { type: 'checkpoint'; id: string }
  | { type: 'barrier'; id: string };

interface RacingCanvasProps {
  mode: Mode;
  track: TrackJson;
  compiled?: CompiledTrack;
  car?: CarState;
  observation?: AgentObservation;
  replay?: ReplayJson;
  replayTime: number;
  displayZoom: number;
  selectedPointId?: string;
  selectedCheckpointId?: string;
  editorTool: EditorTool;
  onTrackChange(track: TrackJson): void;
  onSelectPoint(pointId?: string): void;
  onSelectCheckpoint(checkpointId?: string): void;
}

interface Viewport {
  scale: number;
  offsetX: number;
  offsetY: number;
  width: number;
  height: number;
}

export default function RacingCanvas({
  mode,
  track,
  compiled,
  car,
  observation,
  replay,
  replayTime,
  displayZoom,
  selectedPointId,
  selectedCheckpointId,
  editorTool,
  onTrackChange,
  onSelectPoint,
  onSelectCheckpoint
}: RacingCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const draggingRef = useRef<DragTarget | undefined>(undefined);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return undefined;
    const resize = () => draw();
    const observer = new ResizeObserver(resize);
    observer.observe(canvas);
    draw();
    return () => observer.disconnect();
  });

  useEffect(() => {
    draw();
  }, [mode, track, compiled, car, observation, replay, replayTime, displayZoom, selectedPointId, selectedCheckpointId, editorTool]);

  function draw() {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const nextWidth = Math.max(1, Math.floor(rect.width * dpr));
    const nextHeight = Math.max(1, Math.floor(rect.height * dpr));
    if (canvas.width !== nextWidth || canvas.height !== nextHeight) {
      canvas.width = nextWidth;
      canvas.height = nextHeight;
    }
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);
    ctx.fillStyle = '#1d2c22';
    ctx.fillRect(0, 0, rect.width, rect.height);

    const viewport = getViewport(rect.width, rect.height, displayZoom);
    ctx.save();
    ctx.translate(viewport.offsetX, viewport.offsetY);
    ctx.scale(viewport.scale, viewport.scale);
    drawWorld(ctx);
    if (compiled) {
      drawTrack(ctx, compiled);
      drawReplay(ctx, replay, replayTime);
      if (car && mode !== 'analyze') drawSensors(ctx, car, observation);
      if (car && mode !== 'analyze') drawCar(ctx, car, '#e4574f', '#ffffff');
      if (mode === 'edit') drawEditor(ctx, compiled, track, selectedPointId, selectedCheckpointId, editorTool);
    }
    ctx.restore();
  }

  function handlePointerDown(event: React.PointerEvent<HTMLCanvasElement>) {
    if (mode !== 'edit') return;
    const world = eventToWorld(event);
    if (!world) return;
    event.currentTarget.setPointerCapture(event.pointerId);

    if (editorTool === 'insert') {
      const nextTrack = insertPointAfterNearestSegment(track, world);
      const inserted = nearestPoint(nextTrack.centerline, world);
      onTrackChange(nextTrack);
      onSelectPoint(inserted?.id);
      onSelectCheckpoint(undefined);
      return;
    }

    if (editorTool === 'checkpoint' && compiled) {
      const hit = nearestCheckpoint(compiled, world, 30);
      if (hit) {
        draggingRef.current = { type: 'checkpoint', id: hit.id };
        onSelectCheckpoint(hit.id);
        onSelectPoint(undefined);
        return;
      }
      const nextTrack = addCheckpoint(track, compiled, world);
      const nextCompiled = compileTrack(nextTrack);
      const inserted = nearestCheckpoint(nextCompiled, world);
      onTrackChange(nextTrack);
      onSelectCheckpoint(inserted?.id);
      onSelectPoint(undefined);
      return;
    }

    if (editorTool === 'barrier') {
      const hit = nearestBarrierHandle(track, world, 28);
      if (hit) {
        draggingRef.current = { type: 'barrier', id: hit.point.id };
        onSelectPoint(hit.point.id);
        onSelectCheckpoint(undefined);
        onTrackChange(updatePointWidthFromWorld(track, hit.point.id, world));
        return;
      }
      const point = nearestPoint(track.centerline, world, 24);
      draggingRef.current = point ? { type: 'point', id: point.id } : undefined;
      onSelectPoint(point?.id);
      onSelectCheckpoint(undefined);
      return;
    }

    const hit = nearestPoint(track.centerline, world, 26);
    if (hit) {
      draggingRef.current = { type: 'point', id: hit.id };
      onSelectPoint(hit.id);
      onSelectCheckpoint(undefined);
    } else {
      draggingRef.current = undefined;
      onSelectPoint(undefined);
      onSelectCheckpoint(undefined);
    }
  }

  function handlePointerMove(event: React.PointerEvent<HTMLCanvasElement>) {
    if (mode !== 'edit' || !draggingRef.current) return;
    const world = eventToWorld(event);
    if (!world) return;
    if (draggingRef.current.type === 'point') {
      onTrackChange(updateTrackPoint(track, draggingRef.current.id, world));
    } else if (draggingRef.current.type === 'checkpoint' && compiled) {
      onTrackChange(moveCheckpoint(track, compiled, draggingRef.current.id, world));
    } else if (draggingRef.current.type === 'barrier') {
      onTrackChange(updatePointWidthFromWorld(track, draggingRef.current.id, world));
    }
  }

  function handlePointerUp(event: React.PointerEvent<HTMLCanvasElement>) {
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    draggingRef.current = undefined;
  }

  function handleDoubleClick(event: React.MouseEvent<HTMLCanvasElement>) {
    if (mode !== 'edit') return;
    if (editorTool !== 'select' && editorTool !== 'insert') return;
    const world = eventToWorld(event);
    if (!world) return;
    const nextTrack = insertPointAfterNearestSegment(track, world);
    onTrackChange(nextTrack);
    onSelectPoint(nearestPoint(nextTrack.centerline, world)?.id);
    onSelectCheckpoint(undefined);
  }

  function eventToWorld(event: React.PointerEvent<HTMLCanvasElement> | React.MouseEvent<HTMLCanvasElement>): Vec2 | undefined {
    const canvas = canvasRef.current;
    if (!canvas) return undefined;
    const rect = canvas.getBoundingClientRect();
    const viewport = getViewport(rect.width, rect.height, displayZoom);
    return {
      x: (event.clientX - rect.left - viewport.offsetX) / viewport.scale,
      y: (event.clientY - rect.top - viewport.offsetY) / viewport.scale
    };
  }

  return (
    <div className="canvasWrap">
      <canvas
        ref={canvasRef}
        className={mode === 'edit' ? 'raceCanvas editing' : 'raceCanvas'}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerCancel={handlePointerUp}
        onDoubleClick={handleDoubleClick}
      />
      {mode === 'edit' && editorTool === 'insert' ? (
        <div className="canvasBadge">
          <Plus size={16} />
        </div>
      ) : null}
      {mode === 'edit' && editorTool === 'checkpoint' ? (
        <div className="canvasBadge">
          <CirclePlus size={16} />
        </div>
      ) : null}
    </div>
  );
}

export function deleteSelectedPoint(track: TrackJson, pointId?: string): TrackJson {
  return pointId ? removeTrackPoint(track, pointId) : track;
}

export function deleteSelectedCheckpoint(track: TrackJson, compiled: CompiledTrack | undefined, checkpointId?: string): TrackJson {
  return compiled && checkpointId ? removeCheckpoint(track, compiled, checkpointId) : track;
}

function getViewport(width: number, height: number, displayZoom: number): Viewport {
  const scale = Math.min(width / WORLD_WIDTH, height / WORLD_HEIGHT) * Math.max(0.1, displayZoom);
  const viewWidth = WORLD_WIDTH * scale;
  const viewHeight = WORLD_HEIGHT * scale;
  return {
    scale,
    offsetX: (width - viewWidth) / 2,
    offsetY: (height - viewHeight) / 2,
    width,
    height
  };
}

function drawWorld(ctx: CanvasRenderingContext2D) {
  ctx.fillStyle = '#243928';
  ctx.fillRect(0, 0, WORLD_WIDTH, WORLD_HEIGHT);
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.035)';
  ctx.lineWidth = 1;
  for (let x = 0; x <= WORLD_WIDTH; x += 50) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, WORLD_HEIGHT);
    ctx.stroke();
  }
  for (let y = 0; y <= WORLD_HEIGHT; y += 50) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(WORLD_WIDTH, y);
    ctx.stroke();
  }
}

function drawTrack(ctx: CanvasRenderingContext2D, compiled: CompiledTrack) {
  const path = new Path2D();
  compiled.source.centerline.forEach((point, index) => {
    if (index === 0) path.moveTo(point.x, point.y);
    else path.lineTo(point.x, point.y);
  });
  path.closePath();
  drawTrackBand(ctx, compiled, 22, '#191916');
  drawTrackBand(ctx, compiled, 0, '#5a5953');
  drawTrackBand(ctx, compiled, -20, '#3f3e3a');
  ctx.save();
  ctx.setLineDash([22, 24]);
  ctx.strokeStyle = 'rgba(245, 213, 112, 0.72)';
  ctx.lineWidth = 4;
  ctx.stroke(path);
  ctx.restore();

  for (const [index, checkpoint] of getCheckpoints(compiled).entries()) {
    const half = checkpoint.width / 2;
    const a = { x: checkpoint.point.x - checkpoint.normal.x * half, y: checkpoint.point.y - checkpoint.normal.y * half };
    const b = { x: checkpoint.point.x + checkpoint.normal.x * half, y: checkpoint.point.y + checkpoint.normal.y * half };
    ctx.save();
    ctx.lineWidth = index === 0 ? 7 : 3;
    ctx.strokeStyle = index === 0 ? '#ffffff' : 'rgba(255, 236, 160, 0.35)';
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
    if (index === 0) {
      ctx.setLineDash([10, 10]);
      ctx.strokeStyle = '#111111';
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
    }
    ctx.restore();
  }
}

function drawReplay(ctx: CanvasRenderingContext2D, replay: ReplayJson | undefined, replayTime: number) {
  if (!replay || replay.frames.length === 0) return;
  ctx.save();
  ctx.lineWidth = 4;
  ctx.strokeStyle = 'rgba(39, 185, 170, 0.45)';
  ctx.beginPath();
  replay.frames.forEach((frame, index) => {
    if (index === 0) ctx.moveTo(frame.x, frame.y);
    else if (index % 2 === 0) ctx.lineTo(frame.x, frame.y);
  });
  ctx.stroke();
  const ghost = sampleReplayFrame(replay, replayTime);
  if (ghost) drawCar(ctx, ghost, '#26b8aa', '#eafffb', 0.74);
  ctx.restore();
}

function drawSensors(ctx: CanvasRenderingContext2D, car: CarState, observation?: AgentObservation) {
  if (!observation) return;
  ctx.save();
  ctx.lineWidth = 2;
  observation.rayAngles.forEach((angle, index) => {
    const length = observation.rays[index] ?? DEFAULT_SIM_CONFIG.rayMaxDistance;
    const dir = fromAngle(car.heading + angle);
    ctx.strokeStyle = index === Math.floor(observation.rays.length / 2) ? 'rgba(255, 222, 105, 0.78)' : 'rgba(255, 222, 105, 0.35)';
    ctx.beginPath();
    ctx.moveTo(car.x, car.y);
    ctx.lineTo(car.x + dir.x * length, car.y + dir.y * length);
    ctx.stroke();
  });
  ctx.restore();
}

function drawCar(
  ctx: CanvasRenderingContext2D,
  car: { x: number; y: number; heading: number },
  body: string,
  stripe: string,
  alpha = 1
) {
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.translate(car.x, car.y);
  ctx.rotate(car.heading);
  ctx.fillStyle = 'rgba(0, 0, 0, 0.35)';
  ctx.fillRect(-17, -9, 38, 20);
  ctx.fillStyle = body;
  roundedRect(ctx, -19, -10, 38, 20, 4);
  ctx.fill();
  ctx.fillStyle = stripe;
  roundedRect(ctx, 0, -7, 13, 14, 3);
  ctx.fill();
  ctx.fillStyle = '#202020';
  ctx.fillRect(-13, -13, 8, 4);
  ctx.fillRect(-13, 9, 8, 4);
  ctx.fillRect(8, -13, 8, 4);
  ctx.fillRect(8, 9, 8, 4);
  ctx.restore();
}

function drawEditor(
  ctx: CanvasRenderingContext2D,
  compiled: CompiledTrack,
  track: TrackJson,
  selectedPointId: string | undefined,
  selectedCheckpointId: string | undefined,
  editorTool: EditorTool
) {
  ctx.save();
  ctx.lineWidth = 2;
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.55)';
  ctx.beginPath();
  track.centerline.forEach((point, index) => {
    if (index === 0) ctx.moveTo(point.x, point.y);
    else ctx.lineTo(point.x, point.y);
  });
  ctx.closePath();
  ctx.stroke();

  track.centerline.forEach((point, index) => {
    const isSelected = point.id === selectedPointId;
    const isStart = index === track.start.pointIndex;
    ctx.beginPath();
    ctx.arc(point.x, point.y, isSelected ? 13 : 9, 0, Math.PI * 2);
    ctx.fillStyle = isSelected ? '#f0a72f' : isStart ? '#ffffff' : '#35b38a';
    ctx.fill();
    ctx.lineWidth = isStart ? 4 : 2;
    ctx.strokeStyle = isStart ? '#e4574f' : '#123327';
    ctx.stroke();
  });

  if (editorTool === 'checkpoint') {
    for (const checkpoint of getCheckpointTargets(compiled)) {
      const isSelected = checkpoint.id === selectedCheckpointId;
      const half = checkpoint.sample.width / 2;
      const a = {
        x: checkpoint.sample.point.x - checkpoint.sample.normal.x * half,
        y: checkpoint.sample.point.y - checkpoint.sample.normal.y * half
      };
      const b = {
        x: checkpoint.sample.point.x + checkpoint.sample.normal.x * half,
        y: checkpoint.sample.point.y + checkpoint.sample.normal.y * half
      };
      ctx.lineWidth = isSelected ? 6 : 4;
      ctx.strokeStyle = isSelected ? '#f0a72f' : 'rgba(75, 190, 225, 0.9)';
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(checkpoint.sample.point.x, checkpoint.sample.point.y, isSelected ? 12 : 9, 0, Math.PI * 2);
      ctx.fillStyle = isSelected ? '#f0a72f' : '#4bbee1';
      ctx.fill();
      ctx.strokeStyle = '#12313a';
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }

  if (editorTool === 'barrier') {
    for (const handle of getBarrierHandles(track)) {
      const isSelected = handle.point.id === selectedPointId;
      ctx.beginPath();
      ctx.arc(handle.position.x, handle.position.y, isSelected ? 9 : 7, 0, Math.PI * 2);
      ctx.fillStyle = handle.side === 1 ? '#e4574f' : '#4bbee1';
      ctx.fill();
      ctx.lineWidth = isSelected ? 3 : 2;
      ctx.strokeStyle = '#1b1714';
      ctx.stroke();
    }
  }

  if (editorTool === 'insert') {
    ctx.setLineDash([12, 12]);
    ctx.strokeStyle = 'rgba(240, 167, 47, 0.8)';
    ctx.lineWidth = 4;
    ctx.strokeRect(12, 12, WORLD_WIDTH - 24, WORLD_HEIGHT - 24);
  }
  ctx.restore();
}

function drawTrackBand(ctx: CanvasRenderingContext2D, compiled: CompiledTrack, widthOffset: number, fillStyle: string) {
  const points = compiled.source.centerline;
  if (points.length < 3) return;
  ctx.save();
  ctx.fillStyle = fillStyle;
  points.forEach((point, index) => {
    const next = points[(index + 1) % points.length];
    const normal = rightNormal(normalize(sub(next, point)));
    const halfA = Math.max(6, ((point.width ?? compiled.source.globalWidth) + widthOffset) / 2);
    const halfB = Math.max(6, ((next.width ?? compiled.source.globalWidth) + widthOffset) / 2);
    ctx.beginPath();
    ctx.moveTo(point.x - normal.x * halfA, point.y - normal.y * halfA);
    ctx.lineTo(next.x - normal.x * halfB, next.y - normal.y * halfB);
    ctx.lineTo(next.x + normal.x * halfB, next.y + normal.y * halfB);
    ctx.lineTo(point.x + normal.x * halfA, point.y + normal.y * halfA);
    ctx.closePath();
    ctx.fill();
  });
  for (const point of points) {
    const radius = Math.max(6, ((point.width ?? compiled.source.globalWidth) + widthOffset) / 2);
    ctx.beginPath();
    ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

function getBarrierHandles(track: TrackJson): Array<{ point: TrackPoint; side: -1 | 1; position: Vec2 }> {
  return getTrackPointBarriers(track).flatMap((barrier) => {
    return [
      { point: barrier.point, side: -1 as const, position: barrier.left },
      { point: barrier.point, side: 1 as const, position: barrier.right }
    ];
  });
}

function nearestBarrierHandle(track: TrackJson, target: Vec2, maxDistance = Number.POSITIVE_INFINITY) {
  let best: ReturnType<typeof getBarrierHandles>[number] | undefined;
  let bestDistance = maxDistance;
  for (const handle of getBarrierHandles(track)) {
    const candidateDistance = distance(handle.position, target);
    if (candidateDistance <= bestDistance) {
      best = handle;
      bestDistance = candidateDistance;
    }
  }
  return best;
}

function updatePointWidthFromWorld(track: TrackJson, pointId: string, world: Vec2): TrackJson {
  const index = track.centerline.findIndex((point) => point.id === pointId);
  if (index < 0) return track;
  const point = track.centerline[index];
  const normal = getTrackPointNormal(track.centerline, index);
  const offset = Math.abs(dot(sub(world, point), normal));
  return updateTrackPointWidth(track, pointId, offset * 2);
}

function nearestCheckpoint(compiled: CompiledTrack, target: Vec2, maxDistance = Number.POSITIVE_INFINITY) {
  let best: ReturnType<typeof getCheckpointTargets>[number] | undefined;
  let bestDistance = maxDistance;
  for (const checkpoint of getCheckpointTargets(compiled)) {
    const candidateDistance = distance(checkpoint.sample.point, target);
    if (candidateDistance <= bestDistance) {
      best = checkpoint;
      bestDistance = candidateDistance;
    }
  }
  return best;
}

function nearestPoint(points: TrackPoint[], target: Vec2, maxDistance = Number.POSITIVE_INFINITY): TrackPoint | undefined {
  let best: TrackPoint | undefined;
  let bestDistance = maxDistance;
  for (const point of points) {
    const candidateDistance = distance(point, target);
    if (candidateDistance <= bestDistance) {
      best = point;
      bestDistance = candidateDistance;
    }
  }
  return best;
}

function roundedRect(ctx: CanvasRenderingContext2D, x: number, y: number, width: number, height: number, radius: number) {
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.arcTo(x + width, y, x + width, y + height, radius);
  ctx.arcTo(x + width, y + height, x, y + height, radius);
  ctx.arcTo(x, y + height, x, y, radius);
  ctx.arcTo(x, y, x + width, y, radius);
  ctx.closePath();
}
