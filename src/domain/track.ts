import { clamp, distance, dot, fromAngle, lerp, normalize, projectPointToSegment, rightNormal, sub, wrapDistance } from './geometry';
import type { ClosestTrackPoint, CompiledTrack, TrackCheckpoint, TrackJson, TrackPoint, TrackSample, ValidationResult, Vec2 } from './types';

const WORLD_W = 1200;
const WORLD_H = 800;
const MIN_TRACK_WIDTH = 40;
const MAX_TRACK_WIDTH = 260;

export function createDefaultTrack(): TrackJson {
  const now = new Date().toISOString();
  return {
    version: 1,
    metadata: {
      name: 'Test Loop',
      createdAt: now,
      description: 'Default centerline loop'
    },
    globalWidth: 118,
    start: {
      pointIndex: 0,
      direction: 1
    },
    checkpointCount: 12,
    centerline: [
      { id: 'p0', x: 260, y: 410 },
      { id: 'p1', x: 320, y: 205 },
      { id: 'p2', x: 575, y: 165 },
      { id: 'p3', x: 835, y: 215 },
      { id: 'p4', x: 985, y: 405 },
      { id: 'p5', x: 860, y: 620 },
      { id: 'p6', x: 565, y: 650 },
      { id: 'p7', x: 335, y: 565 }
    ]
  };
}

export function compileTrack(track: TrackJson): CompiledTrack {
  const points = track.centerline;
  if (points.length < 3) {
    throw new Error('A closed track needs at least three centerline points.');
  }

  let cumulative = 0;
  const segments = points.map((a, index) => {
    const b = points[(index + 1) % points.length];
    const segmentLength = Math.max(distance(a, b), 0.0001);
    const heading = Math.atan2(b.y - a.y, b.x - a.x);
    const segment = {
      index,
      a,
      b,
      length: segmentLength,
      cumulative,
      heading,
      widthA: a.width ?? track.globalWidth,
      widthB: b.width ?? track.globalWidth
    };
    cumulative += segmentLength;
    return segment;
  });

  const startIndex = clamp(Math.round(track.start.pointIndex), 0, points.length - 1);
  const startDistance = segments[startIndex]?.cumulative ?? 0;
  return {
    source: track,
    segments,
    totalLength: cumulative,
    startDistance
  };
}

export function sampleTrackAt(compiled: CompiledTrack, absoluteDistance: number): TrackSample {
  const wrappedDistance = wrapDistance(absoluteDistance, compiled.totalLength);
  const segment =
    compiled.segments.find((candidate) => wrappedDistance >= candidate.cumulative && wrappedDistance <= candidate.cumulative + candidate.length) ??
    compiled.segments[compiled.segments.length - 1];
  const t = clamp((wrappedDistance - segment.cumulative) / segment.length, 0, 1);
  const tangent = normalize(sub(segment.b, segment.a));
  const normal = rightNormal(tangent);
  const point = {
    x: lerp(segment.a.x, segment.b.x, t),
    y: lerp(segment.a.y, segment.b.y, t)
  };
  const width = lerp(segment.widthA, segment.widthB, t);
  return {
    point,
    tangent: compiled.source.start.direction === 1 ? tangent : { x: -tangent.x, y: -tangent.y },
    normal,
    heading: segment.heading + (compiled.source.start.direction === 1 ? 0 : Math.PI),
    width,
    absoluteDistance: wrappedDistance,
    progressDistance: toProgressDistance(compiled, wrappedDistance)
  };
}

export function toProgressDistance(compiled: CompiledTrack, absoluteDistance: number): number {
  return wrapDistance((absoluteDistance - compiled.startDistance) * compiled.source.start.direction, compiled.totalLength);
}

export function fromProgressDistance(compiled: CompiledTrack, progressDistance: number): number {
  return wrapDistance(compiled.startDistance + progressDistance * compiled.source.start.direction, compiled.totalLength);
}

export function getCheckpoints(compiled: CompiledTrack): TrackSample[] {
  return getCheckpointTargets(compiled).map((checkpoint) => checkpoint.sample);
}

export function getCheckpointTargets(compiled: CompiledTrack): Array<TrackCheckpoint & { index: number; sample: TrackSample; manual: boolean }> {
  const manualCheckpoints = (compiled.source.checkpoints ?? []).filter(
    (checkpoint) => typeof checkpoint.id === 'string' && checkpoint.id && Number.isFinite(checkpoint.progress)
  );
  if (manualCheckpoints.length >= 2) {
    return manualCheckpoints
      .map((checkpoint, index) => ({
        id: checkpoint.id,
        index,
        progress: wrapDistance(checkpoint.progress, compiled.totalLength),
        sample: sampleTrackAt(compiled, fromProgressDistance(compiled, checkpoint.progress)),
        manual: true
      }))
      .sort((a, b) => a.progress - b.progress)
      .map((checkpoint, index) => ({ ...checkpoint, index }));
  }

  const count = Math.max(2, Math.round(compiled.source.checkpointCount));
  return Array.from({ length: count }, (_, index) => {
    const progress = (compiled.totalLength / count) * index;
    return {
      id: `c${index}`,
      index,
      progress,
      sample: sampleTrackAt(compiled, fromProgressDistance(compiled, progress)),
      manual: false
    };
  });
}

export function closestPointOnTrack(compiled: CompiledTrack, point: Vec2): ClosestTrackPoint {
  let best: ClosestTrackPoint | undefined;

  for (const segment of compiled.segments) {
    const projection = projectPointToSegment(point, segment.a, segment.b);
    const segmentDistance = segment.cumulative + segment.length * projection.t;
    const sample = sampleTrackAt(compiled, segmentDistance);
    const toPoint = sub(point, projection.point);
    const lateralOffset = dot(toPoint, sample.normal);
    const halfWidth = sample.width / 2;
    const candidate: ClosestTrackPoint = {
      ...sample,
      segmentIndex: segment.index,
      segmentT: projection.t,
      lateralOffset,
      distanceToCenter: Math.abs(lateralOffset),
      onTrack: Math.abs(lateralOffset) <= halfWidth
    };

    if (!best || candidate.distanceToCenter < best.distanceToCenter) {
      best = candidate;
    }
  }

  if (!best) {
    throw new Error('Cannot query an empty track.');
  }
  return best;
}

export function isPointOnTrack(compiled: CompiledTrack, point: Vec2, margin = 0): boolean {
  const closest = closestPointOnTrack(compiled, point);
  return Math.abs(closest.lateralOffset) <= closest.width / 2 - margin;
}

export function validateTrackJson(input: unknown): ValidationResult<TrackJson> {
  const warnings: string[] = [];
  const errors: string[] = [];
  const value = input as Partial<TrackJson> | null;

  if (!value || typeof value !== 'object') {
    return { ok: false, warnings, errors: ['Track JSON must be an object.'] };
  }

  if (value.version !== 1) {
    errors.push('Track version must be 1.');
  }

  const rawPoints = Array.isArray(value.centerline) ? value.centerline : [];
  const centerline: TrackPoint[] = rawPoints
    .map<TrackPoint | undefined>((point, index) => {
      const candidate = point as Partial<TrackPoint>;
      const x = Number(candidate.x);
      const y = Number(candidate.y);
      const width = candidate.width === undefined ? undefined : Number(candidate.width);
      if (!Number.isFinite(x) || !Number.isFinite(y)) return undefined;
      const trackPoint: TrackPoint = {
        id: typeof candidate.id === 'string' && candidate.id ? candidate.id : `p${index}`,
        x: clamp(x, 0, WORLD_W),
        y: clamp(y, 0, WORLD_H)
      };
      if (Number.isFinite(width)) {
        trackPoint.width = clamp(width as number, MIN_TRACK_WIDTH, MAX_TRACK_WIDTH);
      }
      return trackPoint;
    })
    .filter((point): point is TrackPoint => Boolean(point));

  if (centerline.length !== rawPoints.length) {
    warnings.push('Some invalid centerline points were ignored.');
  }
  if (centerline.length < 3) {
    errors.push('Track needs at least three valid centerline points.');
  }

  const globalWidth = clamp(Number(value.globalWidth) || 110, MIN_TRACK_WIDTH, MAX_TRACK_WIDTH);
  const checkpointCount = clamp(Math.round(Number(value.checkpointCount) || 10), 2, 64);
  const pointIndex = clamp(Math.round(Number(value.start?.pointIndex) || 0), 0, Math.max(0, centerline.length - 1));
  const direction = value.start?.direction === -1 ? -1 : 1;
  const metadataValue = value.metadata ?? { name: 'Imported Track', createdAt: new Date().toISOString() };
  const name = typeof metadataValue.name === 'string' && metadataValue.name.trim() ? metadataValue.name.trim() : 'Imported Track';
  const createdAt =
    typeof metadataValue.createdAt === 'string' && metadataValue.createdAt.trim() ? metadataValue.createdAt : new Date().toISOString();

  const track: TrackJson = {
    version: 1,
    metadata: {
      name,
      createdAt,
      description: typeof metadataValue.description === 'string' ? metadataValue.description : undefined
    },
    centerline,
    globalWidth,
    start: { pointIndex, direction },
    checkpointCount
  };

  if (!errors.length) {
    try {
      const compiled = compileTrack(track);
      const rawCheckpoints = Array.isArray(value.checkpoints) ? value.checkpoints : [];
      if (rawCheckpoints.length > 0) {
        const checkpoints = rawCheckpoints
          .map<TrackCheckpoint | undefined>((checkpoint, index) => {
            const candidate = checkpoint as Partial<TrackCheckpoint>;
            const progress = Number(candidate.progress);
            if (!Number.isFinite(progress)) return undefined;
            return {
              id: typeof candidate.id === 'string' && candidate.id ? candidate.id : `c${index}`,
              progress: wrapDistance(progress, compiled.totalLength)
            };
          })
          .filter((checkpoint): checkpoint is TrackCheckpoint => Boolean(checkpoint));
        if (checkpoints.length !== rawCheckpoints.length) {
          warnings.push('Some invalid checkpoints were ignored.');
        }
        if (checkpoints.length >= 2) {
          track.checkpoints = checkpoints;
          track.checkpointCount = checkpoints.length;
        } else {
          warnings.push('Manual checkpoints need at least two valid gates; using evenly spaced checkpoints.');
        }
      }
    } catch (error) {
      errors.push(error instanceof Error ? error.message : 'Track could not be compiled.');
    }
  }

  return { ok: errors.length === 0, value: errors.length === 0 ? track : undefined, warnings, errors };
}

export function insertPointAfterNearestSegment(track: TrackJson, point: Vec2): TrackJson {
  const compiled = compileTrack(track);
  let bestIndex = 0;
  let bestDistance = Number.POSITIVE_INFINITY;

  for (const segment of compiled.segments) {
    const projection = projectPointToSegment(point, segment.a, segment.b);
    if (projection.distance < bestDistance) {
      bestDistance = projection.distance;
      bestIndex = segment.index;
    }
  }

  const id = `p${Date.now().toString(36)}${Math.round(point.x)}${Math.round(point.y)}`;
  const centerline = [...track.centerline];
  centerline.splice(bestIndex + 1, 0, {
    id,
    x: clamp(point.x, 0, WORLD_W),
    y: clamp(point.y, 0, WORLD_H)
  });
  return { ...track, centerline };
}

export function updateTrackPoint(track: TrackJson, pointId: string, position: Vec2): TrackJson {
  return {
    ...track,
    centerline: track.centerline.map((point) =>
      point.id === pointId ? { ...point, x: clamp(position.x, 0, WORLD_W), y: clamp(position.y, 0, WORLD_H) } : point
    )
  };
}

export function updateTrackPointWidth(track: TrackJson, pointId: string, width: number): TrackJson {
  return {
    ...track,
    centerline: track.centerline.map((point) =>
      point.id === pointId ? { ...point, width: clamp(width, MIN_TRACK_WIDTH, MAX_TRACK_WIDTH) } : point
    )
  };
}

export function removeTrackPoint(track: TrackJson, pointId: string): TrackJson {
  if (track.centerline.length <= 3) return track;
  const removeIndex = track.centerline.findIndex((point) => point.id === pointId);
  if (removeIndex < 0) return track;
  const centerline = track.centerline.filter((point) => point.id !== pointId);
  return {
    ...track,
    centerline,
    start: {
      ...track.start,
      pointIndex: clamp(track.start.pointIndex > removeIndex ? track.start.pointIndex - 1 : track.start.pointIndex, 0, centerline.length - 1)
    }
  };
}

export function ensureManualCheckpoints(compiled: CompiledTrack): TrackCheckpoint[] {
  return getCheckpointTargets(compiled).map((checkpoint) => ({
    id: checkpoint.id,
    progress: wrapDistance(checkpoint.progress, compiled.totalLength)
  }));
}

export function moveCheckpoint(track: TrackJson, compiled: CompiledTrack, checkpointId: string, point: Vec2): TrackJson {
  const closest = closestPointOnTrack(compiled, point);
  const checkpoints = ensureManualCheckpoints(compiled).map((checkpoint) =>
    checkpoint.id === checkpointId ? { ...checkpoint, progress: closest.progressDistance } : checkpoint
  );
  return {
    ...track,
    checkpoints,
    checkpointCount: checkpoints.length
  };
}

export function addCheckpoint(track: TrackJson, compiled: CompiledTrack, point: Vec2): TrackJson {
  const closest = closestPointOnTrack(compiled, point);
  const checkpoints = [
    ...ensureManualCheckpoints(compiled),
    {
      id: `c${Date.now().toString(36)}${Math.round(point.x)}${Math.round(point.y)}`,
      progress: closest.progressDistance
    }
  ].sort((a, b) => a.progress - b.progress);
  return {
    ...track,
    checkpoints,
    checkpointCount: checkpoints.length
  };
}

export function removeCheckpoint(track: TrackJson, compiled: CompiledTrack, checkpointId: string): TrackJson {
  const checkpoints = ensureManualCheckpoints(compiled).filter((checkpoint) => checkpoint.id !== checkpointId);
  if (checkpoints.length < 2) return track;
  return {
    ...track,
    checkpoints,
    checkpointCount: checkpoints.length
  };
}

export function hashTrack(track: TrackJson): string {
  const stable = JSON.stringify({
    version: track.version,
    width: Math.round(track.globalWidth * 100) / 100,
    start: track.start,
    checkpoints: track.checkpointCount,
    manualCheckpoints: track.checkpoints?.map((checkpoint) => ({
      progress: Math.round(checkpoint.progress * 100) / 100
    })),
    centerline: track.centerline.map((point) => ({
      x: Math.round(point.x * 100) / 100,
      y: Math.round(point.y * 100) / 100,
      width: point.width === undefined ? undefined : Math.round(point.width * 100) / 100
    }))
  });
  let hash = 2166136261;
  for (let index = 0; index < stable.length; index += 1) {
    hash ^= stable.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0).toString(16).padStart(8, '0');
}

export function seededTrackClone(track: TrackJson): TrackJson {
  return JSON.parse(JSON.stringify(track)) as TrackJson;
}
