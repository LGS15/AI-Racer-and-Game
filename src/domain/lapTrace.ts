import type { LapTraceJson, ValidationResult } from './types';

export function validateLapTraceJson(input: unknown): ValidationResult<LapTraceJson> {
  const warnings: string[] = [];
  const errors: string[] = [];
  const value = input as Partial<LapTraceJson> | null;
  if (!value || typeof value !== 'object') {
    return { ok: false, warnings, errors: ['Lap trace JSON must be an object.'] };
  }
  if (value.version !== 1) errors.push('Lap trace version must be 1.');
  if (value.trackVersion !== 1) errors.push('Lap trace trackVersion must be 1.');
  if (typeof value.trackHash !== 'string') errors.push('Lap trace must include a trackHash.');
  if (!value.agent || typeof value.agent.id !== 'string') errors.push('Lap trace must include agent metadata.');
  if (!Number.isFinite(value.lapTime) || Number(value.lapTime) <= 0) errors.push('Lap trace must include a positive lapTime.');
  if (!Array.isArray(value.points) || value.points.length < 2) errors.push('Lap trace must include at least two points.');
  if (Array.isArray(value.points) && value.points.some((point) => !isFinitePoint(point))) {
    errors.push('Lap trace points must include finite t, x, y, speed, and progressDistance values.');
  }

  const trace = value as LapTraceJson;
  return { ok: errors.length === 0, value: errors.length === 0 ? trace : undefined, warnings, errors };
}

function isFinitePoint(point: unknown): boolean {
  const value = point as Record<string, unknown> | null;
  return Boolean(
    value &&
      Number.isFinite(value.t) &&
      Number.isFinite(value.x) &&
      Number.isFinite(value.y) &&
      Number.isFinite(value.speed) &&
      Number.isFinite(value.progressDistance) &&
      typeof value.offTrack === 'boolean'
  );
}
