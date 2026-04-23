import type { Vec2 } from './types';

export const TAU = Math.PI * 2;

export function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

export function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

export function distance(a: Vec2, b: Vec2): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

export function dot(a: Vec2, b: Vec2): number {
  return a.x * b.x + a.y * b.y;
}

export function length(v: Vec2): number {
  return Math.hypot(v.x, v.y);
}

export function normalize(v: Vec2): Vec2 {
  const len = length(v);
  return len <= 0.000001 ? { x: 1, y: 0 } : { x: v.x / len, y: v.y / len };
}

export function add(a: Vec2, b: Vec2): Vec2 {
  return { x: a.x + b.x, y: a.y + b.y };
}

export function sub(a: Vec2, b: Vec2): Vec2 {
  return { x: a.x - b.x, y: a.y - b.y };
}

export function scale(v: Vec2, amount: number): Vec2 {
  return { x: v.x * amount, y: v.y * amount };
}

export function fromAngle(angle: number): Vec2 {
  return { x: Math.cos(angle), y: Math.sin(angle) };
}

export function rightNormal(tangent: Vec2): Vec2 {
  return { x: -tangent.y, y: tangent.x };
}

export function normalizeAngle(angle: number): number {
  let wrapped = angle % TAU;
  if (wrapped > Math.PI) wrapped -= TAU;
  if (wrapped < -Math.PI) wrapped += TAU;
  return wrapped;
}

export function angleDifference(target: number, current: number): number {
  return normalizeAngle(target - current);
}

export function wrapDistance(distanceValue: number, totalLength: number): number {
  if (totalLength <= 0) return 0;
  const wrapped = distanceValue % totalLength;
  return wrapped < 0 ? wrapped + totalLength : wrapped;
}

export function forwardDelta(from: number, to: number, totalLength: number): number {
  let delta = to - from;
  if (delta < -totalLength / 2) delta += totalLength;
  if (delta > totalLength / 2) delta -= totalLength;
  return delta;
}

export function crossedForward(from: number, delta: number, target: number, totalLength: number): boolean {
  if (delta <= 0 || delta >= totalLength / 2) return false;
  const end = wrapDistance(from + delta, totalLength);
  if (end >= from) {
    return target > from && target <= end;
  }
  return target > from || target <= end;
}

export function projectPointToSegment(point: Vec2, a: Vec2, b: Vec2): { point: Vec2; t: number; distance: number } {
  const ab = sub(b, a);
  const abLengthSquared = dot(ab, ab);
  const t = abLengthSquared <= 0.000001 ? 0 : clamp(dot(sub(point, a), ab) / abLengthSquared, 0, 1);
  const projected = add(a, scale(ab, t));
  return { point: projected, t, distance: distance(point, projected) };
}
