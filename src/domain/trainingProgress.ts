import type { ValidationResult } from './types';

export interface TrainingProgressRow {
  event: string;
  step: number;
  epsilon?: number;
  bufferSize?: number;
  meanReward?: number;
  loss?: number;
  bestLapTime?: number;
  bestReward?: number;
  reason?: string;
  metric?: string;
  metricValue?: number;
}

export interface TrainingLapPoint {
  step: number;
  lapTime: number;
  bestLapTime: number;
  isNewBest: boolean;
  reason?: string;
  source: 'lap' | 'checkpoint';
}

const FIELD_ALIASES = {
  bufferSize: 'buffer_size',
  meanReward: 'mean_reward',
  bestLapTime: 'best_lap_time',
  bestReward: 'best_reward',
  metricValue: 'metric_value'
} as const;

export function parseTrainingProgressCsv(input: string): ValidationResult<TrainingProgressRow[]> {
  const warnings: string[] = [];
  const errors: string[] = [];
  const parsed = parseCsv(input).filter((row) => row.some((cell) => cell.trim()));
  if (parsed.length < 2) {
    return { ok: false, warnings, errors: ['Training progress CSV must include a header and at least one data row.'] };
  }

  const headers = parsed[0].map((header) => header.trim());
  if (!headers.includes('event') || !headers.includes('step')) {
    return { ok: false, warnings, errors: ['Training progress CSV must include event and step columns.'] };
  }

  const rows: TrainingProgressRow[] = [];
  parsed.slice(1).forEach((cells, index) => {
    const record = Object.fromEntries(headers.map((header, cellIndex) => [header, cells[cellIndex]?.trim() ?? '']));
    const event = record.event;
    const step = parseFiniteNumber(record.step);
    if (!event || step === undefined) {
      errors.push(`Training progress row ${index + 2} must include an event and finite step.`);
      return;
    }
    rows.push({
      event,
      step,
      epsilon: parseFiniteNumber(record.epsilon),
      bufferSize: parseFiniteNumber(record[FIELD_ALIASES.bufferSize]),
      meanReward: parseFiniteNumber(record[FIELD_ALIASES.meanReward]),
      loss: parseFiniteNumber(record.loss),
      bestLapTime: parseFiniteNumber(record[FIELD_ALIASES.bestLapTime]),
      bestReward: parseFiniteNumber(record[FIELD_ALIASES.bestReward]),
      reason: record.reason || undefined,
      metric: record.metric || undefined,
      metricValue: parseFiniteNumber(record[FIELD_ALIASES.metricValue])
    });
  });

  return { ok: errors.length === 0, value: errors.length === 0 ? rows : undefined, warnings, errors };
}

export function trainingLapSeries(rows: TrainingProgressRow[]): TrainingLapPoint[] {
  const lapRows = rows.filter((row) => row.event === 'lap' && row.metric === 'lap_time' && isPositive(row.metricValue));
  const sourceRows =
    lapRows.length > 0
      ? lapRows.map((row) => ({ row, source: 'lap' as const }))
      : rows
          .filter((row) => row.event === 'checkpoint' && row.metric === 'lap_time' && isPositive(row.metricValue))
          .map((row) => ({ row, source: 'checkpoint' as const }));

  let best = Number.POSITIVE_INFINITY;
  return sourceRows.map(({ row, source }) => {
    const lapTime = row.metricValue as number;
    const previousBest = best;
    const markedBest = /new (best|fastest)/i.test(row.reason ?? '');
    const isNewBest = markedBest || lapTime < previousBest - 0.0005;
    best = Math.min(best, row.bestLapTime ?? lapTime, lapTime);
    return {
      step: row.step,
      lapTime,
      bestLapTime: best,
      isNewBest,
      reason: row.reason,
      source
    };
  });
}

export function trainingIntervalRows(rows: TrainingProgressRow[]): TrainingProgressRow[] {
  return rows.filter((row) => row.event === 'interval');
}

function parseCsv(input: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = '';
  let quoted = false;

  for (let index = 0; index < input.length; index += 1) {
    const char = input[index];
    if (quoted) {
      if (char === '"' && input[index + 1] === '"') {
        field += '"';
        index += 1;
      } else if (char === '"') {
        quoted = false;
      } else {
        field += char;
      }
    } else if (char === '"') {
      quoted = true;
    } else if (char === ',') {
      row.push(field);
      field = '';
    } else if (char === '\n') {
      row.push(field);
      rows.push(row);
      row = [];
      field = '';
    } else if (char !== '\r') {
      field += char;
    }
  }

  if (field || row.length > 0) {
    row.push(field);
    rows.push(row);
  }
  return rows;
}

function parseFiniteNumber(value: string | undefined): number | undefined {
  if (value === undefined || value === '') return undefined;
  const number = Number(value);
  return Number.isFinite(number) ? number : undefined;
}

function isPositive(value: number | undefined): value is number {
  return value !== undefined && value > 0;
}
