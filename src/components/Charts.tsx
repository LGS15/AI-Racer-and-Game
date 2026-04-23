import type { ReplayJson } from '../domain/types';
import { replayActionSeries, replayLapSeries, replaySpeedSeries } from '../domain/replay';

interface MiniLineChartProps {
  title: string;
  values: number[];
  color: string;
  min?: number;
  max?: number;
  suffix?: string;
}

export function MiniLineChart({ title, values, color, min, max, suffix = '' }: MiniLineChartProps) {
  const width = 320;
  const height = 96;
  const localMin = min ?? Math.min(...values, 0);
  const localMax = max ?? Math.max(...values, 1);
  const span = Math.max(0.001, localMax - localMin);
  const points = values
    .map((value, index) => {
      const x = values.length <= 1 ? 0 : (index / (values.length - 1)) * width;
      const y = height - ((value - localMin) / span) * height;
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(' ');
  const latest = values.length ? values[values.length - 1] : 0;

  return (
    <div className="chartBlock">
      <div className="chartHeader">
        <span>{title}</span>
        <strong>
          {latest.toFixed(latest >= 100 ? 0 : 2)}
          {suffix}
        </strong>
      </div>
      <svg className="chartSvg" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={title}>
        <path d={`M 0 ${height - 1} H ${width}`} className="chartAxis" />
        {values.length > 1 ? <polyline points={points} fill="none" stroke={color} strokeWidth="3" strokeLinecap="round" /> : null}
      </svg>
    </div>
  );
}

export function ActionChart({ replay }: { replay?: ReplayJson }) {
  const actions = replayActionSeries(replay);
  const width = 320;
  const height = 116;
  const line = (values: number[], min: number, max: number) => {
    const span = max - min;
    return values
      .map((value, index) => {
        const x = values.length <= 1 ? 0 : (index / (values.length - 1)) * width;
        const y = height - ((value - min) / span) * height;
        return `${x.toFixed(2)},${y.toFixed(2)}`;
      })
      .join(' ');
  };

  return (
    <div className="chartBlock">
      <div className="chartHeader">
        <span>Actions</span>
        <strong>{replay?.frames.length ?? 0}</strong>
      </div>
      <svg className="chartSvg actionChart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Agent actions">
        <path d={`M 0 ${height / 2} H ${width}`} className="chartAxis" />
        <polyline points={line(actions.throttle, -1, 1)} fill="none" stroke="#2aa876" strokeWidth="3" strokeLinecap="round" />
        <polyline points={line(actions.brake, -1, 1)} fill="none" stroke="#c04f48" strokeWidth="3" strokeLinecap="round" />
        <polyline points={line(actions.steer, -1, 1)} fill="none" stroke="#d89b22" strokeWidth="3" strokeLinecap="round" />
      </svg>
      <div className="legend">
        <span>
          <i style={{ background: '#2aa876' }} />
          Throttle
        </span>
        <span>
          <i style={{ background: '#c04f48' }} />
          Brake
        </span>
        <span>
          <i style={{ background: '#d89b22' }} />
          Steer
        </span>
      </div>
    </div>
  );
}

export function ReplayStats({ replay }: { replay?: ReplayJson }) {
  const laps = replayLapSeries(replay);
  const bestLap = laps.length ? Math.min(...laps.map((lap) => lap.time)) : undefined;
  return (
    <div className="statsGrid">
      <div>
        <span>Duration</span>
        <strong>{replay ? `${replay.duration.toFixed(1)}s` : '-'}</strong>
      </div>
      <div>
        <span>Laps</span>
        <strong>{laps.length}</strong>
      </div>
      <div>
        <span>Best</span>
        <strong>{bestLap ? `${bestLap.toFixed(2)}s` : '-'}</strong>
      </div>
      <div>
        <span>Events</span>
        <strong>{replay?.events.length ?? 0}</strong>
      </div>
    </div>
  );
}

export function ReplayCharts({ replay }: { replay?: ReplayJson }) {
  const speeds = replaySpeedSeries(replay);
  const lapTimes = replayLapSeries(replay).map((lap) => lap.time);
  return (
    <div className="analysisStack">
      <ReplayStats replay={replay} />
      <MiniLineChart title="Speed" values={speeds} color="#2f7fbc" max={540} suffix=" px/s" />
      <ActionChart replay={replay} />
      <MiniLineChart title="Lap Times" values={lapTimes} color="#7f5bb8" suffix="s" />
    </div>
  );
}
