import type { ReplayJson } from '../domain/types';
import { replayActionSeries, replayLapSeries, replaySpeedSeries } from '../domain/replay';
import { trainingIntervalRows, trainingLapSeries, type TrainingProgressRow } from '../domain/trainingProgress';

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
        <polyline points={line(actions.throttle, -1, 1)} fill="none" stroke="#12a574" strokeWidth="3" strokeLinecap="round" />
        <polyline points={line(actions.brake, -1, 1)} fill="none" stroke="#e43d52" strokeWidth="3" strokeLinecap="round" />
        <polyline points={line(actions.steer, -1, 1)} fill="none" stroke="#d8ff2f" strokeWidth="3" strokeLinecap="round" />
      </svg>
      <div className="legend">
        <span>
          <i style={{ background: '#12a574' }} />
          Throttle
        </span>
        <span>
          <i style={{ background: '#e43d52' }} />
          Brake
        </span>
        <span>
          <i style={{ background: '#d8ff2f' }} />
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
      <MiniLineChart title="Speed" values={speeds} color="#00b8df" max={540} suffix=" px/s" />
      <ActionChart replay={replay} />
      <MiniLineChart title="Lap Times" values={lapTimes} color="#725cff" suffix="s" />
    </div>
  );
}

export function TrainingProgressCharts({ rows }: { rows?: TrainingProgressRow[] }) {
  const intervals = trainingIntervalRows(rows ?? []);
  const laps = trainingLapSeries(rows ?? []);
  const rewards = intervals.map((row) => row.meanReward).filter(isNumber);
  const losses = intervals.map((row) => row.loss).filter(isNumber);
  const milestones = laps.filter((lap) => lap.isNewBest);

  return (
    <div className="trainingStack">
      <TrainingStats rows={rows} laps={laps} />
      <TrainingLapChart laps={laps} />
      <MiniLineChart title="Mean Reward" values={rewards} color="#12a574" suffix="" />
      <MiniLineChart title="Loss" values={losses} color="#e43d52" suffix="" />
      <TrainingMilestones laps={milestones} />
    </div>
  );
}

function TrainingStats({ rows, laps }: { rows?: TrainingProgressRow[]; laps: ReturnType<typeof trainingLapSeries> }) {
  const intervals = trainingIntervalRows(rows ?? []);
  const latestInterval = intervals[intervals.length - 1];
  const fastest = laps.length ? Math.min(...laps.map((lap) => lap.lapTime)) : undefined;
  const firstLap = laps[0]?.lapTime;
  const improvement = fastest !== undefined && firstLap !== undefined ? firstLap - fastest : undefined;

  return (
    <div className="statsGrid trainingStats">
      <div>
        <span>Laps Logged</span>
        <strong>{laps.length}</strong>
      </div>
      <div>
        <span>Fastest</span>
        <strong>{fastest ? `${fastest.toFixed(2)}s` : '-'}</strong>
      </div>
      <div>
        <span>Improved</span>
        <strong>{improvement && improvement > 0 ? `${improvement.toFixed(2)}s` : '-'}</strong>
      </div>
      <div>
        <span>Latest Reward</span>
        <strong>{formatMetric(latestInterval?.meanReward)}</strong>
      </div>
    </div>
  );
}

function TrainingLapChart({ laps }: { laps: ReturnType<typeof trainingLapSeries> }) {
  const width = 320;
  const height = 142;
  const padding = 14;
  const values = laps.map((lap) => lap.lapTime);
  const minStep = laps[0]?.step ?? 0;
  const maxStep = laps[laps.length - 1]?.step ?? minStep + 1;
  const minLap = values.length ? Math.min(...values) : 0;
  const maxLap = values.length ? Math.max(...values) : 1;
  const stepSpan = Math.max(1, maxStep - minStep);
  const lapSpan = Math.max(0.001, maxLap - minLap);
  const x = (step: number) => padding + ((step - minStep) / stepSpan) * (width - padding * 2);
  const y = (lapTime: number) => padding + ((lapTime - minLap) / lapSpan) * (height - padding * 2);
  const points = laps.map((lap) => `${x(lap.step).toFixed(2)},${y(lap.lapTime).toFixed(2)}`).join(' ');

  return (
    <div className="chartBlock">
      <div className="chartHeader">
        <span>Training Lap Times</span>
        <strong>{laps.length ? `${laps[laps.length - 1].lapTime.toFixed(2)}s` : '-'}</strong>
      </div>
      <svg className="chartSvg trainingLapChart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Training lap times">
        <path d={`M ${padding} ${height - padding} H ${width - padding}`} className="chartAxis" />
        {laps.length > 1 ? <polyline points={points} fill="none" stroke="#725cff" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" /> : null}
        {laps.map((lap) => (
          <circle
            key={`${lap.source}-${lap.step}-${lap.lapTime}`}
            cx={x(lap.step)}
            cy={y(lap.lapTime)}
            r={lap.isNewBest ? 5 : 3}
            className={lap.isNewBest ? 'lapMarker newBest' : 'lapMarker'}
          >
            <title>{`${lap.isNewBest ? 'New fastest lap' : 'Lap'} at step ${Math.round(lap.step)}: ${lap.lapTime.toFixed(3)}s`}</title>
          </circle>
        ))}
      </svg>
    </div>
  );
}

function TrainingMilestones({ laps }: { laps: ReturnType<typeof trainingLapSeries> }) {
  return (
    <div className="milestoneBlock">
      <div className="chartHeader">
        <span>Fastest Lap Updates</span>
        <strong>{laps.length}</strong>
      </div>
      <div className="milestoneList">
        {laps.length ? (
          laps
            .slice(-6)
            .reverse()
            .map((lap) => (
              <div className="milestoneRow" key={`${lap.step}-${lap.lapTime}`}>
                <span>Step {Math.round(lap.step).toLocaleString()}</span>
                <strong>{lap.lapTime.toFixed(3)}s</strong>
              </div>
            ))
        ) : (
          <div className="milestoneRow empty">
            <span>No fastest laps yet</span>
            <strong>-</strong>
          </div>
        )}
      </div>
    </div>
  );
}

function isNumber(value: number | undefined): value is number {
  return value !== undefined;
}

function formatMetric(value: number | undefined): string {
  return value === undefined ? '-' : value.toFixed(Math.abs(value) >= 10 ? 1 : 3);
}
