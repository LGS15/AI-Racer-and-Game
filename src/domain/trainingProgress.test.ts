import { describe, expect, it } from 'vitest';
import { parseTrainingProgressCsv, trainingLapSeries } from './trainingProgress';

describe('training progress parsing', () => {
  it('keeps every logged training lap so the UI can show non-record laps too', () => {
    const csv = [
      'event,step,epsilon,buffer_size,mean_reward,loss,best_lap_time,best_reward,reason,metric,metric_value',
      'interval,1000,0.5,1000,0.2,,12.5,,,,',
      'lap,1200,0.4,1200,,0.1,12.5,,new fastest lap,lap_time,12.5',
      'lap,2400,0.3,2400,,0.2,12.5,,,lap_time,13.1',
      'lap,3600,0.2,3600,,0.3,11.8,,new fastest lap,lap_time,11.8'
    ].join('\n');

    const result = parseTrainingProgressCsv(csv);
    const laps = trainingLapSeries(result.value ?? []);

    expect(result.ok).toBe(true);
    expect(laps.map((lap) => lap.lapTime)).toEqual([12.5, 13.1, 11.8]);
    expect(laps.map((lap) => lap.isNewBest)).toEqual([true, false, true]);
  });

  it('falls back to checkpoint rows from older CSVs that only saved fastest laps', () => {
    const csv = [
      'event,step,epsilon,buffer_size,mean_reward,loss,best_lap_time,best_reward,reason,metric,metric_value',
      'interval,1000,0.5,1000,0.2,,12.5,,,,',
      'checkpoint,2000,0.4,2000,,0.1,9.75,,new best lap time,lap_time,9.75'
    ].join('\n');

    const result = parseTrainingProgressCsv(csv);
    const laps = trainingLapSeries(result.value ?? []);

    expect(result.ok).toBe(true);
    expect(laps).toEqual([
      {
        step: 2000,
        lapTime: 9.75,
        bestLapTime: 9.75,
        isNewBest: true,
        reason: 'new best lap time',
        source: 'checkpoint'
      }
    ]);
  });
});
