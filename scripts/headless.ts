// Headless training driver — steps the simulator as fast as the CPU allows and
// streams the same protocol (§08) to the Python policy on ws://localhost:8765.
// There is no React and no canvas here: "disabling rendering" just means this
// entry point imports the pure domain functions and never imports the UI.
//
// Prereqs:  npm i -D tsx ws      (Node has no built-in WebSocket on this repo's Node 20)
// Run:      npx tsx scripts/headless.ts
//           (start the Python policy first, e.g. `python ai_service/server.py`)
// Tune:     EPISODES=100 STEPS=5000 AI_WS_URL=ws://localhost:8765 npx tsx scripts/headless.ts
// Runs:     RUN_NAME=vision-v2 npx tsx scripts/headless.ts
//           Each run writes its saved best laps to Data/Runs/<RUN_NAME>/, so a
//           later model (e.g. a vision change) lands in its own folder and the
//           notebook can compare runs side by side. Defaults to "lap-traces"
//           to preserve the original flat output location.

import { WebSocket } from 'ws';
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { runEpisode } from '../src/domain/ai/headlessRunner';
import type { LapTraceJson, TrackJson } from '../src/domain/types';
import track from '../Data/Tracks/track-2.track.json';

const URL = process.env.AI_WS_URL ?? 'ws://localhost:8765';
const EPISODES = Number(process.env.EPISODES ?? 50);
const STEPS = Number(process.env.STEPS ?? 3000);
const PROJECT_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
// One folder per training run keeps each model's saved best laps separate so the
// notebook can compare them. Sanitize the name to a single path segment so it
// can't escape Data/Runs/.
const RUN_NAME = (process.env.RUN_NAME ?? 'lap-traces').replace(/[^A-Za-z0-9._-]+/g, '-');
const TRACE_DIR = path.join(PROJECT_ROOT, 'Data', 'Runs', RUN_NAME);

async function openSocket(url: string, timeoutMs = 30000): Promise<WebSocket> {
  const started = Date.now();
  let lastError: unknown;
  while (Date.now() - started < timeoutMs) {
    try {
      return await new Promise((resolve, reject) => {
        const ws = new WebSocket(url);
        ws.once('open', () => resolve(ws));
        ws.once('error', (error) => {
          ws.close();
          reject(error);
        });
      });
    } catch (error) {
      lastError = error;
      await wait(500);
    }
  }
  throw lastError instanceof Error ? lastError : new Error(`Could not connect to ${url}`);
}

function wait(ms: number): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

async function main(): Promise<void> {
  const ws = await openSocket(URL);
  console.log(`connected to ${URL} — running ${EPISODES} episodes × ${STEPS} steps, saving to Data/Runs/${RUN_NAME}/`);

  const started = Date.now();
  let bestLapTime = loadBestLapTime(TRACE_DIR);
  for (let ep = 0; ep < EPISODES; ep++) {
    // The `ws` package's socket implements the WHATWG addEventListener/send API
    // runEpisode relies on; the cast bridges its type to the DOM WebSocket type.
    const result = await runEpisode(track as unknown as TrackJson, ws as unknown as WebSocket, STEPS);
    for (const trace of result.lapTraces) {
      if (trace.lapTime < bestLapTime) {
        bestLapTime = trace.lapTime;
        const savedPath = saveLapTrace(trace, ep + 1);
        console.log(`new fastest lap ${trace.lapTime.toFixed(3)}s saved to ${savedPath}`);
      }
    }
    console.log(`episode ${ep + 1}/${EPISODES} done`);
  }

  const totalSteps = EPISODES * STEPS;
  const secs = (Date.now() - started) / 1000;
  console.log(`done: ${totalSteps} steps in ${secs.toFixed(1)}s = ${Math.round(totalSteps / secs)} steps/s`);
  ws.close();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

function loadBestLapTime(directory: string): number {
  if (!existsSync(directory)) return Number.POSITIVE_INFINITY;
  return readdirSync(directory)
    .filter((file) => file.endsWith('.lap-trace.json'))
    .reduce((best, file) => {
      try {
        const trace = JSON.parse(readFileSync(path.join(directory, file), 'utf8')) as Partial<LapTraceJson>;
        return Number.isFinite(trace.lapTime) && Number(trace.lapTime) > 0 ? Math.min(best, Number(trace.lapTime)) : best;
      } catch {
        return best;
      }
    }, Number.POSITIVE_INFINITY);
}

function saveLapTrace(trace: LapTraceJson, episode: number): string {
  mkdirSync(TRACE_DIR, { recursive: true });
  const lapMillis = Math.round(trace.lapTime * 1000).toString().padStart(6, '0');
  const fileName = `fastest-lap-${lapMillis}ms-episode-${episode}-lap-${trace.lap}.lap-trace.json`;
  const target = path.join(TRACE_DIR, fileName);
  writeFileSync(target, `${JSON.stringify(trace, null, 2)}\n`, 'utf8');
  return path.relative(PROJECT_ROOT, target);
}
