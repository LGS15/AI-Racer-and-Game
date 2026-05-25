// Headless training driver — steps the simulator as fast as the CPU allows and
// streams the same protocol (§08) to the Python policy on ws://localhost:8765.
// There is no React and no canvas here: "disabling rendering" just means this
// entry point imports the pure domain functions and never imports the UI.
//
// Prereqs:  npm i -D tsx ws      (Node has no built-in WebSocket on this repo's Node 20)
// Run:      npx tsx scripts/headless.ts
//           (start the Python policy first, e.g. `python ai_service/server.py`)
// Tune:     EPISODES=100 STEPS=5000 AI_WS_URL=ws://localhost:8765 npx tsx scripts/headless.ts

import { WebSocket } from 'ws';
import { runEpisode } from '../src/domain/ai/headlessRunner';
import type { TrackJson } from '../src/domain/types';
import track from '../Data/Tracks/track-2.track.json';

const URL = process.env.AI_WS_URL ?? 'ws://localhost:8765';
const EPISODES = Number(process.env.EPISODES ?? 50);
const STEPS = Number(process.env.STEPS ?? 3000);

function openSocket(url: string): Promise<WebSocket> {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(url);
    ws.once('open', () => resolve(ws));
    ws.once('error', reject);
  });
}

async function main(): Promise<void> {
  const ws = await openSocket(URL);
  console.log(`connected to ${URL} — running ${EPISODES} episodes × ${STEPS} steps`);

  const started = Date.now();
  for (let ep = 0; ep < EPISODES; ep++) {
    // The `ws` package's socket implements the WHATWG addEventListener/send API
    // runEpisode relies on; the cast bridges its type to the DOM WebSocket type.
    await runEpisode(track as unknown as TrackJson, ws as unknown as WebSocket, STEPS);
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
