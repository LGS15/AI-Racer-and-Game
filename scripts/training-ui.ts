import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process';
import { existsSync } from 'node:fs';
import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

type RunState = 'idle' | 'starting' | 'running' | 'stopping';

interface RunSettings {
  episodes: number;
  steps: number;
}

interface LogEntry {
  source: 'dashboard' | 'python' | 'trainer';
  text: string;
}

const PROJECT_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const PORT = Number(process.env.TRAIN_UI_PORT ?? 8787);
const WS_URL = 'ws://localhost:8765';
const MAX_LOGS = 600;
const VENV_PYTHON = path.join(PROJECT_ROOT, '.venv', 'Scripts', 'python.exe');
const PYTHON_COMMAND = process.env.PYTHON ?? (existsSync(VENV_PYTHON) ? VENV_PYTHON : 'python');

let state: RunState = 'idle';
let settings: RunSettings = { episodes: 50, steps: 3000 };
let pythonProcess: ChildProcessWithoutNullStreams | undefined;
let trainerProcess: ChildProcessWithoutNullStreams | undefined;
let runId = 0;

const logs: LogEntry[] = [];
const clients = new Set<ServerResponse>();

const server = createServer((request, response) => {
  void handleRequest(request, response);
});

server.listen(PORT, '127.0.0.1', () => {
  log('dashboard', `Training dashboard on http://127.0.0.1:${PORT}`);
});

process.on('SIGINT', () => {
  stopRun();
  server.close(() => process.exit(0));
});

process.on('SIGTERM', () => {
  stopRun();
  server.close(() => process.exit(0));
});

async function handleRequest(request: IncomingMessage, response: ServerResponse): Promise<void> {
  const url = new URL(request.url ?? '/', `http://${request.headers.host ?? `127.0.0.1:${PORT}`}`);
  if (request.method === 'GET' && url.pathname === '/') {
    sendHtml(response);
    return;
  }
  if (request.method === 'GET' && url.pathname === '/events') {
    connectEvents(response);
    request.on('close', () => clients.delete(response));
    return;
  }
  if (request.method === 'GET' && url.pathname === '/api/status') {
    sendJson(response, currentStatus());
    return;
  }
  if (request.method === 'POST' && url.pathname === '/api/start') {
    try {
      const body = await readJson(request);
      startRun(parseSettings(body));
      sendJson(response, currentStatus());
    } catch (error) {
      sendJson(response, { error: error instanceof Error ? error.message : 'Invalid start request.' }, 400);
    }
    return;
  }
  if (request.method === 'POST' && url.pathname === '/api/stop') {
    stopRun();
    sendJson(response, currentStatus());
    return;
  }
  response.writeHead(404, { 'content-type': 'text/plain; charset=utf-8' });
  response.end('Not found');
}

function startRun(nextSettings: RunSettings): void {
  if (state !== 'idle') {
    throw new Error('Training is already running.');
  }
  settings = nextSettings;
  state = 'starting';
  runId += 1;
  const activeRun = runId;
  broadcastState();
  log('dashboard', `Starting training: ${settings.episodes} episodes x ${settings.steps} steps.`);

  try {
    pythonProcess = spawnPythonServer();
  } catch (error) {
    log('dashboard', `Python service failed to start: ${error instanceof Error ? error.message : 'unknown error'}`);
    finishRun();
    throw error;
  }
  attachLogs(pythonProcess, 'python');

  pythonProcess.once('exit', (code, signal) => {
    log('dashboard', `Python service exited (${formatExit(code, signal)}).`);
    pythonProcess = undefined;
    if (state !== 'stopping' && trainerProcess) {
      stopProcess(trainerProcess);
    }
    if (!trainerProcess) finishRun();
  });

  if (runId === activeRun && pythonProcess) {
    startTrainer();
  }
}

function spawnPythonServer(): ChildProcessWithoutNullStreams {
  if (process.platform === 'win32') {
    return spawn('powershell.exe', ['-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', `& '${PYTHON_COMMAND}' -u 'ai_service/server.py'`], {
      cwd: PROJECT_ROOT,
      env: { ...process.env, PYTHONUNBUFFERED: '1' },
    });
  }
  return spawn(PYTHON_COMMAND, ['-u', 'ai_service/server.py'], {
    cwd: PROJECT_ROOT,
    env: { ...process.env, PYTHONUNBUFFERED: '1' },
  });
}

function startTrainer(): void {
  state = 'running';
  broadcastState();
  log('dashboard', 'Starting headless trainer.');
  try {
    trainerProcess = spawnHeadlessTrainer();
  } catch (error) {
    log('dashboard', `Headless trainer failed to start: ${error instanceof Error ? error.message : 'unknown error'}`);
    if (pythonProcess) stopProcess(pythonProcess);
    else finishRun();
    throw error;
  }
  attachLogs(trainerProcess, 'trainer');

  trainerProcess.once('exit', (code, signal) => {
    log('dashboard', `Trainer exited (${formatExit(code, signal)}).`);
    trainerProcess = undefined;
    if (pythonProcess) {
      stopProcess(pythonProcess);
    } else {
      finishRun();
    }
  });
}

function spawnHeadlessTrainer(): ChildProcessWithoutNullStreams {
  const env = {
    ...process.env,
    AI_WS_URL: WS_URL,
    EPISODES: String(settings.episodes),
    STEPS: String(settings.steps),
  };
  if (process.platform === 'win32') {
    return spawn('cmd.exe', ['/d', '/s', '/c', 'npx.cmd tsx scripts/headless.ts'], {
      cwd: PROJECT_ROOT,
      env,
    });
  }
  return spawn('npx', ['tsx', 'scripts/headless.ts'], {
    cwd: PROJECT_ROOT,
    env,
  });
}

function stopRun(): void {
  if (state === 'idle') return;
  state = 'stopping';
  runId += 1;
  broadcastState();
  log('dashboard', 'Stopping training.');
  if (trainerProcess) stopProcess(trainerProcess);
  if (pythonProcess) stopProcess(pythonProcess);
  if (!trainerProcess && !pythonProcess) finishRun();
}

function finishRun(): void {
  state = 'idle';
  broadcastState();
  log('dashboard', 'Training stopped.');
}

function stopProcess(child: ChildProcessWithoutNullStreams): void {
  if (!child.killed) child.kill();
}

function attachLogs(child: ChildProcessWithoutNullStreams, source: LogEntry['source']): void {
  child.stdout.on('data', (chunk) => appendChunk(source, chunk));
  child.stderr.on('data', (chunk) => appendChunk(source, chunk));
  child.once('error', (error) => {
    log('dashboard', `${source} failed to start: ${error.message}`);
    if (source === 'python') {
      pythonProcess = undefined;
      finishRun();
    }
    if (source === 'trainer') {
      trainerProcess = undefined;
      if (pythonProcess) stopProcess(pythonProcess);
      else finishRun();
    }
  });
}

function appendChunk(source: LogEntry['source'], chunk: Buffer): void {
  for (const line of chunk.toString('utf8').split(/\r?\n/)) {
    if (line.trim()) log(source, line);
  }
}

function log(source: LogEntry['source'], text: string): void {
  const entry = { source, text };
  logs.push(entry);
  if (logs.length > MAX_LOGS) logs.splice(0, logs.length - MAX_LOGS);
  broadcast({ type: 'log', entry });
}

function connectEvents(response: ServerResponse): void {
  response.writeHead(200, {
    'cache-control': 'no-cache',
    'connection': 'keep-alive',
    'content-type': 'text/event-stream',
  });
  clients.add(response);
  sendEvent(response, { type: 'state', status: currentStatus() });
  for (const entry of logs) sendEvent(response, { type: 'log', entry });
}

function broadcastState(): void {
  broadcast({ type: 'state', status: currentStatus() });
}

function broadcast(payload: unknown): void {
  for (const client of clients) sendEvent(client, payload);
}

function sendEvent(response: ServerResponse, payload: unknown): void {
  response.write(`data: ${JSON.stringify(payload)}\n\n`);
}

function currentStatus() {
  return {
    state,
    settings,
    running: state !== 'idle',
  };
}

async function readJson(request: IncomingMessage): Promise<unknown> {
  const chunks: Buffer[] = [];
  for await (const chunk of request) chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
  return chunks.length ? JSON.parse(Buffer.concat(chunks).toString('utf8')) : {};
}

function parseSettings(input: unknown): RunSettings {
  const value = input as Partial<RunSettings> | null;
  const episodes = Number(value?.episodes);
  const steps = Number(value?.steps);
  if (!Number.isInteger(episodes) || episodes < 1 || episodes > 10000) {
    throw new Error('Episodes must be a whole number from 1 to 10000.');
  }
  if (!Number.isInteger(steps) || steps < 1 || steps > 1000000) {
    throw new Error('Steps must be a whole number from 1 to 1000000.');
  }
  return { episodes, steps };
}

function sendJson(response: ServerResponse, data: unknown, status = 200): void {
  response.writeHead(status, { 'content-type': 'application/json; charset=utf-8' });
  response.end(JSON.stringify(data));
}

function sendHtml(response: ServerResponse): void {
  response.writeHead(200, { 'content-type': 'text/html; charset=utf-8' });
  response.end(HTML);
}

function formatExit(code: number | null, signal: NodeJS.Signals | null): string {
  return signal ? `signal ${signal}` : `code ${code ?? 'unknown'}`;
}

const HTML = `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AI Racer Training</title>
  <style>
    :root { font-family: Inter, ui-sans-serif, system-ui, sans-serif; color: #10131a; background: #edf3f8; }
    body { margin: 0; min-height: 100vh; display: grid; grid-template-rows: auto 1fr; }
    header { padding: 14px 18px; border-bottom: 1px solid #ccd6e3; background: #fff; }
    h1 { margin: 0; font-size: 18px; text-transform: uppercase; }
    main { display: grid; grid-template-columns: 340px minmax(0, 1fr); gap: 14px; padding: 14px; }
    section { border: 1px solid #ccd6e3; border-radius: 8px; background: #fff; padding: 12px; }
    h2 { margin: 0 0 12px; font-size: 12px; text-transform: uppercase; }
    label { display: grid; gap: 6px; margin-bottom: 12px; color: #5d6677; font-size: 12px; font-weight: 800; text-transform: uppercase; }
    input { min-height: 38px; border: 1px solid #ccd6e3; border-radius: 6px; padding: 0 10px; font: inherit; }
    .row { display: flex; gap: 8px; align-items: center; }
    button { min-height: 38px; border: 1px solid #10131a; border-radius: 6px; padding: 0 12px; background: #10131a; color: #fff; cursor: pointer; font-weight: 800; }
    button.secondary { border-color: #ccd6e3; background: #f8fbff; color: #10131a; }
    button:disabled { opacity: .45; cursor: not-allowed; }
    .status { display: grid; grid-template-columns: 1fr auto; gap: 8px; margin-top: 12px; padding-top: 12px; border-top: 1px solid #ccd6e3; }
    .status strong { font-family: Consolas, monospace; }
    pre { height: calc(100vh - 126px); margin: 0; overflow: auto; border-radius: 8px; background: #10131a; color: #edf3f8; padding: 12px; font: 12px/1.45 Consolas, monospace; white-space: pre-wrap; }
    .dashboard { color: #d8ff2f; }
    .python { color: #57a6ff; }
    .trainer { color: #27d17f; }
    @media (max-width: 760px) { main { grid-template-columns: 1fr; } pre { height: 55vh; } }
  </style>
</head>
<body>
  <header><h1>AI Racer Training</h1></header>
  <main>
    <section>
      <h2>Settings</h2>
      <label>Episodes <input id="episodes" type="number" min="1" max="10000" value="50"></label>
      <label>Steps Per Episode <input id="steps" type="number" min="1" max="1000000" value="3000"></label>
      <div class="row">
        <button id="start">Start</button>
        <button id="stop" class="secondary" disabled>Stop</button>
      </div>
      <div class="status">
        <span>Status</span><strong id="state">idle</strong>
        <span>Trace saves</span><strong>Data/Runs/lap-traces</strong>
      </div>
    </section>
    <section>
      <h2>Logs</h2>
      <pre id="logs"></pre>
    </section>
  </main>
  <script>
    const logs = document.getElementById('logs');
    const state = document.getElementById('state');
    const start = document.getElementById('start');
    const stop = document.getElementById('stop');
    const episodes = document.getElementById('episodes');
    const steps = document.getElementById('steps');

    function renderStatus(status) {
      state.textContent = status.state;
      start.disabled = status.running;
      stop.disabled = !status.running;
      episodes.disabled = status.running;
      steps.disabled = status.running;
    }

    function appendLog(entry) {
      const line = document.createElement('span');
      line.className = entry.source;
      line.textContent = '[' + entry.source + '] ' + entry.text + '\\n';
      logs.appendChild(line);
      logs.scrollTop = logs.scrollHeight;
    }

    new EventSource('/events').onmessage = (event) => {
      const message = JSON.parse(event.data);
      if (message.type === 'state') renderStatus(message.status);
      if (message.type === 'log') appendLog(message.entry);
    };

    start.onclick = async () => {
      await fetch('/api/start', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ episodes: Number(episodes.value), steps: Number(steps.value) }),
      });
    };

    stop.onclick = async () => {
      await fetch('/api/stop', { method: 'POST' });
    };
  </script>
</body>
</html>`;
