import asyncio
import json
import random

import websockets

ACTION_COUNT = 7


async def handle_client(websocket):
    print("Game connected.")
    async for raw in websocket:
        msg = json.loads(raw)

        if msg["type"] == "session_start":
            print(f"  session={msg['sessionId']}  state_size={msg['stateSize']}  actions={msg['actionCount']}")
            await websocket.send(json.dumps({
                "type": "session_ready",
                "algorithm": "random",
                "training": False,
            }))

        elif msg["type"] == "action_request":
            action_index = random.randint(0, ACTION_COUNT - 1)
            await websocket.send(json.dumps({
                "type": "action_response",
                "sessionId": msg["sessionId"],
                "step": msg["step"],
                "actionIndex": action_index,
            }))

        elif msg["type"] == "transition":
            pass  # random policy ignores experience


async def main():
    print("Listening on ws://localhost:8765 — start the game and select External.")
    async with websockets.serve(handle_client, "localhost", 8765):
        await asyncio.Future()  # run until Ctrl-C


if __name__ == "__main__":
    asyncio.run(main())
