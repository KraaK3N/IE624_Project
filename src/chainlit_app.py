"""SysBot chat UI (Chainlit).

    python -m chainlit run src/chainlit_app.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chainlit as cl  # noqa: E402

from src.bot import DepartmentChatbot  # noqa: E402

bot = DepartmentChatbot()
_DONE = object()


@cl.set_starters
async def starters():
    return [
        cl.Starter(label="M.Tech course work", message="What is the M.Tech course work in SysCon?"),
        cl.Starter(label="Faculty research areas", message="Who works on stochastic control?"),
        cl.Starter(label="Ph.D. leave", message="How do I apply for leave as a PhD student?"),
        cl.Starter(label="SC 625", message="What is SC 625 about?"),
    ]


@cl.on_chat_start
async def start():
    cl.user_session.set("history", [])


@cl.on_message
async def on_message(message: cl.Message):
    history = cl.user_session.get("history")
    reply = cl.Message(content="", author="SysBot")
    stream = bot.respond_stream(message.content, history)
    while (piece := await asyncio.to_thread(next, stream, _DONE)) is not _DONE:
        await reply.stream_token(piece)
    await reply.send()
    history += [{"role": "user", "content": message.content}, {"role": "assistant", "content": reply.content}]
