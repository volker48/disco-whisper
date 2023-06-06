import asyncio
import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import BinaryIO, cast
from whisper_jax import FlaxWhisperPipline
import jax.numpy as jnp

import aiohttp
import discord

logger = logging.getLogger('discord')

CLIENT_ID = os.getenv("CLIENT_ID")
TOKEN = os.getenv("TOKEN")
MODEL = os.getenv("MODEL", "base.en")

logger.info("loading whisper model")

pipeline = FlaxWhisperPipline(f"openai/whisper-{MODEL}", dtype=jnp.bfloat16)

if TOKEN is None:
    raise Exception("need a token")


class DiscoWhisper(discord.Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.executor = ThreadPoolExecutor()
        self.session = aiohttp.ClientSession()

    async def on_ready(self):
        logger.info(f'Logged in as {self.user} (ID: {self.user.id})')
        logger.info('------')

    async def on_message(self, message: discord.Message):
        # we do not want the bot to reply to itself
        if message.author.id == self.user.id:
            return

        if message.attachments:
            await self.translate(message)

    async def translate(self, message: discord.Message) -> None:
        await message.reply("transcribing...")
        with tempfile.NamedTemporaryFile(mode="w+b", buffering=4096) as fp:
            fp = cast(BinaryIO, fp)
            await self.stream_to_file(message.attachments[0], fp)
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(self.executor, pipeline, fp.name)
            translation = result.get("text")
            if not translation or not isinstance(translation, str):
                logger.error("error during translation")
                return
            logger.info("--- translation result ---")
            logger.info(translation)
            await message.reply(translation)

    async def stream_to_file(self, attachment: discord.Attachment, fp: BinaryIO) -> None:
        async with aiohttp.ClientSession() as session:
            async with session.get(attachment.url) as resp:
                async for chunk in resp.content.iter_chunked(4096):
                    fp.write(chunk)


def main():
    intents = discord.Intents.default()
    intents.message_content = True

    client = DiscoWhisper(intents=intents, application_id=CLIENT_ID, client_id=CLIENT_ID)
    client.run(token=TOKEN)


if __name__ == "__main__":
    main()
