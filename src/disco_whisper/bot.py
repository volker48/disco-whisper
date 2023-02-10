import discord
import os
import whisper
import tempfile
import torch

model = whisper.load_model("base.en")

CLIENT_ID = os.getenv("CLIENT_ID")
TOKEN = os.getenv("TOKEN")

if TOKEN is None:
    raise Exception("need a token")


class MyClient(discord.Client):
    async def on_ready(self):
        print(f'Logged in as {self.user} (ID: {self.user.id})')
        print('------')

    async def on_message(self, message: discord.Message):
        # we do not want the bot to reply to itself
        if message.author.id == self.user.id:
            return

        if message.content.startswith('!hello'):
            await message.reply('Hello!', mention_author=True)
            return
        if message.attachments:
            await message.reply("transcribing...")
            with tempfile.NamedTemporaryFile("wb") as fp:
                data = await message.attachments[0].read()
                fp.write(data)
                result = model.transcribe(fp.name)
                print(result["text"])
                await message.reply(result["text"])


def main():
    intents = discord.Intents.default()
    intents.message_content = True

    client = MyClient(intents=intents, client_id=CLIENT_ID)
    client.run(token=TOKEN)


if __name__ == "__main__":
    main()
