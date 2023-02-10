# Usage

- Go here and follow the instructions to create an app https://discord.com/developers/docs/getting-started#creating-an-app.
- You only need to follow the instructions up to this point https://discord.com/developers/docs/getting-started#installing-your-app to install your app.
- Save your token somewhere like keyring.
- Create a virtualenv and install the requirements with `pip install -e .`
- Run the bot with a command similar to the following: `CLIENT_ID=YOUR_CLIENT_ID TOKEN=$(keyring get disco-whsiper password) bot`.
- Send a DM to the bot with an attached audio file and it will transcribe the audio and reply with text.
