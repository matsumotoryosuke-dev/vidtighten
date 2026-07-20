# VidTighten

A macOS desktop app that removes dead air and filler words from raw video,
transcribes it, and exports directly to Final Cut Pro — so the tedious part
of prepping an edit doesn't eat your evening.

Built out of one YouTube creator's own editing workflow — open-sourced so
other creators who don't have time to build this themselves can use it, fork
it, and improve it together.

## What it does

- **Loads a video**, runs local speech-to-text (Whisper / WhisperX), and
  detects silence and filler words ("um", "えっと", etc. — English and
  Japanese) automatically.
- **Word-level transcript editing** — click any word to seek to it, delete
  individual words or ranges, review every removal with A/B playback before
  committing.
- **Exports FCPXML for Final Cut Pro**: a roughcut assembly with the
  dead air/fillers already cut, and/or a telop (on-screen caption) track that
  matches what you see in VidTighten's own live preview — font, size, line
  spacing, and line-wrapping (Japanese lines wrap only at natural phrase
  boundaries, never mid-word).
- **Transcript find & replace** for bulk-fixing recurring transcription
  typos or homophones (brand names, jargon) without re-running analysis.
- **Optional local-LLM transcript correction** (via [Ollama](https://ollama.com),
  fully opt-in, nothing leaves your machine): scans the transcript for likely
  brand/proper-noun mishearings and proposes fixes you review one-by-one
  before anything is applied — every batch can be reverted in one click. Its
  real-world accuracy is asymmetric: it's reliable for well-known public
  names (it *knows* things like "Notion" or "ChatGPT"), but tends to guess
  the wrong direction for a brand or name it hasn't seen before, so review
  the suggestions rather than trusting them blindly — that's exactly what the
  review step is for.
- **4K-friendly preview** — plays a downscaled proxy for smooth scrubbing on
  large source files without touching your original media.

## Requirements

- **macOS.** The app shell (`pywebview` + a native AppKit menu/file
  integration) is macOS-specific — see [CONTRIBUTING.md](CONTRIBUTING.md) if
  you're interested in porting it elsewhere.
- **Apple Silicon, if you want WhisperX** (the higher-accuracy alignment
  backend — see below). Its dependency chain (torch/torchvision/torchaudio)
  currently ships arm64-only wheels for macOS. The base app and the
  faster-whisper/openai-whisper backends don't have this restriction.
- **Python 3.10–3.13** and **Node.js** — `scripts/bootstrap.sh` handles the
  Python side for you (see Development, below).
- **[ffmpeg](https://ffmpeg.org)** on your `PATH` (`brew install ffmpeg`).

## Development

```bash
git clone https://github.com/matsumotoryosuke-dev/vid-tighten.git
cd vid-tighten
scripts/bootstrap.sh   # installs uv if needed, pins Python 3.13, resolves
                        # every dependency from the committed lockfile
npm install
```

Run it:

```bash
uv run python run_web.py   # opens the app in your browser at http://127.0.0.1:9877
```

Run the tests:

```bash
uv run pytest tests/ -q
npm run test:js -- --run
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide, including what
each optional dependency group (`whisper`, `whisperx`, `japanese`) buys you
and why they're split up.

## Building the .app bundle

```bash
./build_app.sh
```

Produces `VidTighten.app`, using the environment `scripts/bootstrap.sh` set
up.

## Contributing

Bug reports, feature requests, and PRs are welcome — see
[CONTRIBUTING.md](CONTRIBUTING.md). Please read
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) too.

## Security

See [SECURITY.md](SECURITY.md) for the threat model (local-only, no
authentication by design) and how to report a vulnerability privately.

## License

[MIT](LICENSE).

## Acknowledgments

Built on [OpenAI Whisper](https://github.com/openai/whisper),
[faster-whisper](https://github.com/SYSTRAN/faster-whisper),
[WhisperX](https://github.com/m-bain/whisperx),
[fugashi](https://github.com/polm/fugashi) (Japanese morphological analysis),
and [pywebview](https://pywebview.flowrl.com/).
