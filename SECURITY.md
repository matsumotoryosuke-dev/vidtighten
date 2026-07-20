# Security Policy

## Threat model

VidTighten is a **local-only desktop app**. The Flask server it runs binds to
`127.0.0.1` by default and is not reachable from other devices on your network
unless you explicitly pass `--host` *and* `--allow-remote` on the CLI entry
point (`preprod-web`) — which you should only ever do if you understand the
consequence below.

**There is no authentication anywhere in the app.** Every route trusts "this
request came from a process on my own machine." That's a deliberate, accepted
tradeoff for a single-user local tool — not an oversight — but it means:

- If you bind beyond loopback (`--allow-remote`), **anyone who can reach that
  host:port has full read/write access to your files** — no login, no token,
  nothing. Don't do this on an untrusted network.
- The app defends against a malicious *web page* driving its local API via
  [DNS rebinding](https://en.wikipedia.org/wiki/DNS_rebinding) (a page you have
  open in your regular browser resolving a hostname to `127.0.0.1` and issuing
  requests) — every route checks the `Host` header and rejects anything that
  isn't `127.0.0.1` / `localhost` / `::1`. This is app-wide, not just on the
  media-streaming endpoint.

## Trusted input

VidTighten runs `ffmpeg`, `ffprobe`, and Whisper/WhisperX against video and
audio files you load. **These are treated as trusted input — your own
footage** — not untrusted, adversarial input from the network. We do not
attempt to sandbox media decoding against maliciously crafted files; that's a
parser-hardening problem that belongs to ffmpeg/Whisper upstream, not
something this app can meaningfully patch around.

Practical implication: **keep ffmpeg up to date** (`brew upgrade ffmpeg`).
Don't run VidTighten against video files from a source you wouldn't otherwise
trust to open in any other media tool.

All subprocess invocations (ffmpeg, ffprobe, Whisper) use argv-list form —
never a shell string — so filenames (including attacker-chosen ones, if you
ever did process untrusted files) cannot inject shell commands.

## Reporting a vulnerability

If you find a security issue, please open a private report via GitHub's
[Security Advisories](../../security/advisories/new) for this repo rather than
a public issue, so there's time to land a fix before details are public.
