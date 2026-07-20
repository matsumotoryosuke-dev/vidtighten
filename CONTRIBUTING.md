# Contributing to VidTighten

Thanks for considering a contribution. This is a nights-and-weekends project
that grew out of one creator's own editing workflow, opened up so other
creators can fork it, fix what bugs them, and add what they need — please
read this before opening a PR.

## Before you start

- **macOS only, for now.** The app shell is `pywebview` + AppKit-specific
  native menu/file integration — porting to Windows/Linux is a real project,
  not a quick patch. PRs adding cross-platform support are welcome, but
  understand it's a substantial undertaking, not a small tweak.
- **Check open issues first** to avoid duplicate work, especially for
  larger features.
- **For anything bigger than a small fix**, open an issue to discuss the
  approach before writing code — saves you a rewrite if the maintainers see a
  problem with the direction.

## Development setup

Requires Python 3.10–3.13 and Node.js. One command sets up the full Python
environment (installs [`uv`](https://astral.sh/uv) if you don't have it,
pins the right Python version, resolves every dependency from the committed
lockfile):

```bash
scripts/bootstrap.sh
npm install
```

That's it — no manual venv wrangling, no guessing which Python version works.
See the README's "Development" section for what each optional dependency
group buys you (Whisper transcription, WhisperX alignment, Japanese
phrase-boundary detection are all separately installable — `--all-extras`,
which the bootstrap script uses, gets everything).

## Running tests

```bash
uv run pytest tests/ -q                     # Python suite
npm run test:js -- --run                    # JS suite (vitest)
```

Both must stay green. If your change needs new/updated tests, include them in
the same PR — this project runs on trust more than review bandwidth, so
test coverage is what lets a maintainer merge confidently without re-deriving
your reasoning.

## Code style

- No unnecessary comments — code should read clearly from naming; comments
  are reserved for non-obvious *why* (a hidden constraint, a workaround for a
  specific bug, a subtle invariant), not restating *what* the code does.
- Match the existing patterns in the file you're editing before introducing a
  new one.
- Small, focused PRs are much easier to review than large ones that mix
  refactoring with new behavior — please keep them separate.

## Commit messages

Explain *why*, not *what* — the diff already shows what changed. If a commit
fixes a specific bug or regression, say what the bug was.

## Pull requests

- Reference the issue you're addressing, if any.
- Describe what you tested and how.
- Expect review comments — this is a small project maintained part-time, so
  turnaround may take a few days.

## Reporting bugs / requesting features

Use the issue templates — they ask for the information that's actually
needed to act on a report (repro steps, video format/codec if relevant,
console/log output) rather than a blank box.

## Security issues

Please **do not** open a public issue for a security vulnerability — see
[SECURITY.md](SECURITY.md) for how to report privately.
