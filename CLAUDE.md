# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

page2pod converts web pages, HTML files, and Markdown documents to chapter-based podcasts using OpenAI TTS. It generates MP3 files with embedded chapter markers and JSON chapter files for web players.

## Development Commands

```bash
# Activate venv first (required for all Python commands)
. venv/bin/activate

# Run directly
python page2pod.py input.html -o output/

# Install as CLI tool (editable)
pip install -e .

# Then use as command
page2pod input.html -o output/
```

## CLI Options

```bash
page2pod <source> [options]

# Source can be: HTML file, Markdown file, or URL

# Output options
-o, --output DIR       Output directory (default: current directory)
-c, --cache DIR        Cache directory (default: ~/.cache/page2pod)

# Voice options
-v, --voice VOICE      Main content voice (default: onyx)
--code-voice VOICE     Code block voice (default: nova)
--quality MODEL        tts-1 or tts-1-hd (default: tts-1-hd)

# Code block handling
--code-mode skip       Say "Code example skipped" (default)
--code-mode verbatim   Read code aloud with punctuation verbalized
--code-mode describe   AI describes what the code does

# Chapter extraction
--no-ai                Use H2-based extraction instead of AI (default: AI)

# Regeneration options
--force                Force regenerate all chapters
--combine              Just recombine existing chapters (no TTS calls)
--chapter N            Regenerate only chapter N
--list                 List chapters without generating
```

## Available Voices

OpenAI TTS voices: `alloy`, `echo`, `fable`, `onyx` (default), `nova`, `shimmer`

## Project Structure

- **page2pod.py**: Main script (single-file tool)
- **setup.py**: Package configuration for pip install
- **venv/**: Python virtual environment (not committed)

## Cache Structure

Each page gets its own cache directory in `~/.cache/page2pod/<page-id>/`:

```
~/.cache/page2pod/example-com-article/
├── meta.json              # Chapter hashes for change detection
├── source.html            # Original HTML/Markdown
└── chapters/
    ├── 00-introduction.mp3
    ├── 00-introduction.txt
    ├── 01-getting-started.mp3
    └── ...
```

## Output Files

```
output/
├── <page-id>.mp3              # Combined MP3 with ID3 chapter markers
└── <page-id>.chapters.json    # Chapter index for web players
```

## Key Features

- **AI Chapter Extraction**: Uses OpenAI to intelligently identify chapter boundaries (default)
- **Caching**: Only regenerates chapters whose content has changed
- **Code Block Handling**: Three modes for handling code (skip, verbatim, describe)
- **Voice Switching**: Different voices for main content vs code blocks
- **Chapter Markers**: Embedded ID3 CHAP/CTOC frames in MP3 + JSON export
- **Format Support**: HTML, Markdown, and URLs

## Dependencies

- Python 3.x with venv
- openai (TTS API)
- mutagen (MP3 ID3 tags)
- beautifulsoup4 (HTML parsing)
- requests (URL fetching)
- ffmpeg (audio concatenation - system install)

## Environment Variables

- `OPENAI_API_KEY`: Required for TTS generation

## Common Usage Patterns

```bash
# Convert a blog post with default settings (AI chapters, code skipped)
page2pod post.md -o audio/

# Convert with code read verbatim
page2pod post.md -o audio/ --code-mode verbatim

# List chapters without generating (preview)
page2pod post.md --list

# Force full regeneration
page2pod post.md -o audio/ --force

# Regenerate just chapter 5
page2pod post.md -o audio/ --chapter 5
```
