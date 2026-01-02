# page2pod

Convert web pages and articles to chapter-based podcasts using OpenAI TTS.

## Features

- Extracts chapters from H2 headings
- Generates MP3 with embedded chapter markers
- Caches individual chapters - only regenerates changed content
- Outputs chapters.json for web players
- Supports both local HTML files and URLs

## Installation

```bash
pip install openai mutagen beautifulsoup4 requests
```

Or install as a tool:

```bash
pip install -e .
```

## Usage

```bash
# Convert a local HTML file
page2pod index.html

# Convert a URL
page2pod https://example.com/article

# Force regenerate everything
page2pod index.html --force

# Regenerate only chapter 3
page2pod index.html --chapter 3

# Just recombine existing chapters (no TTS calls)
page2pod index.html --combine

# List chapters without generating
page2pod index.html --list

# Custom output directory
page2pod index.html -o ./output

# Use different voice
page2pod index.html -v nova
```

## Voices

OpenAI TTS voices: `alloy`, `echo`, `fable`, `onyx` (default), `nova`, `shimmer`

## Cache Structure

Each page gets its own cache directory in `~/.cache/page2pod/<page-id>/`:

```
~/.cache/page2pod/example-com-article/
├── meta.json              # Chapter hashes for change detection
├── source.html            # Original HTML
└── chapters/
    ├── 00-introduction.mp3
    ├── 00-introduction.txt
    ├── 01-getting-started.mp3
    ├── 01-getting-started.txt
    └── ...
```

## Output

```
./
├── example-com-article.mp3           # Combined MP3 with chapter markers
└── example-com-article.chapters.json # Chapter index for web players
```

## Web Player

The `.chapters.json` file works with players like Plyr.js:

```javascript
fetch('article.chapters.json')
  .then(res => res.json())
  .then(data => {
    data.chapters.forEach(ch => {
      console.log(`${ch.title}: ${ch.start}s - ${ch.end}s`);
    });
  });
```

## Environment

Requires `OPENAI_API_KEY` environment variable.

## License

MIT
