#!/usr/bin/env python3
"""
page2pod - Convert web pages and articles to chapter-based podcasts.

Generates MP3 audio with chapter markers from HTML content.
Caches individual chapters so only changed sections are regenerated.

Usage:
    page2pod <input.html> [options]
    page2pod <url> [options]

Options:
    -o, --output DIR      Output directory (default: current directory)
    -c, --cache DIR       Cache directory (default: ~/.cache/page2pod)
    -v, --voice VOICE     OpenAI TTS voice for main content (default: onyx)
    --code-voice VOICE    OpenAI TTS voice for code blocks (default: nova)
    --code-mode MODE      How to handle code blocks:
                            skip     - Just say "Code example skipped" (default)
                            verbatim - Read code aloud with punctuation verbalized
                            describe - AI describes what the code does
    --force               Force regenerate all chapters
    --combine             Just recombine existing chapters
    --chapter N           Regenerate only chapter N
    --list                List chapters without generating
    --quality MODEL       TTS model: tts-1 or tts-1-hd (default: tts-1-hd)
    --no-ai               Use H2-based extraction instead of AI (default: AI)

Requirements:
    pip install openai mutagen beautifulsoup4 requests

Environment:
    OPENAI_API_KEY        Your OpenAI API key
"""

import os
import re
import sys
import json
import hashlib
import argparse
import subprocess
from pathlib import Path
from urllib.parse import urlparse

try:
    from bs4 import BeautifulSoup, NavigableString
    from openai import OpenAI
    from mutagen.mp3 import MP3
    from mutagen.id3 import ID3, TIT2, TALB, TPE1, CHAP, CTOC, CTOCFlags
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install openai mutagen beautifulsoup4 requests")
    sys.exit(1)

# Default settings
DEFAULT_VOICE = "onyx"
DEFAULT_CODE_VOICE = "nova"  # Different voice for code blocks
DEFAULT_MODEL = "tts-1-hd"
DEFAULT_CACHE = Path.home() / ".cache" / "page2pod"


def get_text_content(element):
    """Extract clean text from an HTML element"""
    if element is None:
        return ""

    texts = []
    for child in element.descendants:
        if isinstance(child, NavigableString):
            parent = child.parent
            if parent.name not in ['script', 'style', 'svg', 'path', 'noscript', 'nav', 'footer']:
                text = str(child).strip()
                if text:
                    texts.append(text)

    return ' '.join(texts)


def clean_for_speech(text):
    """Clean text for natural speech synthesis"""
    text = re.sub(r'\s+', ' ', text)

    # Common abbreviations
    replacements = {
        'IT': 'I.T.', 'AI': 'A.I.', 'UI': 'U.I.', 'API': 'A.P.I.',
        'APIs': 'A.P.I.s', 'SSO': 'S.S.O.', 'MFA': 'M.F.A.',
        'CLI': 'C.L.I.', 'CEO': 'C.E.O.', 'COO': 'C.O.O.',
        'CTO': 'C.T.O.', 'CFO': 'C.F.O.', 'HTML': 'H.T.M.L.',
        'CSS': 'C.S.S.', 'JS': 'JavaScript', 'JSON': 'J.S.O.N.',
        'SQL': 'S.Q.L.', 'URL': 'U.R.L.', 'HTTP': 'H.T.T.P.',
        'HTTPS': 'H.T.T.P.S.', 'REST': 'rest', 'SDK': 'S.D.K.',
        '→': '', '←': '', '✓': '', '•': '', '▸': '',
    }

    for abbr, full in replacements.items():
        text = re.sub(rf'\b{abbr}\b', full, text)

    return text.strip()


def verbalize_code(code):
    """Convert code punctuation to spoken words for TTS, including structure"""
    lines = code.split('\n')
    verbalized_lines = []

    for line in lines:
        if not line.strip():
            continue

        # Count indentation
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        indent_levels = indent // 4  # Assume 4-space indentation

        # Add indent announcement
        if indent_levels > 0:
            prefix = f"indent {indent_levels}, " if indent_levels == 1 else f"indent {indent_levels} levels, "
        else:
            prefix = ""

        # Replace punctuation with words
        replacements = [
            ('==', ' equals equals '), ('!=', ' not equals '),
            ('<=', ' less or equal '), ('>=', ' greater or equal '),
            ('&&', ' and and '), ('||', ' or or '),
            ('->', ' arrow '), ('=>', ' fat arrow '),
            ('::', ' colon colon '), ('...', ' dot dot dot '),
            ('//', ' slash slash '), ('/*', ' slash star '), ('*/', ' star slash '),
            ('++', ' plus plus '), ('--', ' minus minus '),
            ('+=', ' plus equals '), ('-=', ' minus equals '),
            ('*=', ' times equals '), ('/=', ' divide equals '),
            ('{', ' open brace '), ('}', ' close brace '),
            ('(', ' open paren '), (')', ' close paren '),
            ('[', ' open bracket '), (']', ' close bracket '),
            ('<', ' less than '), ('>', ' greater than '),
            ('&', ' ampersand '), ('|', ' pipe '), ('@', ' at '),
            ('#', ' hash '), ('$', ' dollar '), ('%', ' percent '),
            ('!', ' bang '), ('?', ' question '), (':', ' colon '),
            (';', ' semicolon '), (',', ' comma '),
            ('=', ' equals '), ('+', ' plus '), ('-', ' minus '),
            ('*', ' star '), ('/', ' slash '), ('\\', ' backslash '),
            ('_', ' underscore '), ('`', ' backtick '),
            ('~', ' tilde '), ('^', ' caret '),
        ]

        result = stripped
        for char, word in replacements:
            result = result.replace(char, word)

        verbalized_lines.append(prefix + result)

    return ". New line. ".join(verbalized_lines)


def describe_code_ai(code, language, client):
    """Use AI to describe code in plain English"""
    prompt = f"""Describe this {language or 'code'} in plain English for a listener.
Be concise but complete. Explain what it does, not how it's written.
Do not read the code literally - describe its purpose and behavior.
Keep it under 3 sentences unless the code is complex.

Code:
```{language}
{code}
```

Plain English description:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=200
    )

    return response.choices[0].message.content.strip()


def skip_code_text(language):
    """Generate skip announcement for code block"""
    if language:
        return f"{language.capitalize()} code example skipped."
    return "Code example skipped."


def get_code_block_title(code, language=""):
    """Generate descriptive title for code block"""
    if language:
        language = language.capitalize()

    lines = code.strip().split('\n')
    first_line = lines[0] if lines else ""

    # Function definitions
    if re.match(r'^\s*(def|func|function|fn)\s+\w+', first_line):
        match = re.search(r'(def|func|function|fn)\s+(\w+)', first_line)
        if match:
            return f"{language} Function: {match.group(2)}" if language else f"Function: {match.group(2)}"

    # Class definitions
    if re.match(r'^\s*(class|struct|type|interface)\s+\w+', first_line):
        match = re.search(r'(class|struct|type|interface)\s+(\w+)', first_line)
        if match:
            kind = match.group(1).capitalize()
            return f"{language} {kind}: {match.group(2)}" if language else f"{kind}: {match.group(2)}"

    # Import statements
    if re.match(r'^\s*(import|from|require|use|include)', first_line):
        return f"{language} Imports" if language else "Import Statements"

    # Variable/const declarations
    if re.match(r'^\s*(const|let|var|val)\s+', first_line):
        return f"{language} Declaration" if language else "Variable Declaration"

    # Default
    if language:
        return f"{language} Code Example"
    return "Code Example"


def extract_chapters(soup, title=None, code_voice=DEFAULT_CODE_VOICE, main_voice=DEFAULT_VOICE,
                     code_mode="skip", client=None):
    """
    Extract chapters from HTML using H2 headings.
    Code blocks handled based on code_mode: skip, verbatim, or describe.
    Returns list of (title, text, voice) tuples.
    """
    chapters = []

    # Try to get title from H1 or title tag
    if title is None:
        h1 = soup.find('h1')
        if h1:
            title = h1.get_text(strip=True)
        else:
            title_tag = soup.find('title')
            if title_tag:
                title = title_tag.get_text(strip=True)
            else:
                title = "Untitled"

    # Find main content area
    main = soup.find('main') or soup.find('article') or soup.find('body')
    if not main:
        return [(title, clean_for_speech(get_text_content(soup)), main_voice)]

    def process_section(section_html, section_title):
        """Process a section, splitting out code blocks"""
        result = []
        section_soup = BeautifulSoup(section_html, 'html.parser')

        # Find all code blocks (pre > code or pre with class)
        code_blocks = section_soup.find_all('pre')

        if not code_blocks:
            # No code blocks - just text
            text = clean_for_speech(get_text_content(section_soup))
            if text:
                result.append((section_title, text, main_voice))
            return result

        # Replace code blocks with markers and split
        content = str(section_soup)
        code_data = []

        for i, pre in enumerate(code_blocks):
            # Get language from class (e.g., language-python, hljs python)
            language = ""
            code_elem = pre.find('code')
            if code_elem and code_elem.get('class'):
                classes = code_elem.get('class')
                for cls in classes:
                    if cls.startswith('language-'):
                        language = cls[9:]
                        break
                    elif cls in ['python', 'javascript', 'go', 'rust', 'java', 'c', 'cpp', 'bash', 'shell', 'sql', 'html', 'css']:
                        language = cls
                        break

            code_text = pre.get_text()
            code_data.append((language, code_text))
            marker = f"__CODE_BLOCK_{i}__"
            content = content.replace(str(pre), marker, 1)

        # Split by markers
        parts = re.split(r'(__CODE_BLOCK_\d+__)', content)
        text_idx = 0

        for part in parts:
            code_match = re.match(r'__CODE_BLOCK_(\d+)__', part)
            if code_match:
                idx = int(code_match.group(1))
                language, code_text = code_data[idx]
                code_title = get_code_block_title(code_text, language)

                if code_mode == "skip":
                    # Just announce and skip
                    result.append((code_title, skip_code_text(language), main_voice))
                elif code_mode == "verbatim":
                    # Read code with punctuation verbalized
                    verbalized = verbalize_code(code_text)
                    result.append((code_title, verbalized, code_voice))
                elif code_mode == "describe" and client:
                    # AI description
                    description = describe_code_ai(code_text, language, client)
                    result.append((code_title, description, main_voice))
                else:
                    # Fallback to skip
                    result.append((code_title, skip_code_text(language), main_voice))
            else:
                part_soup = BeautifulSoup(part, 'html.parser')
                text = clean_for_speech(get_text_content(part_soup))
                if text:
                    title_suffix = f" (Part {text_idx + 1})" if text_idx > 0 else ""
                    result.append((section_title + title_suffix, text, main_voice))
                    text_idx += 1

        return result

    # Split by H2 headings
    content = str(main)
    h2_pattern = r'<h2[^>]*>(.*?)</h2>'
    parts = re.split(h2_pattern, content, flags=re.IGNORECASE | re.DOTALL)

    # First part is intro (before first H2)
    if parts[0].strip():
        chapters.extend(process_section(parts[0], "Introduction"))

    # Remaining parts alternate: h2_title, content, h2_title, content...
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            h2_title = BeautifulSoup(parts[i], 'html.parser').get_text(separator=' ', strip=True)
            # Clean up anchor link artifacts
            h2_title = re.sub(r'\s*#\s*$', '', h2_title).strip()
            chapters.extend(process_section(parts[i + 1], h2_title))

    # If no chapters found, treat whole page as one chapter
    if not chapters:
        full_text = clean_for_speech(get_text_content(main))
        if full_text:
            chapters.append((title, full_text, main_voice))

    return chapters


def extract_chapters_ai(html, client, code_voice=DEFAULT_CODE_VOICE, main_voice=DEFAULT_VOICE,
                        code_mode="skip"):
    """
    Extract chapters using AI for better semantic understanding.
    Works on pages without clear H2 structure.
    Code blocks handled based on code_mode: skip, verbatim, or describe.
    Returns list of (title, text, voice) tuples.
    """
    soup = BeautifulSoup(html, 'html.parser')

    # Get clean text content
    main = soup.find('main') or soup.find('article') or soup.find('body')
    if not main:
        main = soup

    # Remove nav, footer, scripts, etc.
    for tag in main.find_all(['nav', 'footer', 'script', 'style', 'noscript']):
        tag.decompose()

    text = get_text_content(main)

    # Truncate if too long (keep under token limits)
    if len(text) > 60000:
        text = text[:60000] + "\n\n[Content truncated...]"

    # Adjust prompt based on code_mode
    if code_mode == "skip":
        code_instructions = """
CODE BLOCKS:
- If you encounter code blocks, create a chapter for each one
- Set type to "code" and include the language
- Put a brief note like "Python code example" as content (we'll replace it)"""
    elif code_mode == "describe":
        code_instructions = """
CODE BLOCKS:
- If you encounter code blocks, create a chapter for each one
- Set type to "code" and include the language
- Describe what the code does in plain English (1-3 sentences)
- Do NOT read the code literally, explain its purpose"""
    else:  # verbatim
        code_instructions = """
CODE BLOCKS:
- If you encounter code blocks, create a chapter for each one
- Set type to "code" and include the language
- Convert punctuation to words for TTS (e.g., "{" becomes "open brace")
- Say "new line" between lines, "indent N" for indentation levels"""

    prompt = f"""You are preparing a webpage for audio playback via text-to-speech (TTS).

HOW THIS WILL BE USED:
- The webpage displays an audio player with chapter navigation
- Users see the ORIGINAL webpage content AND an audio player
- Chapters let users jump to specific sections in the audio
- The chapter content you provide will be spoken aloud by TTS
- Users may read along while listening, so content must match the original

CRITICAL UNDERSTANDING:
This is NOT a summary. The chapters appear ALONGSIDE the original content on the same page.
If your audio says different things than what's on screen, it breaks the reading-along experience.
Your job is to COPY the content into chapters, with only minor audio-friendly tweaks.

YOUR TASK:
1. Divide the webpage into logical chapter sections (for navigation)
2. Copy ALL content from each section into the chapter
3. Make only tiny tweaks for audio clarity (see below)
4. Handle code blocks as instructed below

CONTENT PRESERVATION RULES:
- Every sentence from the input MUST appear in your output
- Every paragraph MUST be included, nearly word-for-word
- Your total output should be the SAME LENGTH as the input
- If input is 8000 words, output must be ~8000 words
- Think of yourself as a COPY EDITOR, not a summarizer

MINOR AUDIO TWEAKS ONLY:
- Remove URLs/email addresses (say "contact info on website" if needed)
- Remove "Click here", "Read more", "Back to top" navigation text
- Expand confusing abbreviations for speech
- Remove image/diagram references that won't be visible
{code_instructions}

CHAPTER STRUCTURE:
- 5-15 chapters based on natural topic breaks
- Clear, descriptive titles
- Each chapter contains the COMPLETE text of that section

OUTPUT FORMAT (JSON):
{{
  "chapters": [
    {{"title": "Introduction", "content": "Complete text...", "type": "text"}},
    {{"title": "Python Function: example", "content": "...", "type": "code", "language": "python"}},
    {{"title": "Section Name", "content": "Complete text...", "type": "text"}}
  ]
}}

---

WEBPAGE CONTENT TO CONVERT (include ALL of this in your chapters):

""" + text

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.1
    )

    result = json.loads(response.choices[0].message.content)

    chapters = []
    for ch in result.get("chapters", []):
        title = ch.get("title", "Untitled")
        content = ch.get("content", "")
        ch_type = ch.get("type", "text")
        language = ch.get("language", "")

        if ch_type == "code":
            if code_mode == "skip":
                # Just skip announcement
                chapters.append((title, skip_code_text(language), main_voice))
            elif code_mode == "verbatim":
                # AI should have verbalized, but clean up
                if content:
                    chapters.append((title, content, code_voice))
            elif code_mode == "describe":
                # AI description
                if content:
                    chapters.append((title, content, main_voice))
        else:
            # Regular text
            content = clean_for_speech(content)
            if content:
                chapters.append((title, content, main_voice))

    return chapters


def slugify(text):
    """Convert text to filename-safe slug"""
    return re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')[:50]


def get_hash(text):
    """Get short hash of text"""
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def get_page_id(source):
    """Generate unique ID for a page/URL"""
    if source.startswith(('http://', 'https://')):
        # URL: use host + full path
        parsed = urlparse(source)
        page_id = f"{parsed.netloc}{parsed.path}"
    else:
        # Local file: use absolute path
        page_id = str(Path(source).resolve())

    # Create readable slug with hash suffix for uniqueness
    slug = slugify(page_id)[:60]
    hash_suffix = hashlib.md5(page_id.encode()).hexdigest()[:8]
    return f"{slug}-{hash_suffix}"


class Page2Pod:
    def __init__(self, source, output_dir=None, cache_dir=None, voice=DEFAULT_VOICE,
                 code_voice=DEFAULT_CODE_VOICE, model=DEFAULT_MODEL, code_mode="skip"):
        # Resolve local paths to absolute
        if not source.startswith(('http://', 'https://')):
            source = str(Path(source).resolve())
        self.source = source
        self.voice = voice
        self.code_voice = code_voice
        self.code_mode = code_mode
        self.model = model
        self.client = OpenAI()

        # Set up directories
        self.page_id = get_page_id(source)
        self.cache_dir = (cache_dir or DEFAULT_CACHE) / self.page_id
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Cache files
        self.chapters_dir = self.cache_dir / "chapters"
        self.chapters_dir.mkdir(exist_ok=True)
        self.meta_file = self.cache_dir / "meta.json"
        self.source_file = self.cache_dir / "source.html"

    def load_content(self):
        """Load HTML content from file or URL"""
        if self.source.startswith(('http://', 'https://')):
            import requests
            response = requests.get(self.source, timeout=30)
            response.raise_for_status()
            html = response.text
        else:
            html = Path(self.source).read_text()

        # Save source for reference
        self.source_file.write_text(html)
        return html

    def load_meta(self):
        """Load cached metadata"""
        if self.meta_file.exists():
            return json.loads(self.meta_file.read_text())
        return {"chapters": {}}

    def save_meta(self, meta):
        """Save metadata"""
        self.meta_file.write_text(json.dumps(meta, indent=2))

    def generate_audio(self, text, voice=None):
        """Generate audio using OpenAI TTS"""
        if voice is None:
            voice = self.voice

        chunks = []
        remaining = text

        while remaining:
            chunk = remaining[:4000]
            if len(remaining) > 4000:
                last_period = chunk.rfind('. ')
                if last_period > 2000:
                    chunk = remaining[:last_period + 1]

            response = self.client.audio.speech.create(
                model=self.model,
                voice=voice,
                input=chunk
            )
            chunks.append(response.content)
            remaining = remaining[len(chunk):]

        return b''.join(chunks)

    def get_duration(self, mp3_file):
        """Get MP3 duration in seconds"""
        try:
            return MP3(str(mp3_file)).info.length
        except:
            return mp3_file.stat().st_size / 16000.0

    def add_chapters_to_mp3(self, mp3_path, chapters_with_timestamps, title):
        """Add ID3 chapter markers"""
        audio = MP3(mp3_path, ID3=ID3)
        try:
            audio.add_tags()
        except:
            pass

        audio.tags.add(TIT2(encoding=3, text=title))
        audio.tags.add(TALB(encoding=3, text="page2pod"))

        chapter_ids = []
        for i, (ch_title, start_ms, end_ms) in enumerate(chapters_with_timestamps):
            chapter_id = f"chp{i}"
            chapter_ids.append(chapter_id)

            audio.tags.add(CHAP(
                encoding=3,
                element_id=chapter_id.encode('latin1'),
                start_time=int(start_ms),
                end_time=int(end_ms),
                start_offset=0xFFFFFFFF,
                end_offset=0xFFFFFFFF,
                sub_frames=[TIT2(encoding=3, text=ch_title)]
            ))

        audio.tags.add(CTOC(
            encoding=3,
            element_id=b'toc',
            flags=CTOCFlags.TOP_LEVEL | CTOCFlags.ORDERED,
            child_element_ids=[cid.encode('latin1') for cid in chapter_ids],
            sub_frames=[TIT2(encoding=3, text="Table of Contents")]
        ))

        audio.save()

    def process(self, force=False, combine_only=False, chapter_num=None, list_only=False, use_ai=False):
        """Process the page into podcast"""
        print(f"page2pod: {self.source}")
        print(f"Cache: {self.cache_dir}")
        print(f"Mode: {'AI chapter extraction' if use_ai else 'H2-based extraction'}")
        print(f"Voices: {self.voice} (main), {self.code_voice} (code)")
        print(f"Code: {self.code_mode}")
        print("=" * 50)

        # Load and parse
        html = self.load_content()

        if use_ai:
            print("Analyzing content with AI...")
            chapters = extract_chapters_ai(html, self.client,
                                           code_voice=self.code_voice,
                                           main_voice=self.voice,
                                           code_mode=self.code_mode)
        else:
            soup = BeautifulSoup(html, 'html.parser')
            chapters = extract_chapters(soup,
                                        code_voice=self.code_voice,
                                        main_voice=self.voice,
                                        code_mode=self.code_mode,
                                        client=self.client)

        print(f"Found {len(chapters)} chapters:")
        for i, (title, text, voice) in enumerate(chapters):
            voice_indicator = " [code]" if voice == self.code_voice else ""
            print(f"  [{i}] {title} ({len(text)} chars){voice_indicator}")

        if list_only:
            return

        # Load cached metadata
        meta = self.load_meta()
        chapter_files = []
        generated = 0
        skipped = 0

        for idx, (title, text, voice) in enumerate(chapters):
            filename = f"{idx:02d}-{slugify(title)}.mp3"
            chapter_file = self.chapters_dir / filename
            text_hash = get_hash(text + voice)  # Include voice in hash

            # Save chapter text for reference
            text_file = self.chapters_dir / f"{idx:02d}-{slugify(title)}.txt"
            text_file.write_text(text)

            # Determine if regeneration needed
            cached_hash = meta["chapters"].get(filename, {}).get("hash")
            needs_gen = (
                force or
                (chapter_num is not None and idx == chapter_num) or
                (not combine_only and not chapter_file.exists()) or
                (not combine_only and cached_hash != text_hash)
            )

            if needs_gen and not combine_only:
                voice_label = "code" if voice == self.code_voice else "main"
                print(f"\n[{idx}] Generating: {title} [{voice_label}]...")
                audio_data = self.generate_audio(text, voice=voice)
                chapter_file.write_bytes(audio_data)
                duration = self.get_duration(chapter_file)
                print(f"    Saved: {filename} ({duration:.1f}s)")

                meta["chapters"][filename] = {"hash": text_hash, "title": title, "voice": voice}
                generated += 1
            else:
                if chapter_file.exists():
                    skipped += 1
                else:
                    print(f"[{idx}] WARNING: {filename} missing")

            if chapter_file.exists():
                chapter_files.append((chapter_file, title))

        self.save_meta(meta)
        print(f"\nGenerated: {generated}, Skipped: {skipped}")

        if not chapter_files:
            print("No chapter files to combine")
            return

        # Combine chapters
        print(f"\nCombining {len(chapter_files)} chapters...")

        timestamps = []
        current_ms = 0
        concat_file = self.cache_dir / "concat.txt"

        with open(concat_file, 'w') as f:
            for chapter_file, title in chapter_files:
                f.write(f"file '{chapter_file.absolute()}'\n")
                duration_ms = int(self.get_duration(chapter_file) * 1000)
                timestamps.append((title, current_ms, current_ms + duration_ms))
                current_ms += duration_ms

        # Output file
        output_name = f"{self.page_id}.mp3"
        output_file = self.output_dir / output_name

        subprocess.run([
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
            '-i', str(concat_file),
            '-c', 'copy',
            str(output_file)
        ], capture_output=True, check=True)

        concat_file.unlink()

        # Add chapter markers
        page_title = chapters[0][0] if chapters else "Untitled"
        self.add_chapters_to_mp3(output_file, timestamps, page_title)

        # Write chapters JSON
        chapters_json = {
            "source": self.source,
            "chapters": [
                {"title": t, "start": s/1000, "end": e/1000}
                for t, s, e in timestamps
            ]
        }
        json_file = self.output_dir / f"{self.page_id}.chapters.json"
        json_file.write_text(json.dumps(chapters_json, indent=2))

        total_sec = current_ms / 1000
        print(f"\n{'=' * 50}")
        print(f"Output: {output_file}")
        print(f"Duration: {int(total_sec // 60)}:{int(total_sec % 60):02d}")
        print(f"Chapters: {json_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert web pages to chapter-based podcasts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("source", help="HTML file or URL")
    parser.add_argument("-o", "--output", help="Output directory")
    parser.add_argument("-c", "--cache", help="Cache directory")
    parser.add_argument("-v", "--voice", default=DEFAULT_VOICE,
                        choices=["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
                        help="OpenAI TTS voice for main content")
    parser.add_argument("--code-voice", default=DEFAULT_CODE_VOICE,
                        choices=["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
                        help="OpenAI TTS voice for code (default: nova)")
    parser.add_argument("--code-mode", default="skip",
                        choices=["skip", "verbatim", "describe"],
                        help="How to handle code: skip (default), verbatim, or describe")
    parser.add_argument("--quality", default=DEFAULT_MODEL,
                        choices=["tts-1", "tts-1-hd"],
                        help="TTS model quality")
    parser.add_argument("--force", action="store_true",
                        help="Force regenerate all chapters")
    parser.add_argument("--combine", action="store_true",
                        help="Just recombine existing chapters")
    parser.add_argument("--chapter", type=int,
                        help="Regenerate only this chapter number")
    parser.add_argument("--list", action="store_true",
                        help="List chapters without generating")
    parser.add_argument("--no-ai", action="store_true",
                        help="Use H2-based extraction instead of AI")

    args = parser.parse_args()

    cache_dir = Path(args.cache) if args.cache else None

    converter = Page2Pod(
        source=args.source,
        output_dir=args.output,
        cache_dir=cache_dir,
        voice=args.voice,
        code_voice=args.code_voice,
        code_mode=args.code_mode,
        model=args.quality
    )

    converter.process(
        force=args.force,
        combine_only=args.combine,
        chapter_num=args.chapter,
        list_only=args.list,
        use_ai=not args.no_ai
    )


if __name__ == "__main__":
    main()
