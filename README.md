# search-image-meta

Find images and videos by what's written in their metadata — fast, in
parallel, on macOS, Linux and Windows.

It reads EXIF (including the nested Exif/GPS blocks), XMP, IPTC, JPEG
comments, PNG text chunks, and the metadata of MP4 / MOV / WebM / MKV videos.
It understands **ComfyUI** graphs well enough to search only the nodes that
actually made the image, and it reads Automatic1111 / Forge `parameters` too.

```bash
search-image-meta ~/ComfyUI/output "lighthouse" -r
```

```
/Users/me/ComfyUI/output/2026-09/lighthouse_00012_.png
/Users/me/ComfyUI/output/2026-09/lighthouse_00013_.png

2 match(es) in 18,204 image(s) scanned
```

## Install

With [pipx](https://pipx.pypa.io/) (Python 3.9 or newer):

```bash
pipx install git+https://github.com/wasawi/search-image-meta
```

Or keep it as a single script: copy `search_string_image_meta.py` anywhere
and install its only dependency, `pip install pillow`. Everything below works
the same with `python3 search_string_image_meta.py` instead of
`search-image-meta`.

For HEIC / HEIF photos (iPhones), also install `pillow-heif`
(`pipx install "search-image-meta[heif] @ git+https://github.com/wasawi/search-image-meta"`,
or `pip install pillow-heif` next to the script).

## Searching

```bash
# this folder only, or everything below it
search-image-meta ~/Pictures "sunset"
search-image-meta ~/Pictures "sunset" -r

# every term must appear somewhere in the image (default) …
search-image-meta ~/output "Barcelona" "sunset" -r
# … or any one of them
search-image-meta ~/output "Barcelona" "Girona" -r --any
# … but not these
search-image-meta ~/output "Barcelona" -r --not "night" "rain"

# regular expressions, case-sensitive
search-image-meta ~/output "wan.?2\.2" -r --regex -s
```

Searches ignore case, and an accented letter matches however the file
encoded it (`café` typed in a terminal finds a composed or decomposed `é`;
`cafe` without the accent doesn't). Chinese, Japanese, Korean and emoji work
everywhere, including inside ComfyUI graphs, which store them as `\uXXXX`
escapes.

| Option | What it does |
|---|---|
| `--any` | match images with any of the terms instead of all |
| `--not TERM…` | skip images where any of these turn up |
| `--regex`, `-s` | terms are regular expressions; match case |
| `--scope image\|field\|node` | where several terms must be found: anywhere in the image (default), in one metadata field, or in one ComfyUI node |
| `--fields NAME…` | only these metadata fields, e.g. `prompt workflow` |
| `--ext EXT…` | only these file types, e.g. `png webp mp4` |
| `-r`, `--hidden`, `--follow-symlinks` | walk subfolders, include hidden files, follow linked folders |
| `--exclude-dir NAME…` | skip folders by name, wildcards allowed: `--exclude-dir "old_*" thumbnails` |
| `--since WHEN`, `--until WHEN` | modification date: `2026-09-01`, `"2026-09-01 18:30"`, or an age like `3d`, `12h`, `2w` |
| `--deep` | also read PNG text stored after the image data and XMP deep inside files (about 20× slower; rarely needed) |

## ComfyUI

A ComfyUI image carries its whole graph twice: `prompt` (what ran) and
`workflow` (the editor canvas, including notes, leftovers and disabled
nodes). A plain search looks at all of it, so a checkpoint loader you left
lying around unwired still makes an image match.

```bash
# only nodes wired into the graph: unconnected, muted and bypassed nodes,
# notes and reroutes are ignored
search-image-meta ~/output "krea" -r --only-connected

# stricter: only nodes that lead to a Save / Preview node, so wired-up
# leftovers that feed nothing are ignored too
search-image-meta ~/output "krea" -r --connected-mode output

# both terms inside the same node (one prompt box, one loader …)
search-image-meta ~/output "lighthouse" "dusk" -r --scope node --only-connected

# a LoRA, but only where a LoRA loader uses it
search-image-meta ~/output "darkbrush" -r --node-type Lora

# only prompt text, not file names, titles or settings
search-image-meta ~/output "lighthouse" -r --input text value

# what made each match: prompts, models, LoRAs, seed / steps / sampler
search-image-meta ~/output "fox" -r --show
```

```
/Users/me/ComfyUI/output/fox_00003_.png
    positive: a red fox in snow | cinematic lighting
    models:   flux1-dev.safetensors, t5xxl_fp16.safetensors, clip_l.safetensors, ae.safetensors
    loras:    detail_tweaker.safetensors (0.6), style_a.safetensors (0.9)
    sampler:  seed 7, steps 28, cfg 1, sampler_name euler, scheduler simple, denoise 1
```

What counts as connected:

- Links through **bypassed** nodes, **reroutes** and KJNodes **Set/Get** pairs
  count; the bypassed nodes and plumbing themselves are never searched.
- **Subgraphs** are followed inside; a subgraph's own unwired nodes are
  dropped like any other.
- `--connected-mode output` recognises output nodes by name (Save, Preview,
  Show, Display, VideoCombine, …). If a graph has none it falls back to plain
  `--only-connected` rather than discarding everything.
- `--node-type` matches part of the node's type, any case. `--input` matches
  exact input names; it uses `prompt`, and `workflow` files saved by newer
  frontends (older ones don't record widget names).
- `--show` reads the `prompt` (only nodes that reach an output) or A1111
  `parameters`. It doesn't evaluate switch nodes, so every connected branch is
  listed.

Videos from ComfyUI's SaveVideo / SaveWEBM and VideoHelperSuite, and
ComfyUI's WebP / AVIF images, get all of the above.

## Output

| Option | Output |
|---|---|
| (default) | one path per line; the summary goes to stderr |
| `-v` | also the matching field and a snippet of text around the match |
| `--show` | a summary of prompts, models, LoRAs and sampler under each path |
| `--json` | everything as one JSON document |
| `--csv` | one row per matching field: path, field, snippet, link (plus summary columns with `--show`) |
| `-0` | NUL-separated paths, for `xargs -0` |

The exit status is 0 when something matched and 1 when nothing did, so it
works in scripts: `search-image-meta ~/output "krea" -r > /dev/null && echo found`.

## Collecting matches

```bash
# a link to every match in one folder
search-image-meta ~/output "krea" -r --link-dir ~/matches

# copies named after their relative path (a/b/img.png -> a__b__img.png)
search-image-meta ~/output "krea" -r --link-dir ~/matches --link-type copy --flatten path
```

`--link-type auto` (the default) makes symlinks; on Windows, where symlinks
need Developer Mode or admin rights, it falls back to hardlinks and then to
copies. `alias` makes a Finder alias on macOS and a shortcut (`.lnk`) on
Windows. Links get the original's modification date and, on macOS and
Windows, its creation date (`--no-preserve-dates` to skip).

## Big collections

```bash
# cache what was read: later searches of that drive only read new or changed files
search-image-meta /Volumes/Photos "krea" -r --index ~/data_meta.sqlite

# a resumable log: rerun the same command after an interruption to continue
search-image-meta /Volumes/Photos "krea" -r --results ~/krea_results.txt
```

- `--index FILE` stores the extracted metadata in SQLite, keyed by path, size
  and modification time. It serves any later search, whatever the terms or
  options. It pays off when reading files is the slow part — network shares,
  USB drives, the first search after a reboot — and saves little on a fast
  local disk whose files the system has already cached. The file is large
  (roughly 12 KB per ComfyUI image); delete it to start over.
- `--results FILE` records settings, finished folders and matches; rerunning
  with the same settings skips finished folders (`--no-resume` to start fresh).
- The work runs in a pool of processes, which uses every CPU core: on 55,000
  ComfyUI images that took 14 s against 26 s with threads, and 13 s against
  42 s for a search with accented or Chinese terms. `--pool thread` starts
  instantly and can suit slow network drives. `-j N` sets the number of workers (default: half the CPU cores for processes,
  twice the cores for threads).
- `--progress never` and `--no-count` keep the output quiet and skip the
  counting pass.

## Development

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[test]"
.venv/bin/python -m pytest
```

The tests build their own images and videos, including synthetic ComfyUI
graphs with stray, bypassed, muted, rerouted and subgraph nodes. To look at
those images yourself (drag one into ComfyUI to see its graph):

```bash
python3 tests/samples.py ./samples
```

## License

MIT
