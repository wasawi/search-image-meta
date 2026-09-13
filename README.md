# search-image-meta

Search image metadata for strings, in parallel: EXIF, XMP, IPTC, PNG text
chunks, and ComfyUI `prompt` / `workflow` graphs.

Full documentation is on its way; for now see the examples at the top of
`search_string_image_meta.py` and `--help`.

## Tests

```bash
python3 -m venv .venv
.venv/bin/pip install pillow pytest av
.venv/bin/python -m pytest
```
