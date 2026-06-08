# Example Inputs

Place your input videos here. Any short phone video of a small indoor space works well.

## Recording tips for best results

- Walk slowly, keep the camera steady
- Cover the room from multiple angles — the model needs overlap between frames
- 10–30 seconds is plenty; longer is fine but more frames = more memory
- Good lighting helps (avoid very dark or very bright scenes)
- Avoid pure white walls with no texture — VGGT handles them but results are sparser

## Suggested test clip

If you don't have a video handy, any indoor clip from YouTube works.
Download with `yt-dlp`:

```bash
pip install yt-dlp
yt-dlp -f "bestvideo[height<=720]" <URL> -o examples/room.mp4
```

Then run:

```bash
python run.py examples/room.mp4 --semantic
```
