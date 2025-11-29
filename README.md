# deepvidsumm

A focused video summarization pipeline that turns a long user-provided video and a natural-language prompt into a concise answer clip.

## What the system does
- Input: a long video (≈15 minutes) plus a user question (e.g., "When does Mr. Sloth speak?").
- The video language model scans the video within length limits and returns the answer as a time span (e.g., 23s–54s).
- A timestamp selector pulls the relevant frames and context around the answer span.
- Conditional video generation uses the selected frames and the original prompt to produce a short (~15s) clip that answers the question.
- Output: a new short video that highlights exactly what the user asked for.

## Flow in detail
1. Ingest & prompt  
   - User supplies a long video and a question.  
   - The system prepares the video for model-friendly length limits.
2. Video language model (reasoning)  
   - Processes the clipped/condensed input and the user prompt.  
   - Produces a natural-language answer with start/end timestamps.
3. Timestamp selection (evidence gathering)  
   - Uses the returned timestamps to grab representative frames from the original video around the answer window.
4. Conditional video generation (synthesis)
   - Feeds the selected frames plus the original prompt into a generator to render a concise answer-focused clip.
5. Deliver short clip
   - Returns a new ~15s video that shows exactly the requested content.

## Repository hints
- `reference/video_preprocessor.py`: frame/audio extraction with PyAV and PIL for downstream processing.
- `reference/openrouter.py`: OpenRouter multimodal helper for structured vision/ASR calls.

## Run the Gradio demo
- Install deps with uv: `uv sync` (or `uv pip install -e .`).
- Optional: set `OPENROUTER_API_KEY` (and `OPENROUTER_MODEL`) to use a real vision call; otherwise a mock heuristic runs locally.
- Launch: `uv run python main.py` then open the Gradio link printed to the console.
- Upload a video (mp4 recommended) plus a prompt; the stitched answer clip saves under `runs/session_*`.
- FFmpeg defaults to GPU paths: CUDA decode/scale for frames (`-hwaccel cuda`, `h264_cuvid`, `scale_cuda`) and NVENC for segment encoding (`h264_nvenc`). Adjust `hwaccel`/`decoder`/`encoder` in the preprocessor/composer constructors if you need a different stack (e.g., `vaapi`/`qsv`) or a CPU-only fallback.

## Example prompt/answer pair
- Prompt: "When does Mr. Sloth speak?"
- Output: a ~15s generated clip centered on that interval showing Mr. Sloth speaking.
