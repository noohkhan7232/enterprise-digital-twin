# Recording Plan

A production plan for the 12–15 minute technical walkthrough video described in
`narration_script.md`. It covers preparation, capture, editing and publishing, aimed at a
YouTube-quality result suitable for a portfolio or conference channel.

---

## 1. Equipment and Software

| Need | Minimum | Preferred |
|------|---------|-----------|
| Microphone | USB condenser or quality headset | Cardioid XLR via audio interface |
| Screen capture | OBS Studio (free) | OBS Studio with scene collection |
| Terminal capture | Native screen record | `asciinema` for crisp terminal + OBS overlay |
| Editor | Any non-linear editor | DaVinci Resolve (free tier) |
| Pop filter / treatment | Soft furnishings to reduce echo | Foam panels / treated room |

## 2. Pre-Production

- [ ] Lock the script (`narration_script.md`); rehearse aloud twice; mark breath points.
- [ ] Prepare the architecture figure and any concept diagrams as high-resolution stills.
- [ ] Pre-run every terminal command; save outputs so the live capture is clean and fast.
- [ ] Increase terminal/editor font sizes; choose a high-contrast, calm colour scheme.
- [ ] Silence notifications; close unrelated apps; set a clean desktop.
- [ ] Verify the test suite is green and the observability demos run before recording.

## 3. Scene Plan (maps to narration segments)

| Scene | Source | On-screen | Notes |
|------:|--------|-----------|-------|
| 1 | Slide | Title card → architecture figure | 0.5 s fade-in |
| 2 | Diagram | Before-state (disconnected capabilities) | Build the four boxes on cue |
| 3 | Figure | Ten-layer architecture | Highlight capability vs. operational layers |
| 4 | Figure | Zoom each capability layer | Synchronise zoom with narration |
| 5 | Figure | MLOps + lineage sketch | Animate the lineage links |
| 6 | Diagram | Monitoring concept | Emphasise drift vs. concept drift |
| 7 | Terminal + figure | Health check; topology | Pre-run; paste command |
| 8 | Terminal | Metrics, reliability, readiness demos; rerun once | Show byte-identical output |
| 9 | Terminal | `pytest tests/ -q` summary | Crop to the summary line |
| 10 | Slide | Architecture figure; repo placeholder | Calm outro |

## 4. Capture Workflow

1. **Audio first or separate track.** Record narration as its own track (voice-over) for clean
   editing; capture terminal/video silently, then lay narration over it.
2. **One segment at a time.** Capture each segment independently; retakes are cheap and editing is
   simpler than one long take.
3. **Terminal segments.** Use pre-run outputs; type or paste commands deliberately. For the
   determinism beat, run the same demo twice on camera and let the identical output speak for itself.
4. **Room tone.** Record 10 seconds of silence for noise reduction in post.

## 5. Editing

- [ ] Assemble scenes to the script order; trim dead air.
- [ ] Apply gentle noise reduction and a mild compressor to the voice track; normalise to about
      -16 LUFS for online playback.
- [ ] Add unobtrusive lower-thirds for section titles; keep motion minimal and professional.
- [ ] Caption the video (accurate captions improve accessibility and retention); the script is the
      caption source.
- [ ] Add a brief title and end card; no background music, or very low and neutral if used.
- [ ] Export at 1080p or 1440p, high bitrate; verify legibility of terminal text at target
      resolution.

## 6. Quality Bar

- Terminal text legible at 1080p without zooming.
- Audio free of clipping, hum and echo; consistent levels across segments.
- Pacing matches the script (~130–140 wpm) with deliberate pauses on demos.
- No fabricated figures shown; the determinism and test segments carry the proof.

## 7. Publishing

- [ ] Title: "Enterprise Digital Twin & Decision Intelligence Platform — Technical Walkthrough".
- [ ] Description: one-paragraph summary, chapter timestamps (per segment), and links to the
      repository, research paper and documentation (placeholders until public).
- [ ] Chapters set at each segment boundary for navigability.
- [ ] Thumbnail: the architecture figure with the title; clean and readable.
- [ ] Pin a comment clarifying that the video reports verified engineering properties and configured
      targets, not benchmark results.

## 8. Reusable Cuts

From the same capture, export:
- A **2–3 minute teaser** (segments 1–3 plus the determinism beat) for social or portfolio landing
  pages.
- A **60-second executive cut** (problem → architecture → outcomes) for non-technical audiences.