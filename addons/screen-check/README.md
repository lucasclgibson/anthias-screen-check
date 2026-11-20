# Screen check (local add-on)

Watches what the screen really shows and flags broken screens: error pages, popups and cookie banners, logins,
loading, blank screens, system dialogs and video glitches. An experiment on a fork of Anthias; not part of upstream Anthias.

It changes nothing in Anthias. It runs as its own compose project next to the Anthias stack:

- **Capture:** on Wayland boards (Pi 5, x86) Anthias's viewer runs under the cage compositor. The add-on finds the
  viewer's Wayland socket through `/proc` (it runs with `pid: host` and `SYS_PTRACE`) and takes screenshots with
  `grim`. That is the composited output, video included, as close to HDMI as software gets.
- **Model:** the final screen-state model (`model.fp32.onnx`, MobileNetV3-Small, 12 outcomes, 4.4 MB) at about
  430,000 pixels per screen (864x480 for 16:9), on one CPU thread at low priority (`nice 10`, capped at 1 core and 300 MB). When the screen has not changed
  since the last check (every 16x16 block within 2 grey levels) the model is skipped and its last answer reused; it
  still runs at least every 10 s.
- **Confirmation:** the demo's frame-combining rules with the tuned settings in `settings.json`. A problem must
  persist (1 s, 2 s for loading, 0.8 s for glitches) before it alerts, and clears after 0.5 s of normal frames.

## The model

The model is not in this repository. It lives in the private Hugging Face repo
[`screenly/screen-state-model`](https://huggingface.co/screenly/screen-state-model), so only people with access to the
Screenly organisation on Hugging Face can run the check. `settings.json` pins the revision and the file's SHA-256.

On first start the add-on downloads `model.fp32.onnx` with your Hugging Face token, checks the hash and keeps it in the
`screen-check-data` volume. After that it starts without the token or a network connection. Put a read token in a
`.env` file next to `docker-compose.yml` (git-ignored):

```bash
echo "HF_TOKEN=hf_..." > .env && chmod 600 .env
```

Already have the file? Mount it and set `SCREEN_CHECK_MODEL=/path/in/container/model.fp32.onnx` instead; it is
checked against the same hash. Without a token or a model the status page shows the reason and the container retries
every minute.

## Use

```bash
cd ~/anthias/addons/screen-check        # or wherever this folder is on the player
echo "HF_TOKEN=hf_..." > .env           # first start only: read access to screenly/screen-state-model
docker compose up -d --build            # start (builds the image on the player)
docker compose logs -f                  # ALERT lines when a problem is confirmed or cleared
docker compose down                     # remove
```

- Status page: `http://<player>:8090/` (state, last frame). JSON at `/status`, the last frame at `/frame.jpg`.
- Settings (environment): `SCREEN_CHECK_FPS` (checks per second, default 2), `SCREEN_CHECK_THREADS` (1),
  `SCREEN_CHECK_WEBHOOK` (URL that gets a JSON POST on `problem` and `cleared`).
- Evidence: the frame that triggered each alert is kept in the `screen-check-data` volume (`alerts/`, last 50).

Because it is a separate compose project, Anthias upgrades (`upgrade_containers.sh`, which runs
`up -d --remove-orphans` on the Anthias project) leave it alone.

## Limits

- Wayland boards only (Pi 5, x86, arm64 SBCs). Pi 1/2/3-32 draw into `/dev/fb0` and Pi 3-64/Pi 4 put video on a
  separate DRM plane; both need a different capture.
- It does not know which asset is playing yet; asset changes are handled by the screen-change detector.
- Checked offline on the stress-test video at 5 checks a second: 17 of 21 problems confirmed, no false alerts (the
  same as the tuning). On a Pi 5 at 2 checks a second, live: 15 of 21, no false alerts, about 37% of one core on
  that mostly-video test (a 2560x1440 screen), 150 MB.

## Testing without a screen

`SCREEN_CHECK_SOURCE=/frames` replays a folder of frames at `SCREEN_CHECK_FPS` and prints one JSON line per frame:

```bash
docker run --rm -v "$PWD/frames:/frames:ro" -v screen-check-data:/data -e HF_TOKEN \
  -e SCREEN_CHECK_SOURCE=/frames -e SCREEN_CHECK_FPS=5 anthias-screen-check:local
```
