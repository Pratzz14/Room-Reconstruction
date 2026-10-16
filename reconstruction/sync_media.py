"""Content-addressed browser playback copies; reconstruction inputs remain untouched."""

import hashlib
import json
import math
import os
import uuid
from fractions import Fraction
from pathlib import Path

from .core import digest, read_json, write_json


def byte_range(header, size):
    """Inclusive single byte range, including open-ended and suffix requests."""
    if not header:
        return 0, size - 1, 200
    if not header.startswith("bytes=") or "," in header:
        raise ValueError("Expected one bytes range")
    parts = header[6:].split("-")
    if len(parts) != 2 or not any(parts):
        raise ValueError("Invalid byte range")
    left, right = parts
    if any(part and (not part.isascii() or not part.isdigit()) for part in parts):
        raise ValueError("Invalid byte range")
    if not left:
        count = int(right)
        if count <= 0:
            raise ValueError("Invalid suffix range")
        start, end = max(0, size - count), size - 1
    else:
        start, end = int(left), min(int(right), size - 1) if right else size - 1
    if size <= 0 or start < 0 or start >= size or end < start:
        raise ValueError("Unsatisfiable byte range")
    return start, end, 206


def prepare_playback(source, source_hash, cache_root):
    """Encode H.264/AAC with source PTS spacing and a normalized video origin."""
    import av

    settings = {"version": 2, "av": av.__version__, "video": "libx264", "crf": "20",
                "preset": "veryfast", "pixel_format": "yuv420p", "audio": "aac"}
    key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:16]
    directory = Path(cache_root) / source_hash / key
    directory.mkdir(parents=True, exist_ok=True)
    target, receipt = directory / "playback.mp4", directory / "playback.json"
    if target.is_file() and receipt.is_file():
        try:
            record = read_json(receipt)
            if (record["source_sha256"] == source_hash and record["settings"] == settings
                    and record["sha256"] == digest(target)):
                print(f"Playback cache ready: {source.name}", flush=True)
                return target, record
        except (ValueError, KeyError, OSError):
            pass
    lock = directory / ".encode.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"Playback conversion already locked: {lock}. Check the owning viewer before removing a stale lock.") from exc
    os.close(fd)
    partial = directory / f"{uuid.uuid4().hex}.mp4"
    print(f"Preparing browser playback: {source.name} (first launch only)", flush=True)
    try:
        with av.open(str(source)) as incoming, av.open(str(partial), "w", options={"movflags": "+faststart"}) as outgoing:
            video = incoming.streams.video[0]
            origin = Fraction(video.start_time or 0) * video.time_base
            rate = video.average_rate or Fraction(30)
            output_video = outgoing.add_stream("libx264", rate=rate)
            output_video.time_base = Fraction(1, 90000)
            output_video.codec_context.time_base = Fraction(1, 90000)
            output_video.pix_fmt = "yuv420p"
            output_video.options = {"crf": "20", "preset": "veryfast"}
            output_video.codec_context.max_b_frames = 0
            output_video.codec_context.gop_size = max(1, round(float(rate) * 2))
            audio = incoming.streams.audio[0] if incoming.streams.audio else None
            output_audio = outgoing.add_stream("aac", rate=48000) if audio else None
            resampler = av.AudioResampler(format="fltp", layout="stereo", rate=48000) if audio else None
            if output_audio:
                output_audio.layout = "stereo"
            streams = [video] + ([audio] if audio else [])
            frame_count, rotation, first_time, last_time, next_progress = 0, 0, None, None, 10.0
            durations = {}

            def mux_video(encoded):
                # Encoder packet defaults use average FPS; preserve VFR and the final frame's duration.
                key = round(encoded.pts * encoded.time_base * 90000)
                duration = durations.pop(key, None)
                if duration:
                    encoded.duration = round(duration / encoded.time_base)
                outgoing.mux(encoded)

            for packet in incoming.demux(streams):
                for frame in packet.decode():
                    timestamp = Fraction(frame.pts) * frame.time_base - origin if frame.pts is not None else None
                    if packet.stream == video:
                        if timestamp is None:
                            raise ValueError("Video frame has no presentation timestamp")
                        if frame_count == 0:
                            rotation = round(frame.rotation) % 360
                            if rotation not in (0, 90, 180, 270):
                                raise ValueError(f"Unsupported video rotation: {rotation}")
                            output_video.width = frame.height if rotation in (90, 270) else frame.width
                            output_video.height = frame.width if rotation in (90, 270) else frame.height
                            # Playback-only rescaling to even dimensions for H.264 yuv420p.
                            output_video.width += output_video.width % 2
                            output_video.height += output_video.height % 2
                        duration = frame.duration * frame.time_base
                        if rotation:
                            import numpy as np
                            frame = av.VideoFrame.from_ndarray(np.rot90(frame.to_ndarray(format="rgb24"), rotation // 90), format="rgb24")
                        frame = frame.reformat(width=output_video.width, height=output_video.height, format="yuv420p")
                        frame.pts = round(timestamp * 90000)
                        frame.time_base = Fraction(1, 90000)
                        durations[frame.pts] = duration
                        for encoded in output_video.encode(frame):
                            mux_video(encoded)
                        first_time = float(timestamp) if first_time is None else first_time
                        last_time = float(timestamp)
                        frame_count += 1
                        if last_time >= next_progress:
                            print(f"  {source.name}: {last_time:.0f}s converted", flush=True)
                            next_progress += 10
                    elif output_audio and timestamp is not None:
                        frame.pts = round(timestamp / frame.time_base)
                        for converted in resampler.resample(frame):
                            for encoded in output_audio.encode(converted):
                                outgoing.mux(encoded)
            for encoded in output_video.encode(None):
                mux_video(encoded)
            if output_audio:
                for converted in resampler.resample(None):
                    for encoded in output_audio.encode(converted):
                        outgoing.mux(encoded)
                for encoded in output_audio.encode(None):
                    outgoing.mux(encoded)
            if not frame_count:
                raise ValueError("Source video has no frames")
        with av.open(str(partial)) as check:
            encoded = check.streams.video[0]
            duration = float(encoded.duration * encoded.time_base)
            if encoded.codec_context.name != "h264" or not math.isfinite(duration):
                raise ValueError("Invalid browser playback output")
        record = {"source_sha256": source_hash, "settings": settings, "sha256": digest(partial),
                  "duration": duration, "source_origin_sec": float(origin), "frame_count": frame_count,
                  "first_frame_sec": first_time, "last_frame_sec": last_time, "rotation": rotation}
        partial.replace(target)
        write_json(receipt, record)
        print(f"Playback ready: {source.name} ({duration:.2f}s, {frame_count} frames)", flush=True)
        return target, record
    finally:
        partial.unlink(missing_ok=True)
        lock.unlink(missing_ok=True)
