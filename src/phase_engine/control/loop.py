"""
Egress data-plane loop.

Continuously polls :meth:`PhaseEngine.get_combined_samples` for every active
virtual channel and pushes the combined IQ through the channel's
:class:`RtpStreamer`.  When a backend radiod source drops (no samples
available), the loop *flywheels* by injecting zero-filled IQ at the expected
sample rate so that the downstream client's RTP session stays alive until the
source recovers.
"""

import threading
import time
import logging
import numpy as np

logger = logging.getLogger(__name__)


class EgressLoop:
    """Continuously pulls combined samples from the engine and pushes to RTP streamers."""
    
    def __init__(self, engine, channel_manager):
        self.engine = engine
        self.channel_manager = channel_manager
        self._running = False
        self._thread = None
        
        # Track the last known authoritative timestamp per virtual channel SSRC
        # so we can flywheel (zero-fill) if the source drops, preventing clients from
        # timing out their connection to us while we recover the backend.
        self._last_timestamps = {}
        
        # Track wallclock time of the last stream push to pace the flywheel properly
        self._last_push_times = {}
        
        # How many consecutive failures we've seen per SSRC, to manage logging
        self._flywheel_counts = {}

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="EgressLoop")
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def _loop(self):
        logger.info("Egress data plane loop started")
        
        # Pull the base sample rate from the engine once
        sample_rate = getattr(self.engine, "sample_rate", 24000)
        
        # Process chunks corresponding to 200ms of data maximum at a time
        chunk_size = int(sample_rate * 0.2)
        sleep_interval = 0.05  # Poll at 50ms intervals
        
        # Pre-allocate the zeros array for flywheeling
        zeros_chunk = np.zeros(chunk_size, dtype=np.complex64)

        while self._running:
            active_channels = self.channel_manager.get_active_broadcasts()

            if not active_channels or not self.engine.is_running:
                time.sleep(sleep_interval)
                continue

            now = time.time()

            for ssrc, vchan in active_channels.items():
                result = self.engine.get_combined_samples(vchan, max_samples=chunk_size)

                streamer = self.channel_manager.get_streamer(ssrc)
                if not streamer:
                    continue

                if result is not None and len(result[0]) > 0:
                    # We have valid data from the sources.
                    samples, timestamp = result
                    
                    streamer.stream_samples(samples, timestamp)
                    
                    # Update our flywheel trackers
                    self._last_timestamps[ssrc] = timestamp + len(samples)
                    self._last_push_times[ssrc] = now
                    
                    if self._flywheel_counts.get(ssrc, 0) > 0:
                        logger.info(f"SSRC {ssrc} backend source recovered. Resuming real samples.")
                        self._flywheel_counts[ssrc] = 0
                            
                else:
                    # No data available. Backend source might have dropped.
                    # We zero-fill (flywheel) the stream so the client doesn't disconnect.
                    # ONLY flywheel if it's been more than the chunk duration since our last push
                    # to prevent pumping zeros at an unlimited rate!
                    last_push = self._last_push_times.get(ssrc, now)
                    if now - last_push >= 0.2:
                        last_ts = self._last_timestamps.get(ssrc)
                        if last_ts is not None:
                            streamer.stream_samples(zeros_chunk, last_ts)
                            self._last_timestamps[ssrc] = last_ts + chunk_size
                            self._last_push_times[ssrc] = now
                            
                            count = self._flywheel_counts.get(ssrc, 0)
                            if count == 0:
                                logger.warning(f"SSRC {ssrc} backend source dropped. Flywheeling zero-fill RTP.")
                            self._flywheel_counts[ssrc] = count + 1

            # Sleep so we don't spin CPU
            time.sleep(sleep_interval)
