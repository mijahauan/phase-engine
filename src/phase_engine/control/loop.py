import threading
import time
import logging

logger = logging.getLogger(__name__)


class EgressLoop:
    """Continuously pulls combined samples from the engine and pushes to RTP streamers."""

    def __init__(self, engine, channel_manager):
        self.engine = engine
        self.channel_manager = channel_manager
        self._running = False
        self._thread = None

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

        while self._running:
            active_channels = self.channel_manager.get_active_broadcasts()

            if not active_channels or not self.engine.is_running:
                time.sleep(0.1)
                continue

            # For each active virtual channel, pull combined samples and stream
            # Note: We limit the max_samples here to keep latency low.
            for ssrc, vchan in active_channels.items():
                result = self.engine.get_combined_samples(vchan, max_samples=4800)

                if result is not None:
                    samples, timestamp = result
                    if len(samples) > 0:
                        streamer = self.channel_manager.get_streamer(ssrc)
                        if streamer:
                            streamer.stream_samples(samples, timestamp)

            # Sleep ~half the nominal chunk duration so we wake up when new samples
            # are ready. At 24kHz / 4800-sample chunks that's ~200ms per chunk;
            # sleeping 100ms keeps latency low while not burning CPU on empty polls.
            time.sleep(0.1)
