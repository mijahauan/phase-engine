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
            active_broadcasts = self.channel_manager.get_active_broadcasts()
            
            if not active_broadcasts or not self.engine.is_running:
                time.sleep(0.1)
                continue
                
            # For each active virtual channel, pull combined samples and stream
            # Note: We limit the max_samples here to keep latency low.
            for ssrc, broadcast in active_broadcasts.items():
                samples = self.engine.get_combined_samples(broadcast, max_samples=4800)
                
                if samples is not None and len(samples) > 0:
                    streamer = self.channel_manager.get_streamer(ssrc)
                    if streamer:
                        streamer.stream_samples(samples)
                        
            # Sleep briefly to yield CPU, but keep it tight enough to not overflow ring buffers
            time.sleep(0.01)
