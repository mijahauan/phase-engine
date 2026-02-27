"""
PhaseEngineStream - API-compatible replacement for RadiodStream.

Provides the same interface as ka9q-python's RadiodStream, delivering
phase-combined samples from the array.

Usage:
    from phase_engine.client import PhaseEngineControl, PhaseEngineStream
    
    control = PhaseEngineControl("phase-engine-status.local")
    channel = control.create_channel(frequency_hz=10e6, preset="iq")
    
    def on_samples(samples, quality):
        # Process combined samples
        pass
    
    stream = PhaseEngineStream(channel, on_samples=on_samples)
    stream.start()
    # ... later ...
    stream.stop()
"""

from dataclasses import dataclass
from typing import Callable, Optional, Any
import numpy as np
import threading
import time
import logging

from .control import ChannelInfo

logger = logging.getLogger(__name__)


@dataclass
class StreamQuality:
    """Quality metrics for the stream."""
    completeness_pct: float = 100.0
    gaps: int = 0
    total_samples: int = 0
    combined_snr_db: Optional[float] = None
    beam_gain_db: Optional[float] = None
    null_depth_db: Optional[float] = None


# Callback type: (samples: np.ndarray, quality: StreamQuality) -> None
SampleCallback = Callable[[np.ndarray, StreamQuality], None]


class PhaseEngineStream:
    """
    Stream interface for receiving phase-combined samples.
    
    API-compatible with ka9q-python's RadiodStream. Delivers coherently
    combined samples from the phased array.
    
    Args:
        channel: ChannelInfo from PhaseEngineControl.create_channel()
        on_samples: Callback for sample delivery
        buffer_size: Internal buffer size in samples
    """
    
    def __init__(
        self,
        channel: ChannelInfo,
        on_samples: Optional[SampleCallback] = None,
        buffer_size: int = 24000,
    ):
        """
        Initialize the stream.
        
        Args:
            channel: Channel info from control.create_channel()
            on_samples: Callback invoked with (samples, quality)
            buffer_size: Buffer size in samples
        """
        self.channel = channel
        self.on_samples = on_samples
        self.buffer_size = buffer_size
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Sample buffer (ring buffer for combined samples)
        self._buffer: Optional[np.ndarray] = None
        self._buffer_write_pos = 0
        self._buffer_read_pos = 0
        
        # Quality tracking
        self._total_samples = 0
        self._gaps = 0
        
        # Reference to engine's combined sample source
        self._sample_source = None
        
        logger.debug(f"PhaseEngineStream created for {channel.frequency/1e6:.3f} MHz")
        
    def start(self) -> None:
        """Start receiving samples."""
        if self._running:
            return
            
        self._running = True
        self._buffer = np.zeros(self.buffer_size, dtype=np.complex64)
        self._buffer_write_pos = 0
        self._buffer_read_pos = 0
        self._total_samples = 0
        self._gaps = 0
        
        # Start receive thread
        self._thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._thread.start()
        
        logger.info(f"PhaseEngineStream started for {self.channel.frequency/1e6:.3f} MHz")
        
    def stop(self) -> StreamQuality:
        """
        Stop receiving samples.
        
        Returns:
            StreamQuality with final statistics
        """
        if not self._running:
            return StreamQuality()
            
        self._running = False
        
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
            
        quality = StreamQuality(
            completeness_pct=100.0 if self._gaps == 0 else 
                max(0, 100.0 * (1 - self._gaps / max(1, self._total_samples / 1000))),
            gaps=self._gaps,
            total_samples=self._total_samples,
        )
        
        logger.info(f"PhaseEngineStream stopped: {self._total_samples} samples, "
                   f"{self._gaps} gaps")
                   
        return quality
        
    def _receive_loop(self) -> None:
        """Main receive loop - pulls samples from engine and delivers to callback."""
        while self._running:
            try:
                # Get samples from the engine's combined output
                samples = self._get_combined_samples()
                
                if samples is not None and len(samples) > 0:
                    self._total_samples += len(samples)
                    
                    # Deliver to callback
                    if self.on_samples:
                        quality = StreamQuality(
                            completeness_pct=100.0,
                            gaps=0,
                            total_samples=len(samples),
                        )
                        self.on_samples(samples, quality)
                else:
                    # No samples available, short sleep
                    time.sleep(0.01)
                    
            except Exception as e:
                logger.error(f"Error in receive loop: {e}")
                self._gaps += 1
                time.sleep(0.1)
                
    def _get_combined_samples(self) -> Optional[np.ndarray]:
        """
        Get combined samples from the engine.
        
        In a full implementation, this would:
        1. Subscribe to the engine's RTP multicast for this channel
        2. Receive and decode RTP packets
        3. Return the sample array
        
        For now, this is a placeholder that would be connected to the engine.
        """
        # This will be connected to the actual engine output
        if self._sample_source is not None:
            return self._sample_source()
            
        # Placeholder: sleep to simulate waiting for samples
        time.sleep(0.05)
        return None
        
    def read(self, num_samples: int, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        Read samples synchronously.
        
        Args:
            num_samples: Number of samples to read
            timeout: Timeout in seconds
            
        Returns:
            Sample array, or None if timeout
        """
        samples_collected = []
        samples_needed = num_samples
        start_time = time.time()
        
        while samples_needed > 0:
            if time.time() - start_time > timeout:
                break
                
            samples = self._get_combined_samples()
            if samples is not None and len(samples) > 0:
                samples_collected.append(samples)
                samples_needed -= len(samples)
            else:
                time.sleep(0.01)
                
        if not samples_collected:
            return None
            
        result = np.concatenate(samples_collected)
        if len(result) > num_samples:
            result = result[:num_samples]
            
        return result
        
    @property
    def is_running(self) -> bool:
        """Whether the stream is running."""
        return self._running
        
    @property
    def ssrc(self) -> int:
        """Channel SSRC."""
        return self.channel.ssrc
        
    @property
    def frequency_hz(self) -> float:
        """Channel frequency."""
        return self.channel.frequency
        
    def __repr__(self) -> str:
        status = "running" if self._running else "stopped"
        return f"PhaseEngineStream({self.channel.frequency/1e6:.3f} MHz, {status})"
