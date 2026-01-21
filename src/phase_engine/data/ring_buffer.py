"""
Ring Buffer - Circular buffer for aligning multiple RTP streams.

Since ADCs start at random times (even with shared GPSDO), streams must be
aligned by timestamp. This buffer holds samples until all streams have
data for the same time window.
"""

import numpy as np
import logging
from dataclasses import dataclass
from typing import Optional, Dict
from threading import Lock

logger = logging.getLogger(__name__)


@dataclass
class StreamState:
    """State tracking for a single input stream."""
    name: str
    buffer: np.ndarray  # Complex64 ring buffer
    write_pos: int = 0
    read_pos: int = 0
    timestamp_start: int = 0  # First sample timestamp in buffer
    samples_written: int = 0
    samples_read: int = 0
    overflow_count: int = 0
    underflow_count: int = 0


class AlignedRingBuffer:
    """
    Multi-stream ring buffer with timestamp-based alignment.
    
    Maintains separate buffers for each input stream and provides
    aligned output when all streams have data for the same time window.
    """
    
    def __init__(
        self,
        stream_names: list[str],
        buffer_seconds: float = 2.0,
        sample_rate: int = 24000,
        dtype: np.dtype = np.complex64
    ):
        """
        Initialize aligned ring buffer.
        
        Args:
            stream_names: Names of input streams (e.g., ["north", "south"])
            buffer_seconds: Buffer duration in seconds
            sample_rate: Sample rate in Hz
            dtype: NumPy dtype for samples
        """
        self.sample_rate = sample_rate
        self.buffer_size = int(buffer_seconds * sample_rate)
        self.dtype = dtype
        self._lock = Lock()
        
        # Create per-stream state
        self._streams: Dict[str, StreamState] = {}
        for name in stream_names:
            self._streams[name] = StreamState(
                name=name,
                buffer=np.zeros(self.buffer_size, dtype=dtype)
            )
        
        # Alignment state
        self._aligned_timestamp: Optional[int] = None
        self._alignment_established = False
        
        logger.info(f"AlignedRingBuffer: {len(stream_names)} streams, "
                   f"{buffer_seconds}s @ {sample_rate} Hz")
    
    def write(self, stream_name: str, samples: np.ndarray, timestamp: int) -> int:
        """
        Write samples to a stream's buffer.
        
        Args:
            stream_name: Name of the stream
            samples: Complex samples to write
            timestamp: Timestamp of first sample (in sample units)
            
        Returns:
            Number of samples actually written
        """
        if stream_name not in self._streams:
            logger.error(f"Unknown stream: {stream_name}")
            return 0
        
        with self._lock:
            state = self._streams[stream_name]
            n_samples = len(samples)
            
            # Handle buffer overflow
            if n_samples > self.buffer_size:
                logger.warning(f"{stream_name}: Input too large, truncating")
                samples = samples[-self.buffer_size:]
                n_samples = self.buffer_size
                state.overflow_count += 1
            
            # Update timestamp tracking
            if state.samples_written == 0:
                state.timestamp_start = timestamp
            
            # Write to ring buffer (may wrap)
            end_pos = state.write_pos + n_samples
            
            if end_pos <= self.buffer_size:
                # No wrap
                state.buffer[state.write_pos:end_pos] = samples
            else:
                # Wrap around
                first_part = self.buffer_size - state.write_pos
                state.buffer[state.write_pos:] = samples[:first_part]
                state.buffer[:end_pos - self.buffer_size] = samples[first_part:]
            
            state.write_pos = end_pos % self.buffer_size
            state.samples_written += n_samples
            
            # Check if we can establish alignment
            if not self._alignment_established:
                self._try_establish_alignment()
            
            return n_samples
    
    def _try_establish_alignment(self) -> None:
        """Try to establish timestamp alignment across all streams."""
        # Need data from all streams
        if not all(s.samples_written > 0 for s in self._streams.values()):
            return
        
        # Find the latest start timestamp (all streams must have this)
        latest_start = max(s.timestamp_start for s in self._streams.values())
        
        # Check all streams have enough data past this point
        min_available = float('inf')
        for state in self._streams.values():
            # How many samples does this stream have after latest_start?
            offset = latest_start - state.timestamp_start
            if offset < 0:
                return  # Stream started after latest_start, wait for more data
            available = state.samples_written - offset
            min_available = min(min_available, available)
        
        if min_available > self.sample_rate // 10:  # At least 100ms overlap
            self._aligned_timestamp = latest_start
            self._alignment_established = True
            logger.info(f"Alignment established at timestamp {latest_start}")
    
    def read_aligned(self, n_samples: int) -> Optional[Dict[str, np.ndarray]]:
        """
        Read aligned samples from all streams.
        
        Args:
            n_samples: Number of samples to read from each stream
            
        Returns:
            Dict mapping stream names to sample arrays, or None if not enough data
        """
        if not self._alignment_established:
            return None
        
        with self._lock:
            # Check all streams have enough data
            for state in self._streams.values():
                available = state.samples_written - state.samples_read
                if available < n_samples:
                    return None
            
            # Read from all streams
            result = {}
            for name, state in self._streams.items():
                # Calculate read position in ring buffer
                read_start = state.read_pos
                read_end = read_start + n_samples
                
                if read_end <= self.buffer_size:
                    # No wrap
                    result[name] = state.buffer[read_start:read_end].copy()
                else:
                    # Wrap around
                    first_part = self.buffer_size - read_start
                    result[name] = np.concatenate([
                        state.buffer[read_start:],
                        state.buffer[:read_end - self.buffer_size]
                    ])
                
                state.read_pos = read_end % self.buffer_size
                state.samples_read += n_samples
            
            self._aligned_timestamp += n_samples
            return result
    
    def get_alignment_status(self) -> Dict:
        """Get current alignment status."""
        with self._lock:
            return {
                'aligned': self._alignment_established,
                'timestamp': self._aligned_timestamp,
                'streams': {
                    name: {
                        'written': state.samples_written,
                        'read': state.samples_read,
                        'available': state.samples_written - state.samples_read,
                        'overflows': state.overflow_count,
                        'underflows': state.underflow_count,
                    }
                    for name, state in self._streams.items()
                }
            }
    
    def reset(self) -> None:
        """Reset all buffers and alignment state."""
        with self._lock:
            for state in self._streams.values():
                state.buffer.fill(0)
                state.write_pos = 0
                state.read_pos = 0
                state.timestamp_start = 0
                state.samples_written = 0
                state.samples_read = 0
            
            self._aligned_timestamp = None
            self._alignment_established = False
            logger.info("Ring buffer reset")
