#!/usr/bin/env python3
"""
Command Line Interface for Phase Engine.
"""

import sys
import time
import logging
import argparse
from pathlib import Path

from .config_loader import load_config, get_engine_kwargs
from .engine import PhaseEngine
from .virtual_channel import VirtualChannelManager
from .control.server import ControlServer
from .control.loop import EgressLoop

def run_daemon(args):
    """Run the phase engine daemon."""
    config = load_config(args.config)
    engine_kwargs = get_engine_kwargs(config)
    
    engine_cfg = config.get('engine', {})
    status_addr = engine_cfg.get('status_address', 'phase-engine-status.local')
    control_port = engine_cfg.get('control_port', 5006)
    
    logger = logging.getLogger(__name__)
    logger.info("Initializing Phase Engine Daemon...")
    
    engine = PhaseEngine(**engine_kwargs)
    
    vcm = VirtualChannelManager(engine)
    server = ControlServer(engine, vcm, status_address=status_addr, control_port=control_port)
    loop = EgressLoop(engine, vcm)
    
    try:
        # 1. Connect to sources
        engine.connect()
        
        # 2. Calibrate
        logger.info("Running initial calibration...")
        engine.calibrate(duration_sec=3.0)
        
        # 3. Start engine capture (pulls data from physical radiods)
        engine.start()
        
        # 4. Start control server (listens for hf-timestd requests)
        server.start()
        
        # 5. Start egress loop (pushes combined data back out)
        loop.start()
        
        logger.info("Phase Engine Daemon running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.exception(f"Fatal error in daemon: {e}")
        sys.exit(1)
    finally:
        server.stop()
        loop.stop()
        engine.stop()
        engine.disconnect()

def main():
    """Main entry point for phase-engine command."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(
        description='Phase Engine - Coherent HF Phased Array Middleware',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Daemon command
    daemon_parser = subparsers.add_parser('daemon', help='Run phase engine daemon')
    daemon_parser.add_argument('--config', '-c', required=True, help='Configuration file')
    daemon_parser.add_argument('--debug', '-d', action='store_true', help='Enable debug logging')
    
    # Calibrate command
    cal_parser = subparsers.add_parser('calibrate', help='Run calibration routine')
    cal_parser.add_argument('--config', '-c', required=True, help='Configuration file')
    cal_parser.add_argument('--frequency', '-f', type=float, default=10e6,
                           help='Calibration frequency in Hz (default: 10 MHz)')
    cal_parser.add_argument('--duration', '-t', type=float, default=1.0,
                           help='Calibration duration in seconds (default: 1.0)')
    cal_parser.add_argument('--output', '-o', help='Output file for calibration results')
    cal_parser.add_argument('--debug', '-d', action='store_true', help='Enable debug logging')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show engine status')
    status_parser.add_argument('--config', '-c', required=True, help='Configuration file')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test antenna connectivity')
    test_parser.add_argument('--config', '-c', required=True, help='Configuration file')
    test_parser.add_argument('--debug', '-d', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    if hasattr(args, 'debug') and args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.command == 'daemon':
        run_daemon(args)
        
    elif args.command == 'calibrate':
        print("Calibration routine standalone not yet fully wired to cli")
        sys.exit(0)
        
    elif args.command == 'status':
        print("Status command not yet implemented")
        sys.exit(0)
        
    elif args.command == 'test':
        print("Test command not yet implemented")
        sys.exit(0)

if __name__ == '__main__':
    main()
