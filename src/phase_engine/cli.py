#!/usr/bin/env python3
"""
Command Line Interface for Phase Engine.
"""

import sys
import time
import logging
import argparse
import json
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

    engine_cfg = config.get("engine", {})
    status_addr = engine_cfg.get("status_address", "phase-engine-status.local")
    control_port = engine_cfg.get("control_port", 5006)

    logger = logging.getLogger(__name__)
    logger.info("Initializing Phase Engine Daemon...")

    engine = PhaseEngine(**engine_kwargs)

    vcm = VirtualChannelManager(engine)
    server = ControlServer(engine, vcm, status_address=status_addr, control_port=control_port)
    loop = EgressLoop(engine, vcm)

    try:
        # 1. Connect to sources
        engine.connect()
        
        # Note: Global startup calibration is theoretically invalid due to NCO ambiguity.
        # phase-engine now relies on continuous, per-channel alignment in the data plane.
        
        # 2. Start engine capture (pulls data from physical radiods)
        engine.start()

        # 3. Start control server (listens for hf-timestd requests)
        server.start()

        # 4. Start egress loop (pushes combined data back out)
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
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser(
        description="Phase Engine - Coherent HF Phased Array Middleware",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Daemon command
    daemon_parser = subparsers.add_parser("daemon", help="Run phase engine daemon")
    daemon_parser.add_argument("--config", "-c", required=True, help="Configuration file")
    daemon_parser.add_argument("--debug", "-d", action="store_true", help="Enable debug logging")

    # Calibrate command
    cal_parser = subparsers.add_parser("calibrate", help="Run calibration routine")
    cal_parser.add_argument("--config", "-c", required=True, help="Configuration file")
    cal_parser.add_argument(
        "--frequency",
        "-f",
        type=float,
        default=10e6,
        help="Calibration frequency in Hz (default: 10 MHz)",
    )
    cal_parser.add_argument(
        "--duration",
        "-t",
        type=float,
        default=1.0,
        help="Calibration duration in seconds (default: 1.0)",
    )
    cal_parser.add_argument("--output", "-o", help="Output file for calibration results")
    cal_parser.add_argument("--debug", "-d", action="store_true", help="Enable debug logging")

    # Status command
    status_parser = subparsers.add_parser("status", help="Show engine status")
    status_parser.add_argument("--config", "-c", required=True, help="Configuration file")

    # Test command
    test_parser = subparsers.add_parser("test", help="Test antenna connectivity")
    test_parser.add_argument("--config", "-c", required=True, help="Configuration file")
    test_parser.add_argument("--debug", "-d", action="store_true", help="Enable debug logging")

    # Plot command
    plot_parser = subparsers.add_parser("plot", help="Plot theoretical array radiation pattern")
    plot_parser.add_argument(
        "--config", "-c", required=True, help="Configuration file (for array geometry)"
    )
    plot_parser.add_argument("--freq", "-f", type=float, required=True, help="Frequency in Hz")
    plot_parser.add_argument(
        "--method",
        "-m",
        default="focus",
        choices=["focus", "mvdr", "mrc", "egc", "omni"],
        help="Combining method",
    )
    plot_parser.add_argument(
        "--target",
        "-t",
        action="append",
        help='Target coordinate "Name,Lat,Lon" (can be specified multiple times)',
    )
    plot_parser.add_argument(
        "--null",
        "-n",
        action="append",
        help='Null coordinate "Name,Lat,Lon" (can be specified multiple times)',
    )
    plot_parser.add_argument("--out", "-o", default="pattern.png", help="Output PNG path")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if hasattr(args, "debug") and args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.command == "daemon":
        run_daemon(args)

    elif args.command == "calibrate":
        print("Calibration routine standalone not yet fully wired to cli")
        sys.exit(0)

    elif args.command == "status":
        print("Status command not yet implemented")
        sys.exit(0)

    elif args.command == "test":
        print("Test command not yet implemented")
        sys.exit(0)

    elif args.command == "plot":
        try:
            from phase_engine.dsp.plotter import PatternPlotter
            from phase_engine.dsp.array_geometry import AntennaArray
            from phase_engine.config_loader import load_config
        except ImportError as e:
            logging.error(
                f"Failed to import plotter dependencies. Did you install with 'pip install .[plot]'? Error: {e}"
            )
            sys.exit(1)

        config = load_config(args.config)
        qth_lat = config.get("qth", {}).get("latitude", 0.0)
        qth_lon = config.get("qth", {}).get("longitude", 0.0)

        positions = {}
        ref_name = None
        for ant in config.get("antennas", []):
            positions[ant["name"]] = tuple(ant.get("position", [0, 0, 0]))
            if ant.get("role") == "reference" or ref_name is None:
                ref_name = ant["name"]

        array = AntennaArray(ref_name, positions)
        plotter = PatternPlotter(array, qth_lat, qth_lon)

        def parse_coord(coord_str):
            parts = coord_str.split(",")
            if len(parts) != 3:
                raise ValueError(
                    f"Invalid coordinate format '{coord_str}'. Expected 'Name,Lat,Lon'"
                )
            return {"name": parts[0], "lat": float(parts[1]), "lon": float(parts[2])}

        targets = [parse_coord(t) for t in (args.target or [])]
        nulls = [parse_coord(n) for n in (args.null or [])]

        plotter.generate_plot(args.freq, args.method, targets, nulls, args.out)
        print(f"Plot saved to {args.out}")


if __name__ == "__main__":
    main()
