import toml
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate the TOML config file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    try:
        with open(path, "r") as f:
            config = toml.load(f)
        return config
    except toml.TomlDecodeError as e:
        raise ValueError(f"Failed to parse config file: {e}")


def get_engine_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract arguments for PhaseEngine initialization."""
    kwargs = {}

    # QTH (Location)
    if "qth" in config:
        kwargs["qth_latitude"] = config["qth"].get("latitude", 0.0)
        kwargs["qth_longitude"] = config["qth"].get("longitude", 0.0)
    else:
        logger.warning("No [qth] section in config, using 0,0")
        kwargs["qth_latitude"] = 0.0
        kwargs["qth_longitude"] = 0.0

    # General engine params
    engine_cfg = config.get("engine", {})
    kwargs["sample_rate"] = engine_cfg.get("sample_rate", 12000)

    # Sources / Antennas
    from .engine import SourceConfig

    sources = []
    reference_source = None
    for ant in config.get("antennas", []):
        pos = ant.get("position", [0.0, 0.0, 0.0])
        cfg = SourceConfig(
            name=ant.get("name", "unknown"),
            status_address=ant.get("status_address", ""),
            position=tuple(pos),
            enabled=ant.get("enabled", True),
        )
        sources.append(cfg)

        if ant.get("role") == "reference" or reference_source is None:
            reference_source = cfg.name

    kwargs["sources"] = sources
    kwargs["reference_source"] = reference_source

    return kwargs
