"""Configuration loader for Solana Shares Trader v2."""

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

_CONFIG_DIR = Path(__file__).parent
_DEFAULT_CONFIG = _CONFIG_DIR / "settings.yaml"
_TRADING_CONFIG = _CONFIG_DIR / "trading.yaml"
_config_cache = None
_trading_cache = None


def load_config(config_path: str = None) -> dict:
    """Load and merge configuration from YAML + environment variables."""
    global _config_cache
    if _config_cache is not None and config_path is None:
        return _config_cache

    path = Path(config_path) if config_path else _DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Override from env
    env_overrides = {
        "PRIVATE_KEY": ("_secrets", "private_key"),
        "PROXY_WALLET": ("_secrets", "proxy_wallet"),
        "POLYMARKET_API_KEY": ("_secrets", "poly_api_key"),
        "POLYMARKET_SECRET": ("_secrets", "poly_secret"),
        "POLYMARKET_PASSPHRASE": ("_secrets", "poly_passphrase"),
        "CLICKHOUSE_HOST": ("infrastructure", "clickhouse", "host"),
        "CLICKHOUSE_PASSWORD": ("infrastructure", "clickhouse", "password"),
        "SOLANA_RPC_URL": ("infrastructure", "solana", "rpc_url"),
        "MODE": ("mode",),
        "DRY_RUN": ("dry_run",),
    }

    for env_key, path_keys in env_overrides.items():
        val = os.environ.get(env_key)
        if val is not None:
            obj = cfg
            for k in path_keys[:-1]:
                obj = obj.setdefault(k, {})
            final_key = path_keys[-1]
            if val.lower() in ("true", "false"):
                val = val.lower() == "true"
            obj[final_key] = val

    _config_cache = cfg
    return cfg


def load_trading_config() -> dict:
    """Load trading configuration from trading.yaml."""
    global _trading_cache
    if _trading_cache is not None:
        return _trading_cache

    if not _TRADING_CONFIG.exists():
        _trading_cache = {}
        return _trading_cache

    with open(_TRADING_CONFIG, "r", encoding="utf-8") as f:
        _trading_cache = yaml.safe_load(f) or {}
    return _trading_cache


def get(key: str, default=None):
    """Get a dot-separated config key. E.g. get('models.primary.type')."""
    cfg = load_config()
    keys = key.split(".")
    obj = cfg
    for k in keys:
        if isinstance(obj, dict):
            obj = obj.get(k)
        else:
            return default
        if obj is None:
            return default
    return obj


def get_trading(key: str, default=None):
    """Get a dot-separated key from trading.yaml. E.g. get_trading('entry.min_confidence')."""
    cfg = load_trading_config()
    keys = key.split(".")
    obj = cfg
    for k in keys:
        if isinstance(obj, dict):
            obj = obj.get(k)
        else:
            return default
        if obj is None:
            return default
    return obj


# Auto-load on import
config = load_config()
trading_config = load_trading_config()
