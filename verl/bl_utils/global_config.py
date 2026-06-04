# Stub: global config (no-op for open-source)
_config = {}

def get_config_obj():
    return {}

def get_config(key=None, default=None):
    return default

def set_config(key, value):
    _config[key] = value

def update_full_config(config):
    _config.update(config)
