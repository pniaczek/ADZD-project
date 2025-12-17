from config.loader import load_config

cfg = load_config()
print(cfg.keys())
print(cfg["spark"])
