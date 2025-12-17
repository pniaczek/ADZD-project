import json
from pathlib import Path
from datetime import datetime, UTC

from utils.logger import setup_logger

PAIR = "EUR_USD"
METRIC = "mae"

MODELS_ROOT = Path("models")
REGISTRY_ROOT = MODELS_ROOT / "registry" / PAIR


def load_metric(path: Path) -> float:
    data = json.loads(path.read_text())
    return float(data["metrics"][METRIC])


def main():
    logger = setup_logger(
        name="compare.select_model",
        log_dir=Path("logs/compare") / PAIR,
    )

    logger.info(f"Selecting best model for {PAIR} using metric={METRIC}")

    results = []

    # ======================
    # NAIVE
    # ======================
    naive_meta = Path("models/naive") / PAIR / "latest" / "backtest.json"
    if not naive_meta.exists():
        raise FileNotFoundError(f"Missing NAIVE backtest: {naive_meta}")

    naive_score = load_metric(naive_meta)
    results.append({
        "model": "NAIVE",
        "metric": naive_score,
        "path": naive_meta.parent,
    })

    # ======================
    # SARIMA
    # ======================
    sarima_meta = Path("models/sarima") / PAIR / "latest" / "backtest.json"
    if not sarima_meta.exists():
        raise FileNotFoundError(f"Missing SARIMA backtest: {sarima_meta}")

    sarima_score = load_metric(sarima_meta)
    results.append({
        "model": "SARIMA",
        "metric": sarima_score,
        "path": sarima_meta.parent,
    })

    # ======================
    # SELECT BEST
    # ======================
    best = min(results, key=lambda x: x["metric"])

    active = {
        "pair": PAIR,
        "selected_model": best["model"],
        "metric": METRIC,
        "metric_value": best["metric"],
        "model_path": str(best["path"]),
        "selected_at": datetime.now(UTC).isoformat(),
    }

    # ======================
    # RUNTIME CONTRACT (OBOWIĄZKOWE)
    # ======================
    MODELS_ROOT.mkdir(exist_ok=True)
    active_runtime_path = MODELS_ROOT / "active_model.json"
    active_runtime_path.write_text(json.dumps(active, indent=2))

    # ======================
    # REGISTRY (HISTORY / AUDIT)
    # ======================
    REGISTRY_ROOT.mkdir(parents=True, exist_ok=True)
    (REGISTRY_ROOT / "active.json").write_text(json.dumps(active, indent=2))

    logger.info("Model selection completed")
    logger.info(json.dumps(active, indent=2))


if __name__ == "__main__":
    main()
