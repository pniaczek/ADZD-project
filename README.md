# Forex Spark MLOps — monorepo

Repo zawiera 3 komponenty:
1) `adzd-project/` — pipeline Spark + modele (naive / SARIMA / Prophet) + daily pipeline
2) `mcp-apache-spark-history-server/` — MCP server do Spark History Server
3) `spark-agent/` — agent LLM korzystający z MCP (analiza aplikacji Spark)

## Spis treści
- [Wymagania](#wymagania)
- [Security Group (EC2)](#security-group-ec2)
- [Setup end-to-end (jedna checklista)](#setup-end-to-end-jedna-checklista)
- [1. ADZD Project](#1-adzd-project)
- [2. Spark History Server](#2-spark-history-server)
- [3. MCP Apache Spark History Server](#3-mcp-apache-spark-history-server)
- [4. Spark Agent](#4-spark-agent)
- [Uruchomienie całości (kolejność)](#uruchomienie-całości-kolejność)
- [Logi i artefakty](#logi-i-artefakty)
- [Troubleshooting](#troubleshooting)

---

## Wymagania

- System: Linux (EC2), użytkownik `ec2-user`
- Python: 3.9+ (u Ciebie: 3.9.25)
- Java: 17 (Spark 3.4.2)
- Apache Spark: 3.4.2 (dystrybucja `spark-3.4.2-bin-hadoop3`)
- Spark event logs: `/opt/spark-events` (wymagane dla History Server)
- Porty (lokalnie / przez tunnel):
  - Spark History Server: `18080`
  - MCP server: `18888`
  - Ollama (jeśli używane lokalnie/na EC2): `11434`


## Security Group (EC2)

Poniższe reguły *inbound* są wystarczające do uruchomienia projektu na EC2 i dostępu z Twojego komputera.
Najbezpieczniej jest **nie wystawiać** interfejsów Spark/MCP publicznie, tylko korzystać z tunelowania SSH
(`ssh -N -L ...`). Wtedy publicznie potrzebujesz **tylko portu 22**.

### Minimalny, rekomendowany wariant (tylko SSH)
- `22/tcp` — SSH do EC2 (Source: **Twoje IP** `/32`)

### Jeśli chcesz wystawić UI przez przeglądarkę (niezalecane, ale dopuszczalne)
Ogranicz Source do **Twojego IP** `/32` (tak jak na screenie).
- `22/tcp` — SSH (Source: Twoje IP `/32`)
- `18080/tcp` — Spark History Server UI (Source: Twoje IP `/32`)
- `18888/tcp` — MCP server (Source: Twoje IP `/32`)

> Uwaga: jeśli używasz tuneli SSH:
> - Spark History Server: `ssh -N -L 18080:localhost:18080 ec2-user@<EC2_PUBLIC_IP>`
> - MCP server: `ssh -N -L 18888:localhost:18888 ec2-user@<EC2_PUBLIC_IP>`
> wtedy **nie musisz** otwierać portów `18080` i `18888` w Security Group (wystarczy `22`).


---

## Setup end-to-end

Poniższe kroki zakładają, że jesteś na EC2 i masz w `~` tarball `spark-3.4.2-bin-hadoop3.tgz`.

1) Spark + event logs:

```bash
cd ~
sudo mkdir -p /opt
sudo tar -xzf spark-3.4.2-bin-hadoop3.tgz -C /opt
sudo ln -sfn /opt/spark-3.4.2-bin-hadoop3 /opt/spark
sudo chown -R ec2-user:ec2-user /opt/spark-3.4.2-bin-hadoop3

sudo mkdir -p /opt/spark-events
sudo chown -R ec2-user:ec2-user /opt/spark-events

export SPARK_HOME=/opt/spark
export PATH="$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH"

mkdir -p "$SPARK_HOME/conf"
cat >> "$SPARK_HOME/conf/spark-defaults.conf" << 'EOC'
spark.eventLog.enabled true
spark.eventLog.dir file:/opt/spark-events
spark.history.fs.logDirectory file:/opt/spark-events
EOC

spark-submit --version
```

2) ADZD Project venv:

```bash
# Prerequisite: Python 3.9.25
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
export PYTHONPATH="$(pwd)"
```

3) MCP server venv:

```bash
sudo dnf install -y python3.12 python3.12-venv python3.12-pip
cd ~
python3.12 -m venv spark-mcp
source spark-mcp/bin/activate
pip install mcp-apache-spark-history-server
```

4) Spark Agent venv:

```bash
cd ~/spark-agent
python3.12 -m venv venv
source venv/bin/activate
pip install httpx mcp ollama
```

---

## 1. ADZD Project

### 1.1 Konfiguracja

Edytuj plik:
- `adzd-project/config/config.yaml`

W szczególności ustaw:
- `alpha_vantage.api_key`
- `alpha_vantage.pairs`
- parametry:
  - `models.backtest.daily.test_size`
  - `models.backtest.daily.min_train_size`
  - `models.predict.daily.horizon_days`

### 1.2 Uruchomienie daily pipeline

```bash
cd ~/project_3/ADZD-project/adzd-project
source .venv/bin/activate
export SPARK_HOME=/opt/spark
export PATH="$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH"
export PYTHONPATH="$(pwd)"
bash scripts/run_daily_pipeline.sh
```

---

## 2. Spark History Server

History Server czyta event logi z `/opt/spark-events`.

### 2.1 Start (na EC2)

```bash
export SPARK_HOME=/opt/spark
export PATH="$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH"
$SPARK_HOME/sbin/start-history-server.sh
```

Sprawdzenie:

```bash
curl -I http://localhost:18080 || true
```

### 2.2 SSH tunneling (na komputerze lokalnym)

```bash
ssh -i /path/to/key.pem -N -L 18080:localhost:18080 ec2-user@<EC2_PUBLIC_IP>
```

Otwórz w przeglądarce:
- `http://localhost:18080`

---

## 3. MCP Apache Spark History Server

Ten komponent uruchamia MCP server, który udostępnia narzędzia do odpytywania Spark History Server.

### 3.1 Start MCP server (na EC2)

```bash
cd ~
source spark-mcp/bin/activate
export SPARK_HISTORY_FS_LOG_DIRECTORY=file:///opt/spark-events
python -m spark_history_mcp.core.main
```

MCP endpoint:
- `http://localhost:18888/mcp/`

### 3.2 (Opcjonalnie) tunneling MCP na komputer lokalny

Jeżeli agent działa lokalnie (nie na EC2), zrób tunnel:

```bash
ssh -i /path/to/key.pem -N -L 18888:localhost:18888 ec2-user@<EC2_PUBLIC_IP>
```

---

## 4. Spark Agent

Agent komunikuje się z MCP serverem i wykonuje analizę (LLM).

Wymaga:
- MCP server działający na `localhost:18888`
- (opcjonalnie) Ollama na `localhost:11434` jeśli agent używa lokalnego LLM

### 4.1 Uruchomienie agenta (na EC2)

```bash
cd ~/project/spark-agent
source venv/bin/activate
python agent.py
```

---

## Uruchomienie całości (kolejność)

1) Uruchom daily pipeline:

2) Uruchom Spark History Server:

3) (Lokalnie) tunnel do History Server:

4) Uruchom MCP server:

5) Uruchom agenta:

---

## Logi i artefakty

### Spark event logs (History Server)
- `/opt/spark-events`

### ADZD pipeline logs
- `adzd-project/logs/pipeline/` — logi uruchomień daily pipeline
- `adzd-project/logs/` — logi modułów (naive/sarima/prophet/visualize itd.)

### Dane i artefakty
- `adzd-project/data/` — raw/features/metrics/predictions
- `adzd-project/reports/` — wykresy i podsumowania

---

### Agent nie widzi MCP
- na EC2: MCP_URL powinien być `http://localhost:18888/mcp/`
- lokalnie: użyj tunelu i też `http://localhost:18888/mcp/`
