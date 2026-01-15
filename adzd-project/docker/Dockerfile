FROM apache/spark:3.4.2

USER root

# system tools + python
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# python libs required by Spark jobs
RUN pip3 install --no-cache-dir \
    requests \
    pyyaml \
    pandas

# sanity check (ważne!)
RUN python3 -c "import requests, yaml, pandas; print('Python deps OK')"

USER spark
