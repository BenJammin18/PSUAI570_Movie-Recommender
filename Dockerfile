FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=poll

WORKDIR /app

COPY requirements.txt ./
COPY requirements-deep.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-deep.txt

COPY . .

EXPOSE 8501
ENV PYTHONPATH=/app

CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
