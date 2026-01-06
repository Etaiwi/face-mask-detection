FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV PORT=8080

WORKDIR /app

# Needed for opencv-python-headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN echo "Models dir:" && ls -la /app/models || true

EXPOSE 8080

CMD ["bash", "-c", "streamlit run src/app.py --server.port=${PORT} --server.address=0.0.0.0"]
