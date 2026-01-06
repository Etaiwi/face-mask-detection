FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV PORT=7860

# System deps (opencv headless sometimes needs these)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install python deps first (better caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . /app

# HF Spaces expects port 7860
EXPOSE 7860

# Streamlit must bind to 0.0.0.0 and port 7860
CMD ["streamlit", "run", "src/app.py", "--server.address=0.0.0.0", "--server.port=${PORT}"]
