# Anthias screen check (local add-on). Small on purpose: Debian's grim plus onnxruntime, numpy and Pillow.
FROM python:3.13-slim-trixie
RUN sed -i "s|http://|https://|g" /etc/apt/sources.list.d/debian.sources && apt-get update && apt-get install -y --no-install-recommends grim && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir onnxruntime==1.23.2 numpy==2.3.5 pillow==12.0.0
WORKDIR /app
# No model in the image: it is fetched from Hugging Face on first start (see README).
COPY screen_check.py settings.json ./
CMD ["python", "-u", "screen_check.py"]
