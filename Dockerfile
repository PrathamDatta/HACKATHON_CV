FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 OMP_NUM_THREADS=2 STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 && rm -rf /var/lib/apt/lists/*
# Hugging Face Spaces runs containers as uid 1000; a normal user also keeps Streamlit's cache dirs writable
RUN useradd -m -u 1000 user
WORKDIR /home/user/app
# CPU-only torch keeps the image small (about 1 GB instead of about 4 GB)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY --chown=user . .
USER user
ENV HOME=/home/user
# The app needs only weights/ and samples/ (both committed); the NYU dataset is NOT required to serve it.
EXPOSE 8501
# CORS/XSRF off: required for file uploads when the app is shown inside the Spaces iframe
CMD ["sh", "-c", "streamlit run app.py --server.port ${PORT:-8501} --server.address 0.0.0.0 --server.enableCORS=false --server.enableXsrfProtection=false"]
