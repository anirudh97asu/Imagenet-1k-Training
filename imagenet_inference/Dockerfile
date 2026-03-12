FROM public.ecr.aws/docker/library/python:3.12-slim

# Lambda Web Adapter extension
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.8.4 \
    /lambda-adapter /opt/extensions/lambda-adapter

ENV PORT=8080
WORKDIR /var/task

COPY requirements.txt ./

# Force pip to use PyPI (fixes "No matching distribution found for Pillow")
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir --index-url https://pypi.org/simple -r requirements.txt

# Copy your app folder (contains main.py + model + labels)
COPY app/ ./app/

EXPOSE 8080

# Start uvicorn; Web Adapter will proxy Lambda events to this HTTP server
CMD exec uvicorn --host 0.0.0.0 --port $PORT app.main:app