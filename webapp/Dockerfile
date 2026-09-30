# Built from the staged folder (python webapp/deploy.py stage): the site's files plus coaching_html.py, nothing else.
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY *.py ./
COPY icons ./icons
RUN useradd --create-home site && chown -R site /app
USER site
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 60 "app:create_app()"
