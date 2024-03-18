FROM python:3.10

WORKDIR /app

COPY . .
# COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

# EXPOSE 8000

# CMD ["python", "run.py"]
ENTRYPOINT ["python", "run.py"]