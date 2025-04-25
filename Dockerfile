FROM python:3.7

ENV PORT 8080
ENV HOSTDIR 0.0.0.0

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["python", "app.py"]
