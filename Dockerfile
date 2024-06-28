FROM python:3.9

WORKDIR .

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/

CMD ["streamlit", "run", "--server.port", "8502", "app/app.py"]
