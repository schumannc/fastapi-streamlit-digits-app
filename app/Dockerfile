FROM python:3.10

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY . /app

WORKDIR /app/app

CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]