FROM python:3.10.0-slim

WORKDIR /app


#COPY [ "model2.bin", "model.bin" ]


COPY [ "starter.py", "model2.bin", "model.bin" , "vectorizer.pkl", "./"]

CMD ["python", "starter.py"]