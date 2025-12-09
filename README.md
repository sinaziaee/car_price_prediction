# Steps:

- Preprocess data
- Train and test and save a base model
- Create cli
- Dockerize
- Fast API for API calling
- Bentoml service
- CICD pipeline
- AWS ECS, Lambda, S3
- Web scrape some source for data
- RAG with Gemini and Langchain
- 


Start Docker:
```
sudo systemctl start docker
```

Create docker image:
```
docker build -t car-price:latest .
```

How to Run with docker:
```
docker run -it --rm --entrypoint /bin/bash car-price:latest
```

Now a shell is attached to the container:
```
python main.py preprocess --dataset datasets/ca-dealers-used.csv
python main.py train-eval
python main.py infer --input-file input.json
```


