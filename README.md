# Steps:

- Preprocess data
- Train and test and save a base model
- Create cli
- Dockerize
- Use BentoML and create a service
- Fast API for API calling
- Bentoml service
- CICD pipeline
- AWS ECS, Lambda, S3
- Web scrape some source for data
- RAG with Gemini and Langchain


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
docker run -p 3000:3000 -it --rm --entrypoint /bin/bash car-price:latest
```

Now a shell is attached to the container:
```
python main.py preprocess --dataset datasets/ca-dealers-used.csv
python main.py train-eval
python main.py infer --input-file input.json
```

Before using bentoml, test to see if it exist in the container:
which bentoml

running bentoml service by iteself (with or without docker):
```
bentoml serve bentoml_service:CarPricePrediction --production
```

There is a problem in which bentoml service does not start automatically.

testing the service
```
curl -X POST "http://localhost:3000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "year": 2019,
      "miles": 7120,
      "make": "Acura",
      "model": "NSX",
      "trim": "Base",
      "body_type": "Coupe",
      "vehicle_type": "Car",
      "drivetrain": "AWD",
      "transmission": "Automatic",
      "fuel_type": "Gasoline",
      "engine_block": "V6",
      "engine_size": 3.5,
      "city": "Vancouver",
      "state": "BC"
    }
  }'
```


