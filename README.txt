0. Run Jupyter notebook in virtualenv:
	a. make setup **
	b. source .cloud/bin/activate
	c. make install **
	d. jupyter lab
	e. deactivate (when finished)

0. Run the docker image:
	a. docker compose up

1. Run the Notebook:
	a.  open http://localhost:8888

2. Run the Flask Application:
	a. open http://localhost:8081


2. Run the Flask Application:
	a. docker build -t mldockerimg:v1 .
	b. docker run -dp 8081:5002 -ti \'97name mlContainer mldockerimg:v1
	c. open http://localhost:8081
	d. docker stop mlContainer
	e. docker start mlContainer


3. Pushing docker to ECR:
	a. aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-west-2.amazonaws.com
	b. docker tag webmd_scraper:latest 123456789012.dkr.ecr.us-west-2.amazonaws.com/webmd_scraper:latest
	c. aws ecr create-repository --repository-name webmd_scraper --region YOUR_REGION
	d. docker push 123456789012.dkr.ecr.us-west-2.amazonaws.com/webmd_scraper:latest

3. Create ECS cluster

4. 