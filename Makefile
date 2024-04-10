# Define the name of the virtual environment directory
VENV := .cloud

# Default target executed when no arguments are given to make.
default: setup

# Create virtual environment
setup:
	python3 -m venv $(VENV)
	@echo ">>> Virtual environment created"

# Activate the virtual environment
.PHONY: activate
activate:
	@echo "To activate the virtual environment, execute 'source $(VENV)/bin/activate' in your shell."

# Clean up the virtual environment
#.PHONY: clean
clean:
	rm -rf $(VENV)
	@echo ">>> Virtual environment removed."

# Additional targets can be defined here, for example, to run tests or start the application.
install:
	pip install --upgrade pip
	pip install -r requirements.txt
	ipython kernel install --user --name=$(VENV)
	@echo ">>> Packages installed"

#test:
#	python -m pytest -vv --cov=cli --cov=mlib --cov=utilscli --cov=app test_mlib.py

#format:
#	black *.py

#lint:
#	pylint --disable=R,C,W1203,E1101 mlib cli utilscli
#	#lint Dockerfile
#	#docker run --rm -i hadolint/hadolint < Dockerfile

#deploy:
#	#push to ECR for deploy
#	aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 561744971673.dkr.ecr.us-east-1.amazonaws.com
#	docker build -t mlops .
#	docker tag mlops:latest 561744971673.dkr.ecr.us-east-1.amazonaws.com/mlops:latest
#	docker push 561744971673.dkr.ecr.us-east-1.amazonaws.com/mlops:latest
	
all: install