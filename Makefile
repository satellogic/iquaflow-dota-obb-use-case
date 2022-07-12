
PROJ_NAME=obb
CONTAINER_NAME=${PROJ_NAME}-${USER}

ifndef DS_VOLUME
	DS_VOLUME=/share
endif

ifndef NB_PORT
	NB_PORT=8484
endif

ifndef MLF_PORT
	MLF_PORT=5454
endif

help:
	@echo "build -- builds the docker image"
	@echo "dockershell -- raises an interactive shell docker"
	@echo "notebookshell -- launches a notebook server"
	@echo "mlflow -- launches an mlflow server"

build:
	docker build --no-cache -t $(PROJ_NAME) .

dockershell:
	docker run --shm-size="16g" --rm --name ${CONTAINER_NAME} --gpus all \
	-v $(shell pwd):/work -v $(DS_VOLUME):$(DS_VOLUME) \
	-p ${MLF_PORT}:${MLF_PORT} \
	-p ${NB_PORT}:${NB_PORT} \
	-it $(PROJ_NAME)

notebookshell:
	docker run --shm-size="16g" --gpus all --privileged -itd --rm --name $(CONTAINER_NAME)-nb \
	-p ${NB_PORT}:${NB_PORT} \
	-v $(shell pwd):/work -v $(DS_VOLUME):$(DS_VOLUME) \
	$(PROJ_NAME) \
	jupyter lab \
	--NotebookApp.token='obb' \
	--no-browser \
	--ip=0.0.0.0 \
	--allow-root \
	--port=${NB_PORT}

mlflow:
	docker run --privileged -itd --rm --name $(CONTAINER_NAME) \
	-p ${MLF_PORT}:${MLF_PORT} \
	-v $(shell pwd):/work \
	-v $(DS_VOLUME):$(DS_VOLUME) \
	$(PROJ_NAME) \
	mlflow ui --host 0.0.0.0:$(MLF_PORT)

nbstop:
	docker exec -d --privileged $(CONTAINER_NAME) \
	jupyter lab stop ${NB_PORT}

nbexec:
	docker exec -d --privileged $(CONTAINER_NAME) \
	jupyter lab \
	--NotebookApp.token='obb' \
	--no-browser \
	--ip=0.0.0.0 \
	--allow-root \
	--port=${NB_PORT}

mlfexec:
	docker exec -d $(CONTAINER_NAME) mlflow ui --host 0.0.0.0:$(MLF_PORT)

execshell:
	docker exec -it ${CONTAINER_NAME} /bin/bash
