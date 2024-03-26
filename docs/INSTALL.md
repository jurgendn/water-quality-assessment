# Install

## Conda environment

Firstly, create a new environment using `python=3.10.10`.

```bash
conda create -n water-quality-assessment python=3.10
```

Afterthat, we will install all dependencies via `pip`.

```bash
# install project
cd water-quality-assessment
pip install -r requirements.txt
```

## Virtual Environment

```bash
# clone project
git clone https://github.com/jurgendn/water-quality-assessment

# install project
cd water-quality-assessment
pip install -r requirements.txt
```

## Docker

I also provide a Docker environment for development for those who want to use Docker.

### Install Docker and Docker Compose V2

Please read this instruction to install [`Docker`](https://docs.docker.com/engine/install/ubuntu/) and [`Docker Compose V2`](https://docs.docker.com/compose/migrate/). After that, to verify the installation, run this

```bash
>> docker --version
Docker version 25.0.2, build 29cf629

>> docker compose version
Docker Compose version v2.24.3-desktop.1
```

If you want to access the accelerators, such as GPUs inside the docker environment, read the [Turn on GPU access with Docker Compose](https://docs.docker.com/compose/gpu-support/) to use accelerators.

### The compose

```compose
version: "3.7"

services:
  dev:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: water-assessment-dev
    env_file:
      - ./.env
    volumes:
      - ./:/home/working

    # This part is used to enable GPU support
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

```

An important note here is that the entire working folder is mounted to `/home/working` folder in the container, and we will need to open a shell inside the container, and also attach the working folder into some code editor. In my case, I use **Visual Studio Code** with the extensions `Dev Containers` to attach the opening folder to the corresponding running container.

```lang-sh
# To build then run the container and detach
docker compose up --build -d

# If there is no changes in the Dockerfile, we just need to run
docker compose up -d

# Open the shell
docker compose exec dev bash
```

Then we will be able to access the shell inside a docker container.
