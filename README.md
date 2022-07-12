# Tensornets-gpu-jax-docker
1) Delete dockerfile and docker compose
2) copy to root either mac image for testing on a non gpu machine, or gpu image to train on a gpu machine
3) only if using gpu download cudnn 8.4.x and place it in extras/lib 
4) run docker compose up -d or docker build to make the image
5) for gpu, you'll need to install nvidia drivers and nvidia-docker on your machine
