export HOME_DIR=$PWD

# code
export CODE_DIR=${HOME_DIR}

# cache
export CACHE_DIR=${HOME_DIR}/"tmp"

# data
export dataset_generation_DIR=${CODE_DIR}"/data/ExecTrain"
export final_dataset_DIR=${CODE_DIR}"/data/ExecTrain"

# docker
export docker_HOME_DIR="/home/user"
export docker_CODE_DIR=${docker_HOME_DIR}"/RepoST"
export docker_CACHE_DIR=${docker_HOME_DIR}/"tmp"
export docker_dataset_generation_DIR=${docker_HOME_DIR}"/ExecTrain"
export docker_final_dataset_DIR=${docker_HOME_DIR}"/ExecTrain"