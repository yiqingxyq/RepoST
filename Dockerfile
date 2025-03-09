# Start from python
FROM python:3.11.3

# Install
RUN apt-get update
RUN apt-get install -y zsh tmux wget git autojump unzip vim htop less

# Set the working directory inside the container
WORKDIR /home/user

# Copy files 
RUN mkdir /home/user/RepoST
RUN mkdir /home/user/RepoST/RepoST
COPY utils*.py /home/user/RepoST/
COPY requirements*.txt /home/user/RepoST/
COPY setup*.sh /home/user/RepoST/
COPY RepoST/execution.py /home/user/RepoST/RepoST/

# Install the required packages
# RUN python -m pip install --upgrade pip
RUN pip install --upgrade --no-cache-dir -r /home/user/RepoST/requirements_docker_py311.txt

# override default image starting point (otherwise it starts from python)
CMD /bin/bash
ENTRYPOINT []