#!/bin/bash

# Update the apt lists
sudo apt-get -qq update 
# sudo apt-get -y upgrade 
sudo apt-get -y install apt-transport-https ca-certificates curl software-properties-common

# Add Docker’s official GPG key:

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

# Add Docker’s official apt repo

sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"

# Update the apt lists with the new docker repo, and install Docker CE (Community Edition)
sudo apt-get -qq update
sudo apt-get -y install docker-ce

# Verify the installation by running a few commands.
docker -v

# Remove the requirement of “sudo” when running docker commands, add your user to the docker group. To do so, simply type the following:
sudo usermod -aG docker Username

# Your username is now part of the docker group. To apply changes, either logout and login or type:
su - Username

# Update the apt lists
sudo apt-get -qq update

#install docker-compose 1.29.2
sudo apt purge docker-compose -y
sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
source ~/.bashrc

# restart docker service
sudo systemctl restart docker.service

# init PostgreSql
unzip PostgreSql.zip
cd PostgreSql
sudo bash init.sh

# install pgadmin4
curl https://www.pgadmin.org/static/packages_pgadmin_org.pub | sudo apt-key add
sudo sh -c 'echo "deb https://ftp.postgresql.org/pub/pgadmin/pgadmin4/apt/$(lsb_release -cs) pgadmin4 main" > /etc/apt/sources.list.d/pgadmin4.list && apt update'
sudo apt install pgadmin4-desktop -y

# install pqxx dev C++
# sudo apt-get install libpqxx-dev -y

