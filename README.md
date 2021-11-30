## Disclaimer
Use it at your own risk.

## Getting Started
```bash
sudo apt update
sudo apt upgrade -y
sudo apt install docker-compose -y
sudo mkdir service
cd service
sudo git clone https://github.com/freqtrade/freqtrade.git
cd freqtrade
sudo chmod -R 777 /var/run/docker.sock
docker-compose pull
```

## Oracle Cloud VM port Expose
```bash
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80 -j ACCEPT
sudo netfilter-persistent save
```
- More details here: https://stackoverflow.com/questions/62326988/cant-access-oracle-cloud-always-free-compute-http-port#62343749

## Define your GitHub secrets
```bash
export DOCKER_SSH_HOST=192.168.0.1
export DOCKER_SSH_USERNAME=ubuntu
export DOCKER_SSH_PRIVATE_KEY=<>
export DOCKER_HOST=ubuntu@192.168.0.1
export DOCKER_SSH_PUBLIC_KEY=<>
export GH_PAT=<>
```

## Create a swap file
```bash
sudo fallocate -l 100G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
sudo cp /etc/fstab /etc/fstab.bak
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```
