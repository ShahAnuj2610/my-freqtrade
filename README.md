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
