Configure Nginx

Create directories for Nginx configuration and SSL certificates:

mkdir -p nginx/conf.d
mkdir -p nginx/certbot/conf
mkdir -p nginx/certbot/www


Nginx Configuration File

Located at src/frontend/ngix/config

sudo mkdir src/frontend/credential

sudo cp /etc/letsencrypt/live/dylim.dev/fullchain.pem src/frontend/credential
sudo cp /etc/letsencrypt/live/dylim.dev/privkey.pem src/frontend/credential