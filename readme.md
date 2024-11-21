

git clone https://github.com/Lionel-Lim/RAG_Langchain.git

sudo mkdir src/frontend/credential

sudo cp /etc/letsencrypt/live/dylim.dev/fullchain.pem src/frontend/credential
sudo cp /etc/letsencrypt/live/dylim.dev/privkey.pem src/frontend/credential

sudo docker-compose up -d --build

sudo docker compose up -d --build