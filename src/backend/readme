"""
1. poetry init
2. pipreqs .
3. poetry add $(cat requirements.txt) 
4. export PYTHONPATH="."  
5. export FLASK_APP=src/backend/__init__.py:create_app
6. Flask run

For TESTING:
export PROJECT_ID=ai-sandbox-company-73
export SERVICE_ACCOUNT=dy-local@ai-sandbox-company-73.iam.gserviceaccount.com
export APP_NAME=rag-backend

docker buildx build --platform linux/arm64 -t ${APP_NAME}:latest --load .

For DEPLOYMENT:
export PROJECT_ID=ai-sandbox-company-73
export SERVICE_ACCOUNT=dy-local@ai-sandbox-company-73.iam.gserviceaccount.com
export APP_NAME=rag-backend

docker buildx build --platform linux/amd64 -t ${APP_NAME} --load .
docker tag ${APP_NAME} gcr.io/${PROJECT_ID}/${APP_NAME}
docker push gcr.io/${PROJECT_ID}/${APP_NAME}

gcloud run deploy ${APP_NAME} \
  --image gcr.io/${PROJECT_ID}/${APP_NAME}:latest \
  --region asia-southeast1
"""

