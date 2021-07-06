# cleanup possibly existing files
rm Procfile
rm prediction_service_fastapi.py
rm app.yaml
rm main.py
rm .gcloudignore
rm -f requirements.txt
rm -r output/function_deployment
rm -y Dockerfile
rm -y .dockerignore
rm lambda_function.py
rm setup_sagemaker_endpoint.py
rm -r deployment/aws_sagemaker/docker_context/ml_project
rm deployment/aws_sagemaker/docker_context/model_titanic.tar.gz
rm deployment/aws_sagemaker/docker_context/model_titanic.pkl