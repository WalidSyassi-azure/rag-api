# Use official Python image
FROM python:3.10-slim
# Set working directory
WORKDIR /app
# Copy app files
COPY . /app
# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
# Expose port
EXPOSE 8000
# Run the API with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
