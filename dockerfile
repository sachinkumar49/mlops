# Use Python base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy files to the container
COPY requirements.txt requirements.txt
COPY src/ /app/
COPY model.pkl model.pkl

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]
