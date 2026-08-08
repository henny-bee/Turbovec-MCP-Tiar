FROM python:3.10-slim

# Set up the working directory
WORKDIR /app

# Ensure python output is unbuffered
ENV PYTHONUNBUFFERED=1

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files
COPY main.py vector_db.py tools.py ./

# Set the entrypoint to run the main server script
ENTRYPOINT ["python", "main.py"]
