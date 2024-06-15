# Use the official Python image from the Docker Hub
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR ./src/pjatk_asi/web_app

# Install git, openssh-client, and other necessary packages
RUN apt-get update && \
    apt-get install -y git openssh-client curl && \
    rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python3 -

# Export PATH="/root/.local/bin:$PATH"
RUN export PATH="/root/.local/bin:$PATH"

# Copy your SSH private key to the container
# Make sure your private key is named id_rsa (or provide correct filename)
COPY docker.key /root/.ssh/id_rsa

# Adjust SSH key permissions
RUN chmod 600 /root/.ssh/id_rsa \
    && ssh-keyscan github.com >> /root/.ssh/known_hosts \
    && echo "StrictHostKeyChecking no" >> /etc/ssh/ssh_config

# Clone your Git repository
RUN git clone https://github.com/theConsite/PJATK_ASI.git .

# Copy the requirements file to the working directory
RUN /root/.local/bin/poetry install

# Copy the rest of the application code to the working directory
#COPY . .

# Expose the port the app runs on
EXPOSE 8501

# Command to run the app
CMD ["streamlit", "run", "web_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

