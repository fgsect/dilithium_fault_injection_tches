# Indicate the Gurobi reference image
#FROM gurobi/optimizer:latest
FROM fedora:latest
COPY requirements.txt requirements.txt
RUN dnf update -y && dnf install -y python3 pip vim
RUN pip3 install -r requirements.txt

# Set the application directory
WORKDIR /current_dir

# Copy the application code 
#ADD . /app

# Command used to start the application


