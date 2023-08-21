# Indicate the Gurobi reference image
#FROM gurobi/optimizer:latest
FROM python:3.9

# install requirements
COPY requirements.txt /tmp
RUN pip3 install -r /tmp/requirements.txt

# copy the code
COPY ./code /code

# Set the application directory

# Command used to start the application
WORKDIR /code
ENTRYPOINT ["python3"]
CMD ["dilithium_one_attack.py"]
