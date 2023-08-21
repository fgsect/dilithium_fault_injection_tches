FROM python:3.9

# Install requirements
COPY requirements.txt /tmp
RUN pip3 install -r /tmp/requirements.txt

# Copy the code
COPY ./code /code

# Command used to start the application
WORKDIR /code
ENTRYPOINT ["python3"]
CMD ["dilithium_one_attack.py"]
