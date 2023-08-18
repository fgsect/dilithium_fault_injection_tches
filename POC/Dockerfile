FROM python:3.8.10
RUN apt-get update && \
    apt-get install -y gcc-arm-none-eabi libusb-1.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
COPY requirements.txt /root/
RUN cd /root && pip install pip==23.2.1 && pip install -r requirements.txt
RUN cd /root && \
    git clone -b dilithium https://github.com/Alex1s/chipwhisperer.git chipwhisperer.dilithium && \
    cd chipwhisperer.dilithium && \
    git checkout 433ad95a && \
    git submodule init hardware/victims/firmware/simpleserial-dilithium-ref/dilithium && \
    git submodule update hardware/victims/firmware/simpleserial-dilithium-ref/dilithium && \
    mkdir -p jupyter/courses/fault101 && \
    cp jupyter-dilithium/* jupyter/courses/fault101 && \
    cd jupyter/courses/fault101 && \
    git clone https://github.com/Alex1s/python-dilithium.git && \
    cd python-dilithium && \
    git checkout 07214e8 && \
    rm -rf /root/chipwhisperer.dilithium/.git
COPY dilithium_solver /root/chipwhisperer.dilithium/jupyter/courses/fault101/dilithium_solver
CMD jupyter-notebook --allow-root --ip 0.0.0.0 --port 80 --no-browser /root/chipwhisperer.dilithium/jupyter/courses/fault101/ 
