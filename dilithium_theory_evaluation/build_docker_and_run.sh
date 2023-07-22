#!/bin/bash
set -e 
docker build -t 'dilithium_reproduce_fedora' .
docker run -ti --entrypoint /bin/bash -v $(realpath .):/current_dir dilithium_reproduce_fedora
