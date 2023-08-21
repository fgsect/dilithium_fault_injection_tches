# Reproducing Theoretical Evaluation:
The theoretical evaluation is stored in `dilithium_theory_evaluation`. 
To reproduce our theoretical evaluation, first build the docker container from the `Dockerfile` located in the `dilithium_theory_evaluation` directory. Here we will named the built container `dilithium_theory_evaluation`:
```
cd dilithium_theory_evaluation
docker build . -t dilithium_theory_evaluation
```
Per default this container runs the script `dilithium_one_attack.py`:
```
docker run dilithium_theory_evaluation
```
You can run the script `dilithium.py` like this:
```
docker run dilithium_theory_evaluation dilithium.py
```

The script `dilithium_one_attack.py` will run the theoretical evaluation as described in Section 6 "Simulation and Evaluation of the Attack" of the paper. 
The script will only run one attack, to demonstrate the attack. To run further attacks, run the script `dilithium.py`.
Please note that running `dilithium.py` can take more than a day to finish or even run indefinitely if the computer is not powerful enough. Please refer to our paper for details on the hardware we used. If the output is not a logger output, it is generated by the gurobi solver. The logger outputs JSONs, each one contains information about a single attempted recovery of s_1. To ignore the gurobi output, filter for `:root:`, i.e.:
```
docker run dilithium_theory_evaluation | grep ':root:'
```

The script `dilithium_one_attack.py` is parametrized as follows:
```
nist_security_level = 2
threads = 40
params = Parameters.get_nist_security_level(nist_security_level, threads=threads)
threshold = params.beta
notion_of_success = 2*params.n # How many _correct_ equations (for each secret key polynomial) do we collect before we launch the attack?
#Parameter m measures how many coefficients are NOT faulted
#That means we fault n*l -m coefficients
#Setting m to params.n*params.l -2 will fault exactly TWO coefficients per signature
#The script will inform the user if the attack succeeded
for m in range(params.n*params.l - 2, params.n * params.l -1):
```
If the attack succeeds, you should see the following:
```
INFO:root:Successfully recovered s_1 entry!
INFO:root:Successful recovery!
DEBUG:root:{"entry_results": [{"faulted_coeffs": 535, "total_equations": 267008, "filtered_equations": 669, "duration": 0.8444759845733643, "failure": false, "equations_used": 535.0}, {"faulted_coeffs": 531, "total_equations": 267008, "filtered_equations": 676, "duration": 0.959650993347168, "failure": false, "equations_used": 531.0}, {"faulted_coeffs": 512, "total_equations": 267008, "filtered_equations": 697, "duration": 1.3874258995056152, "failure": false, "equations_used": 512.0}, {"faulted_coeffs": 512, "total_equations": 267008, "filtered_equations": 684, "duration": 1.3388750553131104, "failure": false, "equations_used": 512.0}], "m": 1022, "num_signatures": 1043, "nist_param_level": 2, "threshold": 78, "notion_of_success": 512, "timeout_limit": 300}
INFO:root:m = 1022; notion_of_success = 512 succeeded!
```

The attack itself is implemented in `recover_s1_entry.py`. The data in the "entry_results" dictionary gives detailed information about the attack statistics, such as the number of used equations, number of filtered equations (according to the method described in Section 5.3 Attacking the Protected Implementation of Dilithium to reduce false-positives), or number of faulted coefficients (in total).


# Reproducing Practical Evaluation:

To reproduce the practical evaluation from Section 7, End-To-End Attack Proof-of-Concept, we provide the "chipwhisperer-dilithium" directory. Since some of the code needs a connection to a ChipWhisperer, this code is provided without a docker container. A virtual environment should be sufficient. 

The chipwhisperer.dilithium directory contains:
1. The modified dilithium-firmware (in `hardware/victims/firmware/simpleserial-dilithium-ref/dilithium/`)
2. Two Jupyter notebooks, which:
   2.1 Collect the faulted signatures (`Dilithium - Glitches - Signature - Only - Attack.ipynb`).
   2.2 And run the attack plus print attack statistics (`Stats - Signature - Only - Attack.ipynb`)
Plus some data that we pre-collected in `gc.results.pickled.signature-attacks-2023-03-16_00-14-39.pickle`.

The notebook should execute as is. To do so, please run
```
cd chipwhisperer.dilithium && pip install -r requirements.txt && python setup.py install
```
and then:
```
cd jupyter-dilithium && jupyter notebook .
``` 
You can then connect to the notebook in your browser window.
To collect the data, simply execute all the steps in the `Dilithium_Collect_Signatures` notebook. This notebook requires your computer to be connected to a ChipWhisperer Lite FPGA board, which in turn needs to be connected to a `STM32F4` target board mounted on a ChipWhisperer UFO board.
To run the analysis on our pre-collected data, simply run `Dilithium_Run_Attack`. The cell `Out[13]` depicts the success rate per faulted polynomial, as described in Section 7.3.

