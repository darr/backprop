source ~/virtual_env/py2env/bin/activate
pylint --rcfile=pylint.conf main.py
pylint --rcfile=pylint.conf forward_neural_network.py
pylint --rcfile=pylint.conf mnist_dataset.py
deactivate

source ~/virtual_env/py3env/bin/activate
pylint --rcfile=pylint.conf main.py
pylint --rcfile=pylint.conf forward_neural_network.py
pylint --rcfile=pylint.conf mnist_dataset.py
deactivate
