# MNIST VAE Generation and Classification

### Trains from MNIST Dataset and uses a Variational Auto-Encoder to generate new numbers from 0-9 from the latent space, and trains labels by a FCNN, which is then used to label and export the generated numbers.
#
## Dependencies

- argparse
- json
- os
- torch
- numpy
- matplotlib.pyplot
- itertools
- threading
- time
- sys

## Running `main.py`

To run `main.py`, use

```sh
python main.py -o [RESULTS_DIR] -n [N_GEN_SAMP]

```
