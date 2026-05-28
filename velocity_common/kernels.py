import numpy as np
from velocity_common.poisson_disk import fast_sample

COMMT = '\033[36m'
FIGUR = '\033[32m'
RESET = '\033[0m'


def generate_kernels(lengths, n_kernels):
	# Sample the kernels
	# import lds
	# kernels = lds.sample(n_kernels, np.zeros_like(lengths), lengths)
	kernels = fast_sample(n_kernels, np.zeros_like(lengths), lengths, seed=42)
	n_kernels = len(kernels)
	print(f'{COMMT}[*] Initialized {FIGUR}{n_kernels}{COMMT} kernels{RESET}')

	# Compute the support radius
	if len(lengths) == 2:
		h = (lengths.prod() / n_kernels / np.pi)**.5 * 9
	else:
		h = (lengths.prod() / n_kernels / np.pi * 3 / 4)**(1/3) * 6
	print(f'{COMMT}[*] Initialized support radius: {FIGUR}{h}{COMMT}{RESET}')

	return kernels, h

