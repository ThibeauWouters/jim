"""
Going to check here whether the generated initial guess is reasonable and in the correct format for jim.
"""

import numpy as np
import matplotlib.pyplot as plt
from injection_recovery import PRIOR


clip_initial_guess = True
check_out_of_bounds = True

# Load the data
outdir = "outdir_TaylorF2/injection_144/"
filename = outdir + 'initial_guess.npz'

data = np.load(filename)
initial_guess = data['initial_guess']

print("initial_guess")
print(initial_guess)


### Get the prior bounds
PRIOR = np.array(list(PRIOR.values()))

lower_bounds = PRIOR[:, 0]
upper_bounds = PRIOR[:, 1]

### Test the clipping of the initial guess
if clip_initial_guess:
    print("Clipping the initial guess")
    
    result = np.clip(initial_guess, lower_bounds, upper_bounds)
    initial_guess = result
    
    print("Clipping done: initial_guess")
    print(initial_guess)
    

### Check if any of the initial guesses are outside the prior bounds
if check_out_of_bounds:
    print("Checking if any of the initial guesses are outside the prior bounds")
    for i in range(len(initial_guess)):
        row = initial_guess[i]
        if np.any(row < lower_bounds) or np.any(row > upper_bounds):
            print("initial_guess[{}] is outside the prior bounds".format(i))
            # Now find which elements are outside the bounds
            for j in range(len(row)):
                if row[j] < lower_bounds[j] or row[j] > upper_bounds[j]:
                    print("    initial_guess[{}] is outside the prior bounds".format(i))
                    print("    element {} is outside the prior bounds".format(j))
                    print("    element {} has value {}".format(j, row[j]))
                    print("    lower bound is {}".format(lower_bounds[j]))
                    print("    upper bound is {}".format(upper_bounds[j]))
                    print("    difference is {}".format(row[j] - lower_bounds[j]))
                    print("    difference is {}".format(upper_bounds[j] - row))