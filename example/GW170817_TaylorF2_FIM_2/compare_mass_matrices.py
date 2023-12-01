import jax.numpy as jnp

eps = 1e-3
n_dim = 13
mass_matrix = jnp.eye(n_dim)
mass_matrix = mass_matrix.at[0,0].set(1e-5)
mass_matrix = mass_matrix.at[1,1].set(1e-4)
mass_matrix = mass_matrix.at[2,2].set(1e-3)
mass_matrix = mass_matrix.at[3,3].set(1e-3)
mass_matrix = mass_matrix.at[7,7].set(1e-5)
mass_matrix = mass_matrix.at[11,11].set(1e-2)
mass_matrix = mass_matrix.at[12,12].set(1e-2)

old_mass_matrix = jnp.diag(mass_matrix * eps)
eps = 1e-2
tuned_mass_matrix = eps *  jnp.array([6.73381007e-05, 9.35374439e-05, 2.08901317e-03, 3.16296806e-03,8.59835237e-03, 3.00932838e-02, 7.92793636e-03, 1.00000000e-05,2.42234973e-03, 7.63236126e-03, 4.52767590e-03, 5.98229967e-04,3.12443299e-03])

def format(n):
    a = '%E' % n
    return a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]


naming = ["M_c","q","s1_z","s2_z","lambda_tilde","delta_lambda_tilde","d_L","t_c","phase_c","iota","psi","ra","dec"]

# Now, iterate over naming, old and new mass matrix, and print the results side by side using format, and in scientific notation
for i in range(len(naming)):
    print(f"{naming[i]}: old: {format(old_mass_matrix[i])}, tuned: {format(tuned_mass_matrix[i])}")