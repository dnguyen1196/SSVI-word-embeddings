from Params.PosteriorParamTF import VariationalPosteriorParamsTF
import numpy as np
params = VariationalPosteriorParamsTF(dims=[num_word], D=20)
v, C = params.get_vector_distribution(0, 10)
print(v.shape)
print(C.shape)
params.update_vector_distribution(0, 2, np.zeros((20,)), np.ones((20, 20)))
v, C = params.get_vector_distribution(0, 2)
print(v.shape)
print(C.shape)

