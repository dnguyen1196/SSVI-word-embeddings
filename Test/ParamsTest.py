from Params.PosteriorParamTF import VariationalPosteriorParamsTF
import numpy as np

params = VariationalPosteriorParamsTF(dims=[10], D=20)

params.update_vector_distribution(0, 2, np.zeros((20,)), np.ones((20, 20)))

v, C = params.get_vector_distribution(0, 2)
