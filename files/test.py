import numpy as np
import pystan as ps
import pickle

wheel = ['00','27','10','25','29','12','8','19','31','18','6','21','33',
         '16','4','23','35','14','2','0','28','9','26','30','11','7','20',
         '32','17','5','22','34','15','3','24','36','13','1']

model = """
data {
  int<lower=1> m;
  int counts[m];
  vector<lower=0>[m] priors;
}

parameters {
  simplex[m] p;
}

model {
  p ~ dirichlet(priors);
  counts ~ multinomial(p);
}
"""

sm = ps.StanModel(model_code=model, verbose=True)

with open('mode-file.pkl', 'wb') as f:
        pickle.dump(sm, f)


