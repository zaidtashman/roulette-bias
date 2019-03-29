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