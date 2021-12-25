data {
  int<lower=1> m;               // dimension
  int<lower=1> n;               // n of observations e.g. pixels containing (vh,vv)
  matrix[m, m] Sigma;
  vector[m] mu;
  vector[m] y;
  real alpha;                  // weight of GP prior
}
parameters {
  vector<lower=-10, upper=10>[m] z;
}
transformed parameters {
  vector[m] f = cholesky_decompose(Sigma) * z;
  vector[m] exp_f = exp(f) / sum(exp(f)); // output L1-normalized exp(f)
}
model {
  // z ~ normal(0, 1);
  target += normal_lpdf(z | 0, 1) * alpha;
  target += dot_product(y,f) - n * log(sum(exp(f)));    // likelihood
}
