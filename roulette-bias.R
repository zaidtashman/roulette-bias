require(rstan)

wheel = c('00','27','10','25','29','12','8','19','31','18','6','21','33',
         '16','4','23','35','14','2','0','28','9','26','30','11','7','20',
         '32','17','5','22','34','15','3','24','36','13','1')
wheel.df = data.frame(number=wheel, freq=0)

true.p = rep(1/38, 38) 
true.p[11] = true.p[11] + 1/(2*38)
true.p[12] = true.p[12] - 1/(2*38)

spins = sample(wheel, 100, replace = T, prob = true.p)

for (i in 1:length(spins)){
  k = which(wheel.df$number == spins[i])
  wheel.df$freq[k] = wheel.df$freq[k] + 1
}
counts = wheel.df$freq

stan_model = stan_model("./roulette.stan")

data = list(m=38, counts=counts, priors=rep(1,38))

#fit_vb = vb(stan_model, data = data, output_samples=1000) #, 
fit_sampling = sampling(stan_model, data=data, seed=111) # iter=5000, warmup=1000, chains=2, 

fd_sampling = extract(fit_sampling)
means = colMeans(fd_sampling[["p"]])
wheel.df$prob = means

plot(means)

print(fit_sampling)
plot(fit_sampling, pars = c('p'))
traceplot(fit_sampling)
pairs(fit_sampling)
launch_shinystan(fit_sampling)
