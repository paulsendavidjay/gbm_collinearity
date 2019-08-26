create_data <- function(n_var, n_base, n_obs, r12 = 0.85, r13 = 0.95, r23 = 0.95,
                        v_cor = 6, f_wts = c(9, 0.8, 5, 2.5), snr = 2, seed = 101) {
  require(MASS)
  exp_fun <- Vectorize(function(x) { exp(-3*(1-x)^2)})
  
  # Friedman & Popesdue, 2008 - Predictive Learning via Rule Ensembles
  
  ######################
  # CREATE CORRELATED FEATURES
  set.seed(seed)
  cor_vars <- mvrnorm(n = n_obs,
                      c(0,0,0),
                      Sigma = matrix(c(1,r12, r13,
                                       r12, 1, r23,
                                       r13, r23, 1), nrow = 3))
  
  ######################
  # CREATE DATA FRAME
  x <- sapply(seq(n_var), function(x) { (sample(0:9, n_obs, replace = T) / 10)}) %>%
    data.frame(.)
  
  # ADD CORRELATED VARIABLE TO MODEL
  x[, v_cor] <- cor_vars[, 3]
  
  ######################
  # SET FUNCTION WEIGHTS
  y1_wt <- f_wts[1]
  y2_wt <- f_wts[2]
  y3_wt <- f_wts[3] # 2
  y4_wt <- f_wts[4] # 2.5
  cat(y3_wt)
  # F1: X1 X2 X3
  x[["f1"]] <- sapply(x[,1:3], exp_fun) %>% apply(., MARGIN = 1, FUN = prod) * y1_wt
  # FQ2: X4 X5
  x[["f2"]] <- -y2_wt*exp(-2*(x[,4] - x[,5]))
  # F3: X6
  x[["f3"]] <- y3_wt*sin(pi*x[,6])^2 # use product here for score
  # F4: X7 X8
  x[["f4"]] <- -y4_wt*(x[,7] - x[, 8])
  
  # GET FUNCTION RESULT
  x[["fy"]] <- rowSums(x[, paste0("f", 1:4)])
  
  
  ######################
  # ADD VARIANCE
  variance_f <- var(x[["fy"]])
  noise = rnorm(n = length(x[["fy"]])) # generate standard normal errors
  scale_factor <- sqrt( variance_f / var(noise*sqrt(snr)))
  x[["y"]] = x[["fy"]] + scale_factor*noise
  
  ######################
  # ADD CORRELATED VARIABLES IN
  # first change name of X[v_cor]
  names(x)[v_cor] <- "C3"
  x[[paste0("X", v_cor)]] <- cor_vars[,1] # replace var 6 with one of correlated predictive features
  x[[paste0("X", n_var + 1)]] <- cor_vars[,2]
  x
}
