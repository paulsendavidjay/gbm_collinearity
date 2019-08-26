
library(MASS)
library(ggplot2)
library(magrittr)
# library(gbm)
library(xgboost)
library(data.table)
library(dplyr)
library(parallel)
# library(SHAPforxgboost)

library(doMC)
registerDoMC(7)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

source("~/GBT_Collinearity/create_data.R")

n_var <- 20
n_base <- 7
n_obs <- 5000
snr = 2 # signal-to-noise ratio for error term

# MODELING PARAMETERS
nrounds <- 1500
early_stopping_rounds <- 10
nthread <- 6
print_every_n = 750
params <- list(eta = 0.1,
               max_depth = 4,
               num_parallel_tree = 6)


f3_wt <- c(0.1, 0.2, 0.5, 1, 2, 3) # seq(1, 4, by = 1)
f4_wt <- c(2.5) # seq(1, 4, by = 1) # 0.75, 1.25, 
v_cor <- 6
r13 <- c(0.65, 0.85, 0.95)
seed <- seq(101, 110)
search_grid <- expand.grid(f3_wt = f3_wt,
                           f4_wt = f4_wt,
                           r13 = r13,
                           v_cor = v_cor,
                           seed = seed)

out_data <- list()
log_data <- list()
shap_data <- data.table()
for (i in seq(nrow(search_grid))) {
  
  cat(paste0(i, " of ", nrow(search_grid), "\n"))
  
  # BUILD DATA
  x <- create_data(n_var = n_var, n_base = n_base, n_obs = n_obs,
                   r12 = 0.85,
                   r13 = search_grid[i, "r13"],
                   r23 = 0.85,
                   v_cor = search_grid[i, "v_cor"],
                   f_wts = c(9, 0.8,
                             search_grid[i, "f3_wt"],
                             search_grid[i, "f4_wt"]),
                   snr = 2,
                   seed = search_grid[i, "seed"])
  
  # RUN MODEL 1 - ONLY ONE CORRELATE
  xgb.01 <- xgboost(data = as.matrix(x[, paste0("X", seq(n_var))]),
                    label = x[["y"]], print_every_n = print_every_n,
                    params = params,
                    nrounds = nrounds,
                    nthread = nthread,
                    early_stopping_rounds = early_stopping_rounds)
  
  # RUN MODEL 2 - BOTH CORRELATES
  xgb.02 <- xgboost(data = as.matrix(x[, paste0("X", seq(n_var + 1))]),
                    label = x[["y"]], print_every_n = print_every_n,
                    params = params,
                    nrounds = nrounds,
                    nthread = nthread,
                    early_stopping_rounds = early_stopping_rounds)
  
  # RUN MODEL 3 - C3
  xgb.03 <- xgboost(data = as.matrix(x[, c(paste0("X", seq(n_var)[-6]), "C3")]),
                    label = x[["y"]], print_every_n = print_every_n,
                    params = params,
                    nrounds = nrounds,
                    nthread = nthread,
                    early_stopping_rounds = early_stopping_rounds)
  
  set_list <- rbindlist(list(xgb.importance(model = xgb.01) %>%
                               mutate(., mdl = "A", rnk = rev(rank(Gain))),
                             xgb.importance(model = xgb.02) %>%
                               mutate(., mdl = "B", rnk = rev(rank(Gain))),
                             xgb.importance(model = xgb.03) %>%
                               mutate(., mdl = "C", rnk = rev(rank(Gain))))) %>%
    mutate(., i = i,
           v_cor = search_grid[i, "v_cor"],
           f3_wt = search_grid[i, "f3_wt"],
           f4_wt = search_grid[i, "f4_wt"],
           seed = search_grid[i, "seed"],
           r13 = search_grid[i, "r13"])
  
  log_list <- rbindlist(list(xgb.01$evaluation_log %>%
                               mutate(., mdl = "A"),
                             xgb.02$evaluation_log %>%
                               mutate(., mdl = "B"),
                             xgb.03$evaluation_log %>%
                               mutate(., mdl = "C"))) %>%
    mutate(., i = i,
           v_cor = search_grid[i, "v_cor"],
           f3_wt = search_grid[i, "f3_wt"],
           f4_wt = search_grid[i, "f4_wt"],
           seed = search_grid[i, "seed"],
           r13 = search_grid[i, "r13"])
  
  
  out_data[[i]] <- set_list
  log_data[[i]] <- log_list
  
  # SHAP PREDICTIONS
  x6 <- predict(object = xgb.01, 
                newdata = as.matrix(x[1:1000, paste0("X", seq(n_var))]), 
                predcontrib = TRUE)
  
  # shap_int <- predict(object = xgb.01, 
  #                     newdata = as.matrix(x[1:1000, paste0("X", seq(n_var))]), 
  #                     predinteraction = TRUE)
  # 
  x6_21 <- predict(object = xgb.02, 
                   newdata = as.matrix(x[1:1000, paste0("X", seq(n_var+1))]), 
                   predcontrib = TRUE)
  
  C3 <- predict(object = xgb.03, 
                newdata = as.matrix(x[1:1000, c(paste0("X", seq(n_var)[-6]), "C3")]), 
                predcontrib = TRUE)
  
  cmbd <- data.table(i = i,
                     C3 = C3[,"C3"],
                     A_X6 = x6[,"X6"],
                     B_X6 = x6_21[,"X6"],
                     B_X21 = x6_21[,"X21"]) %>%
    mutate(., B_X6_21 = B_X6 + B_X21)
  
  shap_data <- rbind(shap_data, cmbd)
  
}

out <- rbindlist(out_data)
fwrite(out, "~/GBT_Collinearity/colin_190806_out.csv")

log <- rbindlist(log_data)
fwrite(log, "~/GBT_Collinearity/colin_190806_log.csv")

fwrite(shap_data, "~/GBT_Collinearity/colin_190806_shap.csv")


