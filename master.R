## master.R - replicate all results in the paper and online appendix

##################################################
## produce tables and figures that don't require DTSM estimates

## main body
source("R/plot_data.R")       # Figures 1 (data1.pdf) and 2 (data2.pdf)
source("R/plot_cycles.R")     # Figure 3 (cycles.pdf)
source("R/coint.R")           # Table 1 (coint.tex)
source("R/predict.R")         # Table 2 (returns.tex and returns_subsample.tex)
source("R/predict_cycles.R")  # Table 3 (returns_cycles.tex)

## Online Appendix
source("R/plot_rstar.R")      # OA Figures 1 (rstar_external.pdf) and 2 (rstar_internal.pdf)
source("R/persistence.R")     # OA Table 2 (persistence.tex)
commandArgs <- function(...) "level"
source("R/coint.R")           # OA Table 3 (coint_level.tex)
source("R/predict_mats.R")    # OA Table 4 (returns_mats.tex)
source("R/predict_R2.R")      # OA Table 5 (returns_R2.tex)

##################################################
## DTSM estimation

## estimate OSE model (if necessary)
## - this takes less than one minute
if (!file.exists("results/ose.RData"))
    source("R/ose.R")

## estimate ESE model (if necessary)
## 1) estimate each of five parallel MCMC chains
## - each chain has a different random seed
##   - if no command line argument is given, the default seed is 616
## - this takes about 20 hours FOR EACH CHAIN
##   - better to parallelize, see readme.pdf
for (SEED in c(616, 1, 2, 3, 4))
    if (!file.exists(paste0("results/ese_", SEED, ".RData"))) {
        commandArgs <- function(...) SEED
        source("R/ese.R")
    }

## 2) combine five chains into one MCMC chain
if (!file.exists("results/ese.RData"))
    source("R/ese_combine.R")

##################################################
## produce tables and figures for DTSM results

source("R/plot_dtsm_figures.R")   # Figures 4 (dtsm_istar.pdf), 5 (dtsm_y10.pdf), 7 (dtsm_tp.pdf)
source("R/plot_ose_loadings.R")   # Figure 6 (dtsm_loadings.pdf)
                                  # and OA Figure 3 (istar_loadings_humpshape.pdf)
source("R/sim_returns.R")         # Table 4 (dtsm_returns.tex)

##################################################
## out-of-sample forecasting with OSE model

## source("R/oos.R")              # Table 5 (oos_10y_rmse.tex and oos_10y_bluechip_rmse.tex)
## Blue Chip forecasts are not included in replication files
## can replicate part of Table 5 without it:
source("R/oos_nobluechip.R")      # First panel of Table 5 (oos_10y_rmse.tex)

