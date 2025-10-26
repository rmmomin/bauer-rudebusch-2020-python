## make sure all required R packages are installed
req.packages <- c("KFAS", "MCMCpack", "mvtnorm", "Rcpp", "RcppArmadillo", "numDeriv", "sandwich", "xtable", "urca", "dynlm", "dplyr", "VAR.etp")
new.packages <- req.packages[!(req.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)

## make sure necessary folders exist
if (!file.exists("figures"))
    dir.create("figures")
if (!file.exists("tables"))
    dir.create("tables")
if (!file.exists("results"))
    dir.create("results")

