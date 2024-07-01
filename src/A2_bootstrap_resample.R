
library(dplyr)

if(Sys.info()[['sysname']]=="Windows"){
  path <- "M:/p1402-mauricim/analysis/data/"
} else{
  path  <- "/tsd/p1402/home/p1402-mauricim/analysis/data/"
  }

bootstrap <- function(data, n_resamples, seed = 42, only_indices = TRUE){
  
    require(dplyr)
  
    if (only_indices){
    # select only the study id to save storage and memmory
        data$studyid <- seq(1:nrow(data))
        data <- data %>% select(studyid, Studyid_hn057)
    }
    # Initialize an empty data frame
    bootstrap_samples <- data.frame(matrix(vector(), nrow = 0, ncol = 1 + ncol(data)))

    # Set column names
    if (only_indices){
        # only indices
        colnames(bootstrap_samples) <- c("bs_id", "studyid", "Studyid_hn057")
    } else {
        # use this if you want all data replicated
        colnames(bootstrap_samples) <- c("bs_id", colnames(data))
    }

    bootstrap_samples <- rbind(bootstrap_samples, data %>% mutate(bs_id = 0))

    # Set the seed
    set.seed(seed)

    # Number of rows in original data frame
    n_rows <- nrow(data)

    # For loop to create bootstrap samples
    for(sample in 1:n_resamples){
        # Randomly select rows from the original data frame
        bootstrap_sample <- data[sample(n_rows, replace = TRUE), , drop = FALSE]
        bootstrap_sample$bs_id <- sample

        # Add bootstrap sample to bootstrap_samples
        bootstrap_samples <- rbind(bootstrap_samples, bootstrap_sample, make.row.names=FALSE)
    }
  
    bootstrap_samples <- bootstrap_samples[, c("bs_id", setdiff(names(bootstrap_samples), "bs_id"))]
    
    # prepare ids for python usage (boundaries are from 0 to N-1)
    bootstrap_samples <- bootstrap_samples %>% mutate(studyid = studyid - 1)
    return(bootstrap_samples)
}


df <- read.csv(paste0(path, "BD4QoL_181223_encoded.csv"), stringsAsFactors = TRUE)

bootstrap_samples <- bootstrap(df, 200)
write.csv(bootstrap_samples, "data/bootstrap_ids.csv", row.names=FALSE)
#saveRDS(bootstrap_samples, paste0(path, "bootstrap_samples.rds"))
