#' @title
#' Cancer Detector
#'
#' @description
#' Detect tumor-derived cfDNA and estimate the tumor burden.
#'
#' @param readsBinningFile a file of the fragment-level methylation states 
#' of reads that mapped to the markers.
#' @param tissueMarkersFile a file of paired shape parameters of beta 
#' distributions for markers.
#'
#' @return a list containing the cfDNA tumor burden 
#' and the normal cfDNA fraction.
#'
#' @examples
#' ## input files
#' demo.dir <- system.file("data", package="cfTools")
#' readsBinningFile <- file.path(demo.dir, "CancerDetector.reads.txt.gz")
#' tissueMarkersFile <- file.path(demo.dir, "CancerDetector.markers.txt.gz")
#'
#' CancerDetector(readsBinningFile, tissueMarkersFile)
#'
#' @export
CancerDetector <- function(readsBinningFile, tissueMarkersFile) {

    lambda <- 0.5 # a predefined lambda
    python.script.dir <- system.file("python", package = "cfTools", 
                                    mustWork = TRUE)
    
    extdata.dir <- system.file("data", package = "cfTools", mustWork = TRUE)
    output.dir <- extdata.dir
    
    timeNow <- strsplit(strsplit(as.character(Sys.time()), 
                        " ")[[1]][2], ":")[[1]]
    id <- paste0(timeNow[1], timeNow[2], timeNow[3])
    
    # output.dir <- paste0(extdata.dir, "/tmp/")
    # if (system.file("extdata/tmp", package = "cfTools") == "") {
    #   system2(command = "mkdir", args = output.dir)
    # }
    # id <- strsplit(as.character(Sys.time()), " ")[[1]][2]

    py1 <- paste0(python.script.dir, "/CalcReadLikelihood.py")
    py1.command <- c(py1, readsBinningFile, tissueMarkersFile, 
                         output.dir, id)
    
    id.likelihood <- file.path(output.dir, paste0(id, ".likelihood.txt"))
    tumor.burden <- file.path(output.dir, paste0(id, ".tumor_burden.txt"))
    
    py2 <- paste0(python.script.dir, "/CancerDetector.py")
    py2.command <- c(py2, id.likelihood, lambda, output.dir, id)
    
    proc <- basiliskStart(my_env)
    
    basiliskRun(proc, function() {
        system2(command = "python", args = py1.command)
        system2(command = "python", args = py2.command)
    })
    basiliskStop(proc)
    
    tumor_burden <- read.csv(tumor.burden, header = FALSE, sep = "\t")
    file.remove(id.likelihood)
    file.remove(tumor.burden)
    
    # cat(paste("cfDNA tumor burden:", tumor_burden$V1, "\n"))
    # cat(paste("normal cfDNA fraction:", tumor_burden$V2))
    colnames(tumor_burden) <- c("cfDNA_tumor_burden", 
                                "normal_cfDNA_fraction")
    
    return(tumor_burden)
    # system2(command = "rm", args = id.likelihood)
    # system2(command = "rm", args = tumor.burden)
}
