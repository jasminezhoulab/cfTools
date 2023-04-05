#' Fragment-level methylation state for cancer detection
#'
#' The fragment-level methylation states of reads that mapped 
#' to the cancer-specific markers
#'
#' @name CancerDetector.reads
#' 
#' @return A tibble with 9991 rows and 2 variables
#' 
#' @format A tibble with 9991 rows and 2 variables
#' \describe{
#' \item{markerName}{Name of the marker}
#' \item{methState}{Fragment-level methylation states, 
#' which are represented by a sequence of binary values 
#' (0 represents unmethylated CpG and 1 represents methylated 
#' CpG on the same fragment)}
#' }
#' @usage data("CancerDetector.reads")
#' 
#' @author Ran Hu \email{huran@ucla.edu}
NULL