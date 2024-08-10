#' Fragment-level methylation state for cfSort tissue deconvolution 
#'
#' The fragment-level methylation states of reads that mapped 
#' to the cfSort markers
#'
#' @name cfsort_reads
#' 
#' @return A tibble with 99999 rows and 2 variables
#' 
#' @format A tibble with 99999 rows and 2 variables
#' \describe{
#' \item{markerName}{Name of the cfSort marker}
#' \item{methState}{Fragment-level methylation states, 
#' which are represented by a sequence of binary values 
#' (0 represents unmethylated CpG and 1 represents methylated 
#' CpG on the same fragment)}
#' }
#' @usage data("cfsort_reads")
#' 
#' @author Ran Hu \email{huran@ucla.edu}
NULL