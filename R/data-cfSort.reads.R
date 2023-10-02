#' Fragment-level methylation state for cfSort tissue deconvolution 
#'
#' The fragment-level methylation states of reads that mapped 
#' to the cfSort markers
#'
#' @name cfSort.reads
#' 
#' @return A tibble with 99999 rows and 6 variables
#' 
#' @format A tibble with 99999 rows and 6 variables
#' \describe{
#' \item{markerName}{Name of the cfSort marker}
#' \item{cpgPosition}{Postions of CpG sites on the fragment}
#' \item{methState}{Fragment-level methylation states, 
#' which are represented by a sequence of binary values 
#' (0 represents unmethylated CpG and 1 represents methylated 
#' CpG on the same fragment)}
#' \item{methCount}{Number of methylated CpG sites on the fragment}
#' \item{unmethCount}{Number of unmethylated CpG sites on the fragment}
#' \item{strand}{Strand}
#' }
#' @usage data("cfSort.reads")
#' 
#' @author Ran Hu \email{huran@ucla.edu}
NULL