#' Cancer-specific marker parameter
#'
#' The paired shape parameters of beta distributions 
#' for cancer-specific markers
#'
#' @name CancerDetector.markers
#' 
#' @return A tibble with 1266 rows and 3 variables
#' 
#' @format A tibble with 1266 rows and 3 variables
#' \describe{
#' \item{markerName}{Name of the marker}
#' \item{tumor}{Paired beta distribution shape parameters 
#' for tumor samples}
#' \item{normalPlasma}{Paired beta distribution shape 
#' parameters for normal plasma samples}
#' }
#' @usage data("CancerDetector.markers")
#' 
#' @author Ran Hu \email{huran@ucla.edu}
NULL