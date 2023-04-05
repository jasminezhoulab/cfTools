#' Methylation information for CpG on the original bottom strand (OB)
#'
#' Methylation information for CpG on the original 
#' bottom strand (OB), which is one of the outputs from 
#' 'bismark methylation extractor'
#'
#' @name CpG_OB_demo
#' 
#' @return A tibble with 2224 rows and 5 variables
#' 
#' @format A tibble with 2224 rows and 5 variables
#' \describe{
#' \item{sequence ID}{ID of the sequence}
#' \item{methylation state}{Methylated or unmethylated CpG site}
#' \item{chromosome name}{Chromosome name}
#' \item{chromosome start}{Chromosome start position}
#' \item{methylation call}{Methylation call}
#' }
#' @usage data("CpG_OB_demo")
#' 
#' @author Ran Hu \email{huran@ucla.edu}
NULL
