#' Fragment-level methylation information
#'
#' A BED file of fragment-level methylation information
#'
#' @name demo.fragment_level.meth.bed
#' 
#' @return A tibble with 552 rows and 9 variables
#' 
#' @format A tibble with 552 rows and 9 variables
#' \describe{
#' \item{chr}{Chromosome}
#' \item{start}{Chromosome start}
#' \item{end}{Chromosome end}
#' \item{name}{ID of the sequence}
#' \item{fragmentLength}{Fragment length}
#' \item{strand}{Strand}
#' \item{cpgNumber}{Number of CpG sites on the fragment}
#' \item{cpgPosition}{Postions of CpG sites on the fragment}
#' \item{methState}{A string of methylation states of 
#' CpG sites on the fragment}
#' }
#' @usage data("demo.fragment_level.meth.bed")
#' 
#' @author Ran Hu \email{huran@ucla.edu}
NULL
