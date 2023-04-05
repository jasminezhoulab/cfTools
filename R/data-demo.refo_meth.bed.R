#' Methylation information on fragments
#'
#' A BED file of methylation information on fragments
#'
#' @name demo.refo_meth.bed
#' 
#' @return A tibble with 552 rows and 8 variables
#' 
#' @format A tibble with 552 rows and 8 variables
#' \describe{
#' \item{chr}{Chromosome}
#' \item{cpgStart}{Start postion of first CpG on the fragment}
#' \item{cpgEnd}{End postion of first CpG on the fragment}
#' \item{strand}{Strand}
#' \item{cpgNumber}{Number of CpG sites on the fragment}
#' \item{cpgPosition}{Postions of CpG sites on the fragment}
#' \item{methState}{A string of methylation states of 
#' CpG sites on the fragment}
#' \item{name}{ID of the sequence}
#' }
#' @usage data("demo.refo_meth.bed")
#' 
#' @author Ran Hu \email{huran@ucla.edu}
NULL
