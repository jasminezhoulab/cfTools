#' cfTools: a versatile package for analyzing cell-free DNA data
#' 
#' Given the methylation sequencing data of a cell-free DNA (cfDNA) sample, 
#' for each cancer marker or tissue marker, we deconvolve the tumor-derived 
#' or tissue-specific reads from all reads falling in the marker region. 
#' Our read-based deconvolution algorithm exploits the pervasiveness of 
#' DNA methylation for signal enhancement, therefore can sensitively identify 
#' a trace amount of tumor-specific or tissue-specific cfDNA in plasma.
#' 
#' @details
#' Specifically, cfTools can deconvolve different sources of 
#' cfDNA fragments (or reads) in two contexts:
#' 
#' 1. Cancer detection: separate cfDNA fragments into tumor-derived 
#' fragments and background normal fragments (2 classes), and estimate the 
#' tumor-derived cfDNA fraction.
#' 
#' 2. Tissue deconvolution: separate cfDNA fragments from different tissues 
#' (> 2 classes), and estimate the cfDNA fraction of different tissue types 
#' (including an unknown type) for a plasma cfDNA sample.
#'
#' These functions can serve as foundations for more advanced cfDNA-based studies, 
#' including cancer diagnosis and disease monitoring.
#' 
#' For an overview of the functionality provided by the package, please see the
#' vignette:
#' \code{vignette(package="cfTools")}
#'
#' @author 
#' Ran Hu \email{huran@ucla.edu}, 
#' Mary Louisa Stackpole,
#' Shuo Li,
#' Xianghong Jasmine Zhou \email{XJZhou@mednet.ucla.edu},
#' Wenyuan Li \email{WenyuanLi@mednet.ucla.edu}
#' 
#' @seealso \code{\link{CancerDetector}}, \code{\link{cfDeconvolve}}, 
#' \code{\link{cfSort}}, \code{\link{MergeCpGs}}, \code{\link{MergePEReads}},
#' \code{\link{GenerateFragMeth}}, \code{\link{GenerateMarkerParam}}
#' 
#' @docType package
#' @name cfTools
#' 
#' @keywords internal
NULL