#' @title
#' cfDNA Methylation Read Deconvolution
#'
#' @description
#' Read deconvolution.
#'
#' @param readsBinningFile input cfDNA methylation sequencing reads file, either in plain text or compressed form.
#' @param tissueMarkersFile input markers file in plain text. Each marker for each tissue type has either a median beta value or a paired shape parameters of beta-distribution from the tissue population of tissue types.
#' @param outputType output type: tissueFraction (default), tissueFraction+readCountRaw, tissueFraction+readCountPerMillion, tissueFraction+readCountPerBillion.
#' @param outputFile output file.
#' @param numTissues number of tissue types.
#' @param emAlgorithmType read-based tissue deconvolution EM algorithm type: em.global.unknown (default), em.global.known, em.local.unknown, em.local.known.
#' @param likelihoodRatioThreshold a positive float number. We suggest this number is 2 (default). All reads with likelihood ratio < cutoff (default: -1.0, meanin no reads filtering is used) will not be used. Likelihood ratio is the max(all tissues' likelihoods)/min(all tissues' likelihoods).
#' @param emMaxIterations EM algorithm maximum iteration number. Default is 100.
#'
#' @return an output file.
#'
#' @examples
#' ## input files
#' demo.dir <- system.file("extdata", package="cfTools")
#' readsBinningFile <- file.path(demo.dir, "example.reads.txt")
#' tissueMarkersFile <- file.path(demo.dir, "example.markers.txt")
#' outputType <- "tissueFraction+readCountPerBillion"
#' outputFile <- file.path(demo.dir, "example.profile")
#' numTissues <- 7
#' emAlgorithmType <- "em.global.unknown"
#' likelihoodRatioThreshold <- 2
#'
#' cfSort(readsBinningFile, tissueMarkersFile, outputType, outputFile,
#' numTissues, emAlgorithmType, likelihoodRatioThreshold)
#'
#' @export
cfSort <- function(readsBinningFile, tissueMarkersFile, outputType="tissueFraction", outputFile, numTissues,
                   emAlgorithmType="em.global.unknown", likelihoodRatioThreshold=2, emMaxIterations=100) {

  read_deconvolution_cpp(readsBinningFile, numTissues, likelihoodRatioThreshold, tissueMarkersFile,
                         emAlgorithmType, outputFile, outputType, emMaxIterations)

}
