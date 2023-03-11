#' @title
#' Cancer Detector
#'
#' @description
#' Detect cancer
#'
#' @param readsBinningFile input cfDNA methylation sequencing reads file, either in plain text or compressed form.
#' @param tissueMarkersFile input markers file in plain text. Each marker for each tissue type has either a median beta value or a paired shape parameters of beta-distribution from the tissue population of tissue types.
#' @param output.dir a path to the output directory. Default is "", which means the output will not be written into a file.
#' @param id an ID name for the input data. Default is "", which means the output will not be written into a file.
#' @param python a path to Python 3. Default is "python".
#'
#' @return an output file.
#'
#' @examples
#' ## input files
#' demo.dir <- system.file("extdata", package="cfTools")
#' readsBinningFile <- file.path(demo.dir, "CancerDetector.reads.txt")
#' tissueMarkersFile <- file.path(demo.dir, "CancerDetector.markers.txt")
#' output.dir <- file.path(demo.dir)
#' id <- "CancerDetector"
#'
#' CancerDetector(readsBinningFile, tissueMarkersFile, output.dir, id)
#'
#' @export
CancerDetector <- function(readsBinningFile, tissueMarkersFile, output.dir, id, python="python") {

  lambda = 0.5 # a predefined lambda
  
  python.script.dir <- system.file("python", package = "cfTools", mustWork = TRUE)
  
  py1 <- paste0(python.script.dir, "/CalcReadLikelihood.py")
  py1.command <- paste(py1, readsBinningFile, tissueMarkersFile, output.dir, id)
  system2(command = python, args = py1.command)

  id.likelihood = file.path(output.dir, paste0(id, ".likelihood.txt"))
  
  py2 <- paste0(python.script.dir, "/CancerDetector.py")
  py2.command <- paste(py2, id.likelihood, lambda, output.dir, id)
  system2(command = python, args = py2.command)
  
  # id.tumor_burden = file.path(output.dir, paste0(id, ".tumor_burden.txt"))
}
