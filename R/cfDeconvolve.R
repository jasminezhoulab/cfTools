count_meth_unmeth <- function(methString) {
  methVector <- as.numeric(strsplit(methString, "")[[1]])
  meth_count <- sum(methVector)
  unmeth_count <- length(methVector) - meth_count
  return(c(meth_count, unmeth_count))
}

#' @title
#' cfDNA methylation read deconvolution
#'
#' @description
#' Infer the tissue-type composition of plasma cfDNA.
#'
#' @param readsBinningFile a file of the fragment-level methylation states of reads that mapped to the markers. Either in plain text or compressed form.
#' @param tissueMarkersFile a file of paired shape parameters of beta distributions for markers.
#' @param numTissues a number of tissue types.
#' @param emAlgorithmType a read-based tissue deconvolution EM algorithm type: em.global.unknown (default), em.global.known, em.local.unknown, em.local.known.
#' @param likelihoodRatioThreshold a positive float number. Default is 2. 
#' @param emMaxIterations a number of EM algorithm maximum iteration. Default is 100.
#'
#' @return a data frame containing the cfDNA fractions of different tissue types and an unknown class.
#'
#' @examples
#' ## input files
#' demo.dir <- system.file("extdata", package="cfTools")
#' readsBinningFile <- file.path(demo.dir, "cfDeconvolve.reads.txt")
#' tissueMarkersFile <- file.path(demo.dir, "cfDeconvolve.markers.txt")
#' numTissues <- 7
#' emAlgorithmType <- "em.global.unknown"
#' likelihoodRatioThreshold <- 2
#'
#' cfDeconvolve(readsBinningFile, tissueMarkersFile, numTissues, 
#' emAlgorithmType, likelihoodRatioThreshold)
#'
#' @export
cfDeconvolve <- function(readsBinningFile, tissueMarkersFile, numTissues,
                   emAlgorithmType="em.global.unknown", likelihoodRatioThreshold=2, emMaxIterations=100) {

  extdata.dir <- system.file("extdata", package = "cfTools", mustWork = TRUE)
  output.dir <- extdata.dir
  
  timeNow <- strsplit(strsplit(as.character(Sys.time()), " ")[[1]][2], ":")[[1]]
  id <- paste0(timeNow[1], timeNow[2], timeNow[3])
  
  # output.dir <- paste0(extdata.dir, "/tmp/")
  # if (system.file("extdata/tmp", package = "cfTools") == "") {
  #   system2(command = "mkdir", args = output.dir)
  # }
  # id <- strsplit(as.character(Sys.time()), " ")[[1]][2]
  
  readsBinningFile.count <- file.path(output.dir, paste0(id, ".reads_count.txt"))
  readsBinning <- read.csv(readsBinningFile, header=TRUE, sep="\t", colClasses = "character")
  colnames(readsBinning) <- c("markerName", "methState")
  meth_unmeth_count <- lapply(readsBinning$methState, count_meth_unmeth)
  readsBinningFile.input <- cbind(readsBinning$markerName, data.frame(t(vapply(meth_unmeth_count, c, numeric(2)))))
  colnames(readsBinningFile.input) <- c("marker_index", "meth_count", "unmeth_count")
  write.table(readsBinningFile.input, readsBinningFile.count, sep="\t", row.names=FALSE, quote=FALSE)
  
  outputFile <- file.path(output.dir, paste0(id, ".profile"))
  
  read_deconvolution_cpp(readsBinningFile.count, numTissues, likelihoodRatioThreshold, tissueMarkersFile,
                         emAlgorithmType, outputFile, "tissueFraction", emMaxIterations)

  output <- read.csv(outputFile, header=TRUE, sep="\t")
  # system2(command = "rm", args = outputFile)
  # system2(command = "rm", args = readsBinningFile.count)
  
  file.remove(outputFile)
  file.remove(readsBinningFile.count)
  
  return(output)
  
}