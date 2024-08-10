test_that("cfDeconvolve() works", {
  
    demo.dir <- system.file("data", package="cfTools")
    readsBinningFile <- file.path(demo.dir, "cfDeconvolve.reads.txt.gz")
    tissueMarkersFile <- file.path(demo.dir, "cfDeconvolve.markers.txt.gz")
    numTissues <- 7
    emAlgorithmType <- "em.global.unknown"
    likelihoodRatioThreshold <- 2
    
    result_cfDeconvolve <- cfDeconvolve(readsBinningFile, 
                                        tissueMarkersFile, 
                                        numTissues, emAlgorithmType, 
                                        likelihoodRatioThreshold)
    
    expect_type(result_cfDeconvolve, "list")
})