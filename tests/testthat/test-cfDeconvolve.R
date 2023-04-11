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
    
    expect_equal(result_cfDeconvolve$unknown, 0.00955414)
    expect_equal(result_cfDeconvolve$tissue7, 0.930494)
    expect_equal(result_cfDeconvolve$tissue6, 4.00821e-18)
    expect_equal(result_cfDeconvolve$tissue5, 0.0157226)
    expect_equal(result_cfDeconvolve$tissue4, 1.45249e-21)
    expect_equal(result_cfDeconvolve$tissue3, 0.0442296)
    expect_equal(result_cfDeconvolve$tissue2, 1.35298e-19)
    expect_equal(result_cfDeconvolve$tissue1, 1.58246e-13)
})