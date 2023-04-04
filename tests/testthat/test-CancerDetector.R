test_that("CancerDetector() works", {

    demo.dir <- system.file("extdata", package="cfTools")
    readsBinningFile <- file.path(demo.dir, "CancerDetector.reads.txt")
    tissueMarkersFile <- file.path(demo.dir, "CancerDetector.markers.txt")
    
    result_CancerDetector <- CancerDetector(readsBinningFile, 
                                            tissueMarkersFile)
    
    expect_equal(result_CancerDetector$cfDNA_tumor_burden, 0.055)
    expect_equal(result_CancerDetector$normal_cfDNA_fraction, 0.945)
})