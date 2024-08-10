test_that("cfSort() works", {

    demo.dir <- system.file("data", package="cfTools")
    readsBinningFile <- file.path(demo.dir, "cfsort_reads.txt.gz")
    id <- "test"
    
    output <- cfSort(readsBinningFile, id)
    
    expect_equal(output$adipose_tissue, 0)
    expect_equal(output$WBC, 1)
})