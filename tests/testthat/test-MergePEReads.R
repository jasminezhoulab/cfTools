test_that("MergeCpGs() works", {
  
  demo.dir <- system.file("data", package="cfTools")
  PEReads <- file.path(demo.dir, "demo.sorted.bed.txt.gz")
  
  result_MergePEReads <- MergePEReads(PEReads)
  
  expect_type(result_MergePEReads, "list")
})