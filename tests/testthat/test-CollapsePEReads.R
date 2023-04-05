test_that("CollapseCpGs() works", {
  
  demo.dir <- system.file("data", package="cfTools")
  PEReads <- file.path(demo.dir, "demo.sorted.bed.txt.gz")
  
  result_CollapsePEReads <- CollapsePEReads(PEReads)
  
  expect_type(result_CollapsePEReads, "list")
})