test_that("MergeCpGs() works", {
  
  demo.dir <- system.file("data", package="cfTools")
  CpG_OT <- file.path(demo.dir, "CpG_OT_demo.txt.gz")
  CpG_OB <- file.path(demo.dir, "CpG_OB_demo.txt.gz")
  
  result_MergeCpGs <- MergeCpGs(CpG_OT, CpG_OB)
  
  expect_type(result_MergeCpGs, "list")
})