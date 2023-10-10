#' @title
#' cfSort: tissue deconvolution 
#'
#' @description
#' Tissue deconvolution in cfDNA using DNN models.
#'
#' @param readsBinningFile a file of the fragment-level methylation states 
#' of reads that mapped to the cfSort markers. In compressed form.
#' @param id the sample ID.
#' 
#' @return the tissue composition of the cfDNA sample.
#' 
#' @examples
#' ## input files
#' demo.dir <- system.file("data", package="cfTools")
#' readsBinningFile <- file.path(demo.dir, "cfSort.reads.txt.gz")
#' id <- "test"
#'
#' cfSort(readsBinningFile, id)
#' 
#' @export
cfSort <- function(readsBinningFile, id="sample") {
  
  python.script.dir <- system.file("python", package = "cfTools", 
                                   mustWork = TRUE)
  
  extdata.dir <- system.file("data", package = "cfTools", mustWork = TRUE)
  cfSortMarkerFile <- file.path(extdata.dir, "cfsort_markers.txt.gz")

  # timeNow <- strsplit(strsplit(as.character(Sys.time()), 
  #                              " ")[[1]][2], ":")[[1]]
  # id <- paste0(id, timeNow[1], timeNow[2], timeNow[3])
  
  alpha_value_distr <- file.path(extdata.dir, paste0(id, ".alpha_value_distr.txt.gz"))
  mat <- file.path(extdata.dir, paste0(id, ".mat.txt.gz"))
  depthFile <- file.path(extdata.dir, paste0(id, ".depth.txt"))
  featuresFile <- file.path(extdata.dir, paste0(id, ".features.txt.gz"))
  tfrecords <- file.path(extdata.dir, paste0(id, ".tfrecords"))
  
  py1 <- paste0(python.script.dir, 
                "/convert_binary_meth_to_alpha_value_distribution.py")
  py1.command <- c(py1, readsBinningFile, alpha_value_distr)
  
  py2 <- paste0(python.script.dir, 
                "/generate_data_matrix_by_alpha_value_and_markers.py")
  py2.command <- c(py2, cfSortMarkerFile, alpha_value_distr, mat)
  
  proc <- basiliskStart(my_env)
  
  basiliskRun(proc, function() {
    system2(command = "python", args = py1.command)
    system2(command = "python", args = py2.command)
  })
  basiliskStop(proc)
  
  ### collect read depth at markers
  input_file <- alpha_value_distr
  output_file <- depthFile
  marker_file <- cfSortMarkerFile
  MAX_INDEX <- 1045098
  
  marker <- read.table(marker_file, header = TRUE)$marker_index
  dt <- matrix(NA, ncol = length(marker), nrow = 1)
  rownames(dt) <- input_file
  cur_x <- read.table(input_file, header = TRUE)
  cur_x <- cur_x[cur_x$marker_index %in% marker, ]
  cur_x <- cur_x[! duplicated(cur_x$marker_index),]
  rownames(cur_x) <- as.character(cur_x$marker_index)
  dt[input_file,] <- cur_x[as.character(marker), "num_read"]
  dt[is.na(dt)] <- 0
  dt <- data.frame(dt)
  write.table(dt, output_file, col.names = FALSE, row.names = TRUE, 
              quote = FALSE, sep = "\t")
  
  ## final step
  py3 <- paste0(python.script.dir, 
                "/normalize_local_read_depth_and_cluster_markers.py")
  py3.command <- c(py3, cfSortMarkerFile, mat, depthFile, featuresFile)
  
  norm_params <- paste0(python.script.dir, "/norm_params.npy")
  
  py4 <- paste0(python.script.dir, 
                "/data_prep.for_test_only.py")
  py4.command <- c(py4, featuresFile, tfrecords, norm_params)
  
  proc <- basiliskStart(my_env)
  
  basiliskRun(proc, function() {
    system2(command = "python", args = py3.command)
    system2(command = "python", args = py4.command)
  })
  basiliskStop(proc)
  
  file.remove(mat)
  file.remove(depthFile)
  file.remove(alpha_value_distr)
  file.remove(featuresFile)
  
  # return(tfrecords)
  
  ###### cfSort
  output.dir <- extdata.dir
  tissue.composition.1 <- file.path(output.dir, 
                                    paste0(id, ".tissue_composition.1.txt"))
  tissue.composition.2 <- file.path(output.dir, 
                                    paste0(id, ".tissue_composition.2.txt"))
  
  model_DNN1 <- DNN1()
  model_DNN2 <- DNN2()
  
  py5 <- paste0(python.script.dir, "/model_pred.py")
  py5.command <- c(py5, tfrecords, model_DNN1, tissue.composition.1)
  py6.command <- c(py5, tfrecords, model_DNN2, tissue.composition.2)
  
  proc <- basiliskStart(my_env)
  
  basiliskRun(proc, function() {
      system2(command = "python", args = py5.command)
      system2(command = "python", args = py6.command)
  })
  basiliskStop(proc)
  
  tissue_composition_1 <- read.csv(tissue.composition.1, 
                                   header = FALSE, sep = "\t")
  tissue_composition_2 <- read.csv(tissue.composition.2, 
                                   header = FALSE, sep = "\t")
  
  tissue_composition <- (tissue_composition_1 + tissue_composition_2)/2
  colnames(tissue_composition) <- c("adipose_tissue", "adrenal_gland", "bladder", 
                              "blood_vessel", "breast", "cervix_uteri", 
                              "colon", "esophagus", "fallopian_tube", "heart", 
                              "kidney", "liver", "lung", "muscle", "nerve", 
                              "ovary", "pancreas", "pituitary", "prostate", 
                              "salivary_gland", "skin", "small_intestine", 
                              "spleen", "stomach", "testis", "thyroid", 
                              "uterus", "vagina", "WBC")
  
  file.remove(tfrecords)
  file.remove(tissue.composition.1)
  file.remove(tissue.composition.2)

  return(tissue_composition)
}


