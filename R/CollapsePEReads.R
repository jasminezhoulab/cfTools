#' @title
#' Generate fragment level information for paired-end reads
#'
#' @description
#' Collapse BED file (the output of `bedtools bamtobed`) to 
#' fragment level for paired-end reads.
#'
#' @param bed_file a (sorted) BED file of paired-end reads.
#' @param output.dir a path to the output directory. Default is "", which means the output will not be written into a file.
#' @param id an ID name for the input data. Default is "", which means the output will not be written into a file.
#' @param python a path to Python 3. Default is "python".
#' 
#' @return a data frame in BED file format and/or written to an output BED file.
#'
#' @examples
#' ## input files
#' demo.dir <- system.file("extdata", package="cfTools")
#' bed_file <- file.path(demo.dir, "demo.sorted.bed.gz")
#'
#' output <- CollapsePEReads(bed_file)
#'
#' @export
CollapsePEReads <- function(bed_file, output.dir="", id="", python="python") {
  
  python.script.dir <- system.file("python", package = "cfTools", mustWork = TRUE)
  
  if (output.dir=="" | id=="") {
    extdata.dir <- system.file("extdata", package = "cfTools", mustWork = TRUE)
    output.dir <- paste0(extdata.dir, "/tmp/")
    if (system.file("extdata/tmp", package = "cfTools") == "") {
      system2(command = "mkdir", args = output.dir)
    }
    id <- strsplit(as.character(Sys.time()), " ")[[1]][2]
  }
  
  py1 <- paste0(python.script.dir, "/collapse_bed_file_strand_correct.py")
  refo_frag <- file.path(output.dir, paste0(id, ".refo_frag.bed"))
  py1.command <- paste(py1, bed_file, refo_frag)
  system2(command = python, args = py1.command)
  
  output_bed <- read.csv(refo_frag, sep="\t", header = FALSE)
  output_bed <- as.data.frame(output_bed[order(output_bed$V6),])
  rownames(output_bed) <- NULL
  colnames(output_bed) <- c("chr", "start", "end", "fragmentLength", "strand", "name")
  
  write.table(output_bed, refo_frag, sep="\t", row.names=FALSE, col.names=TRUE, quote=FALSE)
  if (output.dir=="" | id=="") {
    system2(command = "rm", args = refo_frag)
  }
  
  return(output_bed)
}
