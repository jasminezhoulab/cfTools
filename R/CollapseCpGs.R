#' @title
#' Generate fragment-level methylation states of CpGs
#'
#' @description
#' Collapse the methylation states of all CpGs corresponding to the 
#' same fragment onto one line in output.
#'
#' @param CpG_OT a file of methylation information for CpG on the 
#' original top strand (OT), 
#' which is one of the outputs from `bismark methylation extractor`.
#' @param CpG_OB a file of methylation information for CpG on the 
#' original bottom strand (OB), 
#' which is one of the outputs from `bismark methylation extractor`.
#' @param output.dir a path to the output directory. Default is "", 
#' which means the output will not be written into a file.
#' @param id an ID name for the input data. Default is "", 
#' which means the output will not be written into a file.
#' @param python a path to Python 3. Default is "python3".
#' 
#' @return a list in BED file format and/or written to 
#' an output BED file.
#'
#' @examples
#' ## input files
#' demo.dir <- system.file("extdata", package="cfTools")
#' CpG_OT <- file.path(demo.dir, "CpG_OT_demo.txt.gz")
#' CpG_OB <- file.path(demo.dir, "CpG_OB_demo.txt.gz")
#'
#' output <- CollapseCpGs(CpG_OT, CpG_OB)
#'
#' @export
CollapseCpGs <- function(CpG_OT, CpG_OB, output.dir="", id="", 
                        python="python3") {

    # options(scipen = 999)

    python.script.dir <- system.file("python", package = "cfTools", 
                                    mustWork = TRUE)
    hasOutput <- TRUE
    
    if (output.dir=="" | id=="") {
        hasOutput <- FALSE
        extdata.dir <- system.file("extdata", package = "cfTools", 
                                    mustWork = TRUE)
        output.dir <- extdata.dir
        
        timeNow <- strsplit(strsplit(as.character(Sys.time()), 
                                    " ")[[1]][2], ":")[[1]]
        id <- paste0(timeNow[1], timeNow[2], timeNow[3])
        
        # output.dir <- paste0(extdata.dir, "/tmp/")
        # if (system.file("extdata/tmp", package = "cfTools") == "") {
        #   system2(command = "mkdir", args = output.dir)
        # }
        # id <- strsplit(as.character(Sys.time()), " ")[[1]][2]
    }
    
    py2 <- paste0(python.script.dir, "/collapse_CpG.py")
    refo_meth <- file.path(output.dir, paste0(id, ".refo_meth.bed"))
    py2.command <- paste(py2, CpG_OT, CpG_OB, refo_meth)
    system2(command = python, args = py2.command)
    
    output_bed <- read.csv(refo_meth, sep="\t", header = FALSE, 
                            colClasses = "character")
    output_bed <- as.data.frame(output_bed[order(output_bed$V8),])
    # nums<- sapply(output_bed, is.numeric)
    # output_bed$V7 <- format(output_bed$V7, scientific=F)
    # output_bed[ , nums] <- as.data.frame(apply(output_bed[ , nums], 
    #                                            2, as.character))
    rownames(output_bed) <- NULL
    colnames(output_bed) <- c("chr", "cpgStart", "cpgEnd", "strand", 
                            "cpgNumber", "cpgPosition", "methState", "name")
    
    write.table(output_bed, refo_meth, sep="\t", row.names=FALSE, 
                quote = FALSE)
    if (!hasOutput) {
        file.remove(refo_meth)
    }
    
    return(output_bed)
}
