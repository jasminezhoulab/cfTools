#' @title
#' Generate fragment-level information about methylation states
#'
#' @description
#' Join two lists containing the fragment information and the 
#' methylation states on each fragment into one list.
#'
#' @param frag_bed a BED file containing information for every fragment, 
#' which is the output of CollapsePEReads().
#' @param meth_bed a BED file containing methylation states on every fragment, 
#' which is the output of CollapseCpGs().
#' @param output.dir a path to the output directory. Default is "", 
#' which means the output will not be written into a file.
#' @param id an ID name for the input data. Default is "", 
#' which means the output will not be written into a file.
#' 
#' @return a list in BED file format and/or written to 
#' an output BED file.
#'
#' @examples
#' ## input files
#' demo.dir <- system.file("extdata", package="cfTools")
#' frag_bed <- read.delim(file.path(demo.dir, "demo.refo_frag.bed"), 
#' colClasses = "character")
#' meth_bed <- read.delim(file.path(demo.dir, "demo.refo_meth.bed"), 
#' colClasses = "character")
#'
#' output <- GenerateFragMeth(frag_bed, meth_bed)
#'
#' @export
GenerateFragMeth <- function(frag_bed, meth_bed, output.dir="", 
                            id="") {

    # python.script.dir <- system.file("python", package = "cfTools", 
    #                                 mustWork = TRUE)
    hasOutput <- TRUE
    
    if (output.dir=="" | id=="") {
        hasOutput <- FALSE
        extdata.dir <- system.file("extdata", package = "cfTools", 
                                    mustWork = TRUE)
        output.dir <- extdata.dir
        
        timeNow <- strsplit(strsplit(as.character(Sys.time()), 
                                    " ")[[1]][2], ":")[[1]]
        id <- paste0(timeNow[1], timeNow[2], timeNow[3])
        
        # output.dir <- paste0(extdata.dir, "/tmp")
        # if (system.file("extdata/tmp", package = "cfTools") == "") {
        #   system2(command = "mkdir", args = output.dir)
        # }
    }
    
    fragment_level.meth <- file.path(output.dir, 
                                    paste0(id, ".fragment_level.meth.bed"))

    merged_file <- merge(x=frag_bed, y=meth_bed, by = "name")
    output_bed <- as.data.frame(cbind(merged_file$chr.x, merged_file$start, 
                                merged_file$end, merged_file$name, 
                                merged_file$fragmentLength, 
                                merged_file$strand.x, merged_file$cpgNumber,
                                merged_file$cpgPosition, 
                                merged_file$methState))
    colnames(output_bed) <- c("chr", "start", "end", "name", "fragmentLength", 
                            "strand", "cpgNumber", "cpgPosition", "methState")
    
    output_bed <- output_bed[with(output_bed, 
                                order(chr, start)), ] #sort by chr and start
    rownames(output_bed) <- NULL
    
    write.table(output_bed, fragment_level.meth, sep="\t", row.names=FALSE, 
                col.names=TRUE, quote=FALSE)
    if (!hasOutput) {
        # system2(command = "rm", args = fragment_level.meth)
        file.remove(fragment_level.meth)
    }
    
    return(output_bed)
}
