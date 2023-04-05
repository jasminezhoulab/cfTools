
paraEstMoM <- function(meths) {
    mu <- mean(meths,na.rm = TRUE)
    var <- var(meths,na.rm = TRUE)
    mu[mu == 0] <- 1e-5
    mu[mu == 1] <- 1-1e-5
    var[var == 0] <- 1e-9
    momAlpha <- round(-mu*(var+mu*mu-mu)/var, digits = 3)
    momBeta <- round((mu-1)*(var+mu*mu-mu)/var, digits = 3)
    return(c('shape1'=momAlpha, 'shape2'=momBeta))
}

#' @title
#' Generate the methylation pattern of markers
#'
#' @description
#' Output paired shape parameters of beta distributions for methylation markers.
#'
#' @param x a list of methylation levels (e.g., beta values), 
#' where each row is a sample and each column is a marker.
#' @param sample.types a vector of sample types (e.g., tumor or normal, 
#' tissue types) corresponding to the rows of the list.
#' @param marker.names a vector of marker names corresponding to the 
#' columns of the list.
#' @param output.file a character string naming the output file. 
#' Default is "", which means the output will not be written into a file.
#' 
#' @return a list containing the paired shape parameters of 
#' beta distributions for markers and/or written to an output file.
#' 
#' @examples
#' ## input files
#' demo.dir <- system.file("data", package="cfTools")
#' methLevel <- read.table(file.path(demo.dir, "beta_matrix.txt"), 
#' row.names=1, header = TRUE)
#' sampleTypes <- read.table(file.path(demo.dir, "sample_type.txt"), 
#' row.names=1, header = TRUE)$sampleType
#' markerNames <- read.table(file.path(demo.dir, "marker_index.txt"), 
#' row.names=1, header = TRUE)$markerIndex
#'
#' output <- GenerateMarkerParam(methLevel, sampleTypes, markerNames)
#' 
#' @export
GenerateMarkerParam <- function(x, sample.types, marker.names, 
                                output.file="") {

    colnames(x) <- marker.names
    typeTrain <- split(x, sample.types) #split based on sample types
    
    typePara <- list()
    for (type in names(typeTrain)) {
        typePara[[type]] <- do.call('cbind', lapply(as.list(typeTrain[[type]]), 
                                    paraEstMoM))
    }
    
    alphas_betas <- do.call(cbind, lapply(typePara, 
                            function(x) paste0(x[1,], ":", x[2,])))
    alphas_betas_index <- as.data.frame(cbind(marker.names, alphas_betas))
    colnames(alphas_betas_index) <- c("markerName", colnames(alphas_betas))
    
    if (output.file != "") {
        write.table(alphas_betas_index, output.file, sep='\t', row.names=FALSE, 
                    col.names=TRUE, quote=FALSE)
    }
    
    return(alphas_betas_index)
}


