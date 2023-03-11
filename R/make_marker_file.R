
paraEstMoM <- function(meths) {
  mu = mean(meths,na.rm = TRUE)
  var = var(meths,na.rm = TRUE)
  mu[mu == 0] = 1e-5
  mu[mu == 1] = 1-1e-5
  var[var == 0] = 1e-9
  momAlpha = round(-mu*(var+mu*mu-mu)/var, digits = 3)
  momBeta = round((mu-1)*(var+mu*mu-mu)/var, digits = 3)
  return(c('shape1'=momAlpha, 'shape2'=momBeta))
}

#' @title
#' Generate the methylation pattern of markers
#'
#' @description
#' Output paired shape parameters of a beta distribution for markers.
#'
#' @param x a data frame of beta values, where each row is a sample and each column is a marker.
#' @param sample.types a vector of sample types corresponding to the rows of the data frame.
#' @param marker.indexes a vector of marker indexes corresponding to the columns of the data frame.
#' @param output.file a character string naming the output file. Default is "", which means the output will not be written into a file.
#' 
#' @return a data frame containing the paired shape parameters of a beta distribution for markers and/or written to an output file.
#' 
#' @examples
#' ## input files
#' demo.dir <- system.file("extdata", package="cfTools")
#' x <- read.csv(file.path(demo.dir, "beta_matrix.csv"), row.names=1)
#' sample.types <- read.csv(file.path(demo.dir, "sample_type.csv"), row.names=1)$Sample.Type
#' marker.indexes <- read.csv(file.path(demo.dir, "marker_index.csv"), row.names=1)$Marker.Index
#'
#' output <- generateMarkerFile(x, sample.types, marker.indexes)
#' 
#' @export
generateMarkerFile <- function(x, sample.types, marker.indexes, output.file="") {
  
  colnames(x) <- marker.indexes
  typeTrain <- split(x, sample.types) #split based on sample types
  
  typePara <- list()
  for (type in names(typeTrain)) {
    typePara[[type]] <- do.call('cbind', lapply(as.list(typeTrain[[type]]), paraEstMoM))
  }
  
  alphas_betas = do.call(cbind, lapply(typePara, function(x) paste0(x[1,], ":", x[2,])))
  alphas_betas_index <- as.data.frame(cbind(marker.indexes, alphas_betas))
  colnames(alphas_betas_index) <- c("marker.index", colnames(alphas_betas))
  
  if (output.file != "") {
    write.table(alphas_betas_index, output.file, sep='\t', row.names=FALSE, col.names=TRUE, quote=FALSE)
  }
  
  return(alphas_betas_index)
}


