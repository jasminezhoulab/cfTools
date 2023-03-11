library(GenomicRanges)
library(data.table)
demo.dir <- system.file("extdata", package="cfTools")
# demo.dir <- "/Users/huran/Desktop/cfTools_development/cfTools_new/inst/extdata"
frag_bed <- fread(file.path(demo.dir, "demo.fragment_level.meth.bed"))
frag_meth.gr <- GRanges(seqnames=frag_bed$chr, 
                     ranges=IRanges(frag_bed$start, frag_bed$end),
                     strand=frag_bed$strand,
                     methStatus=frag_bed$methStatus)
markers_bed <- fread(file.path(demo.dir, "markers.bed"))
markers.gr <- GRanges(seqnames=markers_bed$chr, 
                      ranges=IRanges(markers_bed$start, markers_bed$end),
                      marker_index=markers_bed$marker_index)

## This file format can be changed, or change to a standard BED format
selected.frag_meth.gr <- subsetByOverlaps(frag_meth.gr, markers.gr, ignore.strand=TRUE)
write.table(selected.frag_meth.gr, file.path(demo.dir, "selected.fragment_level.meth.bed"), 
            sep = "\t", row.names = FALSE, quote = FALSE)