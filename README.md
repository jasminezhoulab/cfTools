# cfTools

cfTools is a versatile toolset of deconvoluting the cell-free DNA fragments (or sequencing reads) into multiple tissue types or tumor types.

## Introduction

Given the methylation sequencing data of a patient's cell-free DNA (cfDNA) sample, for each cancer marker (or tissue marker), we deconvolve the tumor-derived (or tissue-specific) reads from all reads falling in the marker region. The read-based deconvolution algorithm exploits the pervasiveness of DNA methylation for signal enhancement. Specifically, we generalized upon our previous probabilistic read deconvolution algorithm, i.e., CancerDetector, by expanding the 2-class likelihood model to a T-class (T>=2 model, for classifying reads into T different classes, and for a given set of markers, we construct a profile vector where the length of the vector is the number of markers and the value in each entry is the normalized counts of tumor-derived (or tissue-derived) reads. This code can deconvolute the different sources of cfDNA reads in two contexts: (1) separating reads into tumor-derived reads and background reads; and (2) separating the reads from different tissues. This algorithm has been applied to two studies for cancer detection and tissue mapping. 

## Installation

`cfTools` is an `R` package available via the [Bioconductor](http://bioconductor.org) repository for packages. You can install the release version by using the following commands in your `R` session:

```
if (!requireNamespace("BiocManager", quietly = TRUE)) {
    install.packages("BiocManager")
}
BiocManager::install("cfTools")
```

Alternatively, you can install the development version from [GitHub](https://github.com/) :

```
BiocManager::install("jasminezhoulab/cfTools")
```

## Vignettes
See the detailed documentation for `cfTools` using the following commands in your `R` session
```
browseVignettes("cfTools")
```
