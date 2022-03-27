# cfTools

XXXXX

## Introduction

Given the cfMethyl-Seq data of a patient’s cfDNA sample, for each cancer marker (tissue marker), we deconvolve the tumor-derived (tissue-specific) reads from all reads falling in the marker region. The read-based deconvolution exploits the pervasiveness of DNA methylation for signal enhancement. Specifically, we improved upon our previous probabilistic read deconvolution algorithm, i.e., CancerDetector [1], by (1) adding an “unknown” class to represent reads that cannot be classified to any known class (tumor type or tissue type), and (2) expanding the 2-class likelihood model to a k-class (k>=2)  model, for classifying reads into k different classes, and (3) For a given set of markers, we construct a profile vector where the length of the vector is the number of markers and the value in each entry is the normalized counts of tumor-derived (tissue-derived) reads. 

Note that as described in the manuscript, this code can deconvolute the different sources of cfDNA reads in two contexts: (1) separating reads into tumor-derived reads and background reads; and (2) separating the reads from different tissues. 

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

## Citation
XXXXXXXX
