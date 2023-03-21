# cfTools

cfTools is an R package for cell-free DNA (cfDNA) methylation data analysis, including (1) cancer detection: sensitively detect tumor-derived cfDNA and estimate the tumor-derived cfDNA fraction (tumor burden); (2) tissue deconvolution: infer the tissue type composition and the cfDNA fraction of multiple tissue types for a plasma cfDNA sample.

## Introduction

Given the methylation sequencing data of a cfDNA sample, for each cancer marker or tissue marker, we deconvolve the tumor-derived or tissue-specific reads from all reads falling in the marker region. Our read-based deconvolution algorithm exploits the pervasiveness of DNA methylation for signal enhancement, therefore can sensitively identify a trace amount of tumor-specific or tissue-specific cfDNA in plasma. 

Specifically, `cfTools` can deconvolve different sources of cfDNA fragments (or reads) in two contexts:

1. Cancer detection: separate cfDNA fragments into tumor-derived fragments and background normal fragments (2 classes), and estimate the tumor-derived cfDNA fraction $\theta$ ($0\leq \theta < 1$).

2. Tissue deconvolution: separate cfDNA fragments from different tissues (> 2 classes), and estimate the cfDNA fraction $\theta_t$ ($0\leq \theta_t < 1$) of different tissue types (including an unknown type) $t$ for a plasma cfDNA sample.

These functions can serve as foundations for more advanced cfDNA-based studies, including cancer diagnosis and disease monitoring.

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
