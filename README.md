![cellxpedite](img/cellxpedite_logo.png "CELLXPEDITE")

Cloud-Based Platform for Profiling Neuronal Activity from Fluorescence Imaging Data

## Background
Fluorescence microscopy enables the study of neuronal activity with single-cell resolution. 
A single field of view can simultaneously capture the activity of several hundreds of neurons.
High-throughput drug screening that rely on such imaging technique allow scientists to quickly
record the effect of thousands of candidate compounds on diseased cells, with the goal of 
restoring a healthy phenotype or alleviating perturbations induced by the disease.
This technique generates a massive amount of data, and the analysis of such data can quickly exceed 
the computational resources available, creating a bottleneck that directly impact scientific progress.

## Platform Description
CELLXPEDITE was developed with the goal of scaling fluorescence imaging data analysis 
for high-throughput drug screening. Drug screens often involve several multi-well plates, each
associated with their own data containing hundreds of cells. 
This platform enables parallel analysis of plates by assigning separate computing nodes using Amazon Web Services. 
Wells can then be processed in parallel using the available cores on each instance. 
This parallelization scheme can theoretically scale to any number of plates without
increasing processing time beyond the overhead required to spin up the additional computing instances.

## Usage
Instructions for setting up CELLXPEDITE are available on our [wiki page](https://github.com/brunoboivin/cellxpedite/wiki/Setup). 
The platform was originally designed for the purpose of studying the activity of motor neurons imaged under confocal microscopy,
specifically screening for compounds able to revert the perturbances in neuronal activity induced by the SOD1
mutation associated with amyotrophic lateral sclerosis.
We encourage and welcome contributions from the community to extend the use cases of the platform.

## Publication
The development of this tool is the result of a collaboration
between Clifford Woolf's lab at Boston Children's Hospital,
Anne Carpenter's lab at the Broad Institute of Harvard and MIT, 
and Kevin Eggan's lab at Harvard University.
