from __future__ import division
import numpy as np
import pandas as pd
import sys, os, csv
import skimage.io
from src.utils import metadataExtractor, cxpPrinter
from scipy.spatial import distance
import scipy.stats, scipy.cluster
from scipy.cluster.hierarchy import dendrogram
import mlpy
import matplotlib.pyplot as plt
plt.switch_backend('agg')


# euclidean distance between all pairs of points
def euclidean_dst(points):
    return distance.cdist(points,points)


# dynamic-time warping to measure distance between all pairs of signals
def dtw_dst(signals):
    distMatrix = np.zeros((len(signals),len(signals)))
    for i in range(len(signals)):
        for j in range(i,len(signals)):
            dist = mlpy.dtw_std(signals[i], signals[j])
            distMatrix[i][j] = distMatrix[j][i] = dist
    return distMatrix


# element-wise geometric mean of matrices a and b
def geometricMean(a,b):
    return scipy.stats.gmean([a, b], axis=0)


# combine distance metrics via element-wise geometric mean (assumes dst1.shape == dst2.shape)
def combineDistanceMetrics(dst1, dst2):
    return geometricMean(dst1,dst2)


# hierarchical clustering using mlpy
def hclust_mlpy(m,t=0.10):
    hc = mlpy.HCluster()
    hc.linkage(distance.squareform(m))
    clusters = hc.cut(t)
    return clusters


# hierarchical clustering using scipy (more flexible than mlpy, but slower)
def hclust_scipy(m,t=0.10):
    Z = scipy.cluster.hierarchy.linkage(distance.squareform(m),method='complete')  # linkage matrix
    clusters = scipy.cluster.hierarchy.fcluster(Z, t,criterion='distance')
    return clusters, Z


# converts hierarchical clusters into list of segments to merge as [[1,2],[3,5]...]
def listObjectsToMerge(clusters):
    d = {}
    for i in range(clusters.size):
        if clusters[i] in d:
            d[clusters[i]].append(i+1)
        else:
            d[clusters[i]] = [i+1]

    l = []
    for e in d.values():
        if len(e) > 1:
            l.append(e)
    return l


# returns list of segments to merge
def getFragmentsToCombine(well,segmentsdir,tseriesdir):
    hclust_threshold = 0.10

    # Step 1: Compute euclidean distance matrix between center points of all fragments
    # Get image number associated to well (to retrieve appropriate fragments)
    df_img = pd.read_csv(os.path.join(segmentsdir, "gcampsegmentation",'gcampsegmentation_Image.csv'))
    df_img = df_img[df_img['FileName_gcamp'].str.contains(well)]
    df_img = df_img[['ImageNumber']]
    imgNum = df_img.iloc[0].values[0]

    # Get objects and corresponding centers
    df = pd.read_csv(os.path.join(segmentsdir, "gcampsegmentation",'gcampsegmentation_neuronFragment_downsized.csv'))
    df = df[df['ImageNumber'] == imgNum]

    points = list(zip(df['Location_Center_X'], df['Location_Center_Y']))
    eucMatrix = euclidean_dst(points)

    # Step 2: Compute dynamic-time warping matrix for all possible pairs of signals
    with open(os.path.join(tseriesdir,'{0}_fragments_timeseries.csv'.format(well)), 'r') as f:
        reader = csv.reader(f,quoting=csv.QUOTE_NONNUMERIC)
        gcamp_signals = list(reader)
    dtwMatrix = dtw_dst(gcamp_signals)

    # Step 3: Combined distance metrics
    eucMatrix_norm = eucMatrix / np.mean(eucMatrix)
    dtwMatrix_norm = dtwMatrix / np.mean(dtwMatrix)
    distanceMatrix = combineDistanceMetrics(eucMatrix_norm, dtwMatrix_norm)

    # Step 4: Perform hierarchical clustering
    clusters, linkage_matrix = hclust_scipy(distanceMatrix,hclust_threshold)

    # save dendrogram
    fig = plt.figure(figsize=(25, 10))
    plt.title("Well {0} - Hierarchical Clustering Dendrogram".format(well))
    plt.xlabel('Object Index')
    plt.ylabel('Distance')
    dendrogram(
        linkage_matrix,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8.  # font size for the x axis labels
    )
    plt.axhline(y=hclust_threshold, color='black', linestyle='-')
    fig.savefig(os.path.join(segmentsdir, "gcampsegmentation", "{0}_dendrogram.png".format(well)))
    fig.clf()
    plt.close(fig)

    # Step 5: Convert to list of segments to merge
    segmentsToMerge = listObjectsToMerge(clusters)
    return segmentsToMerge


def mergeObjectSegments(config, well):
    cxpPrinter.cxpPrint('Merging segments in well {0}'.format(well))

    # parse config to get plate metadata
    metadata_dict = metadataExtractor.import_metadata(config)
    segmentsdir = metadata_dict["config"]["var"]["segmentsdir"]
    tseriesdir = metadata_dict["config"]["var"]["tseriesdir"]

    # Step 0: Get list of segments to merge
    clusters = getFragmentsToCombine(well,segmentsdir,tseriesdir)
    # save clusters to file
    with open(os.path.join(segmentsdir, "gcampsegmentation", "{0}_clusters.csv".format(well)), "w") as f:
        writer = csv.writer(f)
        writer.writerows(clusters)

    # Step 1: read original segments from image
    segments_img = skimage.io.imread(os.path.join(segmentsdir, "gcampsegmentation", "{0}_maxprojection_neuronFragments.tiff".format(well)))
    numObj_pre = len(np.unique(segments_img[segments_img > 0]))

    # Step 2: Relabel segments that are to be merged
    for cluster in clusters:
        newlabel = min(cluster)
        filter = np.in1d(segments_img, cluster).reshape(segments_img.shape)
        segments_img[filter] = newlabel

    # update labels to avoid discontinuities such as 1,2,3,8 if [4;7] were merged
    newIdx = 1
    for i in sorted(np.unique(segments_img[segments_img > 0])):
        segments_img[segments_img == i] = newIdx
        newIdx = newIdx + 1

    # Step 3: Save new segments image
    skimage.io.imsave(os.path.join(segmentsdir, "gcampsegmentation", "{0}_merged_neuronFragments.tiff".format(well)),segments_img)

    # save clustering statistics
    numObj_post = len(np.unique(segments_img[segments_img > 0]))
    numMergers = numObj_pre - numObj_post
    percentMergers = numMergers / numObj_pre
    with open(os.path.join(segmentsdir, "gcampsegmentation", "{0}_clustering_stats.txt".format(well)), "w") as f:
        f.write(str(numMergers) + ", " + str(percentMergers))


if __name__ == "__main__":
    mergeObjectSegments(*sys.argv[1:])
