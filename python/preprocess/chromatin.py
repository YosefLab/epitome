# This is where we can call code for loading reads in from ADAM
# this loads in datasets we will be using and joins them by
# reference region for featurization


from bdgenomics.adam.adamContext import ADAMContext


class Dataset(object):

    def __init__(self, sc):
        """
        Initializes a CoverageDistribution class.
        Computes the coverage distribution of multiple coverageRDDs.
        :param SparkContext
        """
        self.sc = sc
        self.ac = ADAMContext(sc)


    # matches peaks in peak file with reads
    def pairPeaksWithReads(self, peakFilename, chromatinFilename):
        reads = self.loadChromatin(chromatinFilename)
        peaks = self.loadPeaks(peakFilename)

        # left shuffle join reads with peaks
        #grouped = peaks._jvmRdd.shuffleRegionJoinAndGroupByLeft(reads._jvmRdd)

        #return grouped


    # loads chromatin from ATAC/DNase-seq
    # returns AlignmentRecordRDD
    def loadChromatin(self, filename):
        reads = self.ac.loadAlignments(filename)
        return reads


    # loads called peaks
    # returns FeatureRDD
    def loadPeaks(self, filename):
        peaks = self.ac.loadFeatures(filename)
        return peaks
