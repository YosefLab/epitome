# used for preprocessing data in python

# used for loading 2bit files
import py2bit
import pybedtools

# windows genome into regions, where the size of each regions is defined by windowSize
# returns list of pybedtools.Interval
def window(twobitFile, windowSize = 1000):
    tb = py2bit.open(twobitFile)        
    chroms = tb.chroms()

    windows = []
    for chr, length in chroms.iteritems():
        if "_" not in chr: # filter out all random and gl chromsomes
            start = 0
            while (start < length):
                windows.append(pybedtools.Interval(chr, start, start + windowSize))
                start += windowSize

    return pybedtools.BedTool(windows)
    
    