import numpy as np

COOKIE     = 1211216 # from petscmat.h
IntType    = '>i4'   # big-endian, 4 byte integer
ScalarType = '>f8'   # big-endian, 8 byte real floating

def readmat(filename):
    fh = open(filename, 'rb')
    try:
        header = np.fromfile(fh, dtype=IntType, count=4)
        assert header[0] == COOKIE 
        M, N, nz = header[1:]
        #
        I = np.empty(M+1, dtype=IntType)
        I[0] = 0 
        rownz = np.fromfile(fh, dtype=IntType, count=M)
        np.cumsum(rownz, out=I[1:])
        assert I[-1] == nz
        #
        J = np.fromfile(fh, dtype=IntType,    count=nz)
        V = np.fromfile(fh, dtype=ScalarType, count=nz)
    finally:
        fh.close()
    #
    return (M, N), (I, J, V)

if __name__ == '__main__':
    import sys
    (M, N), (I, J, V) = readmat(sys.argv[1])
    for i in xrange(len(I)-1):
        start, end = I[i], I[i+1]
        colidx = J[start:end]
        values = V[start:end]
        print 'row %d:' % i ,  zip(colidx, values)
