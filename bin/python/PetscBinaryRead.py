import numpy as np
import types

IntType    = '>i4'   # big-endian, 4 byte integer
ScalarType = '>f8'   # big-endian, 8 byte real floating
_classid = {1211216:'Mat',
            1211214:'Vec',
            1211218:'IS',
            1211219:'Bag'}

class DoneWithFile(Exception): pass

def readVec(fh):
    """Reads a PETSc Vec from a binary file handle, returning just the data."""

    nz = np.fromfile(fh, dtype=IntType, count=1)[0]
    try:
        vals = np.fromfile(fh, dtype=ScalarType, count=nz)
    except MemoryError:
        raise IOError('Inconsistent or invalid Vec data in file')
    return vals

def readMatSparse(fh):
    """Reads a PETSc Mat, returning a sparse representation of the data.

    (M,N), (I,J,V) = readMatSparse(fid)

    Input:
      fid : file handle to open binary file.
    Output:
      M,N : matrix size
      I,J : arrays of row and column for each nonzero
      V: nonzero value
    """
    try:
        M,N,nz = np.fromfile(fh, dtype=IntType, count=3)
        #
        I = np.empty(M+1, dtype=IntType)
        I[0] = 0
        rownz = np.fromfile(fh, dtype=IntType, count=M)
        np.cumsum(rownz, out=I[1:])
        assert I[-1] == nz
        #
        J = np.fromfile(fh, dtype=IntType,    count=nz)
        V = np.fromfile(fh, dtype=ScalarType, count=nz)
    except AssertionError, MemoryError:
        raise IOError('Inconsistent or invalid Mat data in file')
    #
    return (M, N), (I, J, V)

def readMatDense(fh):
    """Reads a PETSc Mat, returning a dense represention of the data."""
    try:
        M,N,nz = np.fromfile(fh, dtype=IntType, count=3)
        #
        I = np.empty(M+1, dtype=IntType)
        I[0] = 0
        rownz = np.fromfile(fh, dtype=IntType, count=M)
        np.cumsum(rownz, out=I[1:])
        assert I[-1] == nz
        #
        J = np.fromfile(fh, dtype=IntType,    count=nz)
        V = np.fromfile(fh, dtype=ScalarType, count=nz)
        mat = np.zeros((M,N), dtype=ScalarType)
        for i,j,v in zip(I,J,V):
            mat[i,j] = v
    except AssertionError, MemoryError:
        raise IOError('Inconsistent or invalid Mat data in file')
    #
    return mat

def readMat(fh, mattype='sparse'):
    """Reads a PETSc Mat from binary file handle.

    optional mattype: 'sparse" or 'dense'

    See also: readMatSparse, readMatDense
    """
    if mattype == 'sparse':
        return readMatSparse(fh)
    elif mattype == 'dense':
        return readMatDense(fh)
    else:
        raise RuntimeError('Invalid matrix type requested: choose sparse/dense')

def readIS(fh):
    """Reads a PETSc Index Set from binary file handle."""
    try:
        nz = np.fromfile(fh, dtype=IntType, count=1)[0]
        v = np.fromfile(fh, dtype=IntType, count=nz)
    except MemoryError:
        raise IOError('Inconsistent or invalid IS data in file')
    return v

def readBinaryFile(fid, mattype='sparse'):
    """Reads a PETSc binary file, returning a tuple of objects contained in the file.

    objects = readBinaryFile(fid, mattype='sparse')

    Input:
      fid : either file handle to open binary file, or filename
      mattype : ['sparse'] Read matrices as 'sparse' (row, col, val) or 'dense'.

    Output:
      objects : tuple of objects representing the data.
    """
    close = False

    if type(fid) is types.StringType:
        fid = open(fid, 'rb')
        close = True

    objects = []
    try:
        while True:
            # read header
            try:
                header = np.fromfile(fid, dtype=IntType, count=1)[0]
            except MemoryError:
                raise DoneWithFile
            try:
                objecttype = _classid[header]
            except KeyError:
                raise IOError('Invalid PetscObject CLASSID or CLASSID not yet implemented')

            if objecttype == 'Vec':
                objects.append(('Vec', readVec(fid)))
            elif objecttype == 'IS':
                objects.append(('IS', readIS(fid)))
            elif objecttype == 'Mat':
                objects.append(('Mat', readMat(fid,mattype)))
            elif objecttype == 'Bag':
                raise NotImplementedError('Bag Reader not yet implemented')
    except DoneWithFile:
        pass
    finally:
        if close:
            fid.close()
    #
    return tuple(objects)

if __name__ == '__main__':
    import sys
    objects = readBinaryFile(sys.argv[1])
    for otype, data in objects:
        print 'Read a', otype
        print data
        print '\n'


