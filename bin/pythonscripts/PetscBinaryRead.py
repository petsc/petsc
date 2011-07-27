"""PetscBinaryRead
===============

Provides
  1. PETSc-named objects Vec, Mat, and IS that inherit numpy.ndarray
  2. A function to read these objects from PETSc binary files.

The standard usage of this module should look like:

  >>> import PetscBinaryRead
  >>> objects = PetscBinaryRead.readBinaryFile('filename')

See readBinaryFile.__doc__
"""

import numpy as np
import types

IntType    = '>i4'   # big-endian, 4 byte integer
ScalarType = '>f8'   # big-endian, 8 byte real floating
_classid = {1211216:'Mat',
            1211214:'Vec',
            1211218:'IS',
            1211219:'Bag'}

class DoneWithFile(Exception): pass

class Vec(np.ndarray):
    """Vec represented as 1D numpy array"""
    pass

class MatDense(np.matrix):
    """Mat represented as 2D numpy array"""
    pass

class MatSparse(tuple):
    """Mat represented as CSR tuple ((M, N), (row, col, val))"""
    def __repr__(self):
        return 'MatSparse: %s'%super(MatSparse, self).__repr__()

class IS(np.ndarray):
    """IS represented as 1D numpy array"""
    pass

def readVec(fh):
    """Reads a PETSc Vec from a binary file handle, returning just the data."""

    nz = np.fromfile(fh, dtype=IntType, count=1)[0]
    try:
        vals = np.fromfile(fh, dtype=ScalarType, count=nz)
    except MemoryError:
        raise IOError('Inconsistent or invalid Vec data in file')
    if (len(vals) is 0):
        raise IOError('Inconsistent or invalid Vec data in file')
    return vals.view(Vec)

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
        assert len(J) == nz
        V = np.fromfile(fh, dtype=ScalarType, count=nz)
        assert len(V) == nz
    except (AssertionError, MemoryError, IndexError):
        raise IOError('Inconsistent or invalid Mat data in file')
    #
    return MatSparse(((M, N), (I, J, V)))

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
        J = np.fromfile(fh, dtype=IntType, count=nz)
        assert len(J) == nz
        V = np.fromfile(fh, dtype=ScalarType, count=nz)
        assert len(V) == nz

    except (AssertionError, MemoryError, IndexError):
        raise IOError('Inconsistent or invalid Mat data in file')
    #
    mat = np.zeros((M,N), dtype=ScalarType)
    for row in range(M):
        rstart, rend = I[row:row+2]
        mat[row, J[rstart:rend]] = V[rstart:rend]
    return mat.view(MatDense)

def readMatSciPy(fh):
    from scipy.sparse import csr_matrix
    (M, N), (I, J, V) = readMatSparse(fh)
    return csr_matrix((V, J, I), shape=(M, N))

def readMat(fh, mattype='sparse'):
    """Reads a PETSc Mat from binary file handle.

    optional mattype: 'sparse" or 'dense'

    See also: readMatSparse, readMatDense
    """
    if mattype == 'sparse':
        return readMatSparse(fh)
    elif mattype == 'dense':
        return readMatDense(fh)
    elif mattype == 'scipy.sparse':
        return readMatSciPy(fh)
    else:
        raise RuntimeError('Invalid matrix type requested: choose sparse/dense')

def readIS(fh):
    """Reads a PETSc Index Set from binary file handle."""
    try:
        nz = np.fromfile(fh, dtype=IntType, count=1)[0]
        v = np.fromfile(fh, dtype=IntType, count=nz)
        assert len(v) == nz
    except (MemoryError,IndexError):
        raise IOError('Inconsistent or invalid IS data in file')
    return v.view(IS)

def readBinaryFile(fid, mattype='sparse'):
    """Reads a PETSc binary file, returning a tuple of the contained objects.

    objects = readBinaryFile(fid, mattype='sparse')

    Input:
      fid : either file handle to an open binary file, or filename.
      mattype :
         'sparse': Return matrices as raw CSR: (M, N), (row, col, val).
         'dense': Return matrices as MxN numpy arrays.
         'scipy.sparse': Return matrices as scipy.sparse objects.

    Output:
      objects : tuple of objects representing the data in numpy arrays.
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
            except (MemoryError, IndexError):
                raise DoneWithFile
            try:
                objecttype = _classid[header]
            except KeyError:
                raise IOError('Invalid PetscObject CLASSID or object not implemented for python')

            if objecttype == 'Vec':
                objects.append(readVec(fid))
            elif objecttype == 'IS':
                objects.append(readIS(fid))
            elif objecttype == 'Mat':
                objects.append(readMat(fid,mattype))
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
    petsc_objects = readBinaryFile(sys.argv[1])
    for petsc_obj in petsc_objects:
        print 'Read a', petsc_obj
        print ''


