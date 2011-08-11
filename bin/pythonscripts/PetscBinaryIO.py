"""PetscBinaryIO
===============

Provides
  1. PETSc-named objects Vec, Mat, and IS that inherit numpy.ndarray
  2. Functions to read and write these objects from PETSc binary files.

The standard usage of this module should look like:

  >>> import PetscBinaryIO
  >>> objects = PetscBinaryIO.readBinaryFile('file.dat')

or

  >>> import PetscBinaryIO
  >>> import numpy
  >>> vec = numpy.array([1., 2., 3.]).view(PetscBinaryIO.Vec)
  >>> PetscBinaryIO.writeBinaryFile('file.dat', [vec,])

See also readBinaryFile.__doc__ and writeBinaryFile.__doc__ 
"""

import numpy as np
import types

IntType    = np.dtype('>i4')   # big-endian, 4 byte integer
ScalarType = np.dtype('>f8')   # big-endian, 8 byte real floating
_classid = {1211216:'Mat',
            1211214:'Vec',
            1211218:'IS',
            1211219:'Bag'}

class DoneWithFile(Exception): pass

class Vec(np.ndarray):
    """Vec represented as 1D numpy array

    The best way to instantiate this class for use with writeBinaryFile()
    is through the numpy view method:

    vec = numpy.array([1,2,3]).view(Vec)
    """
    _classid = 1211214

class MatDense(np.matrix):
    """Mat represented as 2D numpy array

    The best way to instantiate this class for use with writeBinaryFile()
    is through the numpy view method:

    mat = numpy.array([[1,0],[0,1]]).view(Mat)
    """
    _classid = 1211216

class MatSparse(tuple):
    """Mat represented as CSR tuple ((M, N), (rowindices, col, val))

    This should be instantiated from a tuple:

    mat = MatSparse( ((M,N), (rowindices,col,val)) )
    """
    _classid = 1211216
    def __repr__(self):
        return 'MatSparse: %s'%super(MatSparse, self).__repr__()

class IS(np.ndarray):
    """IS represented as 1D numpy array

    The best way to instantiate this class for use with writeBinaryFile()
    is through the numpy "view" method:

    an_is = numpy.array([3,4,5]).view(IS)
    """
    _classid = 1211218

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

def writeVec(fh, vec):
    """Writes a PETSc Vec to a binary file handle."""

    metadata = np.array([Vec._classid, len(vec)], dtype=IntType)
    metadata.tofile(fh)
    vec.astype(ScalarType).tofile(fh)
    return

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

def writeMatSparse(fh, mat):
    """Writes a Mat into a PETSc binary file handle"""

    ((M,N), (I,J,V)) = mat
    metadata = np.array([MatSparse._classid,M,N,I[-1]], dtype=IntType)
    rownz = I[1:] - I[:-1]

#    try:
    assert len(J.shape) == len(V.shape) == len(I.shape) == 1
    assert len(J) == len(V) == I[-1] == rownz.sum()
    assert (rownz > -1).all()

#    except AssertionError:
#        raise ValueError('Invalid Mat data given')

    metadata.tofile(fh)
    rownz.astype(IntType).tofile(fh)
    J.astype(IntType).tofile(fh)
    V.astype(ScalarType).tofile(fh)
    return

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

def writeMatSciPy(fh, mat):
    from scipy.sparse import csr_matrix
    assert isinstance(mat, csr_matrix)
    V = mat.data
    M,N = mat.shape
    J = mat.indices
    I = mat.indptr
    return writeMatSparse(fh, (mat.shape, (mat.indptr,mat.indices,mat.data)))

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

def writeIS(fh, anis):
    """Writes a PETSc IS to binary file handle."""

    metadata = np.array([IS._classid, len(anis)], dtype=IntType)
    metadata.tofile(fh)
    anis.astype(IntType).tofile(fh)
    return

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

def writeBinaryFile(fid, objects):
    """Writes a PETSc binary file containing the objects given.

    readBinaryFile(fid, objects)

    Input:
      fid : either file handle to an open binary file, or filename.
      objects : list of objects representing the data in numpy arrays,
                which must be of type Vec, IS, MatSparse, or MatSciPy.
    """
    close = False
    if type(fid) is types.StringType:
        fid = open(fid, 'wb')
        close = True

    for petscobj in objects:
        if (isinstance(petscobj, Vec)):
            writeVec(fid, petscobj)
        elif (isinstance(petscobj, IS)):
            writeIS(fid, petscobj)
        elif (isinstance(petscobj, MatSparse)):
            writeMatSparse(fid, petscobj)
        elif (isinstance(petscobj, MatDense)):
            if close:
                fid.close()
            raise NotImplementedError('Writing a dense matrix is not yet supported')
        else:
            try:
                writeMatSciPy(fid, petscobj)
            except AssertionError:
                if close:
                    fid.close()
                raise TypeError('Object %s is not a valid PETSc object'%(petscobj.__repr__()))
    if close:
        fid.close()
    return
