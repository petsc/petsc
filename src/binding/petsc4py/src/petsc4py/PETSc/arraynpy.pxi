# --------------------------------------------------------------------

cdef extern from "<petsc4py/numpy.h>":

    int import_array "_import_array" () except -1

    ctypedef long npy_intp

    ctypedef extern class numpy.dtype [object PyArray_Descr]:
        pass

    ctypedef extern class numpy.ndarray [object PyArrayObject]:
        pass

    void*     PyArray_DATA(ndarray)
    npy_intp  PyArray_SIZE(ndarray)
    int       PyArray_NDIM(ndarray)
    npy_intp* PyArray_DIMS(ndarray)
    npy_intp  PyArray_DIM(ndarray, int)

    enum: NPY_INTP
    dtype   PyArray_DescrFromType(int)
    object  PyArray_TypeObjectFromType(int)

    enum: NPY_ARRAY_ALIGNED
    enum: NPY_ARRAY_WRITEABLE
    enum: NPY_ARRAY_NOTSWAPPED
    enum: NPY_ARRAY_CARRAY
    enum: NPY_ARRAY_FARRAY

    ndarray PyArray_FROM_O(object)
    ndarray PyArray_FROM_OT(object,int)
    ndarray PyArray_FROM_OTF(object,int,int)

    ndarray PyArray_Copy(ndarray)
    ndarray PyArray_ArangeObj(object,object,object,dtype)
    ndarray PyArray_EMPTY(int,npy_intp[],int,int)
    ndarray PyArray_ZEROS(int,npy_intp[],int,int)

    bint PyArray_ISCONTIGUOUS(ndarray)
    bint PyArray_ISFORTRAN(ndarray)
    ctypedef enum NPY_ORDER:
        NPY_ANYORDER
        NPY_CORDER
        NPY_FORTRANORDER
    ndarray PyArray_NewCopy(ndarray,NPY_ORDER)

    ctypedef struct PyObject
    ctypedef struct PyTypeObject
    ndarray PyArray_New(PyTypeObject*,int,npy_intp[],int,npy_intp[],void*,int,int,PyObject*)
    ndarray PyArray_SimpleNewFromData(int,npy_intp[],int,void*)


cdef extern from "<petsc4py/numpy.h>":

    enum: NPY_INT
    enum: NPY_DOUBLE

    enum: NPY_PETSC_BOOL
    enum: NPY_PETSC_INT
    enum: NPY_PETSC_REAL
    enum: NPY_PETSC_SCALAR
    enum: NPY_PETSC_COMPLEX


# --------------------------------------------------------------------

cdef inline ndarray asarray(object ob):
    return PyArray_FROM_O(ob)

cdef inline ndarray arange(start, stop, stride):
    cdef dtype descr = <dtype> PyArray_DescrFromType(NPY_PETSC_INT)
    return PyArray_ArangeObj(start, stop, stride, descr)

# --------------------------------------------------------------------

cdef inline ndarray empty_i(PetscInt size):
    cdef npy_intp s = <npy_intp> size
    return PyArray_EMPTY(1, &s, NPY_PETSC_INT, 0)

cdef inline ndarray empty_r(PetscInt size):
    cdef npy_intp s = <npy_intp> size
    return PyArray_EMPTY(1, &s, NPY_PETSC_REAL, 0)

cdef inline ndarray empty_s(PetscInt size):
    cdef npy_intp s = <npy_intp> size
    return PyArray_EMPTY(1, &s, NPY_PETSC_SCALAR, 0)

cdef inline ndarray empty_c(PetscInt size):
    cdef npy_intp s = <npy_intp> size
    return PyArray_EMPTY(1, &s, NPY_PETSC_COMPLEX, 0)

cdef inline ndarray empty_p(PetscInt size):
    cdef npy_intp s = <npy_intp> size
    return PyArray_EMPTY(1, &s, NPY_INTP, 0)

# --------------------------------------------------------------------

cdef inline ndarray array_i(PetscInt size, const PetscInt* data):
    cdef npy_intp s = <npy_intp> size
    cdef ndarray ary = PyArray_EMPTY(1, &s, NPY_PETSC_INT, 0)
    if data != NULL:
        memcpy(PyArray_DATA(ary), data, <size_t>size*sizeof(PetscInt))
    return ary

cdef inline ndarray array_r(PetscInt size, const PetscReal* data):
    cdef npy_intp s = <npy_intp> size
    cdef ndarray ary = PyArray_EMPTY(1, &s, NPY_PETSC_REAL, 0)
    if data != NULL:
        memcpy(PyArray_DATA(ary), data, <size_t>size*sizeof(PetscReal))
    return ary

cdef inline ndarray array_b(PetscInt size, const PetscBool* data):
    cdef npy_intp s = <npy_intp> size
    cdef ndarray ary = PyArray_EMPTY(1, &s, NPY_PETSC_BOOL, 0)
    if data != NULL:
        memcpy(PyArray_DATA(ary), data, <size_t>size*sizeof(PetscBool))
    return ary

cdef inline ndarray array_s(PetscInt size, const PetscScalar* data):
    cdef npy_intp s = <npy_intp> size
    cdef ndarray ary = PyArray_EMPTY(1, &s, NPY_PETSC_SCALAR, 0)
    if data != NULL:
        memcpy(PyArray_DATA(ary), data, <size_t>size*sizeof(PetscScalar))
    return ary

# --------------------------------------------------------------------

cdef inline ndarray iarray(object ob, int typenum):
    cdef ndarray ary = PyArray_FROM_OTF(
        ob, typenum, NPY_ARRAY_ALIGNED|NPY_ARRAY_NOTSWAPPED)
    if PyArray_ISCONTIGUOUS(ary): return ary
    if PyArray_ISFORTRAN(ary):    return ary
    return PyArray_Copy(ary)

cdef inline ndarray iarray_i(object ob, PetscInt* size, PetscInt** data):
    cdef ndarray ary = iarray(ob, NPY_PETSC_INT)
    if size != NULL: size[0] = <PetscInt>  PyArray_SIZE(ary)
    if data != NULL: data[0] = <PetscInt*> PyArray_DATA(ary)
    return ary

cdef inline ndarray iarray_r(object ob, PetscInt* size, PetscReal** data):
    cdef ndarray ary = iarray(ob, NPY_PETSC_REAL)
    if size != NULL: size[0] = <PetscInt>   PyArray_SIZE(ary)
    if data != NULL: data[0] = <PetscReal*> PyArray_DATA(ary)
    return ary

cdef inline ndarray iarray_b(object ob, PetscInt* size, PetscBool** data):
    cdef ndarray ary = iarray(ob, NPY_PETSC_BOOL)
    if size != NULL: size[0] = <PetscInt>   PyArray_SIZE(ary)
    if data != NULL: data[0] = <PetscBool*> PyArray_DATA(ary)
    return ary

cdef inline ndarray iarray_s(object ob, PetscInt* size, PetscScalar** data):
    cdef ndarray ary = iarray(ob, NPY_PETSC_SCALAR)
    if size != NULL: size[0] = <PetscInt>     PyArray_SIZE(ary)
    if data != NULL: data[0] = <PetscScalar*> PyArray_DATA(ary)
    return ary

# --------------------------------------------------------------------

cdef inline ndarray oarray(object ob, int typenum):
    cdef ndarray ary = PyArray_FROM_OTF(
        ob, typenum, NPY_ARRAY_ALIGNED|NPY_ARRAY_WRITEABLE|NPY_ARRAY_NOTSWAPPED)
    if PyArray_ISCONTIGUOUS(ary): return ary
    if PyArray_ISFORTRAN(ary):    return ary
    return PyArray_Copy(ary)

cdef inline ndarray oarray_i(object ob, PetscInt* size, PetscInt** data):
    cdef ndarray ary = oarray(ob, NPY_PETSC_INT)
    if size != NULL: size[0] = <PetscInt>  PyArray_SIZE(ary)
    if data != NULL: data[0] = <PetscInt*> PyArray_DATA(ary)
    return ary

cdef inline ndarray oarray_r(object ob, PetscInt* size, PetscReal** data):
    cdef ndarray ary = oarray(ob, NPY_PETSC_REAL)
    if size != NULL: size[0] = <PetscInt>   PyArray_SIZE(ary)
    if data != NULL: data[0] = <PetscReal*> PyArray_DATA(ary)
    return ary

cdef inline ndarray oarray_s(object ob, PetscInt* size, PetscScalar** data):
    cdef ndarray ary = oarray(ob, NPY_PETSC_SCALAR)
    if size != NULL: size[0] = <PetscInt>     PyArray_SIZE(ary)
    if data != NULL: data[0] = <PetscScalar*> PyArray_DATA(ary)
    return ary

cdef inline ndarray oarray_p(object ob, PetscInt* size, void** data):
    cdef ndarray ary = oarray(ob, NPY_INTP)
    if size != NULL: size[0] = <PetscInt> PyArray_SIZE(ary)
    if data != NULL: data[0] = <void*>    PyArray_DATA(ary)
    return ary

# --------------------------------------------------------------------

cdef inline ndarray ocarray_s(object ob, PetscInt* size, PetscScalar** data):
    cdef ndarray ary = PyArray_FROM_OTF(
        ob, NPY_PETSC_SCALAR, NPY_ARRAY_CARRAY|NPY_ARRAY_NOTSWAPPED)
    if size != NULL: size[0] = <PetscInt>     PyArray_SIZE(ary)
    if data != NULL: data[0] = <PetscScalar*> PyArray_DATA(ary)
    return ary

cdef inline ndarray ofarray_s(object ob, PetscInt* size, PetscScalar** data):
    cdef ndarray ary = PyArray_FROM_OTF(
        ob, NPY_PETSC_SCALAR, NPY_ARRAY_FARRAY|NPY_ARRAY_NOTSWAPPED)
    if size != NULL: size[0] = <PetscInt>     PyArray_SIZE(ary)
    if data != NULL: data[0] = <PetscScalar*> PyArray_DATA(ary)
    return ary

# --------------------------------------------------------------------
