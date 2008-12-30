# --------------------------------------------------------------------

cdef extern from "arraynpy.h":

    int import_numpy "_import_array" () except -1

    ctypedef long npy_intp

    ctypedef extern class numpy.dtype [object PyArray_Descr]:
        pass

    ctypedef extern class numpy.ndarray [object PyArrayObject]:
        cdef char*     cdata  "data"
        cdef int       cndim  "nd"
        cdef npy_intp* cshape "dimensions"

    enum: NPY_PETSC_INT
    enum: NPY_PETSC_REAL
    enum: NPY_PETSC_COMPLEX
    enum: NPY_PETSC_SCALAR

    enum: NPY_IN_ARRAY
    enum: NPY_OUT_ARRAY
    enum: NPY_INOUT_ARRAY

    enum: NPY_IN_FARRAY
    enum: NPY_OUT_FARRAY
    enum: NPY_INOUT_FARRAY

    dtype    PyArray_DescrFromType(int)
    object   PyArray_TypeObjectFromType(int)

    ndarray  PyArray_ArangeObj(object,object,object,dtype)
    ndarray  PyArray_EMPTY(int,npy_intp[],int,int)

    ndarray  PyArray_FROM_O(object)
    ndarray  PyArray_FROM_OTF(object,int,int)

    void*    PyArray_DATA(object)
    npy_intp PyArray_SIZE(object)


cdef extern from "arraynpy.h":
    object   numpy_typeobj"PyArray_TypeObjectFromType"(int)


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

# --------------------------------------------------------------------

cdef inline ndarray array_i(PetscInt size, const_PetscInt* data):
    cdef npy_intp s = <npy_intp> size
    cdef ndarray ary = PyArray_EMPTY(1, &s, NPY_PETSC_INT, 0)
    if data != NULL: memcpy(ary.cdata, data, size*sizeof(PetscInt))
    return ary

cdef inline ndarray array_r(PetscInt size, const_PetscReal* data):
    cdef npy_intp s = <npy_intp> size
    cdef ndarray ary = PyArray_EMPTY(1, &s, NPY_PETSC_REAL, 0)
    if data != NULL: memcpy(ary.cdata, data, size*sizeof(PetscReal))
    return ary

cdef inline ndarray array_s(PetscInt size, const_PetscScalar* data):
    cdef npy_intp s = <npy_intp> size
    cdef ndarray ary = PyArray_EMPTY(1, &s, NPY_PETSC_SCALAR, 0)
    if data != NULL: memcpy(ary.cdata, data, size*sizeof(PetscScalar))
    return ary

# --------------------------------------------------------------------

cdef inline ndarray iarray_i(object ob, PetscInt* size, PetscInt** data):
    ob = PyArray_FROM_OTF(ob, NPY_PETSC_INT, NPY_IN_ARRAY)
    cdef ndarray ary = <ndarray> ob
    if size!=NULL: size[0] = <PetscInt>  PyArray_SIZE(ary)
    if data!=NULL: data[0] = <PetscInt*> PyArray_DATA(ary)
    return ary

cdef inline ndarray iarray_s(object ob, PetscInt* size, PetscScalar** data):
    ob = PyArray_FROM_OTF(ob, NPY_PETSC_SCALAR, NPY_IN_ARRAY)
    cdef ndarray ary = <ndarray> ob
    if size!=NULL: size[0] = <PetscInt>     PyArray_SIZE(ary)
    if data!=NULL: data[0] = <PetscScalar*> PyArray_DATA(ary)
    return ary

# --------------------------------------------------------------------

cdef inline ndarray oarray_i(object ob, PetscInt* size, PetscInt** data):
    ob = PyArray_FROM_OTF(ob, NPY_PETSC_INT, NPY_OUT_ARRAY)
    cdef ndarray ary = <ndarray> ob
    if size!=NULL: size[0] = <PetscInt>  PyArray_SIZE(ary)
    if data!=NULL: data[0] = <PetscInt*> PyArray_DATA(ary)
    return ary

cdef inline ndarray oarray_r(object ob, PetscInt* size, PetscReal** data):
    ob = PyArray_FROM_OTF(ob, NPY_PETSC_REAL, NPY_OUT_ARRAY)
    cdef ndarray ary = <ndarray> ob
    if size!=NULL: size[0] = <PetscInt>   PyArray_SIZE(ary)
    if data!=NULL: data[0] = <PetscReal*> PyArray_DATA(ary)
    return ary

cdef inline ndarray oarray_s(object ob, PetscInt* size, PetscScalar** data):
    ob = PyArray_FROM_OTF(ob, NPY_PETSC_SCALAR, NPY_OUT_ARRAY)
    cdef ndarray ary = <ndarray> ob
    if size!=NULL: size[0] = <PetscInt>     PyArray_SIZE(ary)
    if data!=NULL: data[0] = <PetscScalar*> PyArray_DATA(ary)
    return ary

cdef inline ndarray ofarray_s(object ob, PetscInt* size, PetscScalar** data):
    ob = PyArray_FROM_OTF(ob, NPY_PETSC_SCALAR, NPY_OUT_FARRAY)
    cdef ndarray ary = <ndarray> ob
    if size!=NULL: size[0] = <PetscInt>     PyArray_SIZE(ary)
    if data!=NULL: data[0] = <PetscScalar*> PyArray_DATA(ary)
    return ary

# --------------------------------------------------------------------

cdef extern from "arraynpy.h":
    object PetscIS_array_struct(object,PetscIS)
    object PetscVec_array_struct(object,PetscVec)

# --------------------------------------------------------------------
