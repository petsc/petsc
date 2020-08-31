cdef extern from * nogil:

    enum: PETSC_DECIDE
    enum: PETSC_DEFAULT
    enum: PETSC_DETERMINE

    PetscReal PETSC_INFINITY
    PetscReal PETSC_NINFINITY

    ctypedef enum PetscBool:
        PETSC_FALSE
        PETSC_TRUE

    ctypedef enum PetscInsertMode "InsertMode":
        PETSC_NOT_SET_VALUES    "NOT_SET_VALUES"
        PETSC_INSERT_VALUES     "INSERT_VALUES"
        PETSC_ADD_VALUES        "ADD_VALUES"
        PETSC_MAX_VALUES        "MAX_VALUES"
        PETSC_INSERT_ALL_VALUES "INSERT_ALL_VALUES"
        PETSC_ADD_ALL_VALUES    "ADD_ALL_VALUES"
        PETSC_INSERT_BC_VALUES  "INSERT_BC_VALUES"
        PETSC_ADD_BC_VALUES     "ADD_BC_VALUES"

    ctypedef enum PetscScatterMode "ScatterMode":
        PETSC_SCATTER_FORWARD       "SCATTER_FORWARD"
        PETSC_SCATTER_REVERSE       "SCATTER_REVERSE"
        PETSC_SCATTER_FORWARD_LOCAL "SCATTER_FORWARD_LOCAL"
        PETSC_SCATTER_REVERSE_LOCAL "SCATTER_REVERSE_LOCAL"
        PETSC_SCATTER_LOCAL         "SCATTER_LOCAL"

    ctypedef enum  PetscNormType "NormType":
        PETSC_NORM_1          "NORM_1"
        PETSC_NORM_2          "NORM_2"
        PETSC_NORM_1_AND_2    "NORM_1_AND_2"
        PETSC_NORM_FROBENIUS  "NORM_FROBENIUS"
        PETSC_NORM_INFINITY   "NORM_INFINITY"
        PETSC_NORM_MAX        "NORM_MAX"

    ctypedef enum PetscCopyMode:
        PETSC_COPY_VALUES
        PETSC_OWN_POINTER
        PETSC_USE_POINTER
    

cdef extern from * nogil:

    enum: PETSC_ERR_MEM
    enum: PETSC_ERR_SUP
    enum: PETSC_ERR_ORDER
    enum: PETSC_ERR_LIB
    enum: PETSC_ERR_USER
    enum: PETSC_ERR_SYS


cdef inline PetscInsertMode insertmode(object mode) \
    except <PetscInsertMode>(-1):
    if   mode is None:  return PETSC_INSERT_VALUES
    elif mode is True:  return PETSC_ADD_VALUES
    elif mode is False: return PETSC_INSERT_VALUES
    else:               return mode

cdef inline PetscScatterMode scattermode(object mode) \
    except <PetscScatterMode>(-1):
    if mode is None:  return PETSC_SCATTER_FORWARD
    if mode is False: return PETSC_SCATTER_FORWARD
    if mode is True:  return PETSC_SCATTER_REVERSE
    if isinstance(mode, str):
        if mode == 'forward': return PETSC_SCATTER_FORWARD
        if mode == 'reverse': return PETSC_SCATTER_REVERSE
        else: raise ValueError("unknown scatter mode: %s" % mode)
    return mode

