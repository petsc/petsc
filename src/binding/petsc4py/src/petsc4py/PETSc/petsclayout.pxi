# --------------------------------------------------------------------

cdef extern from * nogil:

    struct _n_PetscLayout
    ctypedef _n_PetscLayout* PetscLayout
    PetscErrorCode PetscLayoutSetLocalSize(PetscLayout,PetscInt)
    PetscErrorCode PetscLayoutSetSize(PetscLayout,PetscInt)
    PetscErrorCode PetscLayoutGetBlockSize(PetscLayout,PetscInt*)
    PetscErrorCode PetscLayoutSetBlockSize(PetscLayout,PetscInt)
    PetscErrorCode PetscLayoutSetUp(PetscLayout)

# --------------------------------------------------------------------
