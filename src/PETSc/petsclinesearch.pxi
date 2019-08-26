cdef extern from * nogil:

    ctypedef char* PetscSNESLineSearchType "const char*"
    PetscSNESLineSearchType SNESLINESEARCHBT
    PetscSNESLineSearchType SNESLINESEARCHNLEQERR
    PetscSNESLineSearchType SNESLINESEARCHBASIC
    PetscSNESLineSearchType SNESLINESEARCHL2
    PetscSNESLineSearchType SNESLINESEARCHCP

    int SNESGetLineSearch(PetscSNES,PetscSNESLineSearch*)
    int SNESLineSearchSetFromOptions(PetscSNESLineSearch)
    int SNESLineSearchApply(PetscSNESLineSearch,PetscVec,PetscVec,PetscReal*,PetscVec)
    int SNESLineSearchDestroy(PetscSNESLineSearch*)

    ctypedef int (*PetscSNESPreCheckFunction)(PetscSNESLineSearch,
                                              PetscVec,PetscVec,
                                              PetscBool*,
                                              void*) except PETSC_ERR_PYTHON
    int SNESLineSearchSetPreCheck(PetscSNESLineSearch,PetscSNESPreCheckFunction,void*)
    int SNESLineSearchGetSNES(PetscSNESLineSearch,PetscSNES*)

