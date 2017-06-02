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
