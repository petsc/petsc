# -----------------------------------------------------------------------------

cdef extern from * nogil:

    struct _p_PetscFwk
    ctypedef _p_PetscFwk *PetscFwk
    int PetscFwkCall(PetscFwk, char[])
    int PetscFwkGetURL(PetscFwk, char**)
    int PetscFwkSetURL(PetscFwk, char[])
    #
    int PetscFwkCreate(MPI_Comm,PetscFwk*)
    int PetscFwkView(PetscFwk,PetscViewer)
    int PetscFwkRegisterComponent(PetscFwk,char[])
    int PetscFwkRegisterComponentURL(PetscFwk,char[],char[])
    int PetscFwkRegisterDependence(PetscFwk,char[],char[])
    int PetscFwkGetComponent(PetscFwk,char[],PetscFwk*,PetscBool*)
    int PetscFwkGetParent(PetscFwk,PetscFwk*)
    int PetscFwkVisit(PetscFwk, char[])
    int PetscFwkDestroy(PetscFwk*)
    PetscFwk PETSC_FWK_DEFAULT_(MPI_Comm)

# -----------------------------------------------------------------------------
