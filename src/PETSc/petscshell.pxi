# -----------------------------------------------------------------------------

cdef extern from * nogil:

    struct _p_PetscShell
    ctypedef _p_PetscShell *PetscShell
    int PetscShellCall(PetscShell, char[])
    int PetscShellGetURL(PetscShell, char**)
    int PetscShellSetURL(PetscShell, char[])
    #
    int PetscShellCreate(MPI_Comm,PetscShell*)
    int PetscShellView(PetscShell,PetscViewer)
    int PetscShellRegisterComponentShell(PetscShell,char[],PetscShell)
    int PetscShellRegisterComponentURL(PetscShell,char[],char[])
    int PetscShellRegisterDependence(PetscShell,char[],char[])
    int PetscShellGetComponent(PetscShell,char[],PetscShell*,PetscBool*)
    int PetscShellGetVisitor(PetscShell,PetscShell*)
    int PetscShellVisit(PetscShell, char[])
    int PetscShellDestroy(PetscShell*)
    PetscShell PETSC_SHELL_DEFAULT_(MPI_Comm)

# -----------------------------------------------------------------------------
