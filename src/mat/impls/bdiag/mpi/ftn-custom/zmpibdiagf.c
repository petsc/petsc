#include "zpetsc.h"
#include "petscmat.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define matcreatempibdiag_               MATCREATEMPIBDIAG
#define matmpibdiagsetpreallocation_     MATMPIBDIAGSETPREALLOCATION
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define matcreatempibdiag_               matcreatempibdiag
#define matmpibdiagsetpreallocation_     matmpibdiagsetpreallocation
#endif

EXTERN_C_BEGIN

/* Fortran ignores diagv */
void PETSC_STDCALL matcreatempibdiag_(MPI_Comm *comm,PetscInt *m,PetscInt *M,PetscInt *N,PetscInt *nd,PetscInt *bs,
                        PetscInt *diag,PetscScalar **diagv,Mat *newmat,PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(diag);
  *ierr = MatCreateMPIBDiag((MPI_Comm)PetscToPointerComm(*comm),
                              *m,*M,*N,*nd,*bs,diag,PETSC_NULL,newmat);
}

/* Fortran ignores diagv */
void PETSC_STDCALL matmpibdiagsetpreallocation_(Mat *mat,PetscInt *nd,PetscInt *bs,PetscInt *diag,PetscScalar **diagv,PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(diag);
  *ierr = MatMPIBDiagSetPreallocation(*mat,*nd,*bs,diag,PETSC_NULL);
}


EXTERN_C_END
