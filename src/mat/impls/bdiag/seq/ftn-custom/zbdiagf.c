#include "zpetsc.h"
#include "petscmat.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define matcreateseqbdiag_               MATCREATESEQBDIAG
#define matseqbdiagsetpreallocation_     MATSEQBDIAGSETPREALLOCATION
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define matcreateseqbdiag_               matcreateseqbdiag
#define matseqbdiagsetpreallocation_     matseqbdiagsetpreallocation
#endif

EXTERN_C_BEGIN

/* Fortran ignores diagv */
void PETSC_STDCALL matcreateseqbdiag_(MPI_Comm *comm,PetscInt *m,PetscInt *n,PetscInt *nd,PetscInt *bs,
                        PetscInt *diag,PetscScalar **diagv,Mat *newmat,PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(diag);
  *ierr = MatCreateSeqBDiag((MPI_Comm)PetscToPointerComm(*comm),*m,*n,*nd,*bs,diag,
                               PETSC_NULL,newmat);
}

/* Fortran ignores diagv */
void PETSC_STDCALL matseqbdiagsetpreallocation_(Mat *mat,PetscInt *nd,PetscInt *bs,PetscInt *diag,PetscScalar **diagv,PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(diag);
  *ierr = MatSeqBDiagSetPreallocation(*mat,*nd,*bs,diag,PETSC_NULL);
}
EXTERN_C_END
