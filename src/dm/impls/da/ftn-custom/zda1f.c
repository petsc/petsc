
#include "private/fortranimpl.h"
#include "petscda.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dacreate1d_                  DACREATE1D
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define dacreate1d_                  dacreate1d
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL dacreate1d_(MPI_Comm *comm,DAPeriodicType *wrap,PetscInt *M,PetscInt *w,PetscInt *s,
                 PetscInt *lc,DA *inra,PetscErrorCode *ierr)
{
 CHKFORTRANNULLINTEGER(lc);
  *ierr = DACreate1d(MPI_Comm_f2c(*(MPI_Fint *)&*comm),*wrap,*M,*w,*s,lc,inra);
}
EXTERN_C_END

