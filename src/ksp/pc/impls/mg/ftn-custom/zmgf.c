#include <petsc/private/fortranimpl.h>
#include <petscpc.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define pcmgsetlevels_             PCMGSETLEVELS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define pcmgsetlevels_             pcmgsetlevels
#endif

PETSC_EXTERN void pcmgsetlevels_(PC *pc,PetscInt *levels,MPI_Fint fcomms[], PetscErrorCode *ierr)
{
  MPI_Comm *ccomms = NULL;
  CHKFORTRANNULLMPICOMM(fcomms);
  if (fcomms) {
    *ierr = PetscMalloc1(*levels,&ccomms);if (*ierr) return;
    for (PetscInt i=0; i<*levels; i++) {
      ccomms[i] = MPI_Comm_f2c(fcomms[i]);
    }
  }
  *ierr = PCMGSetLevels(*pc,*levels,ccomms);if (*ierr) return;
  if (fcomms) *ierr = PetscFree(ccomms);
}

