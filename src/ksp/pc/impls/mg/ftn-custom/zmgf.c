#include <petsc-private/fortranimpl.h>
#include <petscpc.h>
#include <petscpcmg.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define pcmgsetlevels_             PCMGSETLEVELS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define pcmgsetlevels_             pcmgsetlevels
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL pcmgsetlevels_(PC *pc,PetscInt *levels,MPI_Comm *comms, PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(comms);
  *ierr = PCMGSetLevels(*pc,*levels,comms);
}

EXTERN_C_END
