#include <petsc/private/ftnimpl.h>
#include <petscdmda.h>

#if PetscDefined(HAVE_FORTRAN_CAPS)
  #define dmdagetprocessorsubset_  DMDAGETPROCESSORSUBSET
  #define dmdagetprocessorsubsets_ DMDAGETPROCESSORSUBSETS
#elif !PetscDefined(HAVE_FORTRAN_UNDERSCORE)
  #define dmdagetprocessorsubset_  dmdagetprocessorsubset
  #define dmdagetprocessorsubsets_ dmdagetprocessorsubsets
#endif

PETSC_EXTERN void dmdagetprocessorsubset_(DM *da, DMDirection *dir, PetscInt *gp, MPI_Fint *fcomm, int *ierr)
{
  MPI_Comm comm;
  *ierr  = DMDAGetProcessorSubset(*da, *dir, *gp, &comm);
  *fcomm = MPI_Comm_c2f(comm);
}
PETSC_EXTERN void dmdagetprocessorsubsets_(DM *da, DMDirection *dir, MPI_Fint *subfcomm, int *ierr)
{
  MPI_Comm subcomm;
  *ierr     = DMDAGetProcessorSubsets(*da, *dir, &subcomm);
  *subfcomm = MPI_Comm_c2f(subcomm);
}
