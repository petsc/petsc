#include <petsc/private/ftnimpl.h>
#include <petscds.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define petscdsgettabulationsetsizes_    PETSCDSGETTABULATIONSETSIZES
  #define petscdsgettabulationsetpointers_ PETSCDSGETTABULATIONSETPOINTERS
  #define f90arraysetrealpointer_          F90ARRAYSETREALPOINTER
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define petscdsgettabulationsetsizes_    petscdsgettabulationsetsizes
  #define petscdsgettabulationsetpointers_ petscdsgettabulationsetpointers
  #define f90arraysetrealpointer_          f90arraysetrealpointer
#endif

PETSC_EXTERN void f90arraysetrealpointer_(const PetscReal *, PetscInt *, PetscInt *, F90Array1d *PETSC_F90_2PTR_PROTO_NOVAR);

typedef struct {
  PetscInt K;
  PetscInt Nr;
  PetscInt Np;
  PetscInt Nb;
  PetscInt Nc;
  PetscInt cdim;
} PetscTabulationFtn;

PETSC_EXTERN void petscdsgettabulationsetsizes_(PetscDS *ds, PetscInt *i, PetscTabulationFtn *tftn, PetscErrorCode *ierr)
{
  PetscTabulation *tab;

  *ierr = PetscDSGetTabulation(*ds, &tab);
  if (*ierr) return;
  *ierr = PetscMemcpy(tftn, tab[*i - 1], sizeof(PetscTabulationFtn));
}

PETSC_EXTERN void petscdsgettabulationsetpointers_(PetscDS *ds, PetscInt *i, F90Array1d *ptrB, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrb))
{
  PetscTabulation *tab;
  PetscInt         size;

  *ierr = PetscDSGetTabulation(*ds, &tab);
  if (*ierr) return;
  size = tab[*i - 1]->Nr * tab[*i - 1]->Np * tab[*i - 1]->Nb * tab[*i - 1]->Nc;

  for (PetscInt j = 0; j <= tab[*i - 1]->K; j++) {
    f90arraysetrealpointer_(tab[*i - 1]->T[j], &size, &j, ptrB PETSC_F90_2PTR_PARAM(ptrb));
    if (*ierr) return;
    size *= tab[*i - 1]->cdim;
  }
}
