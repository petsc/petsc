#include <petsc/private/ftnimpl.h>
#include <petsc/private/dmdaimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define dmdagetownershipranges_     DMDAGETOWNERSHIPRANGES
  #define dmdarestoreownershipranges_ DMDARESTOREOWNERSHIPRANGES
  #define dmdagetneighbors_           DMDAGETNEIGHBORS
  #define dmdarestoreneighbors_       DMDARESTORENEIGHBORS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define dmdagetownershipranges_     dmdagetownershipranges
  #define dmdarestoreownershipranges_ dmdarestoreownershipranges
  #define dmdagetneighbors_           dmdagetneighbors
  #define dmdarestoreneighbors_       dmdarestoreneighbors
#endif

PETSC_EXTERN void dmdagetneighbors_(DM *da, F90Array1d *ptr, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  const PetscMPIInt *r;
  PetscInt           n, dim;

  *ierr = DMDAGetNeighbors(*da, &r);
  if (*ierr) return;
  *ierr = DMGetDimension(*da, &dim);
  if (*ierr) return;
  if (dim == 2) n = 9;
  else n = 27;
  *ierr = F90Array1dCreate((PetscInt *)r, MPI_INT, 1, n, ptr PETSC_F90_2PTR_PARAM(ptrd));
}

PETSC_EXTERN void dmdarestoreneighbors_(DM *da, F90Array1d *ptr, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  *ierr = F90Array1dDestroy(ptr, MPI_INT PETSC_F90_2PTR_PARAM(ptrd));
}

PETSC_EXTERN void dmdagetownershipranges_(DM *da, F90Array1d *lx, F90Array1d *ly, F90Array1d *lz, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(lxd) PETSC_F90_2PTR_PROTO(lyd) PETSC_F90_2PTR_PROTO(lzd))
{
  const PetscInt *gx, *gy, *gz;
  PetscInt        M, N, P;

  *ierr = DMDAGetInfo(*da, NULL, NULL, NULL, NULL, &M, &N, &P, NULL, NULL, NULL, NULL, NULL, NULL);
  if (*ierr) return;
  *ierr = DMDAGetOwnershipRanges(*da, &gx, &gy, &gz);
  if (*ierr) return;
  if ((void *)lx != PETSC_NULL_INTEGER_POINTER_Fortran) {
    *ierr = F90Array1dCreate((PetscInt *)gx, MPIU_INT, 1, M, lx PETSC_F90_2PTR_PARAM(lxd));
    if (*ierr) return;
  }
  if ((void *)ly != PETSC_NULL_INTEGER_POINTER_Fortran) {
    *ierr = F90Array1dCreate((PetscInt *)gy, MPIU_INT, 1, N, ly PETSC_F90_2PTR_PARAM(lyd));
    if (*ierr) return;
  }
  if ((void *)lz != PETSC_NULL_INTEGER_POINTER_Fortran) *ierr = F90Array1dCreate((PetscInt *)gz, MPIU_INT, 1, P, lz PETSC_F90_2PTR_PARAM(lzd));
}

PETSC_EXTERN void dmdarestoreownershipranges_(DM *da, F90Array1d *lx, F90Array1d *ly, F90Array1d *lz, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(lxd) PETSC_F90_2PTR_PROTO(lyd) PETSC_F90_2PTR_PROTO(lzd))
{
  if ((void *)lx != PETSC_NULL_INTEGER_POINTER_Fortran) {
    *ierr = F90Array1dDestroy(lx, MPIU_INT PETSC_F90_2PTR_PARAM(lxd));
    if (*ierr) return;
  }
  if ((void *)ly != PETSC_NULL_INTEGER_POINTER_Fortran) {
    *ierr = F90Array1dDestroy(ly, MPIU_INT PETSC_F90_2PTR_PARAM(lyd));
    if (*ierr) return;
  }
  if ((void *)lz != PETSC_NULL_INTEGER_POINTER_Fortran) *ierr = F90Array1dDestroy(lz, MPIU_INT PETSC_F90_2PTR_PARAM(lzd));
}
