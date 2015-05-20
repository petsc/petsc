#include <petsc/private/fortranimpl.h>
#include <petsc/private/dmdaimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dmdagetownershipranges_        DMDAGETOWNERSHIPRANGES
#define dmdagetneighbors_              DMDAGETNEIGHBORS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define dmdagetownershipranges_        dmdagetownershipranges
#define dmdagetneighbors_              dmdagetneighbors
#endif

PETSC_EXTERN void PETSC_STDCALL dmdagetneighbors_(DM *da,PetscMPIInt *ranks,PetscErrorCode *ierr)
{
  const PetscMPIInt *r;
  PetscInt          n,dim;

  *ierr = DMDAGetNeighbors(*da,&r);if (*ierr) return;
  *ierr = DMGetDimension(*da,&dim);if (*ierr) return;
  if (dim == 2) n = 9;
  else n = 27;
  *ierr = PetscMemcpy(ranks,r,n*sizeof(PetscMPIInt));
}

PETSC_EXTERN void PETSC_STDCALL dmdagetownershipranges_(DM *da,PetscInt lx[],PetscInt ly[],PetscInt lz[],PetscErrorCode *ierr)
{
  const PetscInt *gx,*gy,*gz;
  PetscInt       M,N,P,i;

  CHKFORTRANNULLINTEGER(lx);
  CHKFORTRANNULLINTEGER(ly);
  CHKFORTRANNULLINTEGER(lz);
  *ierr = DMDAGetInfo(*da,0,0,0,0,&M,&N,&P,0,0,0,0,0,0);if (*ierr) return;
  *ierr = DMDAGetOwnershipRanges(*da,&gx,&gy,&gz);if (*ierr) return;
  if (lx) {
    for (i=0; i<M; i++) lx[i] = gx[i];
  }
  if (ly) {
    for (i=0; i<N; i++) ly[i] = gy[i];
  }
  if (lz) {
    for (i=0; i<P; i++) lz[i] = gz[i];
  }
}

