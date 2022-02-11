#include <petscdmda.h>
#include <adolc/adalloc.h>

/*
   REQUIRES configuration of PETSc with option --download-adolc.

   For documentation on ADOL-C, see
     $PETSC_ARCH/externalpackages/ADOL-C-2.6.0/ADOL-C/doc/adolc-manual.pdf
*/

/*
  Wrapper function for allocating contiguous memory in a 2d array

  Input parameters:
  m,n - number of rows and columns of array, respectively

  Output parameter:
  A   - pointer to array for which memory is allocated

  Note: Only arrays of doubles are currently accounted for in ADOL-C's myalloc2 function.
*/
template <class T> PetscErrorCode AdolcMalloc2(PetscInt m,PetscInt n,T **A[])
{
  PetscFunctionBegin;
  *A = myalloc2(m,n);
  PetscFunctionReturn(0);
}

/*
  Wrapper function for freeing memory allocated with AdolcMalloc2

  Input parameter:
  A - array to free memory of

  Note: Only arrays of doubles are currently accounted for in ADOL-C's myfree2 function.
*/
template <class T> PetscErrorCode AdolcFree2(T **A)
{
  PetscFunctionBegin;
  myfree2(A);
  PetscFunctionReturn(0);
}

/*
  Shift indices in an array of type T to endow it with ghost points.
  (e.g. This works for arrays of adoubles or arrays (of structs) thereof.)

  Input parameters:
  da   - distributed array upon which variables are defined
  cgs  - contiguously allocated 1-array with as many entries as there are
         interior and ghost points, in total

  Output parameter:
  array - contiguously allocated array of the appropriate dimension with
          ghost points, pointing to the 1-array
*/
template <class T> PetscErrorCode GiveGhostPoints(DM da,T *cgs,void *array)
{
  PetscErrorCode ierr;
  PetscInt       dim;

  PetscFunctionBegin;
  ierr = DMDAGetInfo(da,&dim,0,0,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  if (dim == 1) {
    ierr = GiveGhostPoints1d(da,(T**)array);CHKERRQ(ierr);
  } else if (dim == 2) {
    ierr = GiveGhostPoints2d(da,cgs,(T***)array);CHKERRQ(ierr);
  } else PetscCheckFalse(dim == 3,PetscObjectComm((PetscObject)da),PETSC_ERR_SUP,"GiveGhostPoints3d not yet implemented"); // TODO
  PetscFunctionReturn(0);
}

/*
  Shift indices in a 1-array of type T to endow it with ghost points.
  (e.g. This works for arrays of adoubles or arrays (of structs) thereof.)

  Input parameters:
  da  - distributed array upon which variables are defined

  Output parameter:
  a1d - contiguously allocated 1-array
*/
template <class T> PetscErrorCode GiveGhostPoints1d(DM da,T *a1d[])
{
  PetscErrorCode ierr;
  PetscInt       gxs;

  PetscFunctionBegin;
  ierr = DMDAGetGhostCorners(da,&gxs,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  *a1d -= gxs;
  PetscFunctionReturn(0);
}

/*
  Shift indices in a 2-array of type T to endow it with ghost points.
  (e.g. This works for arrays of adoubles or arrays (of structs) thereof.)

  Input parameters:
  da  - distributed array upon which variables are defined
  cgs - contiguously allocated 1-array with as many entries as there are
        interior and ghost points, in total

  Output parameter:
  a2d - contiguously allocated 2-array with ghost points, pointing to the
        1-array
*/
template <class T> PetscErrorCode GiveGhostPoints2d(DM da,T *cgs,T **a2d[])
{
  PetscErrorCode ierr;
  PetscInt       gxs,gys,gxm,gym,j;

  PetscFunctionBegin;
  ierr = DMDAGetGhostCorners(da,&gxs,&gys,NULL,&gxm,&gym,NULL);CHKERRQ(ierr);
  for (j=0; j<gym; j++)
    (*a2d)[j] = cgs + j*gxm - gxs;
  *a2d -= gys;
  PetscFunctionReturn(0);
}

/*
  Create a rectangular sub-identity of the m x m identity matrix, as an array.

  Input parameters:
  n - number of (adjacent) rows to take in slice
  s - starting row index

  Output parameter:
  S - resulting n x m submatrix
*/
template <class T> PetscErrorCode Subidentity(PetscInt n,PetscInt s,T **S)
{
  PetscInt       i;

  PetscFunctionBegin;
  for (i=0; i<n; i++) {
    S[i][i+s] = 1.;
  }
  PetscFunctionReturn(0);
}

/*
  Create an identity matrix, as an array.

  Input parameter:
  n - number of rows/columns
  I - n x n array with memory pre-allocated
*/
template <class T> PetscErrorCode Identity(PetscInt n,T **I)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = Subidentity(n,0,I);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
