#include <petsc/private/dmdaimpl.h> /*I      "petscdmda.h"   I*/
#include <petsc/private/isimpl.h>
#include <petscsf.h>

/*@
  DMDASetPreallocationCenterDimension - Determine the topology used to determine adjacency

  Input Parameters:
+ dm                - The `DMDA` object
- preallocCenterDim - The dimension of points which connect adjacent entries

  Level: developer

  Notes:
.vb
     FEM:   Two points p and q are adjacent if q \in closure(star(p)), preallocCenterDim = dim
     FVM:   Two points p and q are adjacent if q \in star(cone(p)),    preallocCenterDim = dim-1
     FVM++: Two points p and q are adjacent if q \in star(closure(p)), preallocCenterDim = 0
.ve

.seealso: [](sec_struct), `DM`, `DMDA`, `DMCreateMatrix()`, `DMDAPreallocateOperator()`
@*/
PetscErrorCode DMDASetPreallocationCenterDimension(DM dm, PetscInt preallocCenterDim)
{
  DM_DA *mesh = (DM_DA *)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMDA);
  mesh->preallocCenterDim = preallocCenterDim;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMDAGetPreallocationCenterDimension - Return the topology used to determine adjacency

  Input Parameter:
. dm - The `DMDA` object

  Output Parameter:
. preallocCenterDim - The dimension of points which connect adjacent entries

  Level: developer

  Notes:
.vb
     FEM:   Two points p and q are adjacent if q \in closure(star(p)), preallocCenterDim = dim
     FVM:   Two points p and q are adjacent if q \in star(cone(p)),    preallocCenterDim = dim-1
     FVM++: Two points p and q are adjacent if q \in star(closure(p)), preallocCenterDim = 0
.ve

.seealso: [](sec_struct), `DM`, `DMDA`, `DMCreateMatrix()`, `DMDAPreallocateOperator()`, `DMDASetPreallocationCenterDimension()`
@*/
PetscErrorCode DMDAGetPreallocationCenterDimension(DM dm, PetscInt *preallocCenterDim)
{
  DM_DA *mesh = (DM_DA *)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMDA);
  PetscAssertPointer(preallocCenterDim, 2);
  *preallocCenterDim = mesh->preallocCenterDim;
  PetscFunctionReturn(PETSC_SUCCESS);
}
