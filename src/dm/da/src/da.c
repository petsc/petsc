#define PETSCDM_DLL
#include "private/daimpl.h"    /*I   "petscda.h"   I*/

#undef __FUNCT__  
#define __FUNCT__ "DASetOptionsPrefix"
/*@C
   DASetOptionsPrefix - Sets the prefix used for searching for all 
   DA options in the database.

   Collective on DA

   Input Parameter:
+  da - the DA context
-  prefix - the prefix to prepend to all option names

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the hyphen.

   Level: advanced

.keywords: DA, set, options, prefix, database

.seealso: DASetFromOptions()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DASetOptionsPrefix(DA da,const char prefix[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)da,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DASetDim"
/*@
  DASetDim - Sets the dimension

  Collective on DA

  Input Parameters:
+ da - the DA
- dim - the dimension (or PETSC_DECIDE)

  Level: intermediate

.seealso: DaGetDim(), DASetSizes()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DASetDim(DA da, PetscInt dim)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, DM_COOKIE, 1);
  if (da->dim > 0 && dim != da->dim) SETERRQ2(PETSC_ERR_ARG_WRONGSTATE,"Cannot change DA dim from %D after it was set to %D",da->dim,dim);
  da->dim = dim;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DASetSizes"
/*@
  DASetSizes - Sets the global sizes

  Collective on DA

  Input Parameters:
+ da - the DA
. M - the global X size (or PETSC_DECIDE)
. N - the global Y size (or PETSC_DECIDE)
- P - the global Z size (or PETSC_DECIDE)

  Level: intermediate

.seealso: DAGetSize(), PetscSplitOwnership()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DASetSizes(DA da, PetscInt M, PetscInt N, PetscInt P)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, DM_COOKIE, 1);
  da->M = M;
  da->N = N;
  da->P = P;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DASetNumProcs"
/*@
  DASetNumProcs - Sets the number of processes in each dimension

  Collective on DA

  Input Parameters:
+ da - the DA
. m - the number of X procs (or PETSC_DECIDE)
. n - the number of Y procs (or PETSC_DECIDE)
- p - the number of Z procs (or PETSC_DECIDE)

  Level: intermediate

.seealso: DASetSizes(), DAGetSize(), PetscSplitOwnership()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DASetNumProcs(DA da, PetscInt m, PetscInt n, PetscInt p)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, DM_COOKIE, 1);
  da->m = m;
  da->n = n;
  da->p = p;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DASetPeriodicity"
/*@
  DASetPeriodicity - Sets the type of periodicity

  Not collective

  Input Parameter:
+ da    - The DA
- ptype - One of DA_NONPERIODIC, DA_XPERIODIC, DA_YPERIODIC, DA_ZPERIODIC, DA_XYPERIODIC, DA_XZPERIODIC, DA_YZPERIODIC, or DA_XYZPERIODIC

  Level: intermediate

.keywords:  distributed array, periodicity
.seealso: DACreate(), DADestroy(), DA, DAPeriodicType
@*/
PetscErrorCode PETSCDM_DLLEXPORT DASetPeriodicity(DA da, DAPeriodicType ptype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  da->wrap = ptype;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DASetDof"
/*@
  DASetDof - Sets the number of degrees of freedom per vertex

  Not collective

  Input Parameter:
+ da  - The DA
- dof - Number of degrees of freedom

  Level: intermediate

.keywords:  distributed array, degrees of freedom
.seealso: DACreate(), DADestroy(), DA
@*/
PetscErrorCode PETSCDM_DLLEXPORT DASetDof(DA da, int dof)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  da->w = dof;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DASetStencilType"
/*@
  DASetStencilType - Sets the type of the communication stencil

  Not collective

  Input Parameter:
+ da    - The DA
- stype - The stencil type, use either DA_STENCIL_BOX or DA_STENCIL_STAR.

  Level: intermediate

.keywords:  distributed array, stencil
.seealso: DACreate(), DADestroy(), DA
@*/
PetscErrorCode PETSCDM_DLLEXPORT DASetStencilType(DA da, DAStencilType stype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  da->stencil_type = stype;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DASetStencilWidth"
/*@
  DASetStencilWidth - Sets the width of the communication stencil

  Not collective

  Input Parameter:
+ da    - The DA
- width - The stencil width

  Level: intermediate

.keywords:  distributed array, stencil
.seealso: DACreate(), DADestroy(), DA
@*/
PetscErrorCode PETSCDM_DLLEXPORT DASetStencilWidth(DA da, int width)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  da->s = width;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DASetVertexDivision"
/*@
  DASetVertexDivision - Sets the number of nodes in each direction on each process

  Not collective

  Input Parameter:
+ da - The DA
. lx - array containing number of nodes in the X direction on each process, or PETSC_NULL. If non-null, must be of length da->m
. ly - array containing number of nodes in the Y direction on each process, or PETSC_NULL. If non-null, must be of length da->n
- lz - array containing number of nodes in the Z direction on each process, or PETSC_NULL. If non-null, must be of length da->p.

  Level: intermediate

.keywords:  distributed array
.seealso: DACreate(), DADestroy(), DA
@*/
PetscErrorCode PETSCDM_DLLEXPORT DASetVertexDivision(DA da, const PetscInt lx[], const PetscInt ly[], const PetscInt lz[])
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  if (lx) {
    if (da->m < 0) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Cannot set vertex division before setting number of procs");
    if (!da->lx) {
      ierr = PetscMalloc(da->m*sizeof(PetscInt), &da->lx);CHKERRQ(ierr);
    }
    ierr = PetscMemcpy(da->lx, lx, da->m*sizeof(PetscInt));CHKERRQ(ierr);
  }
  if (ly) {
    if (da->n < 0) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Cannot set vertex division before setting number of procs");
    if (!da->ly) {
      ierr = PetscMalloc(da->n*sizeof(PetscInt), &da->ly);CHKERRQ(ierr);
    }
    ierr = PetscMemcpy(da->ly, ly, da->n*sizeof(PetscInt));CHKERRQ(ierr);
  }
  if (lz) {
    if (da->p < 0) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Cannot set vertex division before setting number of procs");
    if (!da->lz) {
      ierr = PetscMalloc(da->p*sizeof(PetscInt), &da->lz);CHKERRQ(ierr);
    }
    ierr = PetscMemcpy(da->lz, lz, da->p*sizeof(PetscInt));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DASetInterpolationType"
/*@
       DASetInterpolationType - Sets the type of interpolation that will be 
          returned by DAGetInterpolation()

   Collective on DA

   Input Parameter:
+  da - initial distributed array
.  ctype - DA_Q1 and DA_Q0 are currently the only supported forms

   Level: intermediate

   Notes: you should call this on the coarser of the two DAs you pass to DAGetInterpolation()

.keywords:  distributed array, interpolation

.seealso: DACreate1d(), DACreate2d(), DACreate3d(), DADestroy(), DA, DAInterpolationType
@*/
PetscErrorCode PETSCDM_DLLEXPORT DASetInterpolationType(DA da,DAInterpolationType ctype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  da->interptype = ctype;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DAGetNeighbors"
/*@C
      DAGetNeighbors - Gets an array containing the MPI rank of all the current
        processes neighbors.

    Not Collective

   Input Parameter:
.     da - the DA object

   Output Parameters:
.     ranks - the neighbors ranks, stored with the x index increasing most rapidly.
              this process itself is in the list

   Notes: In 2d the array is of length 9, in 3d of length 27
          Not supported in 1d
          Do not free the array, it is freed when the DA is destroyed.

   Fortran Notes: In fortran you must pass in an array of the appropriate length.

   Level: intermediate

@*/
PetscErrorCode PETSCDM_DLLEXPORT DAGetNeighbors(DA da,const PetscMPIInt *ranks[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  *ranks = da->neighbors;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMGetElements"
/*@C
      DMGetElements - Gets an array containing the indices (in local coordinates) 
                 of all the local elements

    Not Collective

   Input Parameter:
.     dm - the DM object

   Output Parameters:
+     n - number of local elements
-     e - the indices of the elements vertices

   Level: intermediate

.seealso: DMElementType, DMSetElementType(), DMRestoreElements()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DMGetElements(DM dm,PetscInt *n,const PetscInt *e[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_COOKIE,1);
  ierr = (dm->ops->getelements)(dm,n,e);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMRestoreElements"
/*@C
      DMRestoreElements - Returns an array containing the indices (in local coordinates) 
                 of all the local elements obtained with DMGetElements()

    Not Collective

   Input Parameter:
+     dm - the DM object
.     n - number of local elements
-     e - the indices of the elements vertices

   Level: intermediate

.seealso: DMElementType, DMSetElementType(), DMGetElements()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DMRestoreElements(DM dm,PetscInt *n,const PetscInt *e[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_COOKIE,1);
  if (dm->ops->restoreelements) {
    ierr = (dm->ops->restoreelements)(dm,n,e);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DAGetOwnershipRanges"
/*@C
      DAGetOwnershipRanges - Gets the ranges of indices in the x, y and z direction that are owned by each process

    Not Collective

   Input Parameter:
.     da - the DA object

   Output Parameter:
+     lx - ownership along x direction (optional)
.     ly - ownership along y direction (optional)
-     lz - ownership along z direction (optional)

   Level: intermediate

    Note: these correspond to the optional final arguments passed to DACreate(), DACreate2d(), DACreate3d()

    In Fortran one must pass in arrays lx, ly, and lz that are long enough to hold the values; the sixth, seventh and
    eighth arguments from DAGetInfo()

     In C you should not free these arrays, nor change the values in them. They will only have valid values while the
    DA they came from still exists (has not been destroyed).

.seealso: DAGetCorners(), DAGetGhostCorners(), DACreate(), DACreate1d(), DACreate2d(), DACreate3d(), VecGetOwnershipRanges()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DAGetOwnershipRanges(DA da,const PetscInt *lx[],const PetscInt *ly[],const PetscInt *lz[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  if (lx) *lx = da->lx;
  if (ly) *ly = da->ly;
  if (lz) *lz = da->lz;
  PetscFunctionReturn(0);
}

/*@
     DASetRefinementFactor - Set the ratios that the DA grid is refined

    Collective on DA

  Input Parameters:
+    da - the DA object
.    refine_x - ratio of fine grid to coarse in x direction (2 by default)
.    refine_y - ratio of fine grid to coarse in y direction (2 by default)
-    refine_z - ratio of fine grid to coarse in z direction (2 by default)

  Options Database:
+  -da_refine_x - refinement ratio in x direction
.  -da_refine_y - refinement ratio in y direction
-  -da_refine_z - refinement ratio in z direction

  Level: intermediate

    Notes: Pass PETSC_IGNORE to leave a value unchanged

.seealso: DARefine(), DAGetRefinementFactor()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DASetRefinementFactor(DA da, PetscInt refine_x, PetscInt refine_y,PetscInt refine_z)
{
  PetscFunctionBegin;
  if (refine_x > 0) da->refine_x = refine_x;
  if (refine_y > 0) da->refine_y = refine_y;
  if (refine_z > 0) da->refine_z = refine_z;
  PetscFunctionReturn(0);
}

/*@C
     DAGetRefinementFactor - Gets the ratios that the DA grid is refined

    Not Collective

  Input Parameter:
.    da - the DA object

  Output Parameters:
+    refine_x - ratio of fine grid to coarse in x direction (2 by default)
.    refine_y - ratio of fine grid to coarse in y direction (2 by default)
-    refine_z - ratio of fine grid to coarse in z direction (2 by default)

  Level: intermediate

    Notes: Pass PETSC_NULL for values you do not need

.seealso: DARefine(), DASetRefinementFactor()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DAGetRefinementFactor(DA da, PetscInt *refine_x, PetscInt *refine_y,PetscInt *refine_z)
{
  PetscFunctionBegin;
  if (refine_x) *refine_x = da->refine_x;
  if (refine_y) *refine_y = da->refine_y;
  if (refine_z) *refine_z = da->refine_z;
  PetscFunctionReturn(0);
}

/*@C
     DASetGetMatrix - Sets the routine used by the DA to allocate a matrix.

    Collective on DA

  Input Parameters:
+    da - the DA object
-    f - the function that allocates the matrix for that specific DA

  Level: developer

   Notes: See DASetBlockFills() that provides a simple way to provide the nonzero structure for 
       the diagonal and off-diagonal blocks of the matrix

.seealso: DAGetMatrix(), DASetBlockFills()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DASetGetMatrix(DA da,PetscErrorCode (*f)(DA, const MatType,Mat*))
{
  PetscFunctionBegin;
  da->ops->getmatrix = f;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DARefine"
/*@
   DARefine - Creates a new distributed array that is a refinement of a given
   distributed array.

   Collective on DA

   Input Parameter:
+  da - initial distributed array
-  comm - communicator to contain refined DA, must be either same as the da communicator or include the 
          da communicator and be 2, 4, or 8 times larger. Currently ignored

   Output Parameter:
.  daref - refined distributed array

   Level: advanced

   Note:
   Currently, refinement consists of just doubling the number of grid spaces
   in each dimension of the DA.

.keywords:  distributed array, refine

.seealso: DACreate1d(), DACreate2d(), DACreate3d(), DADestroy(), DAGetOwnershipRanges()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DARefine(DA da,MPI_Comm comm,DA *daref)
{
  PetscErrorCode ierr;
  PetscInt       M,N,P;
  DA             da2;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  PetscValidPointer(daref,3);

  if (DAXPeriodic(da->wrap) || da->interptype == DA_Q0){
    M = da->refine_x*da->M;
  } else {
    M = 1 + da->refine_x*(da->M - 1);
  }
  if (DAYPeriodic(da->wrap) || da->interptype == DA_Q0){
    N = da->refine_y*da->N;
  } else {
    N = 1 + da->refine_y*(da->N - 1);
  }
  if (DAZPeriodic(da->wrap) || da->interptype == DA_Q0){
    P = da->refine_z*da->P;
  } else {
    P = 1 + da->refine_z*(da->P - 1);
  }
  if (da->dim == 3) {
    ierr = DACreate3d(((PetscObject)da)->comm,da->wrap,da->stencil_type,M,N,P,da->m,da->n,da->p,da->w,da->s,0,0,0,&da2);CHKERRQ(ierr);
  } else if (da->dim == 2) {
    ierr = DACreate2d(((PetscObject)da)->comm,da->wrap,da->stencil_type,M,N,da->m,da->n,da->w,da->s,0,0,&da2);CHKERRQ(ierr);
  } else if (da->dim == 1) {
    ierr = DACreate1d(((PetscObject)da)->comm,da->wrap,M,da->w,da->s,0,&da2);CHKERRQ(ierr);
  }

  /* allow overloaded (user replaced) operations to be inherited by refinement clones */
  da2->ops->getmatrix        = da->ops->getmatrix;
  da2->ops->getinterpolation = da->ops->getinterpolation;
  da2->ops->getcoloring      = da->ops->getcoloring;
  da2->interptype            = da->interptype;
  
  /* copy fill information if given */
  if (da->dfill) {
    ierr = PetscMalloc((da->dfill[da->w]+da->w+1)*sizeof(PetscInt),&da2->dfill);CHKERRQ(ierr);
    ierr = PetscMemcpy(da2->dfill,da->dfill,(da->dfill[da->w]+da->w+1)*sizeof(PetscInt));CHKERRQ(ierr);
  }
  if (da->ofill) {
    ierr = PetscMalloc((da->ofill[da->w]+da->w+1)*sizeof(PetscInt),&da2->ofill);CHKERRQ(ierr);
    ierr = PetscMemcpy(da2->ofill,da->ofill,(da->ofill[da->w]+da->w+1)*sizeof(PetscInt));CHKERRQ(ierr);
  }
  /* copy the refine information */
  da2->refine_x = da->refine_x;
  da2->refine_y = da->refine_y;
  da2->refine_z = da->refine_z;
  *daref = da2;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DACoarsen"
/*@
   DACoarsen - Creates a new distributed array that is a coarsenment of a given
   distributed array.

   Collective on DA

   Input Parameter:
+  da - initial distributed array
-  comm - communicator to contain coarsend DA. Currently ignored

   Output Parameter:
.  daref - coarsend distributed array

   Level: advanced

   Note:
   Currently, coarsenment consists of just dividing the number of grid spaces
   in each dimension of the DA by refinex_x, refinex_y, ....

.keywords:  distributed array, coarsen

.seealso: DACreate1d(), DACreate2d(), DACreate3d(), DADestroy(), DAGetOwnershipRanges()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DACoarsen(DA da, MPI_Comm comm,DA *daref)
{
  PetscErrorCode ierr;
  PetscInt       M,N,P;
  DA             da2;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  PetscValidPointer(daref,3);

  if (DAXPeriodic(da->wrap) || da->interptype == DA_Q0){
    if(da->refine_x)
      M = da->M / da->refine_x;
    else
      M = da->M;
  } else {
    if(da->refine_x)
      M = 1 + (da->M - 1) / da->refine_x;
    else
      M = da->M;
  }
  if (DAYPeriodic(da->wrap) || da->interptype == DA_Q0){
    if(da->refine_y)
      N = da->N / da->refine_y;
    else
      N = da->N;
  } else {
    if(da->refine_y)
      N = 1 + (da->N - 1) / da->refine_y;
    else
      N = da->M;
  }
  if (DAZPeriodic(da->wrap) || da->interptype == DA_Q0){
    if(da->refine_z)
      P = da->P / da->refine_z;
    else
      P = da->P;
  } else {
    if(da->refine_z)
      P = 1 + (da->P - 1) / da->refine_z;
    else
      P = da->P;
  }
  if (da->dim == 3) {
    ierr = DACreate3d(((PetscObject)da)->comm,da->wrap,da->stencil_type,M,N,P,da->m,da->n,da->p,da->w,da->s,0,0,0,&da2);CHKERRQ(ierr);
  } else if (da->dim == 2) {
    ierr = DACreate2d(((PetscObject)da)->comm,da->wrap,da->stencil_type,M,N,da->m,da->n,da->w,da->s,0,0,&da2);CHKERRQ(ierr);
  } else if (da->dim == 1) {
    ierr = DACreate1d(((PetscObject)da)->comm,da->wrap,M,da->w,da->s,0,&da2);CHKERRQ(ierr);
  }

  /* allow overloaded (user replaced) operations to be inherited by refinement clones */
  da2->ops->getmatrix        = da->ops->getmatrix;
  da2->ops->getinterpolation = da->ops->getinterpolation;
  da2->ops->getcoloring      = da->ops->getcoloring;
  da2->interptype            = da->interptype;
  
  /* copy fill information if given */
  if (da->dfill) {
    ierr = PetscMalloc((da->dfill[da->w]+da->w+1)*sizeof(PetscInt),&da2->dfill);CHKERRQ(ierr);
    ierr = PetscMemcpy(da2->dfill,da->dfill,(da->dfill[da->w]+da->w+1)*sizeof(PetscInt));CHKERRQ(ierr);
  }
  if (da->ofill) {
    ierr = PetscMalloc((da->ofill[da->w]+da->w+1)*sizeof(PetscInt),&da2->ofill);CHKERRQ(ierr);
    ierr = PetscMemcpy(da2->ofill,da->ofill,(da->ofill[da->w]+da->w+1)*sizeof(PetscInt));CHKERRQ(ierr);
  }
  /* copy the refine information */
  da2->refine_x = da->refine_x;
  da2->refine_y = da->refine_y;
  da2->refine_z = da->refine_z;
  *daref = da2;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DARefineHierarchy"
/*@
   DARefineHierarchy - Perform multiple levels of refinement.

   Collective on DA

   Input Parameter:
+  da - initial distributed array
-  nlevels - number of levels of refinement to perform

   Output Parameter:
.  daf - array of refined DAs

   Options Database:
+  -da_refine_hierarchy_x - list of refinement ratios in x direction
.  -da_refine_hierarchy_y - list of refinement ratios in y direction
-  -da_refine_hierarchy_z - list of refinement ratios in z direction

   Level: advanced

.keywords: distributed array, refine

.seealso: DARefine(), DACoarsenHierarchy()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DARefineHierarchy(DA da,PetscInt nlevels,DA daf[])
{
  PetscErrorCode ierr;
  PetscInt i,n,*refx,*refy,*refz;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  if (nlevels < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"nlevels cannot be negative");
  if (nlevels == 0) PetscFunctionReturn(0);
  PetscValidPointer(daf,3);

  /* Get refinement factors, defaults taken from the coarse DA */
  ierr = PetscMalloc3(nlevels,PetscInt,&refx,nlevels,PetscInt,&refy,nlevels,PetscInt,&refz);CHKERRQ(ierr);
  for (i=0; i<nlevels; i++) {
    ierr = DAGetRefinementFactor(da,&refx[i],&refy[i],&refz[i]);CHKERRQ(ierr);
  }
  n = nlevels;
  ierr = PetscOptionsGetIntArray(((PetscObject)da)->prefix,"-da_refine_hierarchy_x",refx,&n,PETSC_NULL);CHKERRQ(ierr);
  n = nlevels;
  ierr = PetscOptionsGetIntArray(((PetscObject)da)->prefix,"-da_refine_hierarchy_y",refy,&n,PETSC_NULL);CHKERRQ(ierr);
  n = nlevels;
  ierr = PetscOptionsGetIntArray(((PetscObject)da)->prefix,"-da_refine_hierarchy_z",refz,&n,PETSC_NULL);CHKERRQ(ierr);

  ierr = DASetRefinementFactor(da,refx[0],refy[0],refz[0]);CHKERRQ(ierr);
  ierr = DARefine(da,((PetscObject)da)->comm,&daf[0]);CHKERRQ(ierr);
  for (i=1; i<nlevels; i++) {
    ierr = DASetRefinementFactor(daf[i-1],refx[i],refy[i],refz[i]);CHKERRQ(ierr);
    ierr = DARefine(daf[i-1],((PetscObject)da)->comm,&daf[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree3(refx,refy,refz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DACoarsenHierarchy"
/*@
   DACoarsenHierarchy - Perform multiple levels of coarsening

   Collective on DA

   Input Parameter:
+  da - initial distributed array
-  nlevels - number of levels of coarsening to perform

   Output Parameter:
.  dac - array of coarsened DAs

   Level: advanced

.keywords: distributed array, coarsen

.seealso: DACoarsen(), DARefineHierarchy()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DACoarsenHierarchy(DA da,PetscInt nlevels,DA dac[])
{
  PetscErrorCode ierr;
  PetscInt i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  if (nlevels < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"nlevels cannot be negative");
  if (nlevels == 0) PetscFunctionReturn(0);
  PetscValidPointer(dac,3);
  ierr = DACoarsen(da,((PetscObject)da)->comm,&dac[0]);CHKERRQ(ierr);
  for (i=1; i<nlevels; i++) {
    ierr = DACoarsen(dac[i-1],((PetscObject)da)->comm,&dac[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
