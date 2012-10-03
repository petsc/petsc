#include <petsc-private/daimpl.h>    /*I   "petscdmda.h"   I*/


#undef __FUNCT__
#define __FUNCT__ "DMDASetDim"
/*@
  DMDASetDim - Sets the dimension

  Collective on DMDA

  Input Parameters:
+ da - the DMDA
- dim - the dimension (or PETSC_DECIDE)

  Level: intermediate

.seealso: DaGetDim(), DMDASetSizes()
@*/
PetscErrorCode  DMDASetDim(DM da, PetscInt dim)
{
  DM_DA *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, DM_CLASSID, 1);
  if (dd->dim > 0 && dim != dd->dim) SETERRQ2(((PetscObject)da)->comm,PETSC_ERR_ARG_WRONGSTATE,"Cannot change DMDA dim from %D after it was set to %D",dd->dim,dim);
  dd->dim = dim;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDASetSizes"
/*@
  DMDASetSizes - Sets the global sizes

  Logically Collective on DMDA

  Input Parameters:
+ da - the DMDA
. M - the global X size (or PETSC_DECIDE)
. N - the global Y size (or PETSC_DECIDE)
- P - the global Z size (or PETSC_DECIDE)

  Level: intermediate

.seealso: DMDAGetSize(), PetscSplitOwnership()
@*/
PetscErrorCode  DMDASetSizes(DM da, PetscInt M, PetscInt N, PetscInt P)
{
  DM_DA *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, DM_CLASSID, 1);
  PetscValidLogicalCollectiveInt(da,M,2);
  PetscValidLogicalCollectiveInt(da,N,3);
  PetscValidLogicalCollectiveInt(da,P,4);
  if (da->setupcalled) SETERRQ(((PetscObject)da)->comm,PETSC_ERR_ARG_WRONGSTATE,"This function must be called before DMSetUp()");

  dd->M = M;
  dd->N = N;
  dd->P = P;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDASetNumProcs"
/*@
  DMDASetNumProcs - Sets the number of processes in each dimension

  Logically Collective on DMDA

  Input Parameters:
+ da - the DMDA
. m - the number of X procs (or PETSC_DECIDE)
. n - the number of Y procs (or PETSC_DECIDE)
- p - the number of Z procs (or PETSC_DECIDE)

  Level: intermediate

.seealso: DMDASetSizes(), DMDAGetSize(), PetscSplitOwnership()
@*/
PetscErrorCode  DMDASetNumProcs(DM da, PetscInt m, PetscInt n, PetscInt p)
{
  DM_DA *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, DM_CLASSID, 1);
  PetscValidLogicalCollectiveInt(da,m,2);
  PetscValidLogicalCollectiveInt(da,n,3);
  PetscValidLogicalCollectiveInt(da,p,4);
  if (da->setupcalled) SETERRQ(((PetscObject)da)->comm,PETSC_ERR_ARG_WRONGSTATE,"This function must be called before DMSetUp()");
  dd->m = m;
  dd->n = n;
  dd->p = p;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDASetBoundaryType"
/*@
  DMDASetBoundaryType - Sets the type of ghost nodes on domain boundaries.

  Not collective

  Input Parameter:
+ da    - The DMDA
- bx,by,bz - One of DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_GHOSTED, DMDA_BOUNDARY_PERIODIC

  Level: intermediate

.keywords:  distributed array, periodicity
.seealso: DMDACreate(), DMDestroy(), DMDA, DMDABoundaryType
@*/
PetscErrorCode  DMDASetBoundaryType(DM da,DMDABoundaryType bx,DMDABoundaryType by,DMDABoundaryType bz)
{
  DM_DA *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  PetscValidLogicalCollectiveEnum(da,bx,2);
  PetscValidLogicalCollectiveEnum(da,by,3);
  PetscValidLogicalCollectiveEnum(da,bz,4);
  if (da->setupcalled) SETERRQ(((PetscObject)da)->comm,PETSC_ERR_ARG_WRONGSTATE,"This function must be called before DMSetUp()");
  dd->bx = bx;
  dd->by = by;
  dd->bz = bz;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDASetDof"
/*@
  DMDASetDof - Sets the number of degrees of freedom per vertex

  Not collective

  Input Parameter:
+ da  - The DMDA
- dof - Number of degrees of freedom

  Level: intermediate

.keywords:  distributed array, degrees of freedom
.seealso: DMDACreate(), DMDestroy(), DMDA
@*/
PetscErrorCode  DMDASetDof(DM da, PetscInt dof)
{
  DM_DA *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  PetscValidLogicalCollectiveInt(da,dof,2);
  if (da->setupcalled) SETERRQ(((PetscObject)da)->comm,PETSC_ERR_ARG_WRONGSTATE,"This function must be called before DMSetUp()");
  dd->w = dof;
  da->bs = dof;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDASetOverlap"
/*@
  DMDASetOverlap - Sets the size of the per-processor overlap.

  Not collective

  Input Parameter:
+ da  - The DMDA
- dof - Number of degrees of freedom

  Level: intermediate

.keywords:  distributed array, degrees of freedom
.seealso: DMDACreate(), DMDestroy(), DMDA
@*/
PetscErrorCode  DMDASetOverlap(DM da, PetscInt overlap)
{
  DM_DA *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  PetscValidLogicalCollectiveInt(da,overlap,2);
  if (da->setupcalled) SETERRQ(((PetscObject)da)->comm,PETSC_ERR_ARG_WRONGSTATE,"This function must be called before DMSetUp()");
  dd->overlap = overlap;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMDASetStencilType"
/*@
  DMDASetStencilType - Sets the type of the communication stencil

  Logically Collective on DMDA

  Input Parameter:
+ da    - The DMDA
- stype - The stencil type, use either DMDA_STENCIL_BOX or DMDA_STENCIL_STAR.

  Level: intermediate

.keywords:  distributed array, stencil
.seealso: DMDACreate(), DMDestroy(), DMDA
@*/
PetscErrorCode  DMDASetStencilType(DM da, DMDAStencilType stype)
{
  DM_DA *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  PetscValidLogicalCollectiveEnum(da,stype,2);
  if (da->setupcalled) SETERRQ(((PetscObject)da)->comm,PETSC_ERR_ARG_WRONGSTATE,"This function must be called before DMSetUp()");
  dd->stencil_type = stype;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDASetStencilWidth"
/*@
  DMDASetStencilWidth - Sets the width of the communication stencil

  Logically Collective on DMDA

  Input Parameter:
+ da    - The DMDA
- width - The stencil width

  Level: intermediate

.keywords:  distributed array, stencil
.seealso: DMDACreate(), DMDestroy(), DMDA
@*/
PetscErrorCode  DMDASetStencilWidth(DM da, PetscInt width)
{
  DM_DA *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  PetscValidLogicalCollectiveInt(da,width,2);
  if (da->setupcalled) SETERRQ(((PetscObject)da)->comm,PETSC_ERR_ARG_WRONGSTATE,"This function must be called before DMSetUp()");
  dd->s = width;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDACheckOwnershipRanges_Private"
static PetscErrorCode DMDACheckOwnershipRanges_Private(DM da,PetscInt M,PetscInt m,const PetscInt lx[])
{
  PetscInt i,sum;

  PetscFunctionBegin;
  if (M < 0) SETERRQ(((PetscObject)da)->comm,PETSC_ERR_ARG_WRONGSTATE,"Global dimension not set");
  for (i=sum=0; i<m; i++) sum += lx[i];
  if (sum != M) SETERRQ2(((PetscObject)da)->comm,PETSC_ERR_ARG_INCOMP,"Ownership ranges sum to %D but global dimension is %D",sum,M);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDASetOwnershipRanges"
/*@
  DMDASetOwnershipRanges - Sets the number of nodes in each direction on each process

  Logically Collective on DMDA

  Input Parameter:
+ da - The DMDA
. lx - array containing number of nodes in the X direction on each process, or PETSC_NULL. If non-null, must be of length da->m
. ly - array containing number of nodes in the Y direction on each process, or PETSC_NULL. If non-null, must be of length da->n
- lz - array containing number of nodes in the Z direction on each process, or PETSC_NULL. If non-null, must be of length da->p.

  Level: intermediate

.keywords:  distributed array
.seealso: DMDACreate(), DMDestroy(), DMDA
@*/
PetscErrorCode  DMDASetOwnershipRanges(DM da, const PetscInt lx[], const PetscInt ly[], const PetscInt lz[])
{
  PetscErrorCode ierr;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  if (da->setupcalled) SETERRQ(((PetscObject)da)->comm,PETSC_ERR_ARG_WRONGSTATE,"This function must be called before DMSetUp()");
  if (lx) {
    if (dd->m < 0) SETERRQ(((PetscObject)da)->comm,PETSC_ERR_ARG_WRONGSTATE,"Cannot set ownership ranges before setting number of procs");
    ierr = DMDACheckOwnershipRanges_Private(da,dd->M,dd->m,lx);CHKERRQ(ierr);
    if (!dd->lx) {
      ierr = PetscMalloc(dd->m*sizeof(PetscInt), &dd->lx);CHKERRQ(ierr);
    }
    ierr = PetscMemcpy(dd->lx, lx, dd->m*sizeof(PetscInt));CHKERRQ(ierr);
  }
  if (ly) {
    if (dd->n < 0) SETERRQ(((PetscObject)da)->comm,PETSC_ERR_ARG_WRONGSTATE,"Cannot set ownership ranges before setting number of procs");
    ierr = DMDACheckOwnershipRanges_Private(da,dd->N,dd->n,ly);CHKERRQ(ierr);
    if (!dd->ly) {
      ierr = PetscMalloc(dd->n*sizeof(PetscInt), &dd->ly);CHKERRQ(ierr);
    }
    ierr = PetscMemcpy(dd->ly, ly, dd->n*sizeof(PetscInt));CHKERRQ(ierr);
  }
  if (lz) {
    if (dd->p < 0) SETERRQ(((PetscObject)da)->comm,PETSC_ERR_ARG_WRONGSTATE,"Cannot set ownership ranges before setting number of procs");
    ierr = DMDACheckOwnershipRanges_Private(da,dd->P,dd->p,lz);CHKERRQ(ierr);
    if (!dd->lz) {
      ierr = PetscMalloc(dd->p*sizeof(PetscInt), &dd->lz);CHKERRQ(ierr);
    }
    ierr = PetscMemcpy(dd->lz, lz, dd->p*sizeof(PetscInt));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDASetInterpolationType"
/*@
       DMDASetInterpolationType - Sets the type of interpolation that will be
          returned by DMCreateInterpolation()

   Logically Collective on DMDA

   Input Parameter:
+  da - initial distributed array
.  ctype - DMDA_Q1 and DMDA_Q0 are currently the only supported forms

   Level: intermediate

   Notes: you should call this on the coarser of the two DMDAs you pass to DMCreateInterpolation()

.keywords:  distributed array, interpolation

.seealso: DMDACreate1d(), DMDACreate2d(), DMDACreate3d(), DMDestroy(), DMDA, DMDAInterpolationType
@*/
PetscErrorCode  DMDASetInterpolationType(DM da,DMDAInterpolationType ctype)
{
  DM_DA *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  PetscValidLogicalCollectiveEnum(da,ctype,2);
  dd->interptype = ctype;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDAGetInterpolationType"
/*@
       DMDAGetInterpolationType - Gets the type of interpolation that will be
          used by DMCreateInterpolation()

   Not Collective

   Input Parameter:
.  da - distributed array

   Output Parameter:
.  ctype - interpolation type (DMDA_Q1 and DMDA_Q0 are currently the only supported forms)

   Level: intermediate

.keywords:  distributed array, interpolation

.seealso: DMDA, DMDAInterpolationType, DMDASetInterpolationType(), DMCreateInterpolation()
@*/
PetscErrorCode  DMDAGetInterpolationType(DM da,DMDAInterpolationType *ctype)
{
  DM_DA *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  PetscValidPointer(ctype,2);
  *ctype = dd->interptype;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDAGetNeighbors"
/*@C
      DMDAGetNeighbors - Gets an array containing the MPI rank of all the current
        processes neighbors.

    Not Collective

   Input Parameter:
.     da - the DMDA object

   Output Parameters:
.     ranks - the neighbors ranks, stored with the x index increasing most rapidly.
              this process itself is in the list

   Notes: In 2d the array is of length 9, in 3d of length 27
          Not supported in 1d
          Do not free the array, it is freed when the DMDA is destroyed.

   Fortran Notes: In fortran you must pass in an array of the appropriate length.

   Level: intermediate

@*/
PetscErrorCode  DMDAGetNeighbors(DM da,const PetscMPIInt *ranks[])
{
  DM_DA *dd = (DM_DA*)da->data;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  *ranks = dd->neighbors;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDAGetOwnershipRanges"
/*@C
      DMDAGetOwnershipRanges - Gets the ranges of indices in the x, y and z direction that are owned by each process

    Not Collective

   Input Parameter:
.     da - the DMDA object

   Output Parameter:
+     lx - ownership along x direction (optional)
.     ly - ownership along y direction (optional)
-     lz - ownership along z direction (optional)

   Level: intermediate

    Note: these correspond to the optional final arguments passed to DMDACreate(), DMDACreate2d(), DMDACreate3d()

    In Fortran one must pass in arrays lx, ly, and lz that are long enough to hold the values; the sixth, seventh and
    eighth arguments from DMDAGetInfo()

     In C you should not free these arrays, nor change the values in them. They will only have valid values while the
    DMDA they came from still exists (has not been destroyed).

.seealso: DMDAGetCorners(), DMDAGetGhostCorners(), DMDACreate(), DMDACreate1d(), DMDACreate2d(), DMDACreate3d(), VecGetOwnershipRanges()
@*/
PetscErrorCode  DMDAGetOwnershipRanges(DM da,const PetscInt *lx[],const PetscInt *ly[],const PetscInt *lz[])
{
  DM_DA *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  if (lx) *lx = dd->lx;
  if (ly) *ly = dd->ly;
  if (lz) *lz = dd->lz;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDASetRefinementFactor"
/*@
     DMDASetRefinementFactor - Set the ratios that the DMDA grid is refined

    Logically Collective on DMDA

  Input Parameters:
+    da - the DMDA object
.    refine_x - ratio of fine grid to coarse in x direction (2 by default)
.    refine_y - ratio of fine grid to coarse in y direction (2 by default)
-    refine_z - ratio of fine grid to coarse in z direction (2 by default)

  Options Database:
+  -da_refine_x - refinement ratio in x direction
.  -da_refine_y - refinement ratio in y direction
-  -da_refine_z - refinement ratio in z direction

  Level: intermediate

    Notes: Pass PETSC_IGNORE to leave a value unchanged

.seealso: DMRefine(), DMDAGetRefinementFactor()
@*/
PetscErrorCode  DMDASetRefinementFactor(DM da, PetscInt refine_x, PetscInt refine_y,PetscInt refine_z)
{
  DM_DA *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  PetscValidLogicalCollectiveInt(da,refine_x,2);
  PetscValidLogicalCollectiveInt(da,refine_y,3);
  PetscValidLogicalCollectiveInt(da,refine_z,4);

  if (refine_x > 0) dd->refine_x = refine_x;
  if (refine_y > 0) dd->refine_y = refine_y;
  if (refine_z > 0) dd->refine_z = refine_z;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDAGetRefinementFactor"
/*@C
     DMDAGetRefinementFactor - Gets the ratios that the DMDA grid is refined

    Not Collective

  Input Parameter:
.    da - the DMDA object

  Output Parameters:
+    refine_x - ratio of fine grid to coarse in x direction (2 by default)
.    refine_y - ratio of fine grid to coarse in y direction (2 by default)
-    refine_z - ratio of fine grid to coarse in z direction (2 by default)

  Level: intermediate

    Notes: Pass PETSC_NULL for values you do not need

.seealso: DMRefine(), DMDASetRefinementFactor()
@*/
PetscErrorCode  DMDAGetRefinementFactor(DM da, PetscInt *refine_x, PetscInt *refine_y,PetscInt *refine_z)
{
  DM_DA *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  if (refine_x) *refine_x = dd->refine_x;
  if (refine_y) *refine_y = dd->refine_y;
  if (refine_z) *refine_z = dd->refine_z;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDASetGetMatrix"
/*@C
     DMDASetGetMatrix - Sets the routine used by the DMDA to allocate a matrix.

    Logically Collective on DMDA

  Input Parameters:
+    da - the DMDA object
-    f - the function that allocates the matrix for that specific DMDA

  Level: developer

   Notes: See DMDASetBlockFills() that provides a simple way to provide the nonzero structure for
       the diagonal and off-diagonal blocks of the matrix

.seealso: DMCreateMatrix(), DMDASetBlockFills()
@*/
PetscErrorCode  DMDASetGetMatrix(DM da,PetscErrorCode (*f)(DM, MatType,Mat*))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  da->ops->creatematrix = f;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDARefineOwnershipRanges"
/*
  Creates "balanced" ownership ranges after refinement, constrained by the need for the
  fine grid boundaries to fall within one stencil width of the coarse partition.

  Uses a greedy algorithm to handle non-ideal layouts, could probably do something better.
*/
static PetscErrorCode DMDARefineOwnershipRanges(DM da,PetscBool periodic,PetscInt stencil_width,PetscInt ratio,PetscInt m,const PetscInt lc[],PetscInt lf[])
{
  PetscInt       i,totalc = 0,remaining,startc = 0,startf = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (ratio < 1) SETERRQ1(((PetscObject)da)->comm,PETSC_ERR_USER,"Requested refinement ratio %D must be at least 1",ratio);
  if (ratio == 1) {
    ierr = PetscMemcpy(lf,lc,m*sizeof(lc[0]));CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  for (i=0; i<m; i++) totalc += lc[i];
  remaining = (!periodic) + ratio * (totalc - (!periodic));
  for (i=0; i<m; i++) {
    PetscInt want = remaining/(m-i) + !!(remaining%(m-i));
    if (i == m-1) lf[i] = want;
    else {
      const PetscInt nextc = startc + lc[i];
      /* Move the first fine node of the next subdomain to the right until the coarse node on its left is within one
       * coarse stencil width of the first coarse node in the next subdomain. */
      while ((startf+want)/ratio < nextc - stencil_width) want++;
      /* Move the last fine node in the current subdomain to the left until the coarse node on its right is within one
       * coarse stencil width of the last coarse node in the current subdomain. */
      while ((startf+want-1+ratio-1)/ratio > nextc-1+stencil_width) want--;
      /* Make sure all constraints are satisfied */
      if (want < 0 || want > remaining || ((startf+want)/ratio < nextc - stencil_width)
          || ((startf+want-1+ratio-1)/ratio > nextc-1+stencil_width)) SETERRQ(((PetscObject)da)->comm,PETSC_ERR_ARG_SIZ,"Could not find a compatible refined ownership range");
    }
    lf[i] = want;
    startc += lc[i];
    startf += lf[i];
    remaining -= lf[i];
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDACoarsenOwnershipRanges"
/*
  Creates "balanced" ownership ranges after coarsening, constrained by the need for the
  fine grid boundaries to fall within one stencil width of the coarse partition.

  Uses a greedy algorithm to handle non-ideal layouts, could probably do something better.
*/
static PetscErrorCode DMDACoarsenOwnershipRanges(DM da,PetscBool periodic,PetscInt stencil_width,PetscInt ratio,PetscInt m,const PetscInt lf[],PetscInt lc[])
{
  PetscInt       i,totalf,remaining,startc,startf;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (ratio < 1) SETERRQ1(((PetscObject)da)->comm,PETSC_ERR_USER,"Requested refinement ratio %D must be at least 1",ratio);
  if (ratio == 1) {
    ierr = PetscMemcpy(lc,lf,m*sizeof(lf[0]));CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  for (i=0,totalf=0; i<m; i++) totalf += lf[i];
  remaining = (!periodic) + (totalf - (!periodic)) / ratio;
  for (i=0,startc=0,startf=0; i<m; i++) {
    PetscInt want = remaining/(m-i) + !!(remaining%(m-i));
    if (i == m-1) lc[i] = want;
    else {
      const PetscInt nextf = startf+lf[i];
      /* Slide first coarse node of next subdomain to the left until the coarse node to the left of the first fine
       * node is within one stencil width. */
      while (nextf/ratio < startc+want-stencil_width) want--;
      /* Slide the last coarse node of the current subdomain to the right until the coarse node to the right of the last
       * fine node is within one stencil width. */
      while ((nextf-1+ratio-1)/ratio > startc+want-1+stencil_width) want++;
      if (want < 0 || want > remaining
          || (nextf/ratio < startc+want-stencil_width) || ((nextf-1+ratio-1)/ratio > startc+want-1+stencil_width)) SETERRQ(((PetscObject)da)->comm,PETSC_ERR_ARG_SIZ,"Could not find a compatible coarsened ownership range");
    }
    lc[i] = want;
    startc += lc[i];
    startf += lf[i];
    remaining -= lc[i];
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMRefine_DA"
PetscErrorCode  DMRefine_DA(DM da,MPI_Comm comm,DM *daref)
{
  PetscErrorCode ierr;
  PetscInt       M,N,P,i;
  DM             da2;
  DM_DA          *dd = (DM_DA*)da->data,*dd2;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  PetscValidPointer(daref,3);

  if (dd->bx == DMDA_BOUNDARY_PERIODIC || dd->interptype == DMDA_Q0){
    M = dd->refine_x*dd->M;
  } else {
    M = 1 + dd->refine_x*(dd->M - 1);
  }
  if (dd->by == DMDA_BOUNDARY_PERIODIC || dd->interptype == DMDA_Q0){
    if (dd->dim > 1) {
      N = dd->refine_y*dd->N;
    } else {
      N = 1;
    }
  } else {
    N = 1 + dd->refine_y*(dd->N - 1);
  }
  if (dd->bz == DMDA_BOUNDARY_PERIODIC || dd->interptype == DMDA_Q0){
    if (dd->dim > 2) {
      P = dd->refine_z*dd->P;
    } else {
      P = 1;
    }
  } else {
    P = 1 + dd->refine_z*(dd->P - 1);
  }
  ierr = DMDACreate(((PetscObject)da)->comm,&da2);CHKERRQ(ierr);
  ierr = DMSetOptionsPrefix(da2,((PetscObject)da)->prefix);CHKERRQ(ierr);
  ierr = DMDASetDim(da2,dd->dim);CHKERRQ(ierr);
  ierr = DMDASetSizes(da2,M,N,P);CHKERRQ(ierr);
  ierr = DMDASetNumProcs(da2,dd->m,dd->n,dd->p);CHKERRQ(ierr);
  ierr = DMDASetBoundaryType(da2,dd->bx,dd->by,dd->bz);CHKERRQ(ierr);
  ierr = DMDASetDof(da2,dd->w);CHKERRQ(ierr);
  ierr = DMDASetStencilType(da2,dd->stencil_type);CHKERRQ(ierr);
  ierr = DMDASetStencilWidth(da2,dd->s);CHKERRQ(ierr);
  if (dd->dim == 3) {
    PetscInt *lx,*ly,*lz;
    ierr = PetscMalloc3(dd->m,PetscInt,&lx,dd->n,PetscInt,&ly,dd->p,PetscInt,&lz);CHKERRQ(ierr);
    ierr = DMDARefineOwnershipRanges(da,(PetscBool)(dd->bx == DMDA_BOUNDARY_PERIODIC || dd->interptype == DMDA_Q0),dd->s,dd->refine_x,dd->m,dd->lx,lx);CHKERRQ(ierr);
    ierr = DMDARefineOwnershipRanges(da,(PetscBool)(dd->by == DMDA_BOUNDARY_PERIODIC || dd->interptype == DMDA_Q0),dd->s,dd->refine_y,dd->n,dd->ly,ly);CHKERRQ(ierr);
    ierr = DMDARefineOwnershipRanges(da,(PetscBool)(dd->bz == DMDA_BOUNDARY_PERIODIC || dd->interptype == DMDA_Q0),dd->s,dd->refine_z,dd->p,dd->lz,lz);CHKERRQ(ierr);
    ierr = DMDASetOwnershipRanges(da2,lx,ly,lz);CHKERRQ(ierr);
    ierr = PetscFree3(lx,ly,lz);CHKERRQ(ierr);
  } else if (dd->dim == 2) {
    PetscInt *lx,*ly;
    ierr = PetscMalloc2(dd->m,PetscInt,&lx,dd->n,PetscInt,&ly);CHKERRQ(ierr);
    ierr = DMDARefineOwnershipRanges(da,(PetscBool)(dd->bx == DMDA_BOUNDARY_PERIODIC || dd->interptype == DMDA_Q0),dd->s,dd->refine_x,dd->m,dd->lx,lx);CHKERRQ(ierr);
    ierr = DMDARefineOwnershipRanges(da,(PetscBool)(dd->by == DMDA_BOUNDARY_PERIODIC || dd->interptype == DMDA_Q0),dd->s,dd->refine_y,dd->n,dd->ly,ly);CHKERRQ(ierr);
    ierr = DMDASetOwnershipRanges(da2,lx,ly,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscFree2(lx,ly);CHKERRQ(ierr);
  } else if (dd->dim == 1) {
    PetscInt *lx;
    ierr = PetscMalloc(dd->m*sizeof(PetscInt),&lx);CHKERRQ(ierr);
    ierr = DMDARefineOwnershipRanges(da,(PetscBool)(dd->bx == DMDA_BOUNDARY_PERIODIC || dd->interptype == DMDA_Q0),dd->s,dd->refine_x,dd->m,dd->lx,lx);CHKERRQ(ierr);
    ierr = DMDASetOwnershipRanges(da2,lx,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscFree(lx);CHKERRQ(ierr);
  }
  dd2 = (DM_DA*)da2->data;

  /* allow overloaded (user replaced) operations to be inherited by refinement clones */
  da2->ops->creatematrix        = da->ops->creatematrix;
  /* da2->ops->createinterpolation = da->ops->createinterpolation; this causes problem with SNESVI */
  da2->ops->getcoloring      = da->ops->getcoloring;
  dd2->interptype            = dd->interptype;

  /* copy fill information if given */
  if (dd->dfill) {
    ierr = PetscMalloc((dd->dfill[dd->w]+dd->w+1)*sizeof(PetscInt),&dd2->dfill);CHKERRQ(ierr);
    ierr = PetscMemcpy(dd2->dfill,dd->dfill,(dd->dfill[dd->w]+dd->w+1)*sizeof(PetscInt));CHKERRQ(ierr);
  }
  if (dd->ofill) {
    ierr = PetscMalloc((dd->ofill[dd->w]+dd->w+1)*sizeof(PetscInt),&dd2->ofill);CHKERRQ(ierr);
    ierr = PetscMemcpy(dd2->ofill,dd->ofill,(dd->ofill[dd->w]+dd->w+1)*sizeof(PetscInt));CHKERRQ(ierr);
  }
  /* copy the refine information */
  dd2->coarsen_x = dd2->refine_x = dd->refine_x;
  dd2->coarsen_y = dd2->refine_y = dd->refine_y;
  dd2->coarsen_z = dd2->refine_z = dd->refine_z;

  /* copy vector type information */
  ierr = PetscFree(da2->vectype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(da->vectype,(char**)&da2->vectype);CHKERRQ(ierr);

  dd2->lf = dd->lf;
  dd2->lj = dd->lj;

  da2->leveldown = da->leveldown;
  da2->levelup   = da->levelup + 1;
  ierr = DMSetFromOptions(da2);CHKERRQ(ierr);
  ierr = DMSetUp(da2);CHKERRQ(ierr);
  ierr = DMView_DA_Private(da2);CHKERRQ(ierr);

  /* interpolate coordinates if they are set on the coarse grid */
  if (da->coordinates) {
    DM  cdaf,cdac;
    Vec coordsc,coordsf;
    Mat II;

    ierr = DMGetCoordinateDM(da,&cdac);CHKERRQ(ierr);
    ierr = DMGetCoordinates(da,&coordsc);CHKERRQ(ierr);
    ierr = DMGetCoordinateDM(da2,&cdaf);CHKERRQ(ierr);
    /* force creation of the coordinate vector */
    ierr = DMDASetUniformCoordinates(da2,0.0,1.0,0.0,1.0,0.0,1.0);CHKERRQ(ierr);
    ierr = DMGetCoordinates(da2,&coordsf);CHKERRQ(ierr);
    ierr = DMCreateInterpolation(cdac,cdaf,&II,PETSC_NULL);CHKERRQ(ierr);
    ierr = MatInterpolate(II,coordsc,coordsf);CHKERRQ(ierr);
    ierr = MatDestroy(&II);CHKERRQ(ierr);
  }

  for (i=0; i<da->bs; i++) {
    const char *fieldname;
    ierr = DMDAGetFieldName(da,i,&fieldname);CHKERRQ(ierr);
    ierr = DMDASetFieldName(da2,i,fieldname);CHKERRQ(ierr);
  }

  *daref = da2;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMCoarsen_DA"
PetscErrorCode  DMCoarsen_DA(DM da, MPI_Comm comm,DM *daref)
{
  PetscErrorCode ierr;
  PetscInt       M,N,P,i;
  DM             da2;
  DM_DA          *dd = (DM_DA*)da->data,*dd2;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  PetscValidPointer(daref,3);

  if (dd->bx == DMDA_BOUNDARY_PERIODIC || dd->interptype == DMDA_Q0){
    M = dd->M / dd->coarsen_x;
  } else {
    M = 1 + (dd->M - 1) / dd->coarsen_x;
  }
  if (dd->by == DMDA_BOUNDARY_PERIODIC || dd->interptype == DMDA_Q0){
    if (dd->dim > 1) {
      N = dd->N / dd->coarsen_y;
    } else {
      N = 1;
    }
  } else {
    N = 1 + (dd->N - 1) / dd->coarsen_y;
  }
  if (dd->bz == DMDA_BOUNDARY_PERIODIC || dd->interptype == DMDA_Q0){
    if (dd->dim > 2) {
      P = dd->P / dd->coarsen_z;
    } else {
      P = 1;
    }
  } else {
    P = 1 + (dd->P - 1) / dd->coarsen_z;
  }
  ierr = DMDACreate(((PetscObject)da)->comm,&da2);CHKERRQ(ierr);
  ierr = DMSetOptionsPrefix(da2,((PetscObject)da)->prefix);CHKERRQ(ierr);
  ierr = DMDASetDim(da2,dd->dim);CHKERRQ(ierr);
  ierr = DMDASetSizes(da2,M,N,P);CHKERRQ(ierr);
  ierr = DMDASetNumProcs(da2,dd->m,dd->n,dd->p);CHKERRQ(ierr);
  ierr = DMDASetBoundaryType(da2,dd->bx,dd->by,dd->bz);CHKERRQ(ierr);
  ierr = DMDASetDof(da2,dd->w);CHKERRQ(ierr);
  ierr = DMDASetStencilType(da2,dd->stencil_type);CHKERRQ(ierr);
  ierr = DMDASetStencilWidth(da2,dd->s);CHKERRQ(ierr);
  if (dd->dim == 3) {
    PetscInt *lx,*ly,*lz;
    ierr = PetscMalloc3(dd->m,PetscInt,&lx,dd->n,PetscInt,&ly,dd->p,PetscInt,&lz);CHKERRQ(ierr);
    ierr = DMDACoarsenOwnershipRanges(da,(PetscBool)(dd->bx == DMDA_BOUNDARY_PERIODIC || dd->interptype == DMDA_Q0),dd->s,dd->coarsen_x,dd->m,dd->lx,lx);CHKERRQ(ierr);
    ierr = DMDACoarsenOwnershipRanges(da,(PetscBool)(dd->by == DMDA_BOUNDARY_PERIODIC || dd->interptype == DMDA_Q0),dd->s,dd->coarsen_y,dd->n,dd->ly,ly);CHKERRQ(ierr);
    ierr = DMDACoarsenOwnershipRanges(da,(PetscBool)(dd->bz == DMDA_BOUNDARY_PERIODIC || dd->interptype == DMDA_Q0),dd->s,dd->coarsen_z,dd->p,dd->lz,lz);CHKERRQ(ierr);
    ierr = DMDASetOwnershipRanges(da2,lx,ly,lz);CHKERRQ(ierr);
    ierr = PetscFree3(lx,ly,lz);CHKERRQ(ierr);
  } else if (dd->dim == 2) {
    PetscInt *lx,*ly;
    ierr = PetscMalloc2(dd->m,PetscInt,&lx,dd->n,PetscInt,&ly);CHKERRQ(ierr);
    ierr = DMDACoarsenOwnershipRanges(da,(PetscBool)(dd->bx == DMDA_BOUNDARY_PERIODIC || dd->interptype == DMDA_Q0),dd->s,dd->coarsen_x,dd->m,dd->lx,lx);CHKERRQ(ierr);
    ierr = DMDACoarsenOwnershipRanges(da,(PetscBool)(dd->by == DMDA_BOUNDARY_PERIODIC || dd->interptype == DMDA_Q0),dd->s,dd->coarsen_y,dd->n,dd->ly,ly);CHKERRQ(ierr);
    ierr = DMDASetOwnershipRanges(da2,lx,ly,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscFree2(lx,ly);CHKERRQ(ierr);
  } else if (dd->dim == 1) {
    PetscInt *lx;
    ierr = PetscMalloc(dd->m*sizeof(PetscInt),&lx);CHKERRQ(ierr);
    ierr = DMDACoarsenOwnershipRanges(da,(PetscBool)(dd->bx == DMDA_BOUNDARY_PERIODIC || dd->interptype == DMDA_Q0),dd->s,dd->coarsen_x,dd->m,dd->lx,lx);CHKERRQ(ierr);
    ierr = DMDASetOwnershipRanges(da2,lx,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscFree(lx);CHKERRQ(ierr);
  }
  dd2 = (DM_DA*)da2->data;

  /* allow overloaded (user replaced) operations to be inherited by refinement clones; why are only some inherited and not all? */
  /* da2->ops->createinterpolation = da->ops->createinterpolation; copying this one causes trouble for DMSetVI */
  da2->ops->creatematrix        = da->ops->creatematrix;
  da2->ops->getcoloring      = da->ops->getcoloring;
  dd2->interptype            = dd->interptype;

  /* copy fill information if given */
  if (dd->dfill) {
    ierr = PetscMalloc((dd->dfill[dd->w]+dd->w+1)*sizeof(PetscInt),&dd2->dfill);CHKERRQ(ierr);
    ierr = PetscMemcpy(dd2->dfill,dd->dfill,(dd->dfill[dd->w]+dd->w+1)*sizeof(PetscInt));CHKERRQ(ierr);
  }
  if (dd->ofill) {
    ierr = PetscMalloc((dd->ofill[dd->w]+dd->w+1)*sizeof(PetscInt),&dd2->ofill);CHKERRQ(ierr);
    ierr = PetscMemcpy(dd2->ofill,dd->ofill,(dd->ofill[dd->w]+dd->w+1)*sizeof(PetscInt));CHKERRQ(ierr);
  }
  /* copy the refine information */
  dd2->coarsen_x = dd2->refine_x = dd->coarsen_x;
  dd2->coarsen_y = dd2->refine_y = dd->coarsen_y;
  dd2->coarsen_z = dd2->refine_z = dd->coarsen_z;

  /* copy vector type information */
  ierr = PetscFree(da2->vectype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(da->vectype,(char**)&da2->vectype);CHKERRQ(ierr);

  dd2->lf = dd->lf;
  dd2->lj = dd->lj;

  da2->leveldown = da->leveldown + 1;
  da2->levelup   = da->levelup;
  ierr = DMSetFromOptions(da2);CHKERRQ(ierr);
  ierr = DMSetUp(da2);CHKERRQ(ierr);
  ierr = DMView_DA_Private(da2);CHKERRQ(ierr);

  /* inject coordinates if they are set on the fine grid */
  if (da->coordinates) {
    DM         cdaf,cdac;
    Vec        coordsc,coordsf;
    VecScatter inject;

    ierr = DMGetCoordinateDM(da,&cdaf);CHKERRQ(ierr);
    ierr = DMGetCoordinates(da,&coordsf);CHKERRQ(ierr);
    ierr = DMGetCoordinateDM(da2,&cdac);CHKERRQ(ierr);
    /* force creation of the coordinate vector */
    ierr = DMDASetUniformCoordinates(da2,0.0,1.0,0.0,1.0,0.0,1.0);CHKERRQ(ierr);
    ierr = DMGetCoordinates(da2,&coordsc);CHKERRQ(ierr);

    ierr = DMCreateInjection(cdac,cdaf,&inject);CHKERRQ(ierr);
    ierr = VecScatterBegin(inject,coordsf,coordsc,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(inject  ,coordsf,coordsc,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterDestroy(&inject);CHKERRQ(ierr);
  }

  for (i=0; i<da->bs; i++) {
    const char *fieldname;
    ierr = DMDAGetFieldName(da,i,&fieldname);CHKERRQ(ierr);
    ierr = DMDASetFieldName(da2,i,fieldname);CHKERRQ(ierr);
  }

  *daref = da2;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMRefineHierarchy_DA"
PetscErrorCode  DMRefineHierarchy_DA(DM da,PetscInt nlevels,DM daf[])
{
  PetscErrorCode ierr;
  PetscInt       i,n,*refx,*refy,*refz;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  if (nlevels < 0) SETERRQ(((PetscObject)da)->comm,PETSC_ERR_ARG_OUTOFRANGE,"nlevels cannot be negative");
  if (nlevels == 0) PetscFunctionReturn(0);
  PetscValidPointer(daf,3);

  /* Get refinement factors, defaults taken from the coarse DMDA */
  ierr = PetscMalloc3(nlevels,PetscInt,&refx,nlevels,PetscInt,&refy,nlevels,PetscInt,&refz);CHKERRQ(ierr);
  for (i=0; i<nlevels; i++) {
    ierr = DMDAGetRefinementFactor(da,&refx[i],&refy[i],&refz[i]);CHKERRQ(ierr);
  }
  n = nlevels;
  ierr = PetscOptionsGetIntArray(((PetscObject)da)->prefix,"-da_refine_hierarchy_x",refx,&n,PETSC_NULL);CHKERRQ(ierr);
  n = nlevels;
  ierr = PetscOptionsGetIntArray(((PetscObject)da)->prefix,"-da_refine_hierarchy_y",refy,&n,PETSC_NULL);CHKERRQ(ierr);
  n = nlevels;
  ierr = PetscOptionsGetIntArray(((PetscObject)da)->prefix,"-da_refine_hierarchy_z",refz,&n,PETSC_NULL);CHKERRQ(ierr);

  ierr = DMDASetRefinementFactor(da,refx[0],refy[0],refz[0]);CHKERRQ(ierr);
  ierr = DMRefine(da,((PetscObject)da)->comm,&daf[0]);CHKERRQ(ierr);
  for (i=1; i<nlevels; i++) {
    ierr = DMDASetRefinementFactor(daf[i-1],refx[i],refy[i],refz[i]);CHKERRQ(ierr);
    ierr = DMRefine(daf[i-1],((PetscObject)da)->comm,&daf[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree3(refx,refy,refz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCoarsenHierarchy_DA"
PetscErrorCode  DMCoarsenHierarchy_DA(DM da,PetscInt nlevels,DM dac[])
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  if (nlevels < 0) SETERRQ(((PetscObject)da)->comm,PETSC_ERR_ARG_OUTOFRANGE,"nlevels cannot be negative");
  if (nlevels == 0) PetscFunctionReturn(0);
  PetscValidPointer(dac,3);
  ierr = DMCoarsen(da,((PetscObject)da)->comm,&dac[0]);CHKERRQ(ierr);
  for (i=1; i<nlevels; i++) {
    ierr = DMCoarsen(dac[i-1],((PetscObject)da)->comm,&dac[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
