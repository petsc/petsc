#define PETSCKSP_DLL

#include "src/ksp/pc/impls/mg/mgimpl.h"       /*I "petscksp.h" I*/
                          /*I "petscmg.h"   I*/

#undef __FUNCT__  
#define __FUNCT__ "MGDefaultResidual"
/*@C
   MGDefaultResidual - Default routine to calculate the residual.

   Collective on Mat and Vec

   Input Parameters:
+  mat - the matrix
.  b   - the right-hand-side
-  x   - the approximate solution
 
   Output Parameter:
.  r - location to store the residual

   Level: advanced

.keywords: MG, default, multigrid, residual

.seealso: MGSetResidual()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT MGDefaultResidual(Mat mat,Vec b,Vec x,Vec r)
{
  PetscErrorCode ierr;
  PetscScalar    mone = -1.0;

  PetscFunctionBegin;
  ierr = MatMult(mat,x,r);CHKERRQ(ierr);
  ierr = VecAYPX(&mone,b,r);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "MGGetCoarseSolve"
/*@C
   MGGetCoarseSolve - Gets the solver context to be used on the coarse grid.

   Not Collective

   Input Parameter:
.  pc - the multigrid context 

   Output Parameter:
.  ksp - the coarse grid solver context 

   Level: advanced

.keywords: MG, multigrid, get, coarse grid
@*/ 
PetscErrorCode PETSCKSP_DLLEXPORT MGGetCoarseSolve(PC pc,KSP *ksp)  
{ 
  MG *mg = (MG*)pc->data;

  PetscFunctionBegin;
  *ksp =  mg[0]->smoothd;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MGSetResidual"
/*@C
   MGSetResidual - Sets the function to be used to calculate the residual 
   on the lth level. 

   Collective on PC and Mat

   Input Parameters:
+  pc       - the multigrid context
.  l        - the level (0 is coarsest) to supply
.  residual - function used to form residual (usually MGDefaultResidual)
-  mat      - matrix associated with residual

   Level: advanced

.keywords:  MG, set, multigrid, residual, level

.seealso: MGDefaultResidual()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT MGSetResidual(PC pc,PetscInt l,PetscErrorCode (*residual)(Mat,Vec,Vec,Vec),Mat mat) 
{
  MG *mg = (MG*)pc->data;

  PetscFunctionBegin;
  if (!mg) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Must set MG levels before calling");

  mg[l]->residual = residual;  
  mg[l]->A        = mat;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MGSetInterpolate"
/*@
   MGSetInterpolate - Sets the function to be used to calculate the 
   interpolation on the lth level. 

   Collective on PC and Mat

   Input Parameters:
+  pc  - the multigrid context
.  mat - the interpolation operator
-  l   - the level (0 is coarsest) to supply

   Level: advanced

   Notes:
          Usually this is the same matrix used also to set the restriction
    for the same level.

          One can pass in the interpolation matrix or its transpose; PETSc figures
    out from the matrix size which one it is.

         If you do not set this, the transpose of the Mat set with MGSetRestriction()
    is used.

.keywords:  multigrid, set, interpolate, level

.seealso: MGSetRestriction()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT MGSetInterpolate(PC pc,PetscInt l,Mat mat)
{ 
  MG             *mg = (MG*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!mg) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Must set MG levels before calling");
  mg[l]->interpolate = mat;  
  ierr = PetscObjectReference((PetscObject)mat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MGSetRestriction"
/*@
   MGSetRestriction - Sets the function to be used to restrict vector
   from level l to l-1. 

   Collective on PC and Mat

   Input Parameters:
+  pc - the multigrid context 
.  mat - the restriction matrix
-  l - the level (0 is coarsest) to supply

   Level: advanced

   Notes: 
          Usually this is the same matrix used also to set the interpolation
    for the same level.

          One can pass in the interpolation matrix or its transpose; PETSc figures
    out from the matrix size which one it is.

         If you do not set this, the transpose of the Mat set with MGSetInterpolate()
    is used.

.keywords: MG, set, multigrid, restriction, level

.seealso: MGSetInterpolate()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT MGSetRestriction(PC pc,PetscInt l,Mat mat)  
{
  PetscErrorCode ierr;
  MG             *mg = (MG*)pc->data;

  PetscFunctionBegin;
  if (!mg) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Must set MG levels before calling");
  mg[l]->restrct  = mat;  
  ierr = PetscObjectReference((PetscObject)mat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MGGetSmoother"
/*@C
   MGGetSmoother - Gets the KSP context to be used as smoother for 
   both pre- and post-smoothing.  Call both MGGetSmootherUp() and 
   MGGetSmootherDown() to use different functions for pre- and 
   post-smoothing.

   Not Collective, KSP returned is parallel if PC is 

   Input Parameters:
+  pc - the multigrid context 
-  l - the level (0 is coarsest) to supply

   Ouput Parameters:
.  ksp - the smoother

   Level: advanced

.keywords: MG, get, multigrid, level, smoother, pre-smoother, post-smoother

.seealso: MGGetSmootherUp(), MGGetSmootherDown()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT MGGetSmoother(PC pc,PetscInt l,KSP *ksp)
{
  MG *mg = (MG*)pc->data;

  PetscFunctionBegin;
  *ksp = mg[l]->smoothd;  
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MGGetSmootherUp"
/*@C
   MGGetSmootherUp - Gets the KSP context to be used as smoother after 
   coarse grid correction (post-smoother). 

   Not Collective, KSP returned is parallel if PC is

   Input Parameters:
+  pc - the multigrid context 
-  l  - the level (0 is coarsest) to supply

   Ouput Parameters:
.  ksp - the smoother

   Level: advanced

.keywords: MG, multigrid, get, smoother, up, post-smoother, level

.seealso: MGGetSmootherUp(), MGGetSmootherDown()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT MGGetSmootherUp(PC pc,PetscInt l,KSP *ksp)
{
  MG             *mg = (MG*)pc->data;
  PetscErrorCode ierr;
  char           *prefix;
  MPI_Comm       comm;

  PetscFunctionBegin;
  /*
     This is called only if user wants a different pre-smoother from post.
     Thus we check if a different one has already been allocated, 
     if not we allocate it.
  */
  if (!l) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"There is no such thing as a up smoother on the coarse grid");
  if (mg[l]->smoothu == mg[l]->smoothd) {
    ierr = PetscObjectGetComm((PetscObject)mg[l]->smoothd,&comm);CHKERRQ(ierr);
    ierr = KSPGetOptionsPrefix(mg[l]->smoothd,&prefix);CHKERRQ(ierr);
    ierr = KSPCreate(comm,&mg[l]->smoothu);CHKERRQ(ierr);
    ierr = KSPSetTolerances(mg[l]->smoothu,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,1);CHKERRQ(ierr);
    ierr = KSPSetOptionsPrefix(mg[l]->smoothu,prefix);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(pc,mg[l]->smoothu);CHKERRQ(ierr);
  }
  if (ksp) *ksp = mg[l]->smoothu;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MGGetSmootherDown"
/*@C
   MGGetSmootherDown - Gets the KSP context to be used as smoother before 
   coarse grid correction (pre-smoother). 

   Not Collective, KSP returned is parallel if PC is

   Input Parameters:
+  pc - the multigrid context 
-  l  - the level (0 is coarsest) to supply

   Ouput Parameters:
.  ksp - the smoother

   Level: advanced

.keywords: MG, multigrid, get, smoother, down, pre-smoother, level

.seealso: MGGetSmootherUp(), MGGetSmoother()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT MGGetSmootherDown(PC pc,PetscInt l,KSP *ksp)
{
  PetscErrorCode ierr;
  MG             *mg = (MG*)pc->data;

  PetscFunctionBegin;
  /* make sure smoother up and down are different */
  ierr = MGGetSmootherUp(pc,l,PETSC_NULL);CHKERRQ(ierr);
  *ksp = mg[l]->smoothd;  
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MGSetCyclesOnLevel"
/*@
   MGSetCyclesOnLevel - Sets the number of cycles to run on this level. 

   Collective on PC

   Input Parameters:
+  pc - the multigrid context 
.  l  - the level (0 is coarsest) this is to be used for
-  n  - the number of cycles

   Level: advanced

.keywords: MG, multigrid, set, cycles, V-cycle, W-cycle, level

.seealso: MGSetCycles()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT MGSetCyclesOnLevel(PC pc,PetscInt l,PetscInt c) 
{
  MG *mg = (MG*)pc->data;

  PetscFunctionBegin;
  if (!mg) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Must set MG levels before calling");
  mg[l]->cycles  = c;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MGSetRhs"
/*@
   MGSetRhs - Sets the vector space to be used to store the right-hand side
   on a particular level. 

   Collective on PC and Vec

   Input Parameters:
+  pc - the multigrid context 
.  l  - the level (0 is coarsest) this is to be used for
-  c  - the space

   Level: advanced

   Notes: If this is not provided PETSc will automatically generate one.

          You do not need to keep a reference to this vector if you do 
          not need it PCDestroy() will properly free it.

.keywords: MG, multigrid, set, right-hand-side, rhs, level

.seealso: MGSetX(), MGSetR()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT MGSetRhs(PC pc,PetscInt l,Vec c)  
{ 
  PetscErrorCode ierr;
  MG             *mg = (MG*)pc->data;

  PetscFunctionBegin;
  if (!mg) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Must set MG levels before calling");
  if (l == mg[0]->levels-1) SETERRQ(PETSC_ERR_ARG_INCOMP,"Do not set rhs for finest level");
  mg[l]->b  = c;
  ierr = PetscObjectReference((PetscObject)c);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MGSetX"
/*@
   MGSetX - Sets the vector space to be used to store the solution on a 
   particular level.

   Collective on PC and Vec

   Input Parameters:
+  pc - the multigrid context 
.  l - the level (0 is coarsest) this is to be used for
-  c - the space

   Level: advanced

   Notes: If this is not provided PETSc will automatically generate one.

          You do not need to keep a reference to this vector if you do 
          not need it PCDestroy() will properly free it.

.keywords: MG, multigrid, set, solution, level

.seealso: MGSetRhs(), MGSetR()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT MGSetX(PC pc,PetscInt l,Vec c)  
{ 
  PetscErrorCode ierr;
  MG             *mg = (MG*)pc->data;

  PetscFunctionBegin;
  if (!mg) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Must set MG levels before calling");
  if (l == mg[0]->levels-1) SETERRQ(PETSC_ERR_ARG_INCOMP,"Do not set rhs for finest level");
  mg[l]->x  = c;
  ierr = PetscObjectReference((PetscObject)c);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MGSetR"
/*@
   MGSetR - Sets the vector space to be used to store the residual on a
   particular level. 

   Collective on PC and Vec

   Input Parameters:
+  pc - the multigrid context 
.  l - the level (0 is coarsest) this is to be used for
-  c - the space

   Level: advanced

   Notes: If this is not provided PETSc will automatically generate one.

          You do not need to keep a reference to this vector if you do 
          not need it PCDestroy() will properly free it.

.keywords: MG, multigrid, set, residual, level
@*/
PetscErrorCode PETSCKSP_DLLEXPORT MGSetR(PC pc,PetscInt l,Vec c)
{ 
  PetscErrorCode ierr;
  MG             *mg = (MG*)pc->data;

  PetscFunctionBegin;
  if (!mg) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Must set MG levels before calling");
  mg[l]->r  = c;
  ierr = PetscObjectReference((PetscObject)c);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}








