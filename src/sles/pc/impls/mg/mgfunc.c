/*$Id: mgfunc.c,v 1.41 2001/08/07 03:03:36 balay Exp $*/

#include "src/sles/pc/impls/mg/mgimpl.h"       /*I "petscsles.h" I*/
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
int MGDefaultResidual(Mat mat,Vec b,Vec x,Vec r)
{
  int    ierr;
  PetscScalar mone = -1.0;

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
.  sles - the coarse grid solver context 

   Level: advanced

.keywords: MG, multigrid, get, coarse grid
@*/ 
int MGGetCoarseSolve(PC pc,SLES *sles)  
{ 
  MG *mg = (MG*)pc->data;

  PetscFunctionBegin;
  *sles =  mg[0]->smoothd;
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
int MGSetResidual(PC pc,int l,int (*residual)(Mat,Vec,Vec,Vec),Mat mat) 
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

.keywords:  multigrid, set, interpolate, level

.seealso: MGSetRestriction()
@*/
int MGSetInterpolate(PC pc,int l,Mat mat)
{ 
  MG *mg = (MG*)pc->data;

  PetscFunctionBegin;
  if (!mg) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Must set MG levels before calling");
  mg[l]->interpolate = mat;  
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

.keywords: MG, set, multigrid, restriction, level

.seealso: MGSetInterpolate()
@*/
int MGSetRestriction(PC pc,int l,Mat mat)  
{
  MG *mg = (MG*)pc->data;

  PetscFunctionBegin;
  if (!mg) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Must set MG levels before calling");
  mg[l]->restrct  = mat;  
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MGGetSmoother"
/*@C
   MGGetSmoother - Gets the SLES context to be used as smoother for 
   both pre- and post-smoothing.  Call both MGGetSmootherUp() and 
   MGGetSmootherDown() to use different functions for pre- and 
   post-smoothing.

   Not Collective, SLES returned is parallel if PC is 

   Input Parameters:
+  pc - the multigrid context 
-  l - the level (0 is coarsest) to supply

   Ouput Parameters:
.  sles - the smoother

   Level: advanced

.keywords: MG, get, multigrid, level, smoother, pre-smoother, post-smoother

.seealso: MGGetSmootherUp(), MGGetSmootherDown()
@*/
int MGGetSmoother(PC pc,int l,SLES *sles)
{
  MG *mg = (MG*)pc->data;

  PetscFunctionBegin;
  *sles = mg[l]->smoothd;  
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MGGetSmootherUp"
/*@C
   MGGetSmootherUp - Gets the SLES context to be used as smoother after 
   coarse grid correction (post-smoother). 

   Not Collective, SLES returned is parallel if PC is

   Input Parameters:
+  pc - the multigrid context 
-  l  - the level (0 is coarsest) to supply

   Ouput Parameters:
.  sles - the smoother

   Level: advanced

.keywords: MG, multigrid, get, smoother, up, post-smoother, level

.seealso: MGGetSmootherUp(), MGGetSmootherDown()
@*/
int MGGetSmootherUp(PC pc,int l,SLES *sles)
{
  MG       *mg = (MG*)pc->data;
  int      ierr;
  char     *prefix;
  KSP      ksp;
  MPI_Comm comm;

  PetscFunctionBegin;
  /*
     This is called only if user wants a different pre-smoother from post.
     Thus we check if a different one has already been allocated, 
     if not we allocate it.
  */
  ierr = PCGetOptionsPrefix(pc,&prefix);CHKERRQ(ierr);

  if (mg[l]->smoothu == mg[l]->smoothd) {
    ierr = PetscObjectGetComm((PetscObject)mg[l]->smoothd,&comm);CHKERRQ(ierr);
    ierr = SLESCreate(comm,&mg[l]->smoothu);CHKERRQ(ierr);
    ierr = SLESGetKSP(mg[l]->smoothd,&ksp);CHKERRQ(ierr);
    ierr = KSPSetTolerances(ksp,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,1);CHKERRQ(ierr);
    ierr = SLESSetOptionsPrefix(mg[l]->smoothu,prefix);CHKERRQ(ierr);
    ierr = SLESAppendOptionsPrefix(mg[l]->smoothd,"mg_levels_");CHKERRQ(ierr);
    PetscLogObjectParent(pc,mg[l]->smoothu);
  }
  if (sles) *sles = mg[l]->smoothu;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MGGetSmootherDown"
/*@C
   MGGetSmootherDown - Gets the SLES context to be used as smoother before 
   coarse grid correction (pre-smoother). 

   Not Collective, SLES returned is parallel if PC is

   Input Parameters:
+  pc - the multigrid context 
-  l  - the level (0 is coarsest) to supply

   Ouput Parameters:
.  sles - the smoother

   Level: advanced

.keywords: MG, multigrid, get, smoother, down, pre-smoother, level

.seealso: MGGetSmootherUp(), MGGetSmoother()
@*/
int MGGetSmootherDown(PC pc,int l,SLES *sles)
{
  int ierr;
  MG  *mg = (MG*)pc->data;

  PetscFunctionBegin;
  /* make sure smoother up and down are different */
  ierr = MGGetSmootherUp(pc,l,PETSC_NULL);CHKERRQ(ierr);
  *sles = mg[l]->smoothd;  
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
int MGSetCyclesOnLevel(PC pc,int l,int c) 
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
   on a particular level.  The user should free this space at the conclusion 
   of multigrid use. 

   Collective on PC and Vec

   Input Parameters:
+  pc - the multigrid context 
.  l  - the level (0 is coarsest) this is to be used for
-  c  - the space

   Level: advanced

.keywords: MG, multigrid, set, right-hand-side, rhs, level

.seealso: MGSetX(), MGSetR()
@*/
int MGSetRhs(PC pc,int l,Vec c)  
{ 
  MG *mg = (MG*)pc->data;

  PetscFunctionBegin;
  if (!mg) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Must set MG levels before calling");
  mg[l]->b  = c;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MGSetX"
/*@
   MGSetX - Sets the vector space to be used to store the solution on a 
   particular level.  The user should free this space at the conclusion 
   of multigrid use.

   Collective on PC and Vec

   Input Parameters:
+  pc - the multigrid context 
.  l - the level (0 is coarsest) this is to be used for
-  c - the space

   Level: advanced

.keywords: MG, multigrid, set, solution, level

.seealso: MGSetRhs(), MGSetR()
@*/
int MGSetX(PC pc,int l,Vec c)  
{ 
  MG *mg = (MG*)pc->data;

  PetscFunctionBegin;
  if (!mg) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Must set MG levels before calling");
  mg[l]->x  = c;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MGSetR"
/*@
   MGSetR - Sets the vector space to be used to store the residual on a
   particular level.  The user should free this space at the conclusion of
   multigrid use.

   Collective on PC and Vec

   Input Parameters:
+  pc - the multigrid context 
.  l - the level (0 is coarsest) this is to be used for
-  c - the space

   Level: advanced

.keywords: MG, multigrid, set, residual, level
@*/
int MGSetR(PC pc,int l,Vec c)
{ 
  MG *mg = (MG*)pc->data;

  PetscFunctionBegin;
  if (!mg) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Must set MG levels before calling");
  mg[l]->r  = c;
  PetscFunctionReturn(0);
}








