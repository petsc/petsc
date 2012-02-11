
#include <../src/ksp/pc/impls/mg/mgimpl.h>       /*I "petscksp.h" I*/
                          /*I "petscpcmg.h"   I*/

#undef __FUNCT__  
#define __FUNCT__ "PCMGDefaultResidual"
/*@C
   PCMGDefaultResidual - Default routine to calculate the residual.

   Collective on Mat and Vec

   Input Parameters:
+  mat - the matrix
.  b   - the right-hand-side
-  x   - the approximate solution
 
   Output Parameter:
.  r - location to store the residual

   Level: advanced

.keywords: MG, default, multigrid, residual

.seealso: PCMGSetResidual()
@*/
PetscErrorCode  PCMGDefaultResidual(Mat mat,Vec b,Vec x,Vec r)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMult(mat,x,r);CHKERRQ(ierr);
  ierr = VecAYPX(r,-1.0,b);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "PCMGGetCoarseSolve"
/*@
   PCMGGetCoarseSolve - Gets the solver context to be used on the coarse grid.

   Not Collective

   Input Parameter:
.  pc - the multigrid context 

   Output Parameter:
.  ksp - the coarse grid solver context 

   Level: advanced

.keywords: MG, multigrid, get, coarse grid
@*/ 
PetscErrorCode  PCMGGetCoarseSolve(PC pc,KSP *ksp)  
{ 
  PC_MG          *mg = (PC_MG*)pc->data;
  PC_MG_Levels   **mglevels = mg->levels;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  *ksp =  mglevels[0]->smoothd;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCMGSetResidual"
/*@C
   PCMGSetResidual - Sets the function to be used to calculate the residual 
   on the lth level. 

   Logically Collective on PC and Mat

   Input Parameters:
+  pc       - the multigrid context
.  l        - the level (0 is coarsest) to supply
.  residual - function used to form residual, if none is provided the previously provide one is used, if no 
              previous one were provided then PCMGDefaultResidual() is used
-  mat      - matrix associated with residual

   Level: advanced

.keywords:  MG, set, multigrid, residual, level

.seealso: PCMGDefaultResidual()
@*/
PetscErrorCode  PCMGSetResidual(PC pc,PetscInt l,PetscErrorCode (*residual)(Mat,Vec,Vec,Vec),Mat mat) 
{
  PC_MG          *mg = (PC_MG*)pc->data;
  PC_MG_Levels   **mglevels = mg->levels;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  if (!mglevels) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_ARG_WRONGSTATE,"Must set MG levels before calling");
  if (residual) {
    mglevels[l]->residual = residual;  
  } if (!mglevels[l]->residual) {
    mglevels[l]->residual = PCMGDefaultResidual;
  }
  if (mat) {ierr = PetscObjectReference((PetscObject)mat);CHKERRQ(ierr);}
  ierr = MatDestroy(&mglevels[l]->A);CHKERRQ(ierr);
  mglevels[l]->A        = mat;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCMGSetInterpolation"
/*@
   PCMGSetInterpolation - Sets the function to be used to calculate the 
   interpolation from l-1 to the lth level

   Logically Collective on PC and Mat

   Input Parameters:
+  pc  - the multigrid context
.  mat - the interpolation operator
-  l   - the level (0 is coarsest) to supply [do not supply 0]

   Level: advanced

   Notes:
          Usually this is the same matrix used also to set the restriction
    for the same level.

          One can pass in the interpolation matrix or its transpose; PETSc figures
    out from the matrix size which one it is.

.keywords:  multigrid, set, interpolate, level

.seealso: PCMGSetRestriction()
@*/
PetscErrorCode  PCMGSetInterpolation(PC pc,PetscInt l,Mat mat)
{ 
  PC_MG          *mg = (PC_MG*)pc->data;
  PC_MG_Levels   **mglevels = mg->levels;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  if (!mglevels) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_ARG_WRONGSTATE,"Must set MG levels before calling");
  if (!l) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Do not set interpolation routine for coarsest level");
  ierr = PetscObjectReference((PetscObject)mat);CHKERRQ(ierr);
  ierr = MatDestroy(&mglevels[l]->interpolate);CHKERRQ(ierr);
  mglevels[l]->interpolate = mat;  
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCMGGetInterpolation"
/*@
   PCMGGetInterpolation - Gets the function to be used to calculate the 
   interpolation from l-1 to the lth level

   Logically Collective on PC

   Input Parameters:
+  pc - the multigrid context 
-  l - the level (0 is coarsest) to supply [Do not supply 0]

   Output Parameter:
.  mat - the interpolation matrix

   Level: advanced

.keywords: MG, get, multigrid, interpolation, level

.seealso: PCMGGetRestriction(), PCMGSetInterpolation(), PCMGGetRScale()
@*/
PetscErrorCode  PCMGGetInterpolation(PC pc,PetscInt l,Mat *mat)
{
  PC_MG          *mg = (PC_MG*)pc->data;
  PC_MG_Levels   **mglevels = mg->levels;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidPointer(mat,3);
  if (!mglevels) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_ARG_WRONGSTATE,"Must set MG levels before calling");
  if (l <= 0 || mg->nlevels <= l) SETERRQ2(((PetscObject)pc)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Level %D must be in range {1,...,%D}",l,mg->nlevels-1);
  if (!mglevels[l]->interpolate) {
    if (!mglevels[l]->restrct) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_ARG_WRONGSTATE,"Must call PCMGSetInterpolation() or PCMGSetInterpolation()");
    ierr = PCMGSetInterpolation(pc,l,mglevels[l]->restrct);CHKERRQ(ierr);
  }
  *mat = mglevels[l]->interpolate;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCMGSetRestriction"
/*@
   PCMGSetRestriction - Sets the function to be used to restrict vector
   from level l to l-1. 

   Logically Collective on PC and Mat

   Input Parameters:
+  pc - the multigrid context 
.  l - the level (0 is coarsest) to supply [Do not supply 0]
-  mat - the restriction matrix

   Level: advanced

   Notes: 
          Usually this is the same matrix used also to set the interpolation
    for the same level.

          One can pass in the interpolation matrix or its transpose; PETSc figures
    out from the matrix size which one it is.

         If you do not set this, the transpose of the Mat set with PCMGSetInterpolation()
    is used.

.keywords: MG, set, multigrid, restriction, level

.seealso: PCMGSetInterpolation()
@*/
PetscErrorCode  PCMGSetRestriction(PC pc,PetscInt l,Mat mat)  
{
  PetscErrorCode ierr;
  PC_MG          *mg = (PC_MG*)pc->data;
  PC_MG_Levels   **mglevels = mg->levels;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(mat,MAT_CLASSID,3);
  if (!mglevels) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_ARG_WRONGSTATE,"Must set MG levels before calling");
  if (!l) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Do not set restriction routine for coarsest level");
  ierr = PetscObjectReference((PetscObject)mat);CHKERRQ(ierr);
  ierr = MatDestroy(&mglevels[l]->restrct);CHKERRQ(ierr);
  mglevels[l]->restrct  = mat;  
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCMGGetRestriction"
/*@
   PCMGGetRestriction - Gets the function to be used to restrict vector
   from level l to l-1. 

   Logically Collective on PC and Mat

   Input Parameters:
+  pc - the multigrid context 
-  l - the level (0 is coarsest) to supply [Do not supply 0]

   Output Parameter:
.  mat - the restriction matrix

   Level: advanced

.keywords: MG, get, multigrid, restriction, level

.seealso: PCMGGetInterpolation(), PCMGSetRestriction(), PCMGGetRScale()
@*/
PetscErrorCode  PCMGGetRestriction(PC pc,PetscInt l,Mat *mat)
{
  PC_MG          *mg = (PC_MG*)pc->data;
  PC_MG_Levels   **mglevels = mg->levels;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidPointer(mat,3);
  if (!mglevels) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_ARG_WRONGSTATE,"Must set MG levels before calling");
  if (l <= 0 || mg->nlevels <= l) SETERRQ2(((PetscObject)pc)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Level %D must be in range {1,...,%D}",l,mg->nlevels-1);
  if (!mglevels[l]->restrct) {
    if (!mglevels[l]->interpolate) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_ARG_WRONGSTATE,"Must call PCMGSetRestriction() or PCMGSetInterpolation()");
    ierr = PCMGSetRestriction(pc,l,mglevels[l]->interpolate);CHKERRQ(ierr);
  }
  *mat = mglevels[l]->restrct;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCMGSetRScale"
/*@
   PCMGSetRScale - Sets the pointwise scaling for the restriction operator from level l to l-1.

   Logically Collective on PC and Vec

   Input Parameters:
+  pc - the multigrid context
-  l - the level (0 is coarsest) to supply [Do not supply 0]
.  rscale - the scaling

   Level: advanced

   Notes: 
       When evaluating a function on a coarse level one does not want to do F( R * x) one does F( rscale * R * x) where rscale is 1 over the row sums of R. 

.keywords: MG, set, multigrid, restriction, level

.seealso: PCMGSetInterpolation(), PCMGSetRestriction(), PCMGGetRScale()
@*/
PetscErrorCode  PCMGSetRScale(PC pc,PetscInt l,Vec rscale)
{
  PetscErrorCode ierr;
  PC_MG          *mg = (PC_MG*)pc->data;
  PC_MG_Levels   **mglevels = mg->levels;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  if (!mglevels) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_ARG_WRONGSTATE,"Must set MG levels before calling");
  if (l <= 0 || mg->nlevels <= l) SETERRQ2(((PetscObject)pc)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Level %D must be in range {1,...,%D}",l,mg->nlevels-1);
  ierr = PetscObjectReference((PetscObject)rscale);CHKERRQ(ierr);
  ierr = VecDestroy(&mglevels[l]->rscale);CHKERRQ(ierr);
  mglevels[l]->rscale  = rscale;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCMGGetRScale"
/*@
   PCMGGetRScale - Gets the pointwise scaling for the restriction operator from level l to l-1.

   Collective on PC

   Input Parameters:
+  pc - the multigrid context
.  rscale - the scaling
-  l - the level (0 is coarsest) to supply [Do not supply 0]

   Level: advanced

   Notes: 
       When evaluating a function on a coarse level one does not want to do F( R * x) one does F( rscale * R * x) where rscale is 1 over the row sums of R. 

.keywords: MG, set, multigrid, restriction, level

.seealso: PCMGSetInterpolation(), PCMGGetRestriction()
@*/
PetscErrorCode PCMGGetRScale(PC pc,PetscInt l,Vec *rscale)
{
  PetscErrorCode ierr;
  PC_MG          *mg = (PC_MG*)pc->data;
  PC_MG_Levels   **mglevels = mg->levels;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  if (!mglevels) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_ARG_WRONGSTATE,"Must set MG levels before calling");
  if (l <= 0 || mg->nlevels <= l) SETERRQ2(((PetscObject)pc)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Level %D must be in range {1,...,%D}",l,mg->nlevels-1);
  if (!mglevels[l]->rscale) {
    Mat R;
    Vec X,Y,coarse,fine;
    PetscInt M,N;
    ierr = PCMGGetRestriction(pc,l,&R);CHKERRQ(ierr);
    ierr = MatGetVecs(R,&X,&Y);CHKERRQ(ierr);
    ierr = MatGetSize(R,&M,&N);CHKERRQ(ierr);
    if (M < N) {fine = X; coarse = Y;}
    else if (N < M) {fine = Y; coarse = X;}
    else SETERRQ(((PetscObject)R)->comm,PETSC_ERR_SUP,"Restriction matrix is square, cannot determine which Vec is coarser");
    ierr = VecSet(fine,1.);CHKERRQ(ierr);
    ierr = MatRestrict(R,fine,coarse);CHKERRQ(ierr);
    ierr = VecDestroy(&fine);CHKERRQ(ierr);
    ierr = VecReciprocal(coarse);CHKERRQ(ierr);
    mglevels[l]->rscale = coarse;
  }
  *rscale = mglevels[l]->rscale;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCMGGetSmoother"
/*@
   PCMGGetSmoother - Gets the KSP context to be used as smoother for 
   both pre- and post-smoothing.  Call both PCMGGetSmootherUp() and 
   PCMGGetSmootherDown() to use different functions for pre- and 
   post-smoothing.

   Not Collective, KSP returned is parallel if PC is 

   Input Parameters:
+  pc - the multigrid context 
-  l - the level (0 is coarsest) to supply

   Ouput Parameters:
.  ksp - the smoother

   Level: advanced

.keywords: MG, get, multigrid, level, smoother, pre-smoother, post-smoother

.seealso: PCMGGetSmootherUp(), PCMGGetSmootherDown()
@*/
PetscErrorCode  PCMGGetSmoother(PC pc,PetscInt l,KSP *ksp)
{
  PC_MG          *mg = (PC_MG*)pc->data;
  PC_MG_Levels   **mglevels = mg->levels;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  *ksp = mglevels[l]->smoothd;  
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCMGGetSmootherUp"
/*@
   PCMGGetSmootherUp - Gets the KSP context to be used as smoother after 
   coarse grid correction (post-smoother). 

   Not Collective, KSP returned is parallel if PC is

   Input Parameters:
+  pc - the multigrid context 
-  l  - the level (0 is coarsest) to supply

   Ouput Parameters:
.  ksp - the smoother

   Level: advanced

   Notes: calling this will result in a different pre and post smoother so you may need to 
         set options on the pre smoother also

.keywords: MG, multigrid, get, smoother, up, post-smoother, level

.seealso: PCMGGetSmootherUp(), PCMGGetSmootherDown()
@*/
PetscErrorCode  PCMGGetSmootherUp(PC pc,PetscInt l,KSP *ksp)
{
  PC_MG          *mg = (PC_MG*)pc->data;
  PC_MG_Levels   **mglevels = mg->levels;
  PetscErrorCode ierr;
  const char     *prefix;
  MPI_Comm       comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  /*
     This is called only if user wants a different pre-smoother from post.
     Thus we check if a different one has already been allocated, 
     if not we allocate it.
  */
  if (!l) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_ARG_OUTOFRANGE,"There is no such thing as a up smoother on the coarse grid");
  if (mglevels[l]->smoothu == mglevels[l]->smoothd) {
    ierr = PetscObjectGetComm((PetscObject)mglevels[l]->smoothd,&comm);CHKERRQ(ierr);
    ierr = KSPGetOptionsPrefix(mglevels[l]->smoothd,&prefix);CHKERRQ(ierr);
    ierr = KSPCreate(comm,&mglevels[l]->smoothu);CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject)mglevels[l]->smoothu,(PetscObject)pc,mglevels[0]->levels-l);CHKERRQ(ierr);
    ierr = KSPSetTolerances(mglevels[l]->smoothu,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,1);CHKERRQ(ierr);
    ierr = KSPSetOptionsPrefix(mglevels[l]->smoothu,prefix);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(pc,mglevels[l]->smoothu);CHKERRQ(ierr);
  }
  if (ksp) *ksp = mglevels[l]->smoothu;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCMGGetSmootherDown"
/*@
   PCMGGetSmootherDown - Gets the KSP context to be used as smoother before 
   coarse grid correction (pre-smoother). 

   Not Collective, KSP returned is parallel if PC is

   Input Parameters:
+  pc - the multigrid context 
-  l  - the level (0 is coarsest) to supply

   Ouput Parameters:
.  ksp - the smoother

   Level: advanced

   Notes: calling this will result in a different pre and post smoother so you may need to 
         set options on the post smoother also

.keywords: MG, multigrid, get, smoother, down, pre-smoother, level

.seealso: PCMGGetSmootherUp(), PCMGGetSmoother()
@*/
PetscErrorCode  PCMGGetSmootherDown(PC pc,PetscInt l,KSP *ksp)
{
  PetscErrorCode ierr;
  PC_MG          *mg = (PC_MG*)pc->data;
  PC_MG_Levels   **mglevels = mg->levels;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  /* make sure smoother up and down are different */
  if (l) {
    ierr = PCMGGetSmootherUp(pc,l,PETSC_NULL);CHKERRQ(ierr);
  }
  *ksp = mglevels[l]->smoothd;  
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCMGSetCyclesOnLevel"
/*@
   PCMGSetCyclesOnLevel - Sets the number of cycles to run on this level. 

   Logically Collective on PC

   Input Parameters:
+  pc - the multigrid context 
.  l  - the level (0 is coarsest) this is to be used for
-  n  - the number of cycles

   Level: advanced

.keywords: MG, multigrid, set, cycles, V-cycle, W-cycle, level

.seealso: PCMGSetCycles()
@*/
PetscErrorCode  PCMGSetCyclesOnLevel(PC pc,PetscInt l,PetscInt c) 
{
  PC_MG          *mg = (PC_MG*)pc->data;
  PC_MG_Levels   **mglevels = mg->levels;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  if (!mglevels) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_ARG_WRONGSTATE,"Must set MG levels before calling");
  PetscValidLogicalCollectiveInt(pc,l,2);
  PetscValidLogicalCollectiveInt(pc,c,3);
  mglevels[l]->cycles  = c;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCMGSetRhs"
/*@
   PCMGSetRhs - Sets the vector space to be used to store the right-hand side
   on a particular level. 

   Logically Collective on PC and Vec

   Input Parameters:
+  pc - the multigrid context 
.  l  - the level (0 is coarsest) this is to be used for
-  c  - the space

   Level: advanced

   Notes: If this is not provided PETSc will automatically generate one.

          You do not need to keep a reference to this vector if you do 
          not need it PCDestroy() will properly free it.

.keywords: MG, multigrid, set, right-hand-side, rhs, level

.seealso: PCMGSetX(), PCMGSetR()
@*/
PetscErrorCode  PCMGSetRhs(PC pc,PetscInt l,Vec c)  
{ 
  PetscErrorCode ierr;
  PC_MG          *mg = (PC_MG*)pc->data;
  PC_MG_Levels   **mglevels = mg->levels;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  if (!mglevels) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_ARG_WRONGSTATE,"Must set MG levels before calling");
  if (l == mglevels[0]->levels-1) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_ARG_INCOMP,"Do not set rhs for finest level");
  ierr = PetscObjectReference((PetscObject)c);CHKERRQ(ierr);
  ierr = VecDestroy(&mglevels[l]->b);CHKERRQ(ierr);
  mglevels[l]->b  = c;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCMGSetX"
/*@
   PCMGSetX - Sets the vector space to be used to store the solution on a 
   particular level.

   Logically Collective on PC and Vec

   Input Parameters:
+  pc - the multigrid context 
.  l - the level (0 is coarsest) this is to be used for
-  c - the space

   Level: advanced

   Notes: If this is not provided PETSc will automatically generate one.

          You do not need to keep a reference to this vector if you do 
          not need it PCDestroy() will properly free it.

.keywords: MG, multigrid, set, solution, level

.seealso: PCMGSetRhs(), PCMGSetR()
@*/
PetscErrorCode  PCMGSetX(PC pc,PetscInt l,Vec c)  
{ 
  PetscErrorCode ierr;
  PC_MG          *mg = (PC_MG*)pc->data;
  PC_MG_Levels   **mglevels = mg->levels;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  if (!mglevels) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_ARG_WRONGSTATE,"Must set MG levels before calling");
  if (l == mglevels[0]->levels-1) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_ARG_INCOMP,"Do not set x for finest level");
  ierr = PetscObjectReference((PetscObject)c);CHKERRQ(ierr);
  ierr = VecDestroy(&mglevels[l]->x);CHKERRQ(ierr);
  mglevels[l]->x  = c;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCMGSetR"
/*@
   PCMGSetR - Sets the vector space to be used to store the residual on a
   particular level. 

   Logically Collective on PC and Vec

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
PetscErrorCode  PCMGSetR(PC pc,PetscInt l,Vec c)
{ 
  PetscErrorCode ierr;
  PC_MG          *mg = (PC_MG*)pc->data;
  PC_MG_Levels   **mglevels = mg->levels;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  if (!mglevels) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_ARG_WRONGSTATE,"Must set MG levels before calling");
  if (!l) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Need not set residual vector for coarse grid");
  ierr = PetscObjectReference((PetscObject)c);CHKERRQ(ierr);
  ierr = VecDestroy(&mglevels[l]->r);CHKERRQ(ierr);
  mglevels[l]->r  = c;
  PetscFunctionReturn(0);
}
