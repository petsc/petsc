#define PETSCKSP_DLL

/*  -------------------------------------------------------------------- 

     This file implements a Jacobi preconditioner in PETSc as part of PC.
     You can use this as a starting point for implementing your own 
     preconditioner that is not provided with PETSc. (You might also consider
     just using PCSHELL)

     The following basic routines are required for each preconditioner.
          PCCreate_XXX()          - Creates a preconditioner context
          PCSetFromOptions_XXX()  - Sets runtime options
          PCApply_XXX()           - Applies the preconditioner
          PCDestroy_XXX()         - Destroys the preconditioner context
     where the suffix "_XXX" denotes a particular implementation, in
     this case we use _Jacobi (e.g., PCCreate_Jacobi, PCApply_Jacobi).
     These routines are actually called via the common user interface
     routines PCCreate(), PCSetFromOptions(), PCApply(), and PCDestroy(), 
     so the application code interface remains identical for all 
     preconditioners.  

     Another key routine is:
          PCSetUp_XXX()           - Prepares for the use of a preconditioner
     by setting data structures and options.   The interface routine PCSetUp()
     is not usually called directly by the user, but instead is called by
     PCApply() if necessary.

     Additional basic routines are:
          PCView_XXX()            - Prints details of runtime options that
                                    have actually been used.
     These are called by application codes via the interface routines
     PCView().

     The various types of solvers (preconditioners, Krylov subspace methods,
     nonlinear solvers, timesteppers) are all organized similarly, so the
     above description applies to these categories also.  One exception is
     that the analogues of PCApply() for these components are KSPSolve(), 
     SNESSolve(), and TSSolve().

     Additional optional functionality unique to preconditioners is left and
     right symmetric preconditioner application via PCApplySymmetricLeft() 
     and PCApplySymmetricRight().  The Jacobi implementation is 
     PCApplySymmetricLeftOrRight_Jacobi().

    -------------------------------------------------------------------- */

/* 
   Include files needed for the Jacobi preconditioner:
     pcimpl.h - private include file intended for use by all preconditioners 
*/

#include "private/pcimpl.h"   /*I "petscpc.h" I*/

/* 
   Private context (data structure) for the Jacobi preconditioner.  
*/
typedef struct {
  Vec        diag;               /* vector containing the reciprocals of the diagonal elements
                                    of the preconditioner matrix */
  Vec        diagsqrt;           /* vector containing the reciprocals of the square roots of
                                    the diagonal elements of the preconditioner matrix (used 
                                    only for symmetric preconditioner application) */
  PetscTruth userowmax;
  PetscTruth userowsum;
  PetscTruth useabs;             /* use the absolute values of the diagonal entries */
} PC_Jacobi;

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCJacobiSetUseRowMax_Jacobi"
PetscErrorCode PETSCKSP_DLLEXPORT PCJacobiSetUseRowMax_Jacobi(PC pc)
{
  PC_Jacobi *j;

  PetscFunctionBegin;
  j            = (PC_Jacobi*)pc->data;
  j->userowmax = PETSC_TRUE;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCJacobiSetUseRowSum_Jacobi"
PetscErrorCode PETSCKSP_DLLEXPORT PCJacobiSetUseRowSum_Jacobi(PC pc)
{
  PC_Jacobi *j;

  PetscFunctionBegin;
  j            = (PC_Jacobi*)pc->data;
  j->userowsum = PETSC_TRUE;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCJacobiSetUseAbs_Jacobi"
PetscErrorCode PETSCKSP_DLLEXPORT PCJacobiSetUseAbs_Jacobi(PC pc)
{
  PC_Jacobi *j;

  PetscFunctionBegin;
  j         = (PC_Jacobi*)pc->data;
  j->useabs = PETSC_TRUE;
  PetscFunctionReturn(0);
}
EXTERN_C_END

/* -------------------------------------------------------------------------- */
/*
   PCSetUp_Jacobi - Prepares for the use of the Jacobi preconditioner
                    by setting data structures and options.   

   Input Parameter:
.  pc - the preconditioner context

   Application Interface Routine: PCSetUp()

   Notes:
   The interface routine PCSetUp() is not usually called directly by
   the user, but instead is called by PCApply() if necessary.
*/
#undef __FUNCT__  
#define __FUNCT__ "PCSetUp_Jacobi"
static PetscErrorCode PCSetUp_Jacobi(PC pc)
{
  PC_Jacobi      *jac = (PC_Jacobi*)pc->data;
  Vec            diag,diagsqrt;
  PetscErrorCode ierr;
  PetscInt       n,i;
  PetscScalar    *x;
  PetscTruth     zeroflag = PETSC_FALSE;

  PetscFunctionBegin;
  /*
       For most preconditioners the code would begin here something like

  if (pc->setupcalled == 0) { allocate space the first time this is ever called
    ierr = MatGetVecs(pc->mat,&jac->diag);CHKERRQ(ierr);
    PetscLogObjectParent(pc,jac->diag);
  }

    But for this preconditioner we want to support use of both the matrix' diagonal
    elements (for left or right preconditioning) and square root of diagonal elements
    (for symmetric preconditioning).  Hence we do not allocate space here, since we
    don't know at this point which will be needed (diag and/or diagsqrt) until the user
    applies the preconditioner, and we don't want to allocate BOTH unless we need
    them both.  Thus, the diag and diagsqrt are allocated in PCSetUp_Jacobi_NonSymmetric()
    and PCSetUp_Jacobi_Symmetric(), respectively.
  */

  /*
    Here we set up the preconditioner; that is, we copy the diagonal values from
    the matrix and put them into a format to make them quick to apply as a preconditioner.
  */
  diag     = jac->diag;
  diagsqrt = jac->diagsqrt;

  if (diag) {
    if (jac->userowmax) {
      ierr = MatGetRowMaxAbs(pc->pmat,diag,PETSC_NULL);CHKERRQ(ierr);
    } else if (jac->userowsum) {
      ierr = MatGetRowSum(pc->pmat,diag);CHKERRQ(ierr);
    } else {
      ierr = MatGetDiagonal(pc->pmat,diag);CHKERRQ(ierr);
    }
    ierr = VecReciprocal(diag);CHKERRQ(ierr);
    ierr = VecGetLocalSize(diag,&n);CHKERRQ(ierr);
    ierr = VecGetArray(diag,&x);CHKERRQ(ierr);
    if (jac->useabs) {
      for (i=0; i<n; i++) {
        x[i]     = PetscAbsScalar(x[i]);
      }
    }
    for (i=0; i<n; i++) {
      if (x[i] == 0.0) {
        x[i]     = 1.0;
        zeroflag = PETSC_TRUE;
      }
    }
    ierr = VecRestoreArray(diag,&x);CHKERRQ(ierr);
  }
  if (diagsqrt) {
    if (jac->userowmax) {
      ierr = MatGetRowMaxAbs(pc->pmat,diagsqrt,PETSC_NULL);CHKERRQ(ierr);
    } else if (jac->userowsum) {
      ierr = MatGetRowSum(pc->pmat,diagsqrt);CHKERRQ(ierr);
    } else {
      ierr = MatGetDiagonal(pc->pmat,diagsqrt);CHKERRQ(ierr);
    }
    ierr = VecGetLocalSize(diagsqrt,&n);CHKERRQ(ierr);
    ierr = VecGetArray(diagsqrt,&x);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      if (x[i] != 0.0) x[i] = 1.0/sqrt(PetscAbsScalar(x[i]));
      else {
        x[i]     = 1.0;
        zeroflag = PETSC_TRUE;
      }
    }
    ierr = VecRestoreArray(diagsqrt,&x);CHKERRQ(ierr);
  }
  if (zeroflag) {
    ierr = PetscInfo(pc,"Zero detected in diagonal of matrix, using 1 at those locations\n");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
/*
   PCSetUp_Jacobi_Symmetric - Allocates the vector needed to store the
   inverse of the square root of the diagonal entries of the matrix.  This
   is used for symmetric application of the Jacobi preconditioner.

   Input Parameter:
.  pc - the preconditioner context
*/
#undef __FUNCT__  
#define __FUNCT__ "PCSetUp_Jacobi_Symmetric"
static PetscErrorCode PCSetUp_Jacobi_Symmetric(PC pc)
{
  PetscErrorCode ierr;
  PC_Jacobi      *jac = (PC_Jacobi*)pc->data;

  PetscFunctionBegin;
  ierr = MatGetVecs(pc->pmat,&jac->diagsqrt,0);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(pc,jac->diagsqrt);CHKERRQ(ierr);
  ierr = PCSetUp_Jacobi(pc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
/*
   PCSetUp_Jacobi_NonSymmetric - Allocates the vector needed to store the
   inverse of the diagonal entries of the matrix.  This is used for left of
   right application of the Jacobi preconditioner.

   Input Parameter:
.  pc - the preconditioner context
*/
#undef __FUNCT__  
#define __FUNCT__ "PCSetUp_Jacobi_NonSymmetric"
static PetscErrorCode PCSetUp_Jacobi_NonSymmetric(PC pc)
{
  PetscErrorCode ierr;
  PC_Jacobi      *jac = (PC_Jacobi*)pc->data;

  PetscFunctionBegin;
  ierr = MatGetVecs(pc->pmat,&jac->diag,0);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(pc,jac->diag);CHKERRQ(ierr);
  ierr = PCSetUp_Jacobi(pc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
/*
   PCApply_Jacobi - Applies the Jacobi preconditioner to a vector.

   Input Parameters:
.  pc - the preconditioner context
.  x - input vector

   Output Parameter:
.  y - output vector

   Application Interface Routine: PCApply()
 */
#undef __FUNCT__  
#define __FUNCT__ "PCApply_Jacobi"
static PetscErrorCode PCApply_Jacobi(PC pc,Vec x,Vec y)
{
  PC_Jacobi      *jac = (PC_Jacobi*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!jac->diag) {
    ierr = PCSetUp_Jacobi_NonSymmetric(pc);CHKERRQ(ierr);
  }
  ierr = VecPointwiseMult(y,x,jac->diag);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
/*
   PCApplySymmetricLeftOrRight_Jacobi - Applies the left or right part of a
   symmetric preconditioner to a vector.

   Input Parameters:
.  pc - the preconditioner context
.  x - input vector

   Output Parameter:
.  y - output vector

   Application Interface Routines: PCApplySymmetricLeft(), PCApplySymmetricRight()
*/
#undef __FUNCT__  
#define __FUNCT__ "PCApplySymmetricLeftOrRight_Jacobi"
static PetscErrorCode PCApplySymmetricLeftOrRight_Jacobi(PC pc,Vec x,Vec y)
{
  PetscErrorCode ierr;
  PC_Jacobi      *jac = (PC_Jacobi*)pc->data;

  PetscFunctionBegin;
  if (!jac->diagsqrt) {
    ierr = PCSetUp_Jacobi_Symmetric(pc);CHKERRQ(ierr);
  }
  VecPointwiseMult(y,x,jac->diagsqrt);
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
/*
   PCDestroy_Jacobi - Destroys the private context for the Jacobi preconditioner
   that was created with PCCreate_Jacobi().

   Input Parameter:
.  pc - the preconditioner context

   Application Interface Routine: PCDestroy()
*/
#undef __FUNCT__  
#define __FUNCT__ "PCDestroy_Jacobi"
static PetscErrorCode PCDestroy_Jacobi(PC pc)
{
  PC_Jacobi      *jac = (PC_Jacobi*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (jac->diag)     {ierr = VecDestroy(jac->diag);CHKERRQ(ierr);}
  if (jac->diagsqrt) {ierr = VecDestroy(jac->diagsqrt);CHKERRQ(ierr);}

  /*
      Free the private data structure that was hanging off the PC
  */
  ierr = PetscFree(jac);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCSetFromOptions_Jacobi"
static PetscErrorCode PCSetFromOptions_Jacobi(PC pc)
{
  PC_Jacobi      *jac = (PC_Jacobi*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("Jacobi options");CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-pc_jacobi_rowmax","Use row maximums for diagonal","PCJacobiSetUseRowMax",jac->userowmax,
                          &jac->userowmax,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-pc_jacobi_rowsum","Use row sums for diagonal","PCJacobiSetUseRowSum",jac->userowsum,
                          &jac->userowsum,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-pc_jacobi_abs","Use absolute values of diagaonal entries","PCJacobiSetUseAbs",jac->useabs,
                          &jac->useabs,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCCreate_Jacobi - Creates a Jacobi preconditioner context, PC_Jacobi, 
   and sets this as the private data within the generic preconditioning 
   context, PC, that was created within PCCreate().

   Input Parameter:
.  pc - the preconditioner context

   Application Interface Routine: PCCreate()
*/

/*MC
     PCJACOBI - Jacobi (i.e. diagonal scaling preconditioning)

   Options Database Key:
+    -pc_jacobi_rowmax - use the maximum absolute value in each row as the scaling factor,
                        rather than the diagonal
.    -pc_jacobi_rowsum - use the maximum absolute value in each row as the scaling factor,
                        rather than the diagonal
-    -pc_jacobi_abs - use the absolute value of the diagaonl entry

   Level: beginner

  Concepts: Jacobi, diagonal scaling, preconditioners

  Notes: By using KSPSetPreconditionerSide(ksp,PC_SYMMETRIC) or -ksp_symmetric_pc you 
         can scale each side of the matrix by the squareroot of the diagonal entries.

         Zero entries along the diagonal are replaced with the value 1.0

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC,
           PCJacobiSetUseRowMax(), PCJacobiSetUseRowSum(), PCJacobiSetUseAbs()
M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCCreate_Jacobi"
PetscErrorCode PETSCKSP_DLLEXPORT PCCreate_Jacobi(PC pc)
{
  PC_Jacobi      *jac;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*
     Creates the private data structure for this preconditioner and
     attach it to the PC object.
  */
  ierr      = PetscNewLog(pc,PC_Jacobi,&jac);CHKERRQ(ierr);
  pc->data  = (void*)jac;

  /*
     Initialize the pointers to vectors to ZERO; these will be used to store
     diagonal entries of the matrix for fast preconditioner application.
  */
  jac->diag          = 0;
  jac->diagsqrt      = 0;
  jac->userowmax     = PETSC_FALSE;
  jac->userowsum     = PETSC_FALSE;
  jac->useabs        = PETSC_FALSE;

  /*
      Set the pointers for the functions that are provided above.
      Now when the user-level routines (such as PCApply(), PCDestroy(), etc.)
      are called, they will automatically call these functions.  Note we
      choose not to provide a couple of these functions since they are
      not needed.
  */
  pc->ops->apply               = PCApply_Jacobi;
  pc->ops->applytranspose      = PCApply_Jacobi;
  pc->ops->setup               = PCSetUp_Jacobi;
  pc->ops->destroy             = PCDestroy_Jacobi;
  pc->ops->setfromoptions      = PCSetFromOptions_Jacobi;
  pc->ops->view                = 0;
  pc->ops->applyrichardson     = 0;
  pc->ops->applysymmetricleft  = PCApplySymmetricLeftOrRight_Jacobi;
  pc->ops->applysymmetricright = PCApplySymmetricLeftOrRight_Jacobi;
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCJacobiSetUseRowMax_C","PCJacobiSetUseRowMax_Jacobi",PCJacobiSetUseRowMax_Jacobi);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCJacobiSetUseRowSum_C","PCJacobiSetUseRowSum_Jacobi",PCJacobiSetUseRowSum_Jacobi);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCJacobiSetUseAbs_C","PCJacobiSetUseAbs_Jacobi",PCJacobiSetUseAbs_Jacobi);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END


#undef __FUNCT__  
#define __FUNCT__ "PCJacobiSetUseAbs"
/*@
   PCJacobiSetUseAbs - Causes the Jacobi preconditioner to use the 
      absolute value of the diagonal to for the preconditioner

   Collective on PC

   Input Parameters:
.  pc - the preconditioner context


   Options Database Key:
.  -pc_jacobi_abs

   Level: intermediate

   Concepts: Jacobi preconditioner

.seealso: PCJacobiaUseRowMax(), PCJacobiaUseRowSum()

@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCJacobiSetUseAbs(PC pc)
{
  PetscErrorCode ierr,(*f)(PC);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCJacobiSetUseAbs_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCJacobiSetUseRowMax"
/*@
   PCJacobiSetUseRowMax - Causes the Jacobi preconditioner to use the 
      maximum entry in each row as the diagonal preconditioner, instead of
      the diagonal entry

   Collective on PC

   Input Parameters:
.  pc - the preconditioner context


   Options Database Key:
.  -pc_jacobi_rowmax 

   Level: intermediate

   Concepts: Jacobi preconditioner

.seealso: PCJacobiaUseAbs()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCJacobiSetUseRowMax(PC pc)
{
  PetscErrorCode ierr,(*f)(PC);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCJacobiSetUseRowMax_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCJacobiSetUseRowSum"
/*@
   PCJacobiSetUseRowSum - Causes the Jacobi preconditioner to use the 
      sum of each row as the diagonal preconditioner, instead of
      the diagonal entry

   Collective on PC

   Input Parameters:
.  pc - the preconditioner context


   Options Database Key:
.  -pc_jacobi_rowsum

   Level: intermediate

   Concepts: Jacobi preconditioner

.seealso: PCJacobiaUseAbs(), PCJacobiaUseRowSum()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCJacobiSetUseRowSum(PC pc)
{
  PetscErrorCode ierr,(*f)(PC);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCJacobiSetUseRowSum_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

