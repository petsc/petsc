
/*  -------------------------------------------------------------------- 

     This file implements a SupportGraph preconditioner in PETSc as part of PC.
     You can use this as a starting point for implementing your own 
     preconditioner that is not provided with PETSc. (You might also consider
     just using PCSHELL)

     The following basic routines are required for each preconditioner.
          PCCreate_XXX()          - Creates a preconditioner context
          PCSetFromOptions_XXX()  - Sets runtime options
          PCApply_XXX()           - Applies the preconditioner
          PCDestroy_XXX()         - Destroys the preconditioner context
     where the suffix "_XXX" denotes a particular implementation, in
     this case we use _SupportGraph (e.g., PCCreate_SupportGraph, PCApply_SupportGraph).
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
     and PCApplySymmetricRight().  The SupportGraph implementation is 
     PCApplySymmetricLeftOrRight_SupportGraph().

    -------------------------------------------------------------------- */

/* 
   Include files needed for the SupportGraph preconditioner:
     pcimpl.h - private include file intended for use by all preconditioners 
     adjacency_list.hpp
*/

#include <petsc-private/pcimpl.h>   /*I "petscpc.h" I*/

/* 
   Private context (data structure) for the SupportGraph preconditioner.  
*/
typedef struct {
  Mat        pre;      /* Cholesky factored preconditioner matrix */
  PetscBool  augment;  /* whether to augment the spanning tree */
  PetscReal  maxCong;  /* create subgraph with at most this much congestion (only used with augment) */
  PetscReal  tol;      /* throw out entries smaller than this */
} PC_SupportGraph;

#undef __FUNCT__  
#define __FUNCT__ "PCView_SupportGraph"
static PetscErrorCode PCView_SupportGraph(PC pc,PetscViewer viewer)
{
  PC_SupportGraph *sg = (PC_SupportGraph*)pc->data;
  PetscErrorCode  ierr;
  PetscBool       iascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  SupportGraph: maxCong = %f\n",sg->maxCong);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  SupportGraph: tol = %f\n",sg->tol);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  Factored Matrix:\n");CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_INFO);CHKERRQ(ierr);  
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = MatView(sg->pre, viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);  
  } else {
    SETERRQ1(((PetscObject)pc)->comm,PETSC_ERR_SUP,"Viewer type %s not supported for PCSupportGraph",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

extern PetscErrorCode AugmentedLowStretchSpanningTree(Mat mat,Mat *pre,PetscBool  augment,PetscReal tol,PetscReal& maxCong);

/* -------------------------------------------------------------------------- */
/*
   PCSetUp_SupportGraph - Prepares for the use of the SupportGraph preconditioner
                    by setting data structures and options.   

   Input Parameter:
.  pc - the preconditioner context

   Application Interface Routine: PCSetUp()

   Notes:
   The interface routine PCSetUp() is not usually called directly by
   the user, but instead is called by PCApply() if necessary.
*/
#undef __FUNCT__  
#define __FUNCT__ "PCSetUp_SupportGraph"
static PetscErrorCode PCSetUp_SupportGraph(PC pc)
{
  PC_SupportGraph  *sg = (PC_SupportGraph*)pc->data;
  PetscBool        isSym;
  PetscErrorCode   ierr;
  /*
  Vec            diag;
  PetscInt       n,i;
  PetscScalar    *x;
  PetscBool      zeroflag = PETSC_FALSE;
  */

  PetscFunctionBegin;
  if (!pc->setupcalled) {
    ierr = MatIsSymmetric(pc->pmat, 1.0e-9, &isSym);CHKERRQ(ierr);
    if (!isSym) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_ARG_WRONG,"matrix must be symmetric");
    /* note that maxCong is being updated */
    ierr = AugmentedLowStretchSpanningTree(pc->pmat, &sg->pre, sg->augment, sg->tol, sg->maxCong);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


/* -------------------------------------------------------------------------- */
/*
   PCApply_SupportGraph - Applies the SupportGraph preconditioner to a vector.

   Input Parameters:
.  pc - the preconditioner context
.  x - input vector

   Output Parameter:
.  y - output vector

   Application Interface Routine: PCApply()
 */
#undef __FUNCT__  
#define __FUNCT__ "PCApply_SupportGraph"
static PetscErrorCode PCApply_SupportGraph(PC pc,Vec x,Vec y)
{
  PC_SupportGraph      *sg = (PC_SupportGraph*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSolve(sg->pre,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
/*
   PCDestroy_SupportGraph - Destroys the private context for the SupportGraph preconditioner
   that was created with PCCreate_SupportGraph().

   Input Parameter:
.  pc - the preconditioner context

   Application Interface Routine: PCDestroy()
*/
#undef __FUNCT__  
#define __FUNCT__ "PCDestroy_SupportGraph"
static PetscErrorCode PCDestroy_SupportGraph(PC pc)
{
  PC_SupportGraph *sg = (PC_SupportGraph*)pc->data;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = MatDestroy(&sg->pre);CHKERRQ(ierr);
  /*
      Free the private data structure that was hanging off the PC
  */
  ierr = PetscFree(pc->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCSetFromOptions_SupportGraph"
static PetscErrorCode PCSetFromOptions_SupportGraph(PC pc)
{
  PC_SupportGraph *sg = (PC_SupportGraph*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("SupportGraph options");CHKERRQ(ierr);
    ierr = PetscOptionsBool("-pc_sg_augment","Max congestion","",sg->augment,&sg->augment,0);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-pc_sg_cong","Max congestion","",sg->maxCong,&sg->maxCong,0);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-pc_sg_tol","Smallest usable value","",sg->tol,&sg->tol,0);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCCreate_SupportGraph - Creates a SupportGraph preconditioner context, PC_SupportGraph, 
   and sets this as the private data within the generic preconditioning 
   context, PC, that was created within PCCreate().

   Input Parameter:
.  pc - the preconditioner context

   Application Interface Routine: PCCreate()
*/

/*MC
     PCSUPPORTGRAPH - SupportGraph (i.e. diagonal scaling preconditioning)

   Options Database Key:
.    -pc_supportgraph_augment - augment the spanning tree

   Level: beginner

  Concepts: SupportGraph, diagonal scaling, preconditioners

  Notes: Zero entries along the diagonal are replaced with the value 1.0

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC
M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCCreate_SupportGraph"
PetscErrorCode  PCCreate_SupportGraph(PC pc)
{
  PC_SupportGraph      *sg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*
     Creates the private data structure for this preconditioner and
     attach it to the PC object.
  */
  ierr      = PetscNewLog(pc,PC_SupportGraph,&sg);CHKERRQ(ierr);
  pc->data  = (void*)sg;

  sg->pre = 0;
  sg->augment = PETSC_TRUE;
  sg->maxCong = 3.0;
  sg->tol = 0;
  

  /*
      Set the pointers for the functions that are provided above.
      Now when the user-level routines (such as PCApply(), PCDestroy(), etc.)
      are called, they will automatically call these functions.  Note we
      choose not to provide a couple of these functions since they are
      not needed.
  */
  pc->ops->apply               = PCApply_SupportGraph;
  pc->ops->applytranspose      = 0;
  pc->ops->setup               = PCSetUp_SupportGraph;
  pc->ops->destroy             = PCDestroy_SupportGraph;
  pc->ops->setfromoptions      = PCSetFromOptions_SupportGraph;
  pc->ops->view                = PCView_SupportGraph;
  pc->ops->applyrichardson     = 0;
  pc->ops->applysymmetricleft  = 0;
  pc->ops->applysymmetricright = 0;
  PetscFunctionReturn(0);
}
EXTERN_C_END
