
/*  --------------------------------------------------------------------

     This file implements a Deflation preconditioner in PETSc as part of PC.
     You can use this as a starting point for implementing your own
     preconditioner that is not provided with PETSc. (You might also consider
     just using PCSHELL)

     The following basic routines are required for each preconditioner.
          PCCreate_XXX()          - Creates a preconditioner context
          PCSetFromOptions_XXX()  - Sets runtime options
          PCApply_XXX()           - Applies the preconditioner
          PCDestroy_XXX()         - Destroys the preconditioner context
     where the suffix "_XXX" denotes a particular implementation, in
     this case we use _Deflation (e.g., PCCreate_Deflation, PCApply_Deflation).
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
     and PCApplySymmetricRight().  The Deflation implementation is
     PCApplySymmetricLeftOrRight_Deflation().

    -------------------------------------------------------------------- */

/*
   Include files needed for the Deflation preconditioner:
     pcimpl.h - private include file intended for use by all preconditioners
*/

#include <petsc/private/pcimpl.h>   /*I "petscpc.h" I*/

const char *const PCDeflationTypes[]    = {"INIT","PRE","POST","PCDeflationType","PC_DEFLATION_",0};

/*
   Private context (data structure) for the deflation preconditioner.
*/
typedef struct {
  PetscBool init;            /* do only init step - error correction of direction is omitted */
  PetscBool pre;             /* start with x0 being the solution in the deflation space */
  PetscBool correct;         /* add CP (Qr) correction to descent direction */
  PetscBool truenorm;
  PetscBool adaptiveconv;
  PetscReal adaptiveconst;
  PetscInt  reductionfact;
  Mat       W,Wt,AW,WtAW;    /* deflation space, coarse problem mats */
  KSP       WtAWinv;         /* deflation coarse problem */
  Vec       *work;

  PCDEFLATIONSpaceType spacetype;
  PetscInt             spacesize;
  PetscInt             nestedlvl;
  PetscInt             maxnestedlvl;
  PetscBool            extendsp;
} PC_Deflation;

static PetscErrorCode  PCDeflationSetType_Deflation(PC pc,PCDeflationType type)
{
  PC_Deflation *def = (PC_Deflation*)pc->data;

  PetscFunctionBegin;
  def->init = PETSC_FALSE;
  def->pre = PETSC_FALSE;
  if (type == PC_DEFLATION_INIT) {
    def->init = PETSC_TRUE;
    def->pre  = PETSC_TRUE;
  } else if (type == PC_DEFLATION_PRE) {
    def->pre  = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

/*@
   PCDeflationSetType - Causes the deflation preconditioner to use only a special
    initial gues or pre/post solve solution update

   Logically Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  type - PC_DEFLATION_PRE, PC_DEFLATION_INIT, PC_DEFLATION_POST

   Options Database Key:
.  -pc_deflation_type <pre,init,post>

   Level: intermediate

   Concepts: Deflation preconditioner

.seealso: PCDeflationGetType()
@*/
PetscErrorCode  PCDeflationSetType(PC pc,PCDeflationType type)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCDeflationSetType_C",(PC,PCDeflationType),(pc,type));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCDeflationGetType_Deflation(PC pc,PCDeflationType *type)
{
  PC_Deflation *def = (PC_Deflation*)pc->data;

  PetscFunctionBegin;
  if (def->init) {
    *type = PC_DEFLATION_INIT;
  } else if (def->pre) {
    *type = PC_DEFLATION_PRE;
  } else {
    *type = PC_DEFLATION_POST;
  }
  PetscFunctionReturn(0);
}

/*@
   PCDeflationGetType - Gets how the diagonal matrix is produced for the preconditioner

   Not Collective on PC

   Input Parameter:
.  pc - the preconditioner context

   Output Parameter:
-  type - PC_DEFLATION_PRE, PC_DEFLATION_INIT, PC_DEFLATION_POST

   Level: intermediate

   Concepts: Deflation preconditioner

.seealso: PCDeflationSetType()
@*/
PetscErrorCode  PCDeflationGetType(PC pc,PCDeflationType *type)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscUseMethod(pc,"PCDeflationGetType_C",(PC,PCDeflationType*),(pc,type));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDeflationSetSpace_Deflation(PC pc,Mat W,PetscBool transpose)
{
  PC_Deflation   *def = (PC_Deflation*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (transpose) {
    def->Wt = W;
    def->W = NULL;
  } else {
    def->W = W;
  }
  ierr = PetscObjectReference((PetscObject)W);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* TODO create PCDeflationSetSpaceTranspose? */
/*@
   PCDeflationSetSpace - Set deflation space matrix (or its transpose).

   Logically Collective on PC

   Input Parameters:
+  pc - the preconditioner context
.  W  - deflation matrix
-  tranpose - indicates that W is an explicit transpose of the deflation matrix

   Level: intermediate

.seealso: PCDEFLATION
@*/
PetscErrorCode PCDeflationSetSpace(PC pc,Mat W,PetscBool transpose)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(W,MAT_CLASSID,2);
  PetscValidLogicalCollectiveBool(pc,transpose,3);
  ierr = PetscTryMethod(pc,"PCDeflationSetSpace_C",(PC,Mat,PetscBool),(pc,W,transpose));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/* -------------------------------------------------------------------------- */
/*
   PCSetUp_Deflation - Prepares for the use of the Deflation preconditioner
                    by setting data structures and options.

   Input Parameter:
.  pc - the preconditioner context

   Application Interface Routine: PCSetUp()

   Notes:
   The interface routine PCSetUp() is not usually called directly by
   the user, but instead is called by PCApply() if necessary.
*/
static PetscErrorCode PCSetUp_Deflation(PC pc)
{
  PC_Deflation      *jac = (PC_Deflation*)pc->data;
  Vec            diag,diagsqrt;
  PetscErrorCode ierr;
  PetscInt       n,i;
  PetscScalar    *x;
  PetscBool      zeroflag = PETSC_FALSE;

  PetscFunctionBegin;
  /*
       For most preconditioners the code would begin here something like

  if (pc->setupcalled == 0) { allocate space the first time this is ever called
    ierr = MatCreateVecs(pc->mat,&jac->diag);CHKERRQ(ierr);
    PetscLogObjectParent((PetscObject)pc,(PetscObject)jac->diag);
  }

    But for this preconditioner we want to support use of both the matrix' diagonal
    elements (for left or right preconditioning) and square root of diagonal elements
    (for symmetric preconditioning).  Hence we do not allocate space here, since we
    don't know at this point which will be needed (diag and/or diagsqrt) until the user
    applies the preconditioner, and we don't want to allocate BOTH unless we need
    them both.  Thus, the diag and diagsqrt are allocated in PCSetUp_Deflation_NonSymmetric()
    and PCSetUp_Deflation_Symmetric(), respectively.
  */

  /*
    Here we set up the preconditioner; that is, we copy the diagonal values from
    the matrix and put them into a format to make them quick to apply as a preconditioner.
  */
  if (zeroflag) {
    ierr = PetscInfo(pc,"Zero detected in diagonal of matrix, using 1 at those locations\n");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
/*
   PCApply_Deflation - Applies the Deflation preconditioner to a vector.

   Input Parameters:
.  pc - the preconditioner context
.  x - input vector

   Output Parameter:
.  y - output vector

   Application Interface Routine: PCApply()
 */
static PetscErrorCode PCApply_Deflation(PC pc,Vec x,Vec y)
{
  PC_Deflation      *jac = (PC_Deflation*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
static PetscErrorCode PCReset_Deflation(PC pc)
{
  PC_Deflation      *jac = (PC_Deflation*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

/*
   PCDestroy_Deflation - Destroys the private context for the Deflation preconditioner
   that was created with PCCreate_Deflation().

   Input Parameter:
.  pc - the preconditioner context

   Application Interface Routine: PCDestroy()
*/
static PetscErrorCode PCDestroy_Deflation(PC pc)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCReset_Deflation(pc);CHKERRQ(ierr);

  /*
      Free the private data structure that was hanging off the PC
  */
  ierr = PetscFree(pc->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetFromOptions_Deflation(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PC_Deflation      *jac = (PC_Deflation*)pc->data;
  PetscErrorCode ierr;
  PetscBool      flg;
  PCDeflationType   deflt,type;

  PetscFunctionBegin;
  ierr = PCDeflationGetType(pc,&deflt);CHKERRQ(ierr);
  ierr = PetscOptionsHead(PetscOptionsObject,"Deflation options");CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-pc_jacobi_type","How to construct diagonal matrix","PCDeflationSetType",PCDeflationTypes,(PetscEnum)deflt,(PetscEnum*)&type,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PCDeflationSetType(pc,type);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
     PCDEFLATION - Deflation preconditioner shifts part of the spectrum to zero (deflates)
     or to a predefined value

   Options Database Key:
+    -pc_deflation_type <init,pre,post> - selects approach to deflation (default: pre)
-    -pc_jacobi_abs - use the absolute value of the diagonal entry

   Level: beginner

  Concepts: Deflation, preconditioners

  Notes:
    todo

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC,
           PCDeflationSetType(), PCDeflationSetSpace()
M*/

PETSC_EXTERN PetscErrorCode PCCreate_Deflation(PC pc)
{
  PC_Deflation   *def;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr     = PetscNewLog(pc,&def);CHKERRQ(ierr);
  pc->data = (void*)def;

  def->init          = PETSC_FALSE;
  def->pre           = PETSC_TRUE;
  def->correct       = PETSC_FALSE;
  def->truenorm      = PETSC_TRUE;
  def->reductionfact = -1;
  def->spacetype     = PC_DEFLATION_SPACE_HAAR;
  def->spacesize     = 1;
  def->extendsp      = PETSC_FALSE;
  def->nestedlvl     = 0;
  def->maxnestedlvl  = 0;
  def->adaptiveconv  = PETSC_FALSE;
  def->adaptiveconst = 1.0;

  /*
      Set the pointers for the functions that are provided above.
      Now when the user-level routines (such as PCApply(), PCDestroy(), etc.)
      are called, they will automatically call these functions.  Note we
      choose not to provide a couple of these functions since they are
      not needed.
  */
  pc->ops->apply               = PCApply_Deflation;
  pc->ops->applytranspose      = PCApply_Deflation;
  pc->ops->setup               = PCSetUp_Deflation;
  pc->ops->reset               = PCReset_Deflation;
  pc->ops->destroy             = PCDestroy_Deflation;
  pc->ops->setfromoptions      = PCSetFromOptions_Deflation;
  pc->ops->view                = 0;
  pc->ops->applyrichardson     = 0;
  pc->ops->applysymmetricleft  = 0;
  pc->ops->applysymmetricright = 0;

  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCDeflationSetType_C",PCDeflationSetType_Deflation);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCDeflationGetType_C",PCDeflationGetType_Deflation);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCDeflationSetSpace_C",PCDeflationSetSpace_Deflation);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

