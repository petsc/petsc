
/*
   This provides a simple shell for Fortran (and C programmers) to
  create their own preconditioner without writing much interface code.
*/

#include <petsc/private/pcimpl.h>        /*I "petscpc.h" I*/

typedef struct {
  void *ctx;                     /* user provided contexts for preconditioner */

  PetscErrorCode (*destroy)(PC);
  PetscErrorCode (*setup)(PC);
  PetscErrorCode (*apply)(PC,Vec,Vec);
  PetscErrorCode (*matapply)(PC,Mat,Mat);
  PetscErrorCode (*applysymmetricleft)(PC,Vec,Vec);
  PetscErrorCode (*applysymmetricright)(PC,Vec,Vec);
  PetscErrorCode (*applyBA)(PC,PCSide,Vec,Vec,Vec);
  PetscErrorCode (*presolve)(PC,KSP,Vec,Vec);
  PetscErrorCode (*postsolve)(PC,KSP,Vec,Vec);
  PetscErrorCode (*view)(PC,PetscViewer);
  PetscErrorCode (*applytranspose)(PC,Vec,Vec);
  PetscErrorCode (*applyrich)(PC,Vec,Vec,Vec,PetscReal,PetscReal,PetscReal,PetscInt,PetscBool,PetscInt*,PCRichardsonConvergedReason*);

  char *name;
} PC_Shell;

/*@C
    PCShellGetContext - Returns the user-provided context associated with a shell PC

    Not Collective

    Input Parameter:
.   pc - should have been created with PCSetType(pc,shell)

    Output Parameter:
.   ctx - the user provided context

    Level: advanced

    Notes:
    This routine is intended for use within various shell routines

   Fortran Notes:
    To use this from Fortran you must write a Fortran interface definition for this
    function that tells Fortran the Fortran derived data type that you are passing in as the ctx argument.

.seealso: PCShellSetContext()
@*/
PetscErrorCode  PCShellGetContext(PC pc,void *ctx)
{
  PetscErrorCode ierr;
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidPointer(ctx,2);
  ierr = PetscObjectTypeCompare((PetscObject)pc,PCSHELL,&flg);CHKERRQ(ierr);
  if (!flg) *(void**)ctx = NULL;
  else      *(void**)ctx = ((PC_Shell*)(pc->data))->ctx;
  PetscFunctionReturn(0);
}

/*@
    PCShellSetContext - sets the context for a shell PC

   Logically Collective on PC

    Input Parameters:
+   pc - the shell PC
-   ctx - the context

   Level: advanced

   Fortran Notes:
    To use this from Fortran you must write a Fortran interface definition for this
    function that tells Fortran the Fortran derived data type that you are passing in as the ctx argument.

.seealso: PCShellGetContext(), PCSHELL
@*/
PetscErrorCode  PCShellSetContext(PC pc,void *ctx)
{
  PC_Shell       *shell = (PC_Shell*)pc->data;
  PetscErrorCode ierr;
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)pc,PCSHELL,&flg);CHKERRQ(ierr);
  if (flg) shell->ctx = ctx;
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetUp_Shell(PC pc)
{
  PC_Shell       *shell = (PC_Shell*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!shell->setup) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_USER,"No setup() routine provided to Shell PC");
  PetscStackCall("PCSHELL user function setup()",ierr = (*shell->setup)(pc);CHKERRQ(ierr));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApply_Shell(PC pc,Vec x,Vec y)
{
  PC_Shell         *shell = (PC_Shell*)pc->data;
  PetscErrorCode   ierr;
  PetscObjectState instate,outstate;

  PetscFunctionBegin;
  if (!shell->apply) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_USER,"No apply() routine provided to Shell PC");
  ierr = PetscObjectStateGet((PetscObject)y, &instate);CHKERRQ(ierr);
  PetscStackCall("PCSHELL user function apply()",ierr = (*shell->apply)(pc,x,y);CHKERRQ(ierr));
  ierr = PetscObjectStateGet((PetscObject)y, &outstate);CHKERRQ(ierr);
  if (instate == outstate) {
    /* increase the state of the output vector since the user did not update its state themselve as should have been done */
    ierr = PetscObjectStateIncrease((PetscObject)y);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCMatApply_Shell(PC pc,Mat X,Mat Y)
{
  PC_Shell         *shell = (PC_Shell*)pc->data;
  PetscErrorCode   ierr;
  PetscObjectState instate,outstate;

  PetscFunctionBegin;
  if (!shell->matapply) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_USER,"No apply() routine provided to Shell PC");
  ierr = PetscObjectStateGet((PetscObject)Y, &instate);CHKERRQ(ierr);
  PetscStackCall("PCSHELL user function apply()",ierr = (*shell->matapply)(pc,X,Y);CHKERRQ(ierr));
  ierr = PetscObjectStateGet((PetscObject)Y, &outstate);CHKERRQ(ierr);
  if (instate == outstate) {
    /* increase the state of the output vector since the user did not update its state themselve as should have been done */
    ierr = PetscObjectStateIncrease((PetscObject)Y);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplySymmetricLeft_Shell(PC pc,Vec x,Vec y)
{
  PC_Shell       *shell = (PC_Shell*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!shell->applysymmetricleft) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_USER,"No apply() routine provided to Shell PC");
  PetscStackCall("PCSHELL user function apply()",ierr = (*shell->applysymmetricleft)(pc,x,y);CHKERRQ(ierr));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplySymmetricRight_Shell(PC pc,Vec x,Vec y)
{
  PC_Shell       *shell = (PC_Shell*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!shell->applysymmetricright) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_USER,"No apply() routine provided to Shell PC");
  PetscStackCall("PCSHELL user function apply()",ierr = (*shell->applysymmetricright)(pc,x,y);CHKERRQ(ierr));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplyBA_Shell(PC pc,PCSide side,Vec x,Vec y,Vec w)
{
  PC_Shell         *shell = (PC_Shell*)pc->data;
  PetscErrorCode   ierr;
  PetscObjectState instate,outstate;

  PetscFunctionBegin;
  if (!shell->applyBA) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_USER,"No applyBA() routine provided to Shell PC");
  ierr = PetscObjectStateGet((PetscObject)w, &instate);CHKERRQ(ierr);
  PetscStackCall("PCSHELL user function applyBA()",ierr = (*shell->applyBA)(pc,side,x,y,w);CHKERRQ(ierr));
  ierr = PetscObjectStateGet((PetscObject)w, &outstate);CHKERRQ(ierr);
  if (instate == outstate) {
    /* increase the state of the output vector since the user did not update its state themselve as should have been done */
    ierr = PetscObjectStateIncrease((PetscObject)w);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCPreSolveChangeRHS_Shell(PC pc,PetscBool* change)
{
  PetscFunctionBegin;
  *change = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode PCPreSolve_Shell(PC pc,KSP ksp,Vec b,Vec x)
{
  PC_Shell       *shell = (PC_Shell*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!shell->presolve) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_USER,"No presolve() routine provided to Shell PC");
  PetscStackCall("PCSHELL user function presolve()",ierr = (*shell->presolve)(pc,ksp,b,x);CHKERRQ(ierr));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCPostSolve_Shell(PC pc,KSP ksp,Vec b,Vec x)
{
  PC_Shell       *shell = (PC_Shell*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!shell->postsolve) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_USER,"No postsolve() routine provided to Shell PC");
  PetscStackCall("PCSHELL user function postsolve()",ierr = (*shell->postsolve)(pc,ksp,b,x);CHKERRQ(ierr));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplyTranspose_Shell(PC pc,Vec x,Vec y)
{
  PC_Shell         *shell = (PC_Shell*)pc->data;
  PetscErrorCode   ierr;
  PetscObjectState instate,outstate;

  PetscFunctionBegin;
  if (!shell->applytranspose) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_USER,"No applytranspose() routine provided to Shell PC");
  ierr = PetscObjectStateGet((PetscObject)y, &instate);CHKERRQ(ierr);
  PetscStackCall("PCSHELL user function applytranspose()",ierr = (*shell->applytranspose)(pc,x,y);CHKERRQ(ierr));
  ierr = PetscObjectStateGet((PetscObject)y, &outstate);CHKERRQ(ierr);
  if (instate == outstate) {
    /* increase the state of the output vector since the user did not update its state themself as should have been done */
    ierr = PetscObjectStateIncrease((PetscObject)y);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplyRichardson_Shell(PC pc,Vec x,Vec y,Vec w,PetscReal rtol,PetscReal abstol, PetscReal dtol,PetscInt it,PetscBool guesszero,PetscInt *outits,PCRichardsonConvergedReason *reason)
{
  PetscErrorCode   ierr;
  PC_Shell         *shell = (PC_Shell*)pc->data;
  PetscObjectState instate,outstate;

  PetscFunctionBegin;
  if (!shell->applyrich) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_USER,"No applyrichardson() routine provided to Shell PC");
  ierr = PetscObjectStateGet((PetscObject)y, &instate);CHKERRQ(ierr);
  PetscStackCall("PCSHELL user function applyrichardson()",ierr = (*shell->applyrich)(pc,x,y,w,rtol,abstol,dtol,it,guesszero,outits,reason);CHKERRQ(ierr));
  ierr = PetscObjectStateGet((PetscObject)y, &outstate);CHKERRQ(ierr);
  if (instate == outstate) {
    /* increase the state of the output vector since the user did not update its state themself as should have been done */
    ierr = PetscObjectStateIncrease((PetscObject)y);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDestroy_Shell(PC pc)
{
  PC_Shell       *shell = (PC_Shell*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(shell->name);CHKERRQ(ierr);
  if (shell->destroy) PetscStackCall("PCSHELL user function destroy()",ierr = (*shell->destroy)(pc);CHKERRQ(ierr));
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCShellSetDestroy_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCShellSetSetUp_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCShellSetApply_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCShellSetMatApply_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCShellSetApplySymmetricLeft_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCShellSetApplySymmetricRight_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCShellSetApplyBA_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCShellSetPreSolve_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCShellSetPostSolve_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCShellSetView_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCShellSetApplyTranspose_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCShellSetName_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCShellGetName_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCShellSetApplyRichardson_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCPreSolveChangeRHS_C",NULL);CHKERRQ(ierr);
  ierr = PetscFree(pc->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCView_Shell(PC pc,PetscViewer viewer)
{
  PC_Shell       *shell = (PC_Shell*)pc->data;
  PetscErrorCode ierr;
  PetscBool      iascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    if (shell->name) {
      ierr = PetscViewerASCIIPrintf(viewer,"  %s\n",shell->name);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  no name\n");CHKERRQ(ierr);
    }
  }
  if (shell->view) {
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = (*shell->view)(pc,viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------------------*/
static PetscErrorCode  PCShellSetDestroy_Shell(PC pc, PetscErrorCode (*destroy)(PC))
{
  PC_Shell *shell= (PC_Shell*)pc->data;

  PetscFunctionBegin;
  shell->destroy = destroy;
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCShellSetSetUp_Shell(PC pc, PetscErrorCode (*setup)(PC))
{
  PC_Shell *shell = (PC_Shell*)pc->data;

  PetscFunctionBegin;
  shell->setup = setup;
  if (setup) pc->ops->setup = PCSetUp_Shell;
  else       pc->ops->setup = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCShellSetApply_Shell(PC pc,PetscErrorCode (*apply)(PC,Vec,Vec))
{
  PC_Shell *shell = (PC_Shell*)pc->data;

  PetscFunctionBegin;
  shell->apply = apply;
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCShellSetMatApply_Shell(PC pc,PetscErrorCode (*matapply)(PC,Mat,Mat))
{
  PC_Shell *shell = (PC_Shell*)pc->data;

  PetscFunctionBegin;
  shell->matapply = matapply;
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCShellSetApplySymmetricLeft_Shell(PC pc,PetscErrorCode (*apply)(PC,Vec,Vec))
{
  PC_Shell *shell = (PC_Shell*)pc->data;

  PetscFunctionBegin;
  shell->applysymmetricleft = apply;
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCShellSetApplySymmetricRight_Shell(PC pc,PetscErrorCode (*apply)(PC,Vec,Vec))
{
  PC_Shell *shell = (PC_Shell*)pc->data;

  PetscFunctionBegin;
  shell->applysymmetricright = apply;
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCShellSetApplyBA_Shell(PC pc,PetscErrorCode (*applyBA)(PC,PCSide,Vec,Vec,Vec))
{
  PC_Shell *shell = (PC_Shell*)pc->data;

  PetscFunctionBegin;
  shell->applyBA = applyBA;
  if (applyBA) pc->ops->applyBA  = PCApplyBA_Shell;
  else         pc->ops->applyBA  = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCShellSetPreSolve_Shell(PC pc,PetscErrorCode (*presolve)(PC,KSP,Vec,Vec))
{
  PC_Shell       *shell = (PC_Shell*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  shell->presolve = presolve;
  if (presolve) {
    pc->ops->presolve = PCPreSolve_Shell;
    ierr = PetscObjectComposeFunction((PetscObject)pc,"PCPreSolveChangeRHS_C",PCPreSolveChangeRHS_Shell);CHKERRQ(ierr);
  } else {
    pc->ops->presolve = NULL;
    ierr = PetscObjectComposeFunction((PetscObject)pc,"PCPreSolveChangeRHS_C",NULL);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCShellSetPostSolve_Shell(PC pc,PetscErrorCode (*postsolve)(PC,KSP,Vec,Vec))
{
  PC_Shell *shell = (PC_Shell*)pc->data;

  PetscFunctionBegin;
  shell->postsolve = postsolve;
  if (postsolve) pc->ops->postsolve = PCPostSolve_Shell;
  else           pc->ops->postsolve = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCShellSetView_Shell(PC pc,PetscErrorCode (*view)(PC,PetscViewer))
{
  PC_Shell *shell = (PC_Shell*)pc->data;

  PetscFunctionBegin;
  shell->view = view;
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCShellSetApplyTranspose_Shell(PC pc,PetscErrorCode (*applytranspose)(PC,Vec,Vec))
{
  PC_Shell *shell = (PC_Shell*)pc->data;

  PetscFunctionBegin;
  shell->applytranspose = applytranspose;
  if (applytranspose) pc->ops->applytranspose = PCApplyTranspose_Shell;
  else                pc->ops->applytranspose = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCShellSetApplyRichardson_Shell(PC pc,PetscErrorCode (*applyrich)(PC,Vec,Vec,Vec,PetscReal,PetscReal,PetscReal,PetscInt,PetscBool ,PetscInt*,PCRichardsonConvergedReason*))
{
  PC_Shell *shell = (PC_Shell*)pc->data;

  PetscFunctionBegin;
  shell->applyrich = applyrich;
  if (applyrich) pc->ops->applyrichardson = PCApplyRichardson_Shell;
  else           pc->ops->applyrichardson = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCShellSetName_Shell(PC pc,const char name[])
{
  PC_Shell       *shell = (PC_Shell*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(shell->name);CHKERRQ(ierr);
  ierr = PetscStrallocpy(name,&shell->name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCShellGetName_Shell(PC pc,const char *name[])
{
  PC_Shell *shell = (PC_Shell*)pc->data;

  PetscFunctionBegin;
  *name = shell->name;
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------*/

/*@C
   PCShellSetDestroy - Sets routine to use to destroy the user-provided
   application context.

   Logically Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  destroy - the application-provided destroy routine

   Calling sequence of destroy:
.vb
   PetscErrorCode destroy (PC)
.ve

.  ptr - the application context

   Notes:
    the function MUST return an error code of 0 on success and nonzero on failure.

   Level: developer

.seealso: PCShellSetApply(), PCShellSetContext()
@*/
PetscErrorCode  PCShellSetDestroy(PC pc,PetscErrorCode (*destroy)(PC))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCShellSetDestroy_C",(PC,PetscErrorCode (*)(PC)),(pc,destroy));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PCShellSetSetUp - Sets routine to use to "setup" the preconditioner whenever the
   matrix operator is changed.

   Logically Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  setup - the application-provided setup routine

   Calling sequence of setup:
.vb
   PetscErrorCode setup (PC pc)
.ve

.  pc - the preconditioner, get the application context with PCShellGetContext()

   Notes:
    the function MUST return an error code of 0 on success and nonzero on failure.

   Level: developer

.seealso: PCShellSetApplyRichardson(), PCShellSetApply(), PCShellSetContext()
@*/
PetscErrorCode  PCShellSetSetUp(PC pc,PetscErrorCode (*setup)(PC))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCShellSetSetUp_C",(PC,PetscErrorCode (*)(PC)),(pc,setup));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PCShellSetView - Sets routine to use as viewer of shell preconditioner

   Logically Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  view - the application-provided view routine

   Calling sequence of view:
.vb
   PetscErrorCode view(PC pc,PetscViewer v)
.ve

+  pc - the preconditioner, get the application context with PCShellGetContext()
-  v   - viewer

   Notes:
    the function MUST return an error code of 0 on success and nonzero on failure.

   Level: developer

.seealso: PCShellSetApplyRichardson(), PCShellSetSetUp(), PCShellSetApplyTranspose()
@*/
PetscErrorCode  PCShellSetView(PC pc,PetscErrorCode (*view)(PC,PetscViewer))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCShellSetView_C",(PC,PetscErrorCode (*)(PC,PetscViewer)),(pc,view));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PCShellSetApply - Sets routine to use as preconditioner.

   Logically Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  apply - the application-provided preconditioning routine

   Calling sequence of apply:
.vb
   PetscErrorCode apply (PC pc,Vec xin,Vec xout)
.ve

+  pc - the preconditioner, get the application context with PCShellGetContext()
.  xin - input vector
-  xout - output vector

   Notes:
    the function MUST return an error code of 0 on success and nonzero on failure.

   Level: developer

.seealso: PCShellSetApplyRichardson(), PCShellSetSetUp(), PCShellSetApplyTranspose(), PCShellSetContext(), PCShellSetApplyBA(), PCShellSetApplySymmetricRight(),PCShellSetApplySymmetricLeft()
@*/
PetscErrorCode  PCShellSetApply(PC pc,PetscErrorCode (*apply)(PC,Vec,Vec))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCShellSetApply_C",(PC,PetscErrorCode (*)(PC,Vec,Vec)),(pc,apply));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PCShellSetMatApply - Sets routine to use as preconditioner on a block of vectors.

   Logically Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  apply - the application-provided preconditioning routine

   Calling sequence of apply:
.vb
   PetscErrorCode apply (PC pc,Mat Xin,Mat Xout)
.ve

+  pc - the preconditioner, get the application context with PCShellGetContext()
.  Xin - input block of vectors
-  Xout - output block of vectors

   Notes:
    the function MUST return an error code of 0 on success and nonzero on failure.

   Level: developer

.seealso: PCShellSetApply()
@*/
PetscErrorCode  PCShellSetMatApply(PC pc,PetscErrorCode (*matapply)(PC,Mat,Mat))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCShellSetMatApply_C",(PC,PetscErrorCode (*)(PC,Mat,Mat)),(pc,matapply));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PCShellSetApplySymmetricLeft - Sets routine to use as left preconditioner (when the PC_SYMMETRIC is used).

   Logically Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  apply - the application-provided left preconditioning routine

   Calling sequence of apply:
.vb
   PetscErrorCode apply (PC pc,Vec xin,Vec xout)
.ve

+  pc - the preconditioner, get the application context with PCShellGetContext()
.  xin - input vector
-  xout - output vector

   Notes:
    the function MUST return an error code of 0 on success and nonzero on failure.

   Level: developer

.seealso: PCShellSetApply(), PCShellSetApplySymmetricLeft(), PCShellSetSetUp(), PCShellSetApplyTranspose(), PCShellSetContext()
@*/
PetscErrorCode  PCShellSetApplySymmetricLeft(PC pc,PetscErrorCode (*apply)(PC,Vec,Vec))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCShellSetApplySymmetricLeft_C",(PC,PetscErrorCode (*)(PC,Vec,Vec)),(pc,apply));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PCShellSetApplySymmetricRight - Sets routine to use as right preconditioner (when the PC_SYMMETRIC is used).

   Logically Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  apply - the application-provided right preconditioning routine

   Calling sequence of apply:
.vb
   PetscErrorCode apply (PC pc,Vec xin,Vec xout)
.ve

+  pc - the preconditioner, get the application context with PCShellGetContext()
.  xin - input vector
-  xout - output vector

   Notes:
    the function MUST return an error code of 0 on success and nonzero on failure.

   Level: developer

.seealso: PCShellSetApply(), PCShellSetApplySymmetricLeft(), PCShellSetSetUp(), PCShellSetApplyTranspose(), PCShellSetContext()
@*/
PetscErrorCode  PCShellSetApplySymmetricRight(PC pc,PetscErrorCode (*apply)(PC,Vec,Vec))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCShellSetApplySymmetricRight_C",(PC,PetscErrorCode (*)(PC,Vec,Vec)),(pc,apply));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PCShellSetApplyBA - Sets routine to use as preconditioner times operator.

   Logically Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  applyBA - the application-provided BA routine

   Calling sequence of applyBA:
.vb
   PetscErrorCode applyBA (PC pc,Vec xin,Vec xout)
.ve

+  pc - the preconditioner, get the application context with PCShellGetContext()
.  xin - input vector
-  xout - output vector

   Notes:
    the function MUST return an error code of 0 on success and nonzero on failure.

   Level: developer

.seealso: PCShellSetApplyRichardson(), PCShellSetSetUp(), PCShellSetApplyTranspose(), PCShellSetContext(), PCShellSetApply()
@*/
PetscErrorCode  PCShellSetApplyBA(PC pc,PetscErrorCode (*applyBA)(PC,PCSide,Vec,Vec,Vec))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCShellSetApplyBA_C",(PC,PetscErrorCode (*)(PC,PCSide,Vec,Vec,Vec)),(pc,applyBA));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PCShellSetApplyTranspose - Sets routine to use as preconditioner transpose.

   Logically Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  apply - the application-provided preconditioning transpose routine

   Calling sequence of apply:
.vb
   PetscErrorCode applytranspose (PC pc,Vec xin,Vec xout)
.ve

+  pc - the preconditioner, get the application context with PCShellGetContext()
.  xin - input vector
-  xout - output vector

   Notes:
    the function MUST return an error code of 0 on success and nonzero on failure.

   Level: developer

   Notes:
   Uses the same context variable as PCShellSetApply().

.seealso: PCShellSetApplyRichardson(), PCShellSetSetUp(), PCShellSetApply(), PCSetContext(), PCShellSetApplyBA()
@*/
PetscErrorCode  PCShellSetApplyTranspose(PC pc,PetscErrorCode (*applytranspose)(PC,Vec,Vec))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCShellSetApplyTranspose_C",(PC,PetscErrorCode (*)(PC,Vec,Vec)),(pc,applytranspose));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PCShellSetPreSolve - Sets routine to apply to the operators/vectors before a KSPSolve() is
      applied. This usually does something like scale the linear system in some application
      specific way.

   Logically Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  presolve - the application-provided presolve routine

   Calling sequence of presolve:
.vb
   PetscErrorCode presolve (PC,KSP ksp,Vec b,Vec x)
.ve

+  pc - the preconditioner, get the application context with PCShellGetContext()
.  xin - input vector
-  xout - output vector

   Notes:
    the function MUST return an error code of 0 on success and nonzero on failure.

   Level: developer

.seealso: PCShellSetApplyRichardson(), PCShellSetSetUp(), PCShellSetApplyTranspose(), PCShellSetPostSolve(), PCShellSetContext()
@*/
PetscErrorCode  PCShellSetPreSolve(PC pc,PetscErrorCode (*presolve)(PC,KSP,Vec,Vec))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCShellSetPreSolve_C",(PC,PetscErrorCode (*)(PC,KSP,Vec,Vec)),(pc,presolve));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PCShellSetPostSolve - Sets routine to apply to the operators/vectors before a KSPSolve() is
      applied. This usually does something like scale the linear system in some application
      specific way.

   Logically Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  postsolve - the application-provided presolve routine

   Calling sequence of postsolve:
.vb
   PetscErrorCode postsolve(PC,KSP ksp,Vec b,Vec x)
.ve

+  pc - the preconditioner, get the application context with PCShellGetContext()
.  xin - input vector
-  xout - output vector

   Notes:
    the function MUST return an error code of 0 on success and nonzero on failure.

   Level: developer

.seealso: PCShellSetApplyRichardson(), PCShellSetSetUp(), PCShellSetApplyTranspose(), PCShellSetPreSolve(), PCShellSetContext()
@*/
PetscErrorCode  PCShellSetPostSolve(PC pc,PetscErrorCode (*postsolve)(PC,KSP,Vec,Vec))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCShellSetPostSolve_C",(PC,PetscErrorCode (*)(PC,KSP,Vec,Vec)),(pc,postsolve));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PCShellSetName - Sets an optional name to associate with a shell
   preconditioner.

   Not Collective

   Input Parameters:
+  pc - the preconditioner context
-  name - character string describing shell preconditioner

   Level: developer

.seealso: PCShellGetName()
@*/
PetscErrorCode  PCShellSetName(PC pc,const char name[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCShellSetName_C",(PC,const char []),(pc,name));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PCShellGetName - Gets an optional name that the user has set for a shell
   preconditioner.

   Not Collective

   Input Parameter:
.  pc - the preconditioner context

   Output Parameter:
.  name - character string describing shell preconditioner (you should not free this)

   Level: developer

.seealso: PCShellSetName()
@*/
PetscErrorCode  PCShellGetName(PC pc,const char *name[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidPointer(name,2);
  ierr = PetscUseMethod(pc,"PCShellGetName_C",(PC,const char*[]),(pc,name));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PCShellSetApplyRichardson - Sets routine to use as preconditioner
   in Richardson iteration.

   Logically Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  apply - the application-provided preconditioning routine

   Calling sequence of apply:
.vb
   PetscErrorCode apply (PC pc,Vec b,Vec x,Vec r,PetscReal rtol,PetscReal abstol,PetscReal dtol,PetscInt maxits)
.ve

+  pc - the preconditioner, get the application context with PCShellGetContext()
.  b - right-hand-side
.  x - current iterate
.  r - work space
.  rtol - relative tolerance of residual norm to stop at
.  abstol - absolute tolerance of residual norm to stop at
.  dtol - if residual norm increases by this factor than return
-  maxits - number of iterations to run

   Notes:
    the function MUST return an error code of 0 on success and nonzero on failure.

   Level: developer

.seealso: PCShellSetApply(), PCShellSetContext()
@*/
PetscErrorCode  PCShellSetApplyRichardson(PC pc,PetscErrorCode (*apply)(PC,Vec,Vec,Vec,PetscReal,PetscReal,PetscReal,PetscInt,PetscBool,PetscInt*,PCRichardsonConvergedReason*))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCShellSetApplyRichardson_C",(PC,PetscErrorCode (*)(PC,Vec,Vec,Vec,PetscReal,PetscReal,PetscReal,PetscInt,PetscBool,PetscInt*,PCRichardsonConvergedReason*)),(pc,apply));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
   PCSHELL - Creates a new preconditioner class for use with your
              own private data storage format.

   Level: advanced

  Usage:
$             extern PetscErrorCode apply(PC,Vec,Vec);
$             extern PetscErrorCode applyba(PC,PCSide,Vec,Vec,Vec);
$             extern PetscErrorCode applytranspose(PC,Vec,Vec);
$             extern PetscErrorCode setup(PC);
$             extern PetscErrorCode destroy(PC);
$
$             PCCreate(comm,&pc);
$             PCSetType(pc,PCSHELL);
$             PCShellSetContext(pc,ctx)
$             PCShellSetApply(pc,apply);
$             PCShellSetApplyBA(pc,applyba);               (optional)
$             PCShellSetApplyTranspose(pc,applytranspose); (optional)
$             PCShellSetSetUp(pc,setup);                   (optional)
$             PCShellSetDestroy(pc,destroy);               (optional)

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC,
           MATSHELL, PCShellSetSetUp(), PCShellSetApply(), PCShellSetView(),
           PCShellSetApplyTranspose(), PCShellSetName(), PCShellSetApplyRichardson(),
           PCShellGetName(), PCShellSetContext(), PCShellGetContext(), PCShellSetApplyBA()
M*/

PETSC_EXTERN PetscErrorCode PCCreate_Shell(PC pc)
{
  PetscErrorCode ierr;
  PC_Shell       *shell;

  PetscFunctionBegin;
  ierr     = PetscNewLog(pc,&shell);CHKERRQ(ierr);
  pc->data = (void*)shell;

  pc->ops->destroy         = PCDestroy_Shell;
  pc->ops->view            = PCView_Shell;
  pc->ops->apply           = PCApply_Shell;
  pc->ops->matapply        = PCMatApply_Shell;
  pc->ops->applysymmetricleft  = PCApplySymmetricLeft_Shell;
  pc->ops->applysymmetricright = PCApplySymmetricRight_Shell;
  pc->ops->applytranspose  = NULL;
  pc->ops->applyrichardson = NULL;
  pc->ops->setup           = NULL;
  pc->ops->presolve        = NULL;
  pc->ops->postsolve       = NULL;

  shell->apply          = NULL;
  shell->applytranspose = NULL;
  shell->name           = NULL;
  shell->applyrich      = NULL;
  shell->presolve       = NULL;
  shell->postsolve      = NULL;
  shell->ctx            = NULL;
  shell->setup          = NULL;
  shell->view           = NULL;
  shell->destroy        = NULL;
  shell->applysymmetricleft  = NULL;
  shell->applysymmetricright = NULL;

  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCShellSetDestroy_C",PCShellSetDestroy_Shell);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCShellSetSetUp_C",PCShellSetSetUp_Shell);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCShellSetApply_C",PCShellSetApply_Shell);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCShellSetMatApply_C",PCShellSetMatApply_Shell);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCShellSetApplySymmetricLeft_C",PCShellSetApplySymmetricLeft_Shell);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCShellSetApplySymmetricRight_C",PCShellSetApplySymmetricRight_Shell);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCShellSetApplyBA_C",PCShellSetApplyBA_Shell);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCShellSetPreSolve_C",PCShellSetPreSolve_Shell);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCShellSetPostSolve_C",PCShellSetPostSolve_Shell);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCShellSetView_C",PCShellSetView_Shell);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCShellSetApplyTranspose_C",PCShellSetApplyTranspose_Shell);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCShellSetName_C",PCShellSetName_Shell);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCShellGetName_C",PCShellGetName_Shell);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCShellSetApplyRichardson_C",PCShellSetApplyRichardson_Shell);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
