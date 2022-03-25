
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
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidPointer(ctx,2);
  PetscCall(PetscObjectTypeCompare((PetscObject)pc,PCSHELL,&flg));
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
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscCall(PetscObjectTypeCompare((PetscObject)pc,PCSHELL,&flg));
  if (flg) shell->ctx = ctx;
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetUp_Shell(PC pc)
{
  PC_Shell       *shell = (PC_Shell*)pc->data;

  PetscFunctionBegin;
  PetscCheck(shell->setup,PetscObjectComm((PetscObject)pc),PETSC_ERR_USER,"No setup() routine provided to Shell PC");
  PetscStackCall("PCSHELL user function setup()",PetscCall((*shell->setup)(pc)));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApply_Shell(PC pc,Vec x,Vec y)
{
  PC_Shell         *shell = (PC_Shell*)pc->data;
  PetscObjectState instate,outstate;

  PetscFunctionBegin;
  PetscCheck(shell->apply,PetscObjectComm((PetscObject)pc),PETSC_ERR_USER,"No apply() routine provided to Shell PC");
  PetscCall(PetscObjectStateGet((PetscObject)y, &instate));
  PetscStackCall("PCSHELL user function apply()",PetscCall((*shell->apply)(pc,x,y)));
  PetscCall(PetscObjectStateGet((PetscObject)y, &outstate));
  if (instate == outstate) {
    /* increase the state of the output vector since the user did not update its state themselve as should have been done */
    PetscCall(PetscObjectStateIncrease((PetscObject)y));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCMatApply_Shell(PC pc,Mat X,Mat Y)
{
  PC_Shell         *shell = (PC_Shell*)pc->data;
  PetscObjectState instate,outstate;

  PetscFunctionBegin;
  PetscCheck(shell->matapply,PetscObjectComm((PetscObject)pc),PETSC_ERR_USER,"No apply() routine provided to Shell PC");
  PetscCall(PetscObjectStateGet((PetscObject)Y, &instate));
  PetscStackCall("PCSHELL user function apply()",PetscCall((*shell->matapply)(pc,X,Y)));
  PetscCall(PetscObjectStateGet((PetscObject)Y, &outstate));
  if (instate == outstate) {
    /* increase the state of the output vector since the user did not update its state themselve as should have been done */
    PetscCall(PetscObjectStateIncrease((PetscObject)Y));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplySymmetricLeft_Shell(PC pc,Vec x,Vec y)
{
  PC_Shell       *shell = (PC_Shell*)pc->data;

  PetscFunctionBegin;
  PetscCheck(shell->applysymmetricleft,PetscObjectComm((PetscObject)pc),PETSC_ERR_USER,"No apply() routine provided to Shell PC");
  PetscStackCall("PCSHELL user function apply()",PetscCall((*shell->applysymmetricleft)(pc,x,y)));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplySymmetricRight_Shell(PC pc,Vec x,Vec y)
{
  PC_Shell       *shell = (PC_Shell*)pc->data;

  PetscFunctionBegin;
  PetscCheck(shell->applysymmetricright,PetscObjectComm((PetscObject)pc),PETSC_ERR_USER,"No apply() routine provided to Shell PC");
  PetscStackCall("PCSHELL user function apply()",PetscCall((*shell->applysymmetricright)(pc,x,y)));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplyBA_Shell(PC pc,PCSide side,Vec x,Vec y,Vec w)
{
  PC_Shell         *shell = (PC_Shell*)pc->data;
  PetscObjectState instate,outstate;

  PetscFunctionBegin;
  PetscCheck(shell->applyBA,PetscObjectComm((PetscObject)pc),PETSC_ERR_USER,"No applyBA() routine provided to Shell PC");
  PetscCall(PetscObjectStateGet((PetscObject)w, &instate));
  PetscStackCall("PCSHELL user function applyBA()",PetscCall((*shell->applyBA)(pc,side,x,y,w)));
  PetscCall(PetscObjectStateGet((PetscObject)w, &outstate));
  if (instate == outstate) {
    /* increase the state of the output vector since the user did not update its state themselve as should have been done */
    PetscCall(PetscObjectStateIncrease((PetscObject)w));
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

  PetscFunctionBegin;
  PetscCheck(shell->presolve,PetscObjectComm((PetscObject)pc),PETSC_ERR_USER,"No presolve() routine provided to Shell PC");
  PetscStackCall("PCSHELL user function presolve()",PetscCall((*shell->presolve)(pc,ksp,b,x)));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCPostSolve_Shell(PC pc,KSP ksp,Vec b,Vec x)
{
  PC_Shell       *shell = (PC_Shell*)pc->data;

  PetscFunctionBegin;
  PetscCheck(shell->postsolve,PetscObjectComm((PetscObject)pc),PETSC_ERR_USER,"No postsolve() routine provided to Shell PC");
  PetscStackCall("PCSHELL user function postsolve()",PetscCall((*shell->postsolve)(pc,ksp,b,x)));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplyTranspose_Shell(PC pc,Vec x,Vec y)
{
  PC_Shell         *shell = (PC_Shell*)pc->data;
  PetscObjectState instate,outstate;

  PetscFunctionBegin;
  PetscCheck(shell->applytranspose,PetscObjectComm((PetscObject)pc),PETSC_ERR_USER,"No applytranspose() routine provided to Shell PC");
  PetscCall(PetscObjectStateGet((PetscObject)y, &instate));
  PetscStackCall("PCSHELL user function applytranspose()",PetscCall((*shell->applytranspose)(pc,x,y)));
  PetscCall(PetscObjectStateGet((PetscObject)y, &outstate));
  if (instate == outstate) {
    /* increase the state of the output vector since the user did not update its state themself as should have been done */
    PetscCall(PetscObjectStateIncrease((PetscObject)y));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplyRichardson_Shell(PC pc,Vec x,Vec y,Vec w,PetscReal rtol,PetscReal abstol, PetscReal dtol,PetscInt it,PetscBool guesszero,PetscInt *outits,PCRichardsonConvergedReason *reason)
{
  PC_Shell         *shell = (PC_Shell*)pc->data;
  PetscObjectState instate,outstate;

  PetscFunctionBegin;
  PetscCheck(shell->applyrich,PetscObjectComm((PetscObject)pc),PETSC_ERR_USER,"No applyrichardson() routine provided to Shell PC");
  PetscCall(PetscObjectStateGet((PetscObject)y, &instate));
  PetscStackCall("PCSHELL user function applyrichardson()",PetscCall((*shell->applyrich)(pc,x,y,w,rtol,abstol,dtol,it,guesszero,outits,reason)));
  PetscCall(PetscObjectStateGet((PetscObject)y, &outstate));
  if (instate == outstate) {
    /* increase the state of the output vector since the user did not update its state themself as should have been done */
    PetscCall(PetscObjectStateIncrease((PetscObject)y));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDestroy_Shell(PC pc)
{
  PC_Shell       *shell = (PC_Shell*)pc->data;

  PetscFunctionBegin;
  PetscCall(PetscFree(shell->name));
  if (shell->destroy) PetscStackCall("PCSHELL user function destroy()",PetscCall((*shell->destroy)(pc)));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCShellSetDestroy_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCShellSetSetUp_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCShellSetApply_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCShellSetMatApply_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCShellSetApplySymmetricLeft_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCShellSetApplySymmetricRight_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCShellSetApplyBA_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCShellSetPreSolve_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCShellSetPostSolve_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCShellSetView_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCShellSetApplyTranspose_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCShellSetName_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCShellGetName_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCShellSetApplyRichardson_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCPreSolveChangeRHS_C",NULL));
  PetscCall(PetscFree(pc->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCView_Shell(PC pc,PetscViewer viewer)
{
  PC_Shell       *shell = (PC_Shell*)pc->data;
  PetscBool      iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    if (shell->name) {
      PetscCall(PetscViewerASCIIPrintf(viewer,"  %s\n",shell->name));
    } else {
      PetscCall(PetscViewerASCIIPrintf(viewer,"  no name\n"));
    }
  }
  if (shell->view) {
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall((*shell->view)(pc,viewer));
    PetscCall(PetscViewerASCIIPopTab(viewer));
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

  PetscFunctionBegin;
  shell->presolve = presolve;
  if (presolve) {
    pc->ops->presolve = PCPreSolve_Shell;
    PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCPreSolveChangeRHS_C",PCPreSolveChangeRHS_Shell));
  } else {
    pc->ops->presolve = NULL;
    PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCPreSolveChangeRHS_C",NULL));
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

  PetscFunctionBegin;
  PetscCall(PetscFree(shell->name));
  PetscCall(PetscStrallocpy(name,&shell->name));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscCall(PetscTryMethod(pc,"PCShellSetDestroy_C",(PC,PetscErrorCode (*)(PC)),(pc,destroy)));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscCall(PetscTryMethod(pc,"PCShellSetSetUp_C",(PC,PetscErrorCode (*)(PC)),(pc,setup)));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscCall(PetscTryMethod(pc,"PCShellSetView_C",(PC,PetscErrorCode (*)(PC,PetscViewer)),(pc,view)));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscCall(PetscTryMethod(pc,"PCShellSetApply_C",(PC,PetscErrorCode (*)(PC,Vec,Vec)),(pc,apply)));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscCall(PetscTryMethod(pc,"PCShellSetMatApply_C",(PC,PetscErrorCode (*)(PC,Mat,Mat)),(pc,matapply)));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscCall(PetscTryMethod(pc,"PCShellSetApplySymmetricLeft_C",(PC,PetscErrorCode (*)(PC,Vec,Vec)),(pc,apply)));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscCall(PetscTryMethod(pc,"PCShellSetApplySymmetricRight_C",(PC,PetscErrorCode (*)(PC,Vec,Vec)),(pc,apply)));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscCall(PetscTryMethod(pc,"PCShellSetApplyBA_C",(PC,PetscErrorCode (*)(PC,PCSide,Vec,Vec,Vec)),(pc,applyBA)));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscCall(PetscTryMethod(pc,"PCShellSetApplyTranspose_C",(PC,PetscErrorCode (*)(PC,Vec,Vec)),(pc,applytranspose)));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscCall(PetscTryMethod(pc,"PCShellSetPreSolve_C",(PC,PetscErrorCode (*)(PC,KSP,Vec,Vec)),(pc,presolve)));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscCall(PetscTryMethod(pc,"PCShellSetPostSolve_C",(PC,PetscErrorCode (*)(PC,KSP,Vec,Vec)),(pc,postsolve)));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscCall(PetscTryMethod(pc,"PCShellSetName_C",(PC,const char []),(pc,name)));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidPointer(name,2);
  PetscCall(PetscUseMethod(pc,"PCShellGetName_C",(PC,const char*[]),(pc,name)));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscCall(PetscTryMethod(pc,"PCShellSetApplyRichardson_C",(PC,PetscErrorCode (*)(PC,Vec,Vec,Vec,PetscReal,PetscReal,PetscReal,PetscInt,PetscBool,PetscInt*,PCRichardsonConvergedReason*)),(pc,apply)));
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
  PC_Shell       *shell;

  PetscFunctionBegin;
  PetscCall(PetscNewLog(pc,&shell));
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

  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCShellSetDestroy_C",PCShellSetDestroy_Shell));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCShellSetSetUp_C",PCShellSetSetUp_Shell));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCShellSetApply_C",PCShellSetApply_Shell));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCShellSetMatApply_C",PCShellSetMatApply_Shell));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCShellSetApplySymmetricLeft_C",PCShellSetApplySymmetricLeft_Shell));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCShellSetApplySymmetricRight_C",PCShellSetApplySymmetricRight_Shell));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCShellSetApplyBA_C",PCShellSetApplyBA_Shell));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCShellSetPreSolve_C",PCShellSetPreSolve_Shell));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCShellSetPostSolve_C",PCShellSetPostSolve_Shell));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCShellSetView_C",PCShellSetView_Shell));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCShellSetApplyTranspose_C",PCShellSetApplyTranspose_Shell));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCShellSetName_C",PCShellSetName_Shell));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCShellGetName_C",PCShellGetName_Shell));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCShellSetApplyRichardson_C",PCShellSetApplyRichardson_Shell));
  PetscFunctionReturn(0);
}
