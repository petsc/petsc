#include <petsc-private/linesearchimpl.h> /*I "petscsnes.h" I*/

PetscBool  SNESLineSearchRegisterAllCalled = PETSC_FALSE;
PetscFList SNESLineSearchList              = PETSC_NULL;

PetscClassId   SNESLINESEARCH_CLASSID;
PetscLogEvent  SNESLineSearch_Apply;

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchCreate"
/*@
   SNESLineSearchCreate - Creates the line search context.

   Logically Collective on Comm

   Input Parameters:
.  comm - MPI communicator for the line search (typically from the associated SNES context).

   Output Parameters:
.  outlinesearch - the new linesearch context

   Level: beginner

.keywords: LineSearch, create, context

.seealso: LineSearchDestroy()
@*/

PetscErrorCode SNESLineSearchCreate(MPI_Comm comm, SNESLineSearch *outlinesearch) {
  PetscErrorCode      ierr;
  SNESLineSearch     linesearch;
  PetscFunctionBegin;
  PetscValidPointer(outlinesearch,2);
  *outlinesearch = PETSC_NULL;
  ierr = PetscHeaderCreate(linesearch,_p_LineSearch,struct _LineSearchOps,SNESLINESEARCH_CLASSID, 0,
                           "SNESLineSearch","Linesearch","SNESLineSearch",comm,SNESLineSearchDestroy,SNESLineSearchView);CHKERRQ(ierr);

  linesearch->ops->precheckstep = PETSC_NULL;
  linesearch->ops->postcheckstep = PETSC_NULL;

  linesearch->vec_sol_new   = PETSC_NULL;
  linesearch->vec_func_new  = PETSC_NULL;
  linesearch->vec_sol       = PETSC_NULL;
  linesearch->vec_func      = PETSC_NULL;
  linesearch->vec_update    = PETSC_NULL;

  linesearch->lambda        = 1.0;
  linesearch->fnorm         = 1.0;
  linesearch->ynorm         = 1.0;
  linesearch->xnorm         = 1.0;
  linesearch->success       = PETSC_TRUE;
  linesearch->norms         = PETSC_TRUE;
  linesearch->keeplambda    = PETSC_FALSE;
  linesearch->damping       = 1.0;
  linesearch->maxstep       = 1e8;
  linesearch->steptol       = 1e-12;
  linesearch->rtol          = 1e-8;
  linesearch->atol          = 1e-15;
  linesearch->ltol          = 1e-8;
  linesearch->precheckctx   = PETSC_NULL;
  linesearch->postcheckctx  = PETSC_NULL;
  linesearch->max_its       = 1;
  linesearch->setupcalled   = PETSC_FALSE;
  *outlinesearch            = linesearch;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchSetUp"
/*@
   SNESLineSearchSetUp - Prepares the line search for being applied by allocating
   any required vectors.

   Collective on SNESLineSearch

   Input Parameters:
.  linesearch - The LineSearch instance.

   Notes:
   For most cases, this needn't be called outside of SNESLineSearchApply().
   The only current case where this is called outside of this is for the VI
   solvers, which modify the solution and work vectors before the first call
   of SNESLineSearchApply, requiring the SNESLineSearch work vectors to be
   allocated upfront.


   Level: advanced

.keywords: SNESLineSearch, SetUp

.seealso: SNESLineSearchReset()
@*/

PetscErrorCode SNESLineSearchSetUp(SNESLineSearch linesearch) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (!((PetscObject)linesearch)->type_name) {
    ierr = SNESLineSearchSetType(linesearch,SNESLINESEARCHBASIC);CHKERRQ(ierr);
  }
  if (!linesearch->setupcalled) {
    if (!linesearch->vec_sol_new) {
      ierr = VecDuplicate(linesearch->vec_sol, &linesearch->vec_sol_new);CHKERRQ(ierr);
    }
    if (!linesearch->vec_func_new) {
      ierr = VecDuplicate(linesearch->vec_sol, &linesearch->vec_func_new);CHKERRQ(ierr);
    }
    if (linesearch->ops->setup) {
      ierr = (*linesearch->ops->setup)(linesearch);CHKERRQ(ierr);
    }
    linesearch->lambda = linesearch->damping;
    linesearch->setupcalled = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchReset"

/*@
   SNESLineSearchReset - Undoes the SetUp and deletes any Vecs or Mats allocated by the line search.

   Collective on SNESLineSearch

   Input Parameters:
.  linesearch - The LineSearch instance.

   Level: intermediate

.keywords: SNESLineSearch, Reset

.seealso: SNESLineSearchSetUp()
@*/

PetscErrorCode SNESLineSearchReset(SNESLineSearch linesearch) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (linesearch->ops->reset) {
    (*linesearch->ops->reset)(linesearch);
  }
  ierr = VecDestroy(&linesearch->vec_sol_new);CHKERRQ(ierr);
  ierr = VecDestroy(&linesearch->vec_func_new);CHKERRQ(ierr);

  ierr = VecDestroyVecs(linesearch->nwork, &linesearch->work);CHKERRQ(ierr);
  linesearch->nwork = 0;
  linesearch->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchSetPreCheck"
/*@C
   SNESLineSearchSetPreCheck - Sets a pre-check function for the line search routine.

   Logically Collective on SNESLineSearch

   Input Parameters:
+  linesearch - the SNESLineSearch context
.  func       - [optional] function evaluation routine
-  ctx        - [optional] user-defined context for private data for the
                function evaluation routine (may be PETSC_NULL)

   Calling sequence of func:
$    func (SNESLineSearch snes,Vec x,Vec y, PetscBool *changed);

+  x - solution vector
.  y - search direction vector
-  changed - flag to indicate the precheck changed x or y.

   Level: intermediate

.keywords: set, linesearch, pre-check

.seealso: SNESLineSearchSetPostCheck()
@*/
PetscErrorCode  SNESLineSearchSetPreCheck(SNESLineSearch linesearch, SNESLineSearchPreCheckFunc func,void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch,SNESLINESEARCH_CLASSID,1);
  if (func) linesearch->ops->precheckstep = func;
  if (ctx) linesearch->precheckctx = ctx;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchGetPreCheck"
/*@C
   SNESLineSearchSetPreCheck - Gets the pre-check function for the line search routine.

   Input Parameters:
.  linesearch - the SNESLineSearch context

   Output Parameters:
+  func       - [optional] function evaluation routine
-  ctx        - [optional] user-defined context for private data for the
                function evaluation routine (may be PETSC_NULL)

   Level: intermediate

.keywords: get, linesearch, pre-check

.seealso: SNESLineSearchGetPostCheck(), SNESLineSearchSetPreCheck()
@*/
PetscErrorCode  SNESLineSearchGetPreCheck(SNESLineSearch linesearch, SNESLineSearchPreCheckFunc *func,void **ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch,SNESLINESEARCH_CLASSID,1);
  if (func) *func = linesearch->ops->precheckstep;
  if (ctx) *ctx = linesearch->precheckctx;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchSetPostCheck"
/*@C
   SNESLineSearchSetPostCheck - Sets a post-check function for the line search routine.

   Logically Collective on SNESLineSearch

   Input Parameters:
+  linesearch - the SNESLineSearch context
.  func       - [optional] function evaluation routine
-  ctx        - [optional] user-defined context for private data for the
                function evaluation routine (may be PETSC_NULL)

   Calling sequence of func:
$    func (SNESLineSearch linesearch,Vec x,Vec y,  Vec w, *changed_y, PetscBool *changed_w);

+  x - old solution vector
.  y - search direction vector
.  w - new solution vector
.  changed_y - indicates that the line search changed y
.  changed_w - indicates that the line search changed w

   Level: intermediate

.keywords: set, linesearch, post-check

.seealso: SNESLineSearchSetPreCheck()
@*/
PetscErrorCode  SNESLineSearchSetPostCheck(SNESLineSearch linesearch, SNESLineSearchPostCheckFunc func,void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch,SNESLINESEARCH_CLASSID,1);
  if (func) linesearch->ops->postcheckstep = func;
  if (ctx) linesearch->postcheckctx = ctx;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchGetPostCheck"
/*@C
   SNESLineSearchGetPostCheck - Gets the post-check function for the line search routine.

   Input Parameters:
.  linesearch - the SNESLineSearch context

   Output Parameters:
+  func       - [optional] function evaluation routine
-  ctx        - [optional] user-defined context for private data for the
                function evaluation routine (may be PETSC_NULL)

   Level: intermediate

.keywords: get, linesearch, post-check

.seealso: SNESLineSearchGetPreCheck(), SNESLineSearchSetPostCheck()
@*/
PetscErrorCode  SNESLineSearchGetPostCheck(SNESLineSearch linesearch, SNESLineSearchPostCheckFunc *func,void **ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch,SNESLINESEARCH_CLASSID,1);
  if (func) *func = linesearch->ops->postcheckstep;
  if (ctx) *ctx = linesearch->postcheckctx;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchPreCheck"
/*@
   SNESLineSearchPreCheck - Prepares the line search for being applied.

   Logically Collective on SNESLineSearch

   Input Parameters:
+  linesearch - The linesearch instance.
.  X - The current solution
-  Y - The step direction

   Output Parameters:
.  changed - Indicator that the precheck routine has changed anything

   Level: Beginner

.keywords: SNESLineSearch, Create

.seealso: SNESLineSearchPostCheck()
@*/
PetscErrorCode SNESLineSearchPreCheck(SNESLineSearch linesearch,Vec X,Vec Y,PetscBool *changed)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  *changed = PETSC_FALSE;
  if (linesearch->ops->precheckstep) {
    ierr = (*linesearch->ops->precheckstep)(linesearch, X, Y, changed, linesearch->precheckctx);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchPostCheck"
/*@
   SNESLineSearchPostCheck - Prepares the line search for being applied.

   Logically Collective on SNESLineSearch

   Input Parameters:
+  linesearch - The linesearch context
.  X - The last solution
.  Y - The step direction
-  W - The updated solution, W = X + lambda*Y for some lambda

   Output Parameters:
+  changed_Y - Indicator if the direction Y has been changed.
-  changed_W - Indicator if the new candidate solution W has been changed.

   Level: Intermediate

.keywords: SNESLineSearch, Create

.seealso: SNESLineSearchPreCheck()
@*/
PetscErrorCode SNESLineSearchPostCheck(SNESLineSearch linesearch,Vec X,Vec Y,Vec W,PetscBool *changed_Y,PetscBool *changed_W)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  *changed_Y = PETSC_FALSE;
  *changed_W = PETSC_FALSE;
  if (linesearch->ops->postcheckstep) {
    ierr = (*linesearch->ops->postcheckstep)(linesearch,X,Y,W,changed_Y,changed_W,linesearch->postcheckctx);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchPreCheckPicard"
/*@C
   SNESLineSearchPreCheckPicard - Implements a correction that is sometimes useful to improve the convergence rate of Picard iteration

   Logically Collective on SNESLineSearch

   Input Arguments:
+  linesearch - linesearch context
.  X - base state for this step
.  Y - initial correction

   Output Arguments:
+  Y - correction, possibly modified
-  changed - flag indicating that Y was modified

   Options Database Key:
+  -snes_linesearch_precheck_picard - activate this routine
-  -snes_linesearch_precheck_picard_angle - angle

   Level: advanced

   Notes:
   This function should be passed to SNESLineSearchSetPreCheck()

   The justification for this method involves the linear convergence of a Picard iteration
   so the Picard linearization should be provided in place of the "Jacobian". This correction
   is generally not useful when using a Newton linearization.

   Reference:
   Hindmarsh and Payne (1996) Time step limits for stable solutions of the ice sheet equation, Annals of Glaciology.

.seealso: SNESLineSearchSetPreCheck()
@*/
PetscErrorCode SNESLineSearchPreCheckPicard(SNESLineSearch linesearch,Vec X,Vec Y,PetscBool *changed,void *ctx)
{
  PetscErrorCode ierr;
  PetscReal      angle = *(PetscReal*)linesearch->precheckctx;
  Vec            Ylast;
  PetscScalar    dot;
  
  PetscInt       iter;
  PetscReal      ynorm,ylastnorm,theta,angle_radians;
  SNES           snes;

  PetscFunctionBegin;
  ierr = SNESLineSearchGetSNES(linesearch, &snes);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject)snes,"SNESLineSearchPreCheckPicard_Ylast",(PetscObject*)&Ylast);CHKERRQ(ierr);
  if (!Ylast) {
    ierr = VecDuplicate(Y,&Ylast);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)snes,"SNESLineSearchPreCheckPicard_Ylast",(PetscObject)Ylast);CHKERRQ(ierr);
    ierr = PetscObjectDereference((PetscObject)Ylast);CHKERRQ(ierr);
  }
  ierr = SNESGetIterationNumber(snes,&iter);CHKERRQ(ierr);
  if (iter < 2) {
    ierr = VecCopy(Y,Ylast);CHKERRQ(ierr);
    *changed = PETSC_FALSE;
    PetscFunctionReturn(0);
  }

  ierr = VecDot(Y,Ylast,&dot);CHKERRQ(ierr);
  ierr = VecNorm(Y,NORM_2,&ynorm);CHKERRQ(ierr);
  ierr = VecNorm(Ylast,NORM_2,&ylastnorm);CHKERRQ(ierr);
  /* Compute the angle between the vectors Y and Ylast, clip to keep inside the domain of acos() */
  theta = acos((double)PetscClipInterval(PetscAbsScalar(dot) / (ynorm * ylastnorm),-1.0,1.0));
  angle_radians = angle * PETSC_PI / 180.;
  if (PetscAbsReal(theta) < angle_radians || PetscAbsReal(theta - PETSC_PI) < angle_radians) {
    /* Modify the step Y */
    PetscReal alpha,ydiffnorm;
    ierr = VecAXPY(Ylast,-1.0,Y);CHKERRQ(ierr);
    ierr = VecNorm(Ylast,NORM_2,&ydiffnorm);CHKERRQ(ierr);
    alpha = ylastnorm / ydiffnorm;
    ierr = VecCopy(Y,Ylast);CHKERRQ(ierr);
    ierr = VecScale(Y,alpha);CHKERRQ(ierr);
    ierr = PetscInfo3(snes,"Angle %G degrees less than threshold %G, corrected step by alpha=%G\n",theta*180./PETSC_PI,angle,alpha);CHKERRQ(ierr);
  } else {
    ierr = PetscInfo2(snes,"Angle %G degrees exceeds threshold %G, no correction applied\n",theta*180./PETSC_PI,angle);CHKERRQ(ierr);
    ierr = VecCopy(Y,Ylast);CHKERRQ(ierr);
    *changed = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchApply"
/*@
   SNESLineSearchApply - Computes the line-search update.

   Collective on SNESLineSearch

   Input Parameters:
+  linesearch - The linesearch context
.  X - The current solution
.  F - The current function
.  fnorm - The current norm
-  Y - The search direction

   Output Parameters:
+  X - The new solution
.  F - The new function
-  fnorm - The new function norm

   Options Database Keys:
+ -snes_linesearch_type - basic, bt, l2, cp, shell
. -snes_linesearch_monitor - Print progress of line searches
. -snes_linesearch_damping - The linesearch damping parameter
. -snes_linesearch_norms   - Turn on/off the linesearch norms
. -snes_linesearch_keeplambda - Keep the previous search length as the initial guess
- -snes_linesearch_max_it - The number of iterations for iterative line searches

   Notes:
   This is typically called from within a SNESSolve() implementation in order to
   help with convergence of the nonlinear method.  Various SNES types use line searches
   in different ways, but the overarching theme is that a line search is used to determine
   an optimal damping parameter of a step at each iteration of the method.  Each
   application of the line search may invoke SNESComputeFunction several times, and
   therefore may be fairly expensive.

   Level: Intermediate

.keywords: SNESLineSearch, Create

.seealso: SNESLineSearchCreate(), SNESLineSearchPreCheck(), SNESLineSearchPostCheck(), SNESSolve(), SNESComputeFunction()
@*/
PetscErrorCode SNESLineSearchApply(SNESLineSearch linesearch, Vec X, Vec F, PetscReal * fnorm, Vec Y) {
  PetscErrorCode ierr;
  PetscFunctionBegin;

  /* check the pointers */
  PetscValidHeaderSpecific(linesearch,SNESLINESEARCH_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,2);
  PetscValidHeaderSpecific(F,VEC_CLASSID,3);
  PetscValidHeaderSpecific(Y,VEC_CLASSID,4);

  linesearch->success = PETSC_TRUE;

  linesearch->vec_sol = X;
  linesearch->vec_update = Y;
  linesearch->vec_func = F;

  ierr = SNESLineSearchSetUp(linesearch);CHKERRQ(ierr);

  if (!linesearch->keeplambda)
    linesearch->lambda = linesearch->damping; /* set the initial guess to lambda */

  if (fnorm) {
    linesearch->fnorm = *fnorm;
  } else {
    ierr = VecNorm(F, NORM_2, &linesearch->fnorm);CHKERRQ(ierr);
  }

  ierr = PetscLogEventBegin(SNESLineSearch_Apply,linesearch,X,F,Y);CHKERRQ(ierr);

  ierr = (*linesearch->ops->apply)(linesearch);CHKERRQ(ierr);

  ierr = PetscLogEventEnd(SNESLineSearch_Apply,linesearch,X,F,Y);CHKERRQ(ierr);

  if (fnorm)
    *fnorm = linesearch->fnorm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchDestroy"
/*@
   SNESLineSearchDestroy - Destroys the line search instance.

   Collective on SNESLineSearch

   Input Parameters:
.  linesearch - The linesearch context

   Level: Intermediate

.keywords: SNESLineSearch, Destroy

.seealso: SNESLineSearchCreate(), SNESLineSearchReset(), SNESDestroy()
@*/
PetscErrorCode SNESLineSearchDestroy(SNESLineSearch * linesearch) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (!*linesearch) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*linesearch),SNESLINESEARCH_CLASSID,1);
  if (--((PetscObject)(*linesearch))->refct > 0) {*linesearch = 0; PetscFunctionReturn(0);}
  ierr = PetscObjectDepublish((*linesearch));CHKERRQ(ierr);
  ierr = SNESLineSearchReset(*linesearch);
  if ((*linesearch)->ops->destroy) {
    (*linesearch)->ops->destroy(*linesearch);
  }
  ierr = PetscViewerDestroy(&(*linesearch)->monitor);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(linesearch);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchSetMonitor"
/*@
   SNESLineSearchSetMonitor - Turns on/off printing useful information and debugging output about the line search.

   Input Parameters:
+  snes - nonlinear context obtained from SNESCreate()
-  flg - PETSC_TRUE to monitor the line search

   Logically Collective on SNES

   Options Database:
.   -snes_linesearch_monitor - enables the monitor

   Level: intermediate


.seealso: SNESLineSearchGetMonitor(), PetscViewer
@*/
PetscErrorCode  SNESLineSearchSetMonitor(SNESLineSearch linesearch, PetscBool flg)
{

  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (flg && !linesearch->monitor) {
    ierr = PetscViewerASCIIOpen(((PetscObject)linesearch)->comm,"stdout",&linesearch->monitor);CHKERRQ(ierr);
  } else if (!flg && linesearch->monitor) {
    ierr = PetscViewerDestroy(&linesearch->monitor);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchGetMonitor"
/*@
   SNESLineSearchGetMonitor - Gets the PetscViewer instance for the line search monitor.

   Input Parameters:
.  linesearch - linesearch context

   Input Parameters:
.  monitor - monitor context

   Logically Collective on SNES


   Options Database Keys:
.   -snes_linesearch_monitor - enables the monitor

   Level: intermediate


.seealso: SNESLineSearchSetMonitor(), PetscViewer
@*/
PetscErrorCode  SNESLineSearchGetMonitor(SNESLineSearch linesearch, PetscViewer *monitor)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch,SNESLINESEARCH_CLASSID,1);
  if (monitor) {
    PetscValidPointer(monitor, 2);
    *monitor = linesearch->monitor;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchSetFromOptions"
/*@
   SNESLineSearchSetFromOptions - Sets options for the line search

   Input Parameters:
.  linesearch - linesearch context

   Options Database Keys:
+ -snes_linesearch_type <type> - basic, bt, l2, cp, shell
. -snes_linesearch_order <order> - 1, 2, 3.  Most types only support certain orders (bt supports 2 or 3)
. -snes_linesearch_norms   - Turn on/off the linesearch norms for the basic linesearch type
. -snes_linesearch_minlambda - The minimum step length
. -snes_linesearch_maxstep - The maximum step size
. -snes_linesearch_rtol - Relative tolerance for iterative line searches
. -snes_linesearch_atol - Absolute tolerance for iterative line searches
. -snes_linesearch_ltol - Change in lambda tolerance for iterative line searches
. -snes_linesearch_max_it - The number of iterations for iterative line searches
. -snes_linesearch_monitor - Print progress of line searches
. -snes_linesearch_damping - The linesearch damping parameter
. -snes_linesearch_keeplambda - Keep the previous search length as the initial guess.
. -snes_linesearch_precheck_picard - Use precheck that speeds up convergence of picard method
- -snes_linesearch_precheck_picard_angle - Angle used in picard precheck method

   Logically Collective on SNESLineSearch

   Level: intermediate


.seealso: SNESLineSearchCreate(), SNESLineSearchSetOrder(), SNESLineSearchSetType(), SNESLineSearchSetTolerances(), SNESLineSearchSetDamping(), SNESLineSearchPreCheckPicard()
@*/
PetscErrorCode SNESLineSearchSetFromOptions(SNESLineSearch linesearch) {
  PetscErrorCode ierr;
  const char     *deft = SNESLINESEARCHBASIC;
  char           type[256];
  PetscBool      flg, set;
  PetscFunctionBegin;
  if (!SNESLineSearchRegisterAllCalled) {ierr = SNESLineSearchRegisterAll(PETSC_NULL);CHKERRQ(ierr);}

  ierr = PetscObjectOptionsBegin((PetscObject)linesearch);CHKERRQ(ierr);
  if (((PetscObject)linesearch)->type_name) {
    deft = ((PetscObject)linesearch)->type_name;
  }
  ierr = PetscOptionsList("-snes_linesearch_type","Linesearch type","SNESLineSearchSetType",SNESLineSearchList,deft,type,256,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = SNESLineSearchSetType(linesearch,type);CHKERRQ(ierr);
  } else if (!((PetscObject)linesearch)->type_name) {
    ierr = SNESLineSearchSetType(linesearch,deft);CHKERRQ(ierr);
  }

  ierr = PetscOptionsBool("-snes_linesearch_monitor","Print progress of line searches","SNESSNESLineSearchSetMonitor",
                          linesearch->monitor ? PETSC_TRUE : PETSC_FALSE,&flg,&set);CHKERRQ(ierr);
  if (set) {ierr = SNESLineSearchSetMonitor(linesearch,flg);CHKERRQ(ierr);}

  /* tolerances */
  ierr = PetscOptionsReal("-snes_linesearch_minlambda","Minimum step length","SNESLineSearchSetTolerances",linesearch->steptol,&linesearch->steptol,0);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-snes_linesearch_maxstep","Maximum step size","SNESLineSearchSetTolerances",linesearch->maxstep,&linesearch->maxstep,0);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-snes_linesearch_rtol","Relative tolerance for iterative line search","SNESLineSearchSetTolerances",linesearch->rtol,&linesearch->rtol,0);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-snes_linesearch_atol","Absolute tolerance for iterative line search","SNESLineSearchSetTolerances",linesearch->atol,&linesearch->atol,0);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-snes_linesearch_ltol","Change in lambda tolerance for iterative line search","SNESLineSearchSetTolerances",linesearch->ltol,&linesearch->ltol,0);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-snes_linesearch_max_it","Maximum iterations for iterative line searches","SNESLineSearchSetTolerances",linesearch->max_its,&linesearch->max_its,0);CHKERRQ(ierr);

  /* damping parameters */
  ierr = PetscOptionsReal("-snes_linesearch_damping","Line search damping and initial step guess","SNESLineSearchSetDamping",linesearch->damping,&linesearch->damping,0);CHKERRQ(ierr);

  ierr = PetscOptionsBool("-snes_linesearch_keeplambda","Use previous lambda as damping","SNESLineSearchSetKeepLambda",linesearch->keeplambda,&linesearch->keeplambda,0);CHKERRQ(ierr);

  /* precheck */
  ierr = PetscOptionsBool("-snes_linesearch_precheck_picard","Use a correction that sometimes improves convergence of Picard iteration","SNESLineSearchPreCheckPicard",flg,&flg,&set);CHKERRQ(ierr);
  if (set) {
    if (flg) {
      linesearch->precheck_picard_angle = 10.; /* correction only active if angle is less than 10 degrees */
      ierr = PetscOptionsReal("-snes_linesearch_precheck_picard_angle","Maximum angle at which to activate the correction",
                              "none",linesearch->precheck_picard_angle,&linesearch->precheck_picard_angle,PETSC_NULL);CHKERRQ(ierr);
      ierr = SNESLineSearchSetPreCheck(linesearch,SNESLineSearchPreCheckPicard,&linesearch->precheck_picard_angle);CHKERRQ(ierr);
    } else {
      ierr = SNESLineSearchSetPreCheck(linesearch,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    }
  }
  ierr = PetscOptionsInt("-snes_linesearch_order","Order of approximation used in the line search","SNESLineSearchSetOrder",linesearch->order,&linesearch->order,0);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-snes_linesearch_norms","Compute final norms in line search","SNESLineSearchSetComputeNorms",linesearch->norms,&linesearch->norms,0);CHKERRQ(ierr);

  if (linesearch->ops->setfromoptions) {
    (*linesearch->ops->setfromoptions)(linesearch);CHKERRQ(ierr);
  }

  ierr = PetscObjectProcessOptionsHandlers((PetscObject)linesearch);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchView"
/*@
   SNESLineSearchView - Prints useful information about the line search not
   related to an individual call.

   Input Parameters:
.  linesearch - linesearch context

   Logically Collective on SNESLineSearch

   Level: intermediate

.seealso: SNESLineSearchCreate()
@*/
PetscErrorCode SNESLineSearchView(SNESLineSearch linesearch, PetscViewer viewer) {
  PetscErrorCode ierr;
  PetscBool      iascii;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch,SNESLINESEARCH_CLASSID,1);
  if (!viewer) {
    ierr = PetscViewerASCIIGetStdout(((PetscObject)linesearch)->comm,&viewer);CHKERRQ(ierr);
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(linesearch,1,viewer,2);

  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscObjectPrintClassNamePrefixType((PetscObject)linesearch,viewer,"SNESLineSearch Object");CHKERRQ(ierr);
    if (linesearch->ops->view) {
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = (*linesearch->ops->view)(linesearch,viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"  maxstep=%e, minlambda=%e\n", linesearch->maxstep,linesearch->steptol);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  tolerances: relative=%e, absolute=%e, lambda=%e\n", linesearch->rtol,linesearch->atol,linesearch->ltol);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  maximum iterations=%D\n", linesearch->max_its);CHKERRQ(ierr);
    if (linesearch->ops->precheckstep) {
      if (linesearch->ops->precheckstep == SNESLineSearchPreCheckPicard) {
        ierr = PetscViewerASCIIPrintf(viewer,"  using precheck step to speed up Picard convergence\n", linesearch->max_its);CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIIPrintf(viewer,"  using user-defined precheck step\n", linesearch->max_its);CHKERRQ(ierr);
      }
    }
    if (linesearch->ops->postcheckstep) {
      ierr = PetscViewerASCIIPrintf(viewer,"  using user-defined postcheck step\n", linesearch->max_its);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchSetType"
/*@C
   SNESLineSearchSetType - Sets the linesearch type

   Input Parameters:
+  linesearch - linesearch context
-  type - The type of line search to be used

   Available Types:
+  basic - Simple damping line search.
.  bt - Backtracking line search over the L2 norm of the function
.  l2 - Secant line search over the L2 norm of the function
.  cp - Critical point secant line search assuming F(x) = grad G(x) for some unknown G(x)
-  shell - User provided SNESLineSearch implementation

   Logically Collective on SNESLineSearch

   Level: intermediate


.seealso: SNESLineSearchCreate()
@*/
PetscErrorCode SNESLineSearchSetType(SNESLineSearch linesearch, const SNESLineSearchType type)
{

  PetscErrorCode ierr,(*r)(SNESLineSearch);
  PetscBool      match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch,SNESLINESEARCH_CLASSID,1);
  PetscValidCharPointer(type,2);

  ierr = PetscObjectTypeCompare((PetscObject)linesearch,type,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  ierr =  PetscFListFind(SNESLineSearchList,((PetscObject)linesearch)->comm,type,PETSC_TRUE,(void (**)(void)) &r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unable to find requested Line Search type %s",type);
  /* Destroy the previous private linesearch context */
  if (linesearch->ops->destroy) {
    ierr = (*(linesearch)->ops->destroy)(linesearch);CHKERRQ(ierr);
    linesearch->ops->destroy = PETSC_NULL;
  }
  /* Reinitialize function pointers in SNESLineSearchOps structure */
  linesearch->ops->apply          = 0;
  linesearch->ops->view           = 0;
  linesearch->ops->setfromoptions = 0;
  linesearch->ops->destroy        = 0;

  ierr = PetscObjectChangeTypeName((PetscObject)linesearch,type);CHKERRQ(ierr);
  ierr = (*r)(linesearch);CHKERRQ(ierr);
#if defined(PETSC_HAVE_AMS)
  if (PetscAMSPublishAll) {
    ierr = PetscObjectAMSPublish((PetscObject)linesearch);CHKERRQ(ierr);
  }
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchSetSNES"
/*@
   SNESLineSearchSetSNES - Sets the SNES for the linesearch for function evaluation.

   Input Parameters:
+  linesearch - linesearch context
-  snes - The snes instance

   Level: developer

   Notes:
   This happens automatically when the line search is gotten/created with
   SNESGetSNESLineSearch().  This routine is therefore mainly called within SNES
   implementations.

   Level: developer

.seealso: SNESLineSearchGetSNES(), SNESLineSearchSetVecs(), SNES
@*/
PetscErrorCode  SNESLineSearchSetSNES(SNESLineSearch linesearch, SNES snes){
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch,SNESLINESEARCH_CLASSID,1);
  PetscValidHeaderSpecific(snes,SNES_CLASSID,2);
  linesearch->snes = snes;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchGetSNES"
/*@
   SNESLineSearchGetSNES - Gets the SNES instance associated with the line search.
   Having an associated SNES is necessary because most line search implementations must be able to
   evaluate the function using SNESComputeFunction() for the associated SNES.  This routine
   is used in the line search implementations when one must get this associated SNES instance.

   Input Parameters:
.  linesearch - linesearch context

   Output Parameters:
.  snes - The snes instance

   Level: developer

.seealso: SNESLineSearchGetSNES(), SNESLineSearchSetVecs(), SNES
@*/
PetscErrorCode  SNESLineSearchGetSNES(SNESLineSearch linesearch, SNES *snes){
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch,SNESLINESEARCH_CLASSID,1);
  PetscValidPointer(snes, 2);
  *snes = linesearch->snes;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchGetLambda"
/*@
   SNESLineSearchGetLambda - Gets the last linesearch steplength discovered.

   Input Parameters:
.  linesearch - linesearch context

   Output Parameters:
.  lambda - The last steplength computed during SNESLineSearchApply()

   Level: advanced

   Notes:
   This is useful in methods where the solver is ill-scaled and
   requires some adaptive notion of the difference in scale between the
   solution and the function.  For instance, SNESQN may be scaled by the
   line search lambda using the argument -snes_qn_scaling ls.


.seealso: SNESLineSearchSetLambda(), SNESLineSearchGetDamping(), SNESLineSearchApply()
@*/
PetscErrorCode  SNESLineSearchGetLambda(SNESLineSearch linesearch,PetscReal *lambda)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch,SNESLINESEARCH_CLASSID,1);
  PetscValidPointer(lambda, 2);
  *lambda = linesearch->lambda;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchSetLambda"
/*@
   SNESLineSearchSetLambda - Sets the linesearch steplength.

   Input Parameters:
+  linesearch - linesearch context
-  lambda - The last steplength.

   Notes:
   This routine is typically used within implementations of SNESLineSearchApply
   to set the final steplength.  This routine (and SNESLineSearchGetLambda()) were
   added in order to facilitate Quasi-Newton methods that use the previous steplength
   as an inner scaling parameter.

   Level: advanced

.seealso: SNESLineSearchGetLambda()
@*/
PetscErrorCode  SNESLineSearchSetLambda(SNESLineSearch linesearch, PetscReal lambda)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch,SNESLINESEARCH_CLASSID,1);
  linesearch->lambda = lambda;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "SNESLineSearchGetTolerances"
/*@
   SNESLineSearchGetTolerances - Gets the tolerances for the linesearch.  These include
   tolerances for the relative and absolute change in the function norm, the change
   in lambda for iterative line searches, the minimum steplength, the maximum steplength,
   and the maximum number of iterations the line search procedure may take.

   Input Parameters:
.  linesearch - linesearch context

   Output Parameters:
+  steptol - The minimum steplength
.  maxstep - The maximum steplength
.  rtol    - The relative tolerance for iterative line searches
.  atol    - The absolute tolerance for iterative line searches
.  ltol    - The change in lambda tolerance for iterative line searches
-  max_it  - The maximum number of iterations of the line search

   Level: intermediate

   Notes:
   Different line searches may implement these parameters slightly differently as
   the type requires.

.seealso: SNESLineSearchSetTolerances()
@*/
PetscErrorCode  SNESLineSearchGetTolerances(SNESLineSearch linesearch,PetscReal *steptol,PetscReal *maxstep, PetscReal *rtol, PetscReal *atol, PetscReal *ltol, PetscInt *max_its)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch,SNESLINESEARCH_CLASSID,1);
  if (steptol) {
    PetscValidPointer(steptol, 2);
    *steptol = linesearch->steptol;
  }
  if (maxstep) {
    PetscValidPointer(maxstep, 3);
    *maxstep = linesearch->maxstep;
  }
  if (rtol) {
    PetscValidPointer(rtol, 4);
    *rtol = linesearch->rtol;
  }
  if (atol) {
    PetscValidPointer(atol, 5);
    *atol = linesearch->atol;
  }
  if (ltol) {
    PetscValidPointer(ltol, 6);
    *ltol = linesearch->ltol;
  }
  if (max_its) {
    PetscValidPointer(max_its, 7);
    *max_its = linesearch->max_its;
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "SNESLineSearchSetTolerances"
/*@
   SNESLineSearchSetTolerances -  Gets the tolerances for the linesearch.  These include
   tolerances for the relative and absolute change in the function norm, the change
   in lambda for iterative line searches, the minimum steplength, the maximum steplength,
   and the maximum number of iterations the line search procedure may take.

   Input Parameters:
+  linesearch - linesearch context
.  steptol - The minimum steplength
.  maxstep - The maximum steplength
.  rtol    - The relative tolerance for iterative line searches
.  atol    - The absolute tolerance for iterative line searches
.  ltol    - The change in lambda tolerance for iterative line searches
-  max_it  - The maximum number of iterations of the line search

   Notes:
   The user may choose to not set any of the tolerances using PETSC_DEFAULT in
   place of an argument.

   Level: intermediate

.seealso: SNESLineSearchGetTolerances()
@*/
PetscErrorCode  SNESLineSearchSetTolerances(SNESLineSearch linesearch,PetscReal steptol,PetscReal maxstep, PetscReal rtol, PetscReal atol, PetscReal ltol, PetscInt max_its)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch,SNESLINESEARCH_CLASSID,1);
  PetscValidLogicalCollectiveReal(linesearch,steptol,2);
  PetscValidLogicalCollectiveReal(linesearch,maxstep,3);
  PetscValidLogicalCollectiveReal(linesearch,rtol,4);
  PetscValidLogicalCollectiveReal(linesearch,atol,5);
  PetscValidLogicalCollectiveReal(linesearch,ltol,6);
  PetscValidLogicalCollectiveInt(linesearch,max_its,7);

  if ( steptol!= PETSC_DEFAULT) {
    if ( steptol < 0.0) SETERRQ1(((PetscObject)linesearch)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Minimum step length %G must be non-negative",steptol);
    linesearch->steptol = steptol;
  }

  if ( maxstep!= PETSC_DEFAULT) {
    if ( maxstep < 0.0) SETERRQ1(((PetscObject)linesearch)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Maximum step length %G must be non-negative",maxstep);
    linesearch->maxstep = maxstep;
  }

  if (rtol != PETSC_DEFAULT) {
    if (rtol < 0.0 || 1.0 <= rtol) SETERRQ1(((PetscObject)linesearch)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Relative tolerance %G must be non-negative and less than 1.0",rtol);
    linesearch->rtol = rtol;
  }

  if (atol != PETSC_DEFAULT) {
    if (atol < 0.0) SETERRQ1(((PetscObject)linesearch)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Absolute tolerance %G must be non-negative",atol);
    linesearch->atol = atol;
  }

  if (ltol != PETSC_DEFAULT) {
    if (ltol < 0.0) SETERRQ1(((PetscObject)linesearch)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Labmda tolerance %G must be non-negative",ltol);
  linesearch->ltol = ltol;
  }

  if (max_its != PETSC_DEFAULT) {
    if (max_its < 0) SETERRQ1(((PetscObject)linesearch)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Maximum number of iterations %D must be non-negative",max_its);
    linesearch->max_its = max_its;
  }

  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchGetDamping"
/*@
   SNESLineSearchGetDamping - Gets the line search damping parameter.

   Input Parameters:
.  linesearch - linesearch context

   Output Parameters:
.  damping - The damping parameter

   Level: advanced

.seealso: SNESLineSearchGetStepTolerance(), SNESQN
@*/

PetscErrorCode  SNESLineSearchGetDamping(SNESLineSearch linesearch,PetscReal *damping)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch,SNESLINESEARCH_CLASSID,1);
  PetscValidPointer(damping, 2);
  *damping = linesearch->damping;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchSetDamping"
/*@
   SNESLineSearchSetDamping - Sets the line search damping paramter.

   Input Parameters:
.  linesearch - linesearch context
.  damping - The damping parameter

   Level: intermediate

   Notes:
   The basic line search merely takes the update step scaled by the damping parameter.
   The use of the damping parameter in the l2 and cp line searches is much more subtle;
   it is used as a starting point in calculating the secant step. However, the eventual
   step may be of greater length than the damping parameter.  In the bt line search it is
   used as the maximum possible step length, as the bt line search only backtracks.

.seealso: SNESLineSearchGetDamping()
@*/
PetscErrorCode  SNESLineSearchSetDamping(SNESLineSearch linesearch,PetscReal damping)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch,SNESLINESEARCH_CLASSID,1);
  linesearch->damping = damping;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchGetOrder"
/*@
   SNESLineSearchGetOrder - Gets the line search approximation order.

   Input Parameters:
.  linesearch - linesearch context

   Output Parameters:
.  order - The order

   Possible Values for order:
+  1 or SNES_LINESEARCH_ORDER_LINEAR - linear order
.  2 or SNES_LINESEARCH_ORDER_QUADRATIC - quadratic order
-  3 or SNES_LINESEARCH_ORDER_CUBIC - cubic order

   Level: intermediate

.seealso: SNESLineSearchSetOrder()
@*/

PetscErrorCode  SNESLineSearchGetOrder(SNESLineSearch linesearch,PetscInt *order)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch,SNESLINESEARCH_CLASSID,1);
  PetscValidPointer(order, 2);
  *order = linesearch->order;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchSetOrder"
/*@
   SNESLineSearchSetOrder - Sets the line search damping paramter.

   Input Parameters:
.  linesearch - linesearch context
.  order - The damping parameter

   Level: intermediate

   Possible Values for order:
+  1 or SNES_LINESEARCH_ORDER_LINEAR - linear order
.  2 or SNES_LINESEARCH_ORDER_QUADRATIC - quadratic order
-  3 or SNES_LINESEARCH_ORDER_CUBIC - cubic order

   Notes:
   Variable orders are supported by the following line searches:
+  bt - cubic and quadratic
-  cp - linear and quadratic

.seealso: SNESLineSearchGetOrder()
@*/
PetscErrorCode  SNESLineSearchSetOrder(SNESLineSearch linesearch,PetscInt order)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch,SNESLINESEARCH_CLASSID,1);
  linesearch->order = order;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchGetNorms"
/*@
   SNESLineSearchGetNorms - Gets the norms for for X, Y, and F.

   Input Parameters:
.  linesearch - linesearch context

   Output Parameters:
+  xnorm - The norm of the current solution
.  fnorm - The norm of the current function
-  ynorm - The norm of the current update

   Notes:
   This function is mainly called from SNES implementations.

   Level: developer

.seealso: SNESLineSearchSetNorms() SNESLineSearchGetVecs()
@*/
PetscErrorCode  SNESLineSearchGetNorms(SNESLineSearch linesearch, PetscReal * xnorm, PetscReal * fnorm, PetscReal * ynorm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch,SNESLINESEARCH_CLASSID,1);
  if (xnorm) {
    *xnorm = linesearch->xnorm;
  }
  if (fnorm) {
    *fnorm = linesearch->fnorm;
  }
  if (ynorm) {
    *ynorm = linesearch->ynorm;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchSetNorms"
/*@
   SNESLineSearchSetNorms - Gets the computed norms for for X, Y, and F.

   Input Parameters:
+  linesearch - linesearch context
.  xnorm - The norm of the current solution
.  fnorm - The norm of the current function
-  ynorm - The norm of the current update

   Level: advanced

.seealso: SNESLineSearchGetNorms(), SNESLineSearchSetVecs()
@*/
PetscErrorCode  SNESLineSearchSetNorms(SNESLineSearch linesearch, PetscReal xnorm, PetscReal fnorm, PetscReal ynorm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch,SNESLINESEARCH_CLASSID,1);
  linesearch->xnorm = xnorm;
  linesearch->fnorm = fnorm;
  linesearch->ynorm = ynorm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchComputeNorms"
/*@
   SNESLineSearchComputeNorms - Computes the norms of X, F, and Y.

   Input Parameters:
.  linesearch - linesearch context

   Options Database Keys:
.   -snes_linesearch_norms - turn norm computation on or off

   Level: intermediate

.seealso: SNESLineSearchGetNorms, SNESLineSearchSetNorms(), SNESLineSearchSetComputeNorms()
@*/
PetscErrorCode SNESLineSearchComputeNorms(SNESLineSearch linesearch)
{
  PetscErrorCode ierr;
  SNES snes;
  PetscFunctionBegin;
  if (linesearch->norms) {
    if (linesearch->ops->vinorm) {
      ierr = SNESLineSearchGetSNES(linesearch, &snes);CHKERRQ(ierr);
      ierr = VecNorm(linesearch->vec_sol, NORM_2, &linesearch->xnorm);CHKERRQ(ierr);
      ierr = VecNorm(linesearch->vec_update, NORM_2, &linesearch->ynorm);CHKERRQ(ierr);
      ierr = (*linesearch->ops->vinorm)(snes, linesearch->vec_func, linesearch->vec_sol, &linesearch->fnorm);CHKERRQ(ierr);
    } else {
      ierr = VecNormBegin(linesearch->vec_func,   NORM_2, &linesearch->fnorm);CHKERRQ(ierr);
      ierr = VecNormBegin(linesearch->vec_sol,    NORM_2, &linesearch->xnorm);CHKERRQ(ierr);
      ierr = VecNormBegin(linesearch->vec_update, NORM_2, &linesearch->ynorm);CHKERRQ(ierr);
      ierr = VecNormEnd(linesearch->vec_func,     NORM_2, &linesearch->fnorm);CHKERRQ(ierr);
      ierr = VecNormEnd(linesearch->vec_sol,      NORM_2, &linesearch->xnorm);CHKERRQ(ierr);
      ierr = VecNormEnd(linesearch->vec_update,   NORM_2, &linesearch->ynorm);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchSetComputeNorms"
/*@
   SNESLineSearchSetComputeNorms - Turns on or off the computation of final norms in the line search.

   Input Parameters:
+  linesearch  - linesearch context
-  flg  - indicates whether or not to compute norms

   Options Database Keys:
.   -snes_linesearch_norms - turn norm computation on or off

   Notes:
   This is most relevant to the SNESLINESEARCHBASIC line search type.

   Level: intermediate

.seealso: SNESLineSearchGetNorms(), SNESLineSearchSetNorms(), SNESLineSearchComputeNorms(), SNESLINESEARCHBASIC
@*/
PetscErrorCode SNESLineSearchSetComputeNorms(SNESLineSearch linesearch, PetscBool flg)
{
  PetscFunctionBegin;
  linesearch->norms = flg;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchGetVecs"
/*@
   SNESLineSearchGetVecs - Gets the vectors from the SNESLineSearch context

   Input Parameters:
.  linesearch - linesearch context

   Output Parameters:
+  X - The old solution
.  F - The old function
.  Y - The search direction
.  W - The new solution
-  G - The new function

   Level: advanced

.seealso: SNESLineSearchGetNorms(), SNESLineSearchSetVecs()
@*/
PetscErrorCode SNESLineSearchGetVecs(SNESLineSearch linesearch,Vec *X,Vec *F, Vec *Y,Vec *W,Vec *G) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch,SNESLINESEARCH_CLASSID,1);
  if (X) {
    PetscValidPointer(X, 2);
    *X = linesearch->vec_sol;
  }
  if (F) {
    PetscValidPointer(F, 3);
    *F = linesearch->vec_func;
  }
  if (Y) {
    PetscValidPointer(Y, 4);
    *Y = linesearch->vec_update;
  }
  if (W) {
    PetscValidPointer(W, 5);
    *W = linesearch->vec_sol_new;
  }
  if (G) {
    PetscValidPointer(G, 6);
    *G = linesearch->vec_func_new;
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchSetVecs"
/*@
   SNESLineSearchSetVecs - Sets the vectors on the SNESLineSearch context

   Input Parameters:
+  linesearch - linesearch context
.  X - The old solution
.  F - The old function
.  Y - The search direction
.  W - The new solution
-  G - The new function

   Level: advanced

.seealso: SNESLineSearchSetNorms(), SNESLineSearchGetVecs()
@*/
PetscErrorCode SNESLineSearchSetVecs(SNESLineSearch linesearch,Vec X,Vec F,Vec Y,Vec W, Vec G) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch,SNESLINESEARCH_CLASSID,1);
  if (X) {
    PetscValidHeaderSpecific(X,VEC_CLASSID,2);
    linesearch->vec_sol = X;
  }
  if (F) {
    PetscValidHeaderSpecific(F,VEC_CLASSID,3);
    linesearch->vec_func = F;
  }
  if (Y) {
    PetscValidHeaderSpecific(Y,VEC_CLASSID,4);
    linesearch->vec_update = Y;
  }
  if (W) {
    PetscValidHeaderSpecific(W,VEC_CLASSID,5);
    linesearch->vec_sol_new = W;
  }
  if (G) {
    PetscValidHeaderSpecific(G,VEC_CLASSID,6);
    linesearch->vec_func_new = G;
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchAppendOptionsPrefix"
/*@C
   SNESLineSearchAppendOptionsPrefix - Appends to the prefix used for searching for all
   SNES options in the database.

   Logically Collective on SNESLineSearch

   Input Parameters:
+  snes - the SNES context
-  prefix - the prefix to prepend to all option names

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the hyphen.

   Level: advanced

.keywords: SNESLineSearch, append, options, prefix, database

.seealso: SNESGetOptionsPrefix()
@*/
PetscErrorCode  SNESLineSearchAppendOptionsPrefix(SNESLineSearch linesearch,const char prefix[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch,SNESLINESEARCH_CLASSID,1);
  ierr = PetscObjectAppendOptionsPrefix((PetscObject)linesearch,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchGetOptionsPrefix"
/*@C
   SNESLineSearchGetOptionsPrefix - Sets the prefix used for searching for all
   SNESLineSearch options in the database.

   Not Collective

   Input Parameter:
.  linesearch - the SNESLineSearch context

   Output Parameter:
.  prefix - pointer to the prefix string used

   Notes:
   On the fortran side, the user should pass in a string 'prefix' of
   sufficient length to hold the prefix.

   Level: advanced

.keywords: SNESLineSearch, get, options, prefix, database

.seealso: SNESAppendOptionsPrefix()
@*/
PetscErrorCode  SNESLineSearchGetOptionsPrefix(SNESLineSearch linesearch,const char *prefix[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch,SNESLINESEARCH_CLASSID,1);
  ierr = PetscObjectGetOptionsPrefix((PetscObject)linesearch,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchGetWork"
/*@
   SNESLineSearchGetWork - Gets work vectors for the line search.

   Input Parameter:
+  linesearch - the SNESLineSearch context
-  nwork - the number of work vectors

   Level: developer

   Notes:
   This is typically called at the beginning of a SNESLineSearch or SNESLineSearchShell implementation.

.keywords: SNESLineSearch, work, vector

.seealso: SNESDefaultGetWork()
@*/
PetscErrorCode  SNESLineSearchGetWork(SNESLineSearch linesearch, PetscInt nwork)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (linesearch->vec_sol) {
    ierr = VecDuplicateVecs(linesearch->vec_sol, nwork, &linesearch->work);CHKERRQ(ierr);
  } else {
    SETERRQ(((PetscObject)linesearch)->comm, PETSC_ERR_USER, "Cannot get linesearch work-vectors without setting a solution vec!");
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchGetSuccess"
/*@
   SNESLineSearchGetSuccess - Gets the success/failure status of the last line search application

   Input Parameters:
.  linesearch - linesearch context

   Output Parameters:
.  success - The success or failure status

   Notes:
   This is typically called after SNESLineSearchApply in order to determine if the line-search failed
   (and set the SNES convergence accordingly).

   Level: intermediate

.seealso: SNESLineSearchSetSuccess()
@*/
PetscErrorCode  SNESLineSearchGetSuccess(SNESLineSearch linesearch, PetscBool *success)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch,SNESLINESEARCH_CLASSID,1);
  PetscValidPointer(success, 2);
  if (success) {
    *success = linesearch->success;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchSetSuccess"
/*@
   SNESLineSearchSetSuccess - Sets the success/failure status of the last line search application

   Input Parameters:
+  linesearch - linesearch context
-  success - The success or failure status

   Notes:
   This is typically called in a SNESLineSearchApply() or SNESLineSearchShell implementation to set
   the success or failure of the line search method.

   Level: developer

.seealso: SNESLineSearchGetSuccess()
@*/
PetscErrorCode  SNESLineSearchSetSuccess(SNESLineSearch linesearch, PetscBool success)
{
  PetscValidHeaderSpecific(linesearch,SNESLINESEARCH_CLASSID,1);
  PetscFunctionBegin;
  linesearch->success = success;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchSetVIFunctions"
/*@C
   SNESLineSearchSetVIFunctions - Sets VI-specific functions for line search computation.

   Input Parameters:
+  snes - nonlinear context obtained from SNESCreate()
.  projectfunc - function for projecting the function to the bounds
-  normfunc - function for computing the norm of an active set

   Logically Collective on SNES

   Calling sequence of projectfunc:
.vb
   projectfunc (SNES snes, Vec X)
.ve

    Input parameters for projectfunc:
+   snes - nonlinear context
-   X - current solution

    Output parameters for projectfunc:
.   X - Projected solution

   Calling sequence of normfunc:
.vb
   projectfunc (SNES snes, Vec X, Vec F, PetscScalar * fnorm)
.ve

    Input parameters for normfunc:
+   snes - nonlinear context
.   X - current solution
-   F - current residual

    Output parameters for normfunc:
.   fnorm - VI-specific norm of the function

    Notes:
    The VI solvers require projection of the solution to the feasible set.  projectfunc should implement this.

    The VI solvers require special evaluation of the function norm such that the norm is only calculated
    on the inactive set.  This should be implemented by normfunc.

    Level: developer

.keywords: SNES, line search, VI, nonlinear, set, line search

.seealso: SNESLineSearchGetVIFunctions(), SNESLineSearchSetPostCheck(), SNESLineSearchSetPreCheck()
@*/
extern PetscErrorCode SNESLineSearchSetVIFunctions(SNESLineSearch linesearch, SNESLineSearchVIProjectFunc projectfunc, SNESLineSearchVINormFunc normfunc)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch,SNESLINESEARCH_CLASSID,1);
  if (projectfunc) linesearch->ops->viproject = projectfunc;
  if (normfunc) linesearch->ops->vinorm = normfunc;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchGetVIFunctions"
/*@C
   SNESLineSearchGetVIFunctions - Sets VI-specific functions for line search computation.

   Input Parameters:
.  snes - nonlinear context obtained from SNESCreate()

   Output Parameters:
+  projectfunc - function for projecting the function to the bounds
-  normfunc - function for computing the norm of an active set

   Logically Collective on SNES

    Level: developer

.keywords: SNES, line search, VI, nonlinear, get, line search

.seealso: SNESLineSearchSetVIFunctions(), SNESLineSearchGetPostCheck(), SNESLineSearchGetPreCheck()
@*/
extern PetscErrorCode SNESLineSearchGetVIFunctions(SNESLineSearch linesearch, SNESLineSearchVIProjectFunc *projectfunc, SNESLineSearchVINormFunc *normfunc)
{
  PetscFunctionBegin;
  if (projectfunc) *projectfunc = linesearch->ops->viproject;
  if (normfunc) *normfunc = linesearch->ops->vinorm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchRegister"
/*@C
  SNESLineSearchRegister - See SNESLineSearchRegisterDynamic()

  Level: advanced
@*/
PetscErrorCode  SNESLineSearchRegister(const char sname[],const char path[],const char name[],PetscErrorCode (*function)(SNESLineSearch))
{
  char           fullname[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFListConcat(path,name,fullname);CHKERRQ(ierr);
  ierr = PetscFListAdd(&SNESLineSearchList,sname,fullname,(void (*)(void))function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
