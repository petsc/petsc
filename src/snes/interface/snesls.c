#include <private/snesimpl.h>      /*I "petscsnes.h"  I*/

const char *const SNESLineSearchTypes[] = {"BASIC","BASICNONORMS","QUADRATIC","CUBIC","EXACT", "TEST", "SECANT", "USERDEFINED", "SNESLineSearchType","SNES_LS_",0};

const char *SNESLineSearchTypeName(SNESLineSearchType type)
{
  return (SNES_LS_BASIC <= type && type <= SNES_LS_TEST) ? SNESLineSearchTypes[type] : "unknown";
}

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchSet"
/*@C
   SNESLineSearchSet - Sets the line search routine to be used
   by the method SNESLS.

   Input Parameters:
+  snes - nonlinear context obtained from SNESCreate()
.  lsctx - optional user-defined context for use by line search
-  func - pointer to int function

   Logically Collective on SNES

   Available Routines:
+  SNESLineSearchCubic() - default line search
.  SNESLineSearchQuadratic() - quadratic line search
.  SNESLineSearchNo() - the full Newton step (actually not a line search)
-  SNESLineSearchNoNorms() - the full Newton step (calculating no norms; faster in parallel)

    Options Database Keys:
+   -snes_ls [cubic,quadratic,basic,basicnonorms] - Selects line search
.   -snes_ls_alpha <alpha> - Sets alpha used in determining if reduction in function norm is sufficient
.   -snes_ls_maxstep <maxstep> - Sets maximum step the line search will use (if the 2-norm(y) > maxstep then scale y to be y = (maxstep/2-norm(y)) *y)
-   -snes_ls_minlambda <minlambda> - Sets the minimum lambda the line search will use  minlambda / max_i ( y[i]/x[i] )
 
   Calling sequence of func:
.vb
   func (SNES snes,void *lsctx,Vec x,Vec f,Vec g,Vec y,Vec w,PetscReal fnorm,PetscReal xnorm,PetscReal *ynorm,PetscReal *gnorm,PetscBool  *flag)
.ve

    Input parameters for func:
+   snes - nonlinear context
.   lsctx - optional user-defined context for line search
.   x - current iterate
.   f - residual evaluated at x
.   y - search direction
.   fnorm - 2-norm of f
-   xnorm - 2-norm of f

    Output parameters for func:
+   g - residual evaluated at new iterate y
.   w - new iterate
.   gnorm - 2-norm of g
.   ynorm - 2-norm of search length
-   flag - set to PETSC_TRUE if the line search succeeds; PETSC_FALSE on failure.

    Level: advanced

.keywords: SNES, nonlinear, set, line search, routine

.seealso: SNESLineSearchCubic(), SNESLineSearchQuadratic(), SNESLineSearchNo(), SNESLineSearchNoNorms(),
          SNESLineSearchSetPostCheck(), SNESLineSearchSetParams(), SNESLineSearchGetParams(), SNESLineSearchSetPreCheck()
@*/
PetscErrorCode  SNESLineSearchSet(SNES snes,PetscErrorCode (*func)(SNES,void*,Vec,Vec,Vec,PetscReal,PetscReal,Vec,Vec,PetscReal*,PetscReal*,PetscBool *),void *lsctx)
{
  PetscFunctionBegin;
  snes->ls_type = SNES_LS_USER_DEFINED; /* user defined */
  snes->ops->linesearch = func;
  snes->lsP = lsctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchSetType"
/*@
   SNESLineSearchSetType - Sets the line search type to be  used by the method SNES solver.

   Input Parameters:
+  snes - nonlinear context obtained from SNESCreate()
-  type - line search type from among several standard types
   Logically Collective on SNES

   Available Types:
+  SNES_LS_CUBIC - cubic - default line search
.  SNES_LS_QUADRATIC - quadratic - quadratic line search
.  SNES_LS_NO - basic - the full Newton step (actually not a line search)
.  SNES_LS_NONORMS - basicnonorms - the full Newton step (calculating no norms; faster in parallel)
.  SNES_LS_EXACT - exact - exact line search
-  SNES_LS_TEST - test  - test line search dependent on method (such as the linear step length used for nonlinear CG)

    Level: advanced

.keywords: SNES, nonlinear, set, line search, routine

.seealso: SNESLineSearchCubic(), SNESLineSearchQuadratic(), SNESLineSearchNo(), SNESLineSearchNoNorms(),
          SNESLineSearchSetPostCheck(), SNESLineSearchSetParams(), SNESLineSearchGetParams(), SNESLineSearchSetPreCheck()
@*/
PetscErrorCode  SNESLineSearchSetType(SNES snes,SNESLineSearchType type)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTryMethod(snes,"SNESLineSearchSetType_C",(SNES,SNESLineSearchType),(snes,type));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchSetParams"
/*@
   SNESLineSearchSetParams - Sets the parameters associated with the line search
   routine in the Newton-based method SNESLS.

   Logically Collective on SNES

   Input Parameters:
+  snes    - The nonlinear context obtained from SNESCreate()
.  alpha   - The scalar such that .5*f_{n+1} . f_{n+1} <= .5*f_n . f_n - alpha |p_n . J . f_n|
.  maxstep - The maximum norm of the update vector
-  steptol - lambda is not allowed to be smaller than minlambda/( max_i y[i]/x[i]) 

   Level: intermediate

   Note:
   Pass in PETSC_DEFAULT for any parameter you do not wish to change.

   We are finding the zero of f() so the one dimensional minimization problem we are
   solving in the line search is minimize .5*f(x_n + lambda*step_direction) . f(x_n + lambda*step_direction)


.keywords: SNES, nonlinear, set, line search params

.seealso: SNESLineSearchGetParams(), SNESLineSearchSet()
@*/
PetscErrorCode  SNESLineSearchSetParams(SNES snes,PetscReal alpha,PetscReal maxstep,PetscReal steptol)
{
  PetscFunctionBegin;
  PetscValidLogicalCollectiveReal(snes,alpha,2);
  PetscValidLogicalCollectiveReal(snes,maxstep,3);
  PetscValidLogicalCollectiveReal(snes,steptol,4);
  if (alpha   >= 0.0) snes->ls_alpha       = alpha;
  if (maxstep >= 0.0) snes->maxstep        = maxstep;
  if (steptol >= 0.0) snes->steptol        = steptol;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchGetParams"
/*@C
   SNESLineSearchGetParams - Gets the parameters associated with the line search
     routine in the Newton-based method SNESLS.

   Not collective, but any processor will return the same values

   Input Parameter:
.  snes    - The nonlinear context obtained from SNESCreate()

   Output Parameters:
+  alpha   - The scalar such that .5*f_{n+1} . f_{n+1} <= .5*f_n . f_n - alpha |p_n . J . f_n|
.  maxstep - The maximum norm of the update vector
-  steptol - The step length constant is not allowed to be smaller than steptol/( max_i y[i]/x[i])

   Level: intermediate

   Note:
    To not get a certain parameter, pass in PETSC_NULL

   We are finding the zero of f() so the one dimensional minimization problem we are
   solving in the line search is minimize .5*f(x_n + lambda*step_direction) . f(x_n + lambda*step_direction)

.keywords: SNES, nonlinear, set, line search parameters

.seealso: SNESLineSearchSetParams(), SNESLineSearchSet()
@*/
PetscErrorCode  SNESLineSearchGetParams(SNES snes,PetscReal *alpha,PetscReal *maxstep,PetscReal *steptol)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  if (alpha) {
    PetscValidDoublePointer(alpha,2);
    *alpha   = snes->ls_alpha;
  }
  if (maxstep) {
    PetscValidDoublePointer(maxstep,3);
    *maxstep = snes->maxstep;
  }
  if (steptol) {
    PetscValidDoublePointer(steptol,3);
    *steptol = snes->steptol;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchSetPreCheck"
/*@C
   SNESLineSearchSetPreCheck - Sets a routine to check the validity of a new direction given by the linear solve
         before the line search is called.

   Input Parameters:
+  snes - nonlinear context obtained from SNESCreate()
.  func - pointer to function
-  checkctx - optional user-defined context for use by step checking routine

   Logically Collective on SNES

   Calling sequence of func:
.vb
   int func (SNES snes, Vec x,Vec y,void *checkctx, PetscBool  *changed_y)
.ve
   where func returns an error code of 0 on success and a nonzero
   on failure.

   Input parameters for func:
+  snes - nonlinear context
.  checkctx - optional user-defined context for use by step checking routine
.  x - previous iterate
-  y - new search direction and length

   Output parameters for func:
+  y - search direction (possibly changed)
-  changed_y - indicates search direction was changed by this routine

   Level: advanced

   Notes: All line searches accept the new iterate computed by the line search checking routine.

.keywords: SNES, nonlinear, set, line search check, step check, routine

.seealso: SNESLineSearchSet(), SNESLineSearchSetPostCheck(), SNESSetUpdate()
@*/
PetscErrorCode  SNESLineSearchSetPreCheck(SNES snes,PetscErrorCode (*func)(SNES,Vec,Vec,void*,PetscBool *),void *checkctx)
{
  PetscFunctionBegin;
  snes->ops->precheckstep  = func;
  snes->precheck           = checkctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchPreCheckPicard"
/*@C
   SNESLineSearchPreCheckPicard - Implements a correction that is sometimes useful to improve the convergence rate of Picard iteration

   Logically Collective

   Input Arguments:
+  snes - nonlinear solver
.  X - base state for this step
.  Y - initial correction
-  ctx - context, should be a pointer to PetscReal containing the angle in degrees below which to activate the correction

   Output Arguments:
+  Y - correction, possibly modifide
-  changed - flag indicating that Y was modified

   Options Database Key:
+  -snes_ls_precheck_picard - activate this routine
-  -snes_ls_precheck_picard_angle - angle

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
PetscErrorCode SNESLineSearchPreCheckPicard(SNES snes,Vec X,Vec Y,void *ctx,PetscBool *changed)
{
  PetscErrorCode ierr;
  PetscReal      angle = *(PetscReal*)ctx;
  Vec            Ylast;
  PetscScalar    dot;
  PetscInt       iter;
  PetscReal      ynorm,ylastnorm,theta,angle_radians;

  PetscFunctionBegin;
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
#define __FUNCT__ "SNESLineSearchSetPostCheck"
/*@C
   SNESLineSearchSetPostCheck - Sets a routine to check the validity of new iterate computed
   by the line search routine in the Newton-based method SNESLS.

   Input Parameters:
+  snes - nonlinear context obtained from SNESCreate()
.  func - pointer to function
-  checkctx - optional user-defined context for use by step checking routine

   Logically Collective on SNES

   Calling sequence of func:
.vb
   int func (SNES snes, Vec x,Vec y,Vec w,void *checkctx, PetscBool  *changed_y,PetscBool  *changed_w)
.ve
   where func returns an error code of 0 on success and a nonzero
   on failure.

   Input parameters for func:
+  snes - nonlinear context
.  checkctx - optional user-defined context for use by step checking routine 
.  x - previous iterate
.  y - new search direction and length
-  w - current candidate iterate

   Output parameters for func:
+  y - search direction (possibly changed)
.  w - current iterate (possibly modified)
.  changed_y - indicates search direction was changed by this routine
-  changed_w - indicates current iterate was changed by this routine

   Level: advanced

   Notes: All line searches accept the new iterate computed by the line search checking routine.

   Only one of changed_y and changed_w can  be PETSC_TRUE

   On input w = x - y

   SNESLineSearchNo() and SNESLineSearchNoNorms() (1) compute a candidate iterate u_{i+1}, (2) pass control
   to the checking routine, and then (3) compute the corresponding nonlinear
   function f(u_{i+1}) with the (possibly altered) iterate u_{i+1}.

   SNESLineSearchQuadratic() and SNESLineSearchCubic() (1) compute a candidate iterate u_{i+1} as well as a
   candidate nonlinear function f(u_{i+1}), (2) pass control to the checking
   routine, and then (3) force a re-evaluation of f(u_{i+1}) if any changes
   were made to the candidate iterate in the checking routine (as indicated
   by flag=PETSC_TRUE).  The overhead of this extra function re-evaluation can be
   very costly, so use this feature with caution!

.keywords: SNES, nonlinear, set, line search check, step check, routine

.seealso: SNESLineSearchSet(), SNESLineSearchSetPreCheck()
@*/
PetscErrorCode  SNESLineSearchSetPostCheck(SNES snes,PetscErrorCode (*func)(SNES,Vec,Vec,Vec,void*,PetscBool *,PetscBool *),void *checkctx)
{
  PetscFunctionBegin;
  snes->ops->postcheckstep = func;
  snes->postcheck          = checkctx;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchSetMonitor"
/*@C
   SNESLineSearchSetMonitor - Prints information about the progress or lack of progress of the line search

   Input Parameters:
+  snes - nonlinear context obtained from SNESCreate()
-  flg - PETSC_TRUE to monitor the line search

   Logically Collective on SNES

   Options Database:
.   -snes_ls_monitor

   Level: intermediate


.seealso: SNESLineSearchSet(), SNESLineSearchSetPostCheck(), SNESSetUpdate()
@*/
PetscErrorCode  SNESLineSearchSetMonitor(SNES snes,PetscBool flg)
{

  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (flg && !snes->ls_monitor) {
    ierr = PetscViewerASCIIOpen(((PetscObject)snes)->comm,"stdout",&snes->ls_monitor);CHKERRQ(ierr);
  } else if (!flg && snes->ls_monitor) {
    ierr = PetscViewerDestroy(&snes->ls_monitor);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchNo"

/*@C
   SNESLineSearchNo - This routine is not a line search at all;
   it simply uses the full Newton step.  Thus, this routine is intended
   to serve as a template and is not recommended for general use.

   Logically Collective on SNES and Vec

   Input Parameters:
+  snes - nonlinear context
.  lsctx - optional context for line search (not used here)
.  x - current iterate
.  f - residual evaluated at x
.  y - search direction
.  fnorm - 2-norm of f
-  xnorm - norm of x if known, otherwise 0

   Output Parameters:
+  g - residual evaluated at new iterate y
.  w - new iterate
.  gnorm - 2-norm of g
.  ynorm - 2-norm of search length
-  flag - PETSC_TRUE on success, PETSC_FALSE on failure

   Options Database Key:
.  -snes_ls basic - Activates SNESLineSearchNo()

   Level: advanced

.keywords: SNES, nonlinear, line search, cubic

.seealso: SNESLineSearchCubic(), SNESLineSearchQuadratic(),
          SNESLineSearchSet(), SNESLineSearchNoNorms()
@*/
PetscErrorCode  SNESLineSearchNo(SNES snes,void *lsctx,Vec x,Vec f,Vec y,PetscReal fnorm,PetscReal xnorm,Vec g,Vec w,PetscReal *ynorm,PetscReal *gnorm,PetscBool  *flag)
{
  PetscErrorCode ierr;
  PetscBool      changed_w = PETSC_FALSE,changed_y = PETSC_FALSE;

  PetscFunctionBegin;
  *flag = PETSC_TRUE;
  ierr = PetscLogEventBegin(SNES_LineSearch,snes,x,f,g);CHKERRQ(ierr);
  ierr = VecNorm(y,NORM_2,ynorm);CHKERRQ(ierr);         /* ynorm = || y || */
  ierr = VecWAXPY(w,-snes->damping,y,x);CHKERRQ(ierr);            /* w <- x - y   */
  if (snes->ops->postcheckstep) {
   ierr = (*snes->ops->postcheckstep)(snes,x,y,w,snes->postcheck,&changed_y,&changed_w);CHKERRQ(ierr);
  }
  if (changed_y) {
    ierr = VecWAXPY(w,-snes->damping,y,x);CHKERRQ(ierr);            /* w <- x - y   */
  }
  ierr = SNESComputeFunction(snes,w,g);CHKERRQ(ierr);
  if (!snes->domainerror) {
    ierr = VecNorm(g,NORM_2,gnorm);CHKERRQ(ierr);  /* gnorm = || g || */
    if (PetscIsInfOrNanReal(*gnorm)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FP,"User provided compute function generated a Not-a-Number");
  }
  ierr = PetscLogEventEnd(SNES_LineSearch,snes,x,f,g);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchNoNorms"

/*@C
   SNESLineSearchNoNorms - This routine is not a line search at
   all; it simply uses the full Newton step. This version does not
   even compute the norm of the function or search direction; this
   is intended only when you know the full step is fine and are
   not checking for convergence of the nonlinear iteration (for
   example, you are running always for a fixed number of Newton steps).

   Logically Collective on SNES and Vec

   Input Parameters:
+  snes - nonlinear context
.  lsctx - optional context for line search (not used here)
.  x - current iterate
.  f - residual evaluated at x
.  y - search direction
.  fnorm - 2-norm of f
-  xnorm - norm of x if known, otherwise 0

   Output Parameters:
+  g - residual evaluated at new iterate y
.  w - new iterate
.  gnorm - not changed
.  ynorm - not changed
-  flag - set to PETSC_TRUE indicating a successful line search

   Options Database Key:
.  -snes_ls basicnonorms - Activates SNESLineSearchNoNorms()

   Notes:
   SNESLineSearchNoNorms() must be used in conjunction with
   either the options
$     -snes_no_convergence_test -snes_max_it <its>
   or alternatively a user-defined custom test set via
   SNESSetConvergenceTest(); or a -snes_max_it of 1,
   otherwise, the SNES solver will generate an error.

   During the final iteration this will not evaluate the function at
   the solution point. This is to save a function evaluation while
   using pseudo-timestepping.

   The residual norms printed by monitoring routines such as
   SNESMonitorDefault() (as activated via -snes_monitor) will not be
   correct, since they are not computed.

   Level: advanced

.keywords: SNES, nonlinear, line search, cubic

.seealso: SNESLineSearchCubic(), SNESLineSearchQuadratic(),
          SNESLineSearchSet(), SNESLineSearchNo()
@*/
PetscErrorCode  SNESLineSearchNoNorms(SNES snes,void *lsctx,Vec x,Vec f,Vec y,PetscReal fnorm,PetscReal xnorm,Vec g,Vec w,PetscReal *ynorm,PetscReal *gnorm,PetscBool  *flag)
{
  PetscErrorCode ierr;
  PetscBool      changed_w = PETSC_FALSE,changed_y = PETSC_FALSE;

  PetscFunctionBegin;
  *flag = PETSC_TRUE;
  ierr = PetscLogEventBegin(SNES_LineSearch,snes,x,f,g);CHKERRQ(ierr);
  ierr = VecWAXPY(w,-snes->damping,y,x);CHKERRQ(ierr);            /* w <- x - y      */
  if (snes->ops->postcheckstep) {
   ierr = (*snes->ops->postcheckstep)(snes,x,y,w,snes->postcheck,&changed_y,&changed_w);CHKERRQ(ierr);
  }
  if (changed_y) {
    ierr = VecWAXPY(w,-snes->damping,y,x);CHKERRQ(ierr);            /* w <- x - y   */
  }

  /* don't evaluate function the last time through */
  if (snes->iter < snes->max_its-1) {
    ierr = SNESComputeFunction(snes,w,g);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(SNES_LineSearch,snes,x,f,g);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchQuadratic"
PetscErrorCode  SNESLineSearchQuadratic(SNES snes,void *lsctx,Vec x,Vec f,Vec y,PetscReal fnorm,PetscReal xnorm,Vec g,Vec w,PetscReal *ynorm,PetscReal *gnorm,PetscBool  *flag) {
  PetscFunctionBegin;
  SETERRQ(((PetscObject)snes)->comm, PETSC_ERR_SUP, "SNESLineSearchQuadratic for general nonlinear solvers not yet implemented");
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchCubic"
PetscErrorCode  SNESLineSearchCubic(SNES snes,void *lsctx,Vec x,Vec f,Vec y,PetscReal fnorm,PetscReal xnorm,Vec g,Vec w,PetscReal *ynorm,PetscReal *gnorm,PetscBool  *flag) {
  PetscFunctionBegin;
  SETERRQ(((PetscObject)snes)->comm, PETSC_ERR_SUP, "SNESLineSearchCubic for general nonlinear solvers not yet implemented");
  PetscFunctionReturn(0);
}

/*@ SNESLineSearchSecant - This routine calculates a search length based upon secant descent assuming that F(X +
 lambdaY) is minimized in some problem-specific fashion when F(X + lambdaY) dot Y = 0.  This minimization is performed
 using secant descent on lambda; that is,

lambda_i+1 = lambda_i + (F dot Y)*(lambda_i - lambda_i-1) / (F dot Y - F_old dot Y)

   Input Parameters:
+  snes - nonlinear context
.  lsctx - optional context for line search (not used here)
.  X - current iterate
.  F - residual evaluated at x
.  Y - search direction
.  fnorm - 2-norm of f
-  xnorm - norm of x if known, otherwise 0

   Output Parameters:
+  G - residual evaluated at new iterate y
.  W - new iterate
.  gnorm - norm of G
.  ynorm - not changed
-  flag - set to PETSC_TRUE indicating a successful line search

   Options Database Keys:
.  -snes_ls secant - Activates SNESLineSearchSecant() on selected solvers.
.  -snes_ls_it 1        - Sets the number of steps of the secant iteration occur.
.  -snes_ls_damping 1.0 - Sets a "safe" initial secant step.

   Notes: Works best with the Quasi-Newton and Nonlinear Conjugate Gradient solvers.  Is akin to satisfying the second
   strong Wolfe condition for an unknown problem-specific optimization quantity for which F is the gradient.

   Level: advanced

.keywords: SNES, nonlinear, line search, secant

.seealso: SNESLineSearchCubic(), SNESLineSearchQuadratic(),
          SNESLineSearchSet(), SNESLineSearchNo()
@*/

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchSecant"
PetscErrorCode  SNESLineSearchSecant(SNES snes,void *lsctx,Vec X,Vec F,Vec Y,PetscReal fnorm,PetscReal xnorm,Vec G,Vec W,PetscReal *ynorm,PetscReal *gnorm,PetscBool  *flag) {
  PetscErrorCode ierr;
  PetscReal      lambda = snes->damping;
  PetscReal      lambda_old = 0.0;
  PetscReal      lambda_update;
  PetscReal      fnrm;
  PetscScalar    fty;
  PetscScalar    fty_old;
  PetscInt       i;

  PetscFunctionBegin;
  ierr = VecDot(F, Y, &fty_old);CHKERRQ(ierr);
  *flag = PETSC_TRUE;
  for (i = 0; i < snes->ls_its; i++) {
    /* convergence in the step tolerance */

    if (fabs((lambda - lambda_old) / lambda) < snes->steptol) break;

    ierr = VecCopy(X, W);CHKERRQ(ierr);
    ierr = VecAXPY(W, lambda, Y);CHKERRQ(ierr);
    ierr = SNESComputeFunction(snes, W, G);CHKERRQ(ierr);
    ierr = VecNorm(G, NORM_2, &fnrm);CHKERRQ(ierr);
    ierr = VecDot(G, Y, &fty);CHKERRQ(ierr);

    if (fabs((lambda - lambda_old) / lambda) < snes->steptol) break;

    /* lambda = lambda - (F, Y)*(lambda - lambda_old) / ((F, Y) - (F_old, Y)) */
    lambda_update =  PetscRealPart((fty*lambda_old - fty_old*lambda) / (fty - fty_old));
    if (lambda_update > snes->maxstep || lambda_update < snes->steptol) {
      break;
    }
    if (snes->ls_monitor) {
      ierr = PetscViewerASCIIAddTab(snes->ls_monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(snes->ls_monitor,"    Line search: lambda = [%g, %g] f dot y = [%g, %g], curvature = %g \n",
                                    lambda, lambda_old, PetscRealPart(fty), PetscRealPart(fty_old), 
                                    PetscRealPart((fty - fty_old) / (lambda - lambda_old)));CHKERRQ(ierr);
      ierr = PetscViewerASCIISubtractTab(snes->ls_monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
    }
    lambda_old = lambda;
    lambda = lambda_update;
    fty_old = fty;
  }
  ierr = VecCopy(X, W);CHKERRQ(ierr);
  ierr = VecAXPY(W, lambda, Y);CHKERRQ(ierr);
  ierr = SNESComputeFunction(snes, W, G);CHKERRQ(ierr);
  ierr = VecNorm(G, NORM_2, gnorm);CHKERRQ(ierr);
  if (snes->ls_monitor) {
    ierr = PetscViewerASCIIAddTab(snes->ls_monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(snes->ls_monitor,"    Line search termination: lambda = %g\n", lambda);CHKERRQ(ierr);
    ierr = PetscViewerASCIISubtractTab(snes->ls_monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
  }
  if (lambda <= snes->steptol) {
    *flag = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchQuadraticSecant"
/*@
 SNESLineSearchQuadraticSecant - This routine is an approximate secant line search on the L_2 norm of the residual.

   Input Parameters:
+  snes - nonlinear context
.  lsctx - optional context for line search (not used here)
.  X - current iterate
.  F - residual evaluated at x
.  Y - search direction
.  fnorm - 2-norm of f
-  xnorm - norm of x if known, otherwise 0

   Output Parameters:
+  G - residual evaluated at new iterate y
.  W - new iterate
.  gnorm - norm of G
.  ynorm - not changed
-  flag - set to PETSC_TRUE indicating a successful line search


   Options Database Keys:
.  -snes_ls    secant   - Activates SNESLineSearchSecant() on selected solvers.
.  -snes_ls_it 1        - Sets the number of steps of the secant iteration occur.
.  -snes_ls_damping 1.0 - Sets a "safe" initial secant step.

   Notes: A single step of this behaves exactly like LineSearchIntervalQuadratic, except that it is able to step outside
   the unit interval, delivering a more optimal search step in the cases where the update is ill-scaled.  This occurs
   frequently for methods such as the nonlinear Richardson solver.

   Level: advanced

.keywords: SNES, nonlinear, line search, quadratic, secant

.seealso: SNESLineSearchQuadratic(), SNESLineSearchSecant(),
          SNESLineSearchSet(), SNESLineSearchNo()
@*/
PetscErrorCode  SNESLineSearchQuadraticSecant(SNES snes,void *lsctx,Vec X,Vec F,Vec Y,PetscReal fnorm,PetscReal xnorm,Vec G,Vec W,PetscReal *ynorm,PetscReal *gnorm,PetscBool  *flag) {
  PetscReal        alpha, alpha_old, alpha_mid, alpha_update, delAlpha;
  PetscScalar      fnrm, fnrm_old, fnrm_mid;
  PetscScalar      delFnrm, delFnrm_old, del2Fnrm;
  PetscInt         i;
  PetscErrorCode    ierr;
  PetscFunctionBegin;
  alpha = snes->damping;
  alpha_old = 0.0;
  fnrm_old = fnorm*fnorm;
  alpha_mid = 0.5*(alpha + alpha_old);
  *flag = PETSC_TRUE;
  for (i = 0; i < snes->ls_its; i++) {

  /* compute the norm at the midpoint */
  ierr = VecCopy(X, W);CHKERRQ(ierr);
  ierr = VecAXPY(W, alpha_mid, Y);CHKERRQ(ierr);
  ierr = SNESComputeFunction(snes, W, G);CHKERRQ(ierr);
  ierr = VecDot(G, G, &fnrm_mid);CHKERRQ(ierr);

  /* compute the norm at alpha */
  ierr = VecCopy(X, W);CHKERRQ(ierr);
  ierr = VecAXPY(W, alpha, Y);CHKERRQ(ierr);
  ierr = SNESComputeFunction(snes, W, G);CHKERRQ(ierr);
  ierr = VecDot(G, G, &fnrm);CHKERRQ(ierr);

  /* this gives us the derivatives at the endpoints -- compute them from here

   a = x - a0

   p_0(x) = (x / dA - 1)(2x / dA - 1)
   p_1(x) = 4(x / dA)(1 - x / dA)
   p_2(x) = (x / dA)(2x / dA - 1)

   dp_0[0] / dx = 3 / dA
   dp_1[0] / dx = -4 / dA
   dp_2[0] / dx = 1 / dA

   dp_0[dA] / dx = -1 / dA
   dp_1[dA] / dx = 4 / dA
   dp_2[dA] / dx = -3 / dA

   d^2p_0[0] / dx^2 =  4 / dA^2
   d^2p_1[0] / dx^2 = -8 / dA^2
   d^2p_2[0] / dx^2 =  4 / dA^2
     */

    delAlpha    = alpha - alpha_old;
    delFnrm      = (3.*fnrm - 4.*fnrm_mid + 1.*fnrm_old) / delAlpha;
    delFnrm_old  = (-3.*fnrm_old + 4.*fnrm_mid -1.*fnrm) / delAlpha;
    del2Fnrm = (delFnrm - delFnrm_old) / delAlpha;

    /* check for positive curvature -- looking for that root wouldn't be a good thing. */
    while ((PetscRealPart(del2Fnrm) < 0.0) && (fabs(delAlpha) > snes->steptol)) {
      fnrm_old = fnrm_mid;
      alpha_old = alpha_mid;
      alpha_mid = 0.5*(alpha_old + alpha);
      ierr = VecCopy(X, W);CHKERRQ(ierr);
      ierr = VecAXPY(W, alpha_mid, Y);CHKERRQ(ierr);
      ierr = SNESComputeFunction(snes, W, G);CHKERRQ(ierr);
      ierr = VecDot(G, G, &fnrm_mid);CHKERRQ(ierr);
      delAlpha    = alpha - alpha_old;
      delFnrm      = (3.*fnrm - 4.*fnrm_mid + 1.*fnrm_old) / delAlpha;
      delFnrm_old  = (-3.*fnrm_old + 4.*fnrm_mid -1.*fnrm) / delAlpha;
      del2Fnrm = (delFnrm - delFnrm_old) / delAlpha;
    }

    if (snes->ls_monitor) {
      ierr = PetscViewerASCIIAddTab(snes->ls_monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(snes->ls_monitor,"    Line search: alphas = [%g, %g, %g], fnorms = [%g, %g, %g]\n",
                                    alpha, alpha_mid, alpha_old, PetscSqrtReal(PetscRealPart(fnrm)), PetscSqrtReal(PetscRealPart(fnrm_mid)), PetscSqrtReal(PetscRealPart(fnrm_old)));CHKERRQ(ierr);
      ierr = PetscViewerASCIISubtractTab(snes->ls_monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
    }

    /* compute the search direction */
    alpha_update = alpha - PetscRealPart(delFnrm)*delAlpha / PetscRealPart(delFnrm - delFnrm_old);
    if (PetscIsInfOrNanScalar(alpha_update)) break;
    if (alpha_update > snes->maxstep) {
      break;
    }

    /* compute the new state of the line search */
    alpha_old = alpha;
    alpha = alpha_update;
    fnrm_old = fnrm;
    alpha_mid = 0.5*(alpha + alpha_old);
  }
  ierr = VecCopy(X, W);CHKERRQ(ierr);
  ierr = VecAXPY(W, alpha, Y);CHKERRQ(ierr);
  ierr = SNESComputeFunction(snes, W, G);CHKERRQ(ierr);
  ierr = VecNorm(G, NORM_2, gnorm);CHKERRQ(ierr);
  if (snes->ls_monitor) {
    ierr = PetscViewerASCIIAddTab(snes->ls_monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(snes->ls_monitor,"    Line search terminated: alpha = %g, fnorms = %g\n", alpha, *gnorm);CHKERRQ(ierr);
    ierr = PetscViewerASCIISubtractTab(snes->ls_monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
  }
  if (alpha <= snes->steptol) {
    *flag = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}
