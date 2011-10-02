#include <private/snesimpl.h>      /*I "petscsnes.h"  I*/

const char *const SNESLineSearchTypes[] = {"BASIC","BASICNONORMS","QUADRATIC","CUBIC","EXACT", "TEST", "SNESLineSearchType","SNES_LS_",0};

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
  snes->ops->linesearch = func;
  snes->lsP = lsctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchSetType"
/*@C
   SNESLineSearchSetType - Sets the line search routine to be used
   by the method SNESLS.

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
  switch (type) {
  case SNES_LS_BASIC:
    ierr = SNESLineSearchSet(snes,snes->ops->linesearchno,PETSC_NULL);CHKERRQ(ierr);
    break;
  case SNES_LS_BASIC_NONORMS:
    ierr = SNESLineSearchSet(snes,snes->ops->linesearchnonorms,PETSC_NULL);CHKERRQ(ierr);
    break;
  case SNES_LS_QUADRATIC:
    ierr = SNESLineSearchSet(snes,snes->ops->linesearchquadratic,PETSC_NULL);CHKERRQ(ierr);
    break;
  case SNES_LS_CUBIC:
    ierr = SNESLineSearchSet(snes,snes->ops->linesearchcubic,PETSC_NULL);CHKERRQ(ierr);
    break;
  case SNES_LS_EXACT:
    ierr = SNESLineSearchSet(snes,snes->ops->linesearchexact,PETSC_NULL);CHKERRQ(ierr);
    break;
  case SNES_LS_TEST:
    ierr = SNESLineSearchSet(snes,snes->ops->linesearchtest,PETSC_NULL);CHKERRQ(ierr);
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP,"Unknown line search type.");
    break;
  }
  if (!snes->ops->linesearch) {
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_SUP,"Line search type %s unsupported for SNES type.", SNESLineSearchTypeName(type));;
  } else {
    snes->ls_type = type;
  }
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
   were made to the candidate iterate in the checking routine (as indicate 
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
PetscErrorCode  SNESLineSearchSetMonitor(SNES snes,PetscBool  flg)
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

PetscErrorCode SNESLineSearchNo(SNES snes,void *lsctx,Vec X,Vec F,Vec Y,PetscReal fnorm,PetscReal dummyXnorm,Vec dummyG,Vec W,PetscReal *dummyYnorm,PetscReal *gnorm,PetscBool *flag)
{
  PetscErrorCode   ierr;
  PetscFunctionBegin;
  ierr = VecAXPY(X, snes->damping, Y);CHKERRQ(ierr);
  ierr = SNESComputeFunction(snes, X, F);CHKERRQ(ierr);
  ierr = VecNorm(F, NORM_2, gnorm);CHKERRQ(ierr);
  if (PetscIsInfOrNanReal(*gnorm)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FP,"Infinite or not-a-number generated norm");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchNoNorms"

PetscErrorCode SNESLineSearchNoNorms(SNES snes,void *lsctx,Vec X,Vec F,Vec Y,PetscReal fnorm,PetscReal dummyXnorm,Vec dummyG,Vec W,PetscReal *dummyYnorm,PetscReal *gnorm,PetscBool *flag)
{
  PetscErrorCode   ierr;
  PetscFunctionBegin;
  ierr = VecAXPY(X, snes->damping, Y);CHKERRQ(ierr);
  ierr = SNESComputeFunction(snes, X, F);CHKERRQ(ierr);
  ierr = VecNorm(F, NORM_2, gnorm);CHKERRQ(ierr);
  if (PetscIsInfOrNanReal(*gnorm)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FP,"Infinite or not-a-number generated norm");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchQuadratic"

PetscErrorCode SNESLineSearchQuadratic(SNES snes,void *lsctx,Vec X,Vec F,Vec Y,PetscReal fnorm,PetscReal dummyXnorm,Vec G,Vec W,PetscReal *dummyYnorm,PetscReal *gnorm,PetscBool *flag)
{
  PetscInt         i;
  PetscReal        alphas[3] = {0.0, 0.5, 1.0};
  PetscReal        norms[3];
  PetscReal        alpha,a,b;
  PetscErrorCode   ierr;
  PetscFunctionBegin;
  norms[0]  = fnorm;
  for(i=1; i < 3; ++i) {
    ierr = VecWAXPY(W, alphas[i], Y, X);CHKERRQ(ierr);     /* W =  X^n - \alpha Y */
    ierr = SNESComputeFunction(snes, W, F);CHKERRQ(ierr);
    ierr = VecNorm(F, NORM_2, &norms[i]);CHKERRQ(ierr);
  }
  for(i = 0; i < 3; ++i) {
    norms[i] = PetscSqr(norms[i]);
  }
  /* Fit a quadratic:
       If we have x_{0,1,2} = 0, x_1, x_2 which generate norms y_{0,1,2}
       a = (x_1 y_2 - x_2 y_1 + (x_2 - x_1) y_0)/(x^2_2 x_1 - x_2 x^2_1)
       b = (x^2_1 y_2 - x^2_2 y_1 + (x^2_2 - x^2_1) y_0)/(x_2 x^2_1 - x^2_2 x_1)
       c = y_0
       x_min = -b/2a

       If we let x_{0,1,2} = 0, 0.5, 1.0
       a = 2 y_2 - 4 y_1 + 2 y_0
       b =  -y_2 + 4 y_1 - 3 y_0
       c =   y_0
  */
  a = (alphas[1]*norms[2] - alphas[2]*norms[1] + (alphas[2] - alphas[1])*norms[0])/(PetscSqr(alphas[2])*alphas[1] - alphas[2]*PetscSqr(alphas[1]));
  b = (PetscSqr(alphas[1])*norms[2] - PetscSqr(alphas[2])*norms[1] + (PetscSqr(alphas[2]) - PetscSqr(alphas[1]))*norms[0])/(alphas[2]*PetscSqr(alphas[1]) - PetscSqr(alphas[2])*alphas[1]);
  /* Check for positive a (concave up) */
  if (a >= 0.0) {
    alpha = -b/(2.0*a);
    alpha = PetscMin(alpha, alphas[2]);
    alpha = PetscMax(alpha, alphas[0]);
  } else {
    alpha = 1.0;
  }
  if (snes->ls_monitor) {
    ierr = PetscViewerASCIIAddTab(snes->ls_monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(snes->ls_monitor,"    Line search: norms[0] = %g, norms[1] = %g, norms[2] = %g alpha %g\n", sqrt(norms[0]),sqrt(norms[1]),sqrt(norms[2]),alpha);CHKERRQ(ierr);
    ierr = PetscViewerASCIISubtractTab(snes->ls_monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
  }
  ierr = VecAXPY(X, alpha, Y);CHKERRQ(ierr);
  if (alpha != 1.0) {
    ierr = SNESComputeFunction(snes, X, F);CHKERRQ(ierr);
    ierr = VecNorm(F, NORM_2, gnorm);CHKERRQ(ierr);
  } else {
    *gnorm = PetscSqrtReal(norms[2]);
  }
  if (alpha == 0.0) *flag = PETSC_FALSE;
  else              *flag = PETSC_TRUE;
  PetscFunctionReturn(0);
}
