/*$Id: snesmfj2.c,v 1.35 2001/08/21 21:03:55 bsmith Exp $*/

#include "src/snes/snesimpl.h"   /*I  "petscsnes.h"   I*/

EXTERN int DiffParameterCreate_More(SNES,Vec,void**);
EXTERN int DiffParameterCompute_More(SNES,void*,Vec,Vec,PetscReal*,PetscReal*);
EXTERN int DiffParameterDestroy_More(void*);

typedef struct {  /* default context for matrix-free SNES */
  SNES         snes;             /* SNES context */
  Vec          w;                /* work vector */
  MatNullSpace sp;               /* null space context */
  PetscReal    error_rel;        /* square root of relative error in computing function */
  PetscReal    umin;             /* minimum allowable u'a value relative to |u|_1 */
  int          jorge;            /* flag indicating use of Jorge's method for determining
                                   the differencing parameter */
  PetscReal    h;                /* differencing parameter */
  int          need_h;           /* flag indicating whether we must compute h */
  int          need_err;         /* flag indicating whether we must currently compute error_rel */
  int          compute_err;      /* flag indicating whether we must ever compute error_rel */
  int          compute_err_iter; /* last iter where we've computer error_rel */
  int          compute_err_freq; /* frequency of computing error_rel */
  void         *data;            /* implementation-specific data */
} MFCtx_Private;

#undef __FUNCT__  
#define __FUNCT__ "SNESMatrixFreeDestroy2_Private" /* ADIC Ignore */
int SNESMatrixFreeDestroy2_Private(Mat mat)
{
  int           ierr;
  MFCtx_Private *ctx;

  PetscFunctionBegin;
  ierr = MatShellGetContext(mat,(void **)&ctx);
  ierr = VecDestroy(ctx->w);CHKERRQ(ierr);
  if (ctx->sp) {ierr = MatNullSpaceDestroy(ctx->sp);CHKERRQ(ierr);}
  if (ctx->jorge || ctx->compute_err) {ierr = DiffParameterDestroy_More(ctx->data);CHKERRQ(ierr);}
  ierr = PetscFree(ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESMatrixFreeView2_Private" /* ADIC Ignore */
/*
   SNESMatrixFreeView2_Private - Views matrix-free parameters.
 */
int SNESMatrixFreeView2_Private(Mat J,PetscViewer viewer)
{
  int           ierr;
  MFCtx_Private *ctx;
  PetscTruth    isascii;

  PetscFunctionBegin;
  ierr = MatShellGetContext(J,(void **)&ctx);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
     ierr = PetscViewerASCIIPrintf(viewer,"  SNES matrix-free approximation:\n");CHKERRQ(ierr);
     if (ctx->jorge) {
       ierr = PetscViewerASCIIPrintf(viewer,"    using Jorge's method of determining differencing parameter\n");CHKERRQ(ierr);
     }
     ierr = PetscViewerASCIIPrintf(viewer,"    err=%g (relative error in function evaluation)\n",ctx->error_rel);CHKERRQ(ierr);
     ierr = PetscViewerASCIIPrintf(viewer,"    umin=%g (minimum iterate parameter)\n",ctx->umin);CHKERRQ(ierr);
     if (ctx->compute_err) {
       ierr = PetscViewerASCIIPrintf(viewer,"    freq_err=%d (frequency for computing err)\n",ctx->compute_err_freq);CHKERRQ(ierr);
     }
  } else {
    SETERRQ1(1,"Viewer type %s not supported by SNES matrix free Jorge",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESMatrixFreeMult2_Private"
/*
  SNESMatrixFreeMult2_Private - Default matrix-free form for Jacobian-vector
  product, y = F'(u)*a:
        y = (F(u + ha) - F(u)) /h, 
  where F = nonlinear function, as set by SNESSetFunction()
        u = current iterate
        h = difference interval
*/
int SNESMatrixFreeMult2_Private(Mat mat,Vec a,Vec y)
{
  MFCtx_Private *ctx;
  SNES          snes;
  PetscReal     h,norm,sum,umin,noise;
  PetscScalar   hs,dot,mone = -1.0;
  Vec           w,U,F;
  int           ierr,iter,(*eval_fct)(SNES,Vec,Vec);
  MPI_Comm      comm;

  PetscFunctionBegin;

  /* We log matrix-free matrix-vector products separately, so that we can
     separate the performance monitoring from the cases that use conventional
     storage.  We may eventually modify event logging to associate events
     with particular objects, hence alleviating the more general problem. */
  ierr = PetscLogEventBegin(MAT_MultMatrixFree,a,y,0,0);CHKERRQ(ierr);

  ierr = PetscObjectGetComm((PetscObject)mat,&comm);CHKERRQ(ierr);
  ierr = MatShellGetContext(mat,(void **)&ctx);CHKERRQ(ierr);
  snes = ctx->snes;
  w    = ctx->w;
  umin = ctx->umin;

  ierr = SNESGetSolution(snes,&U);CHKERRQ(ierr);
  if (snes->method_class == SNES_NONLINEAR_EQUATIONS) {
    eval_fct = SNESComputeFunction;
    ierr = SNESGetFunction(snes,&F,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  }
  else if (snes->method_class == SNES_UNCONSTRAINED_MINIMIZATION) {
    eval_fct = SNESComputeGradient;
    ierr = SNESGetGradient(snes,&F,PETSC_NULL);CHKERRQ(ierr);
  }
  else SETERRQ(1,"Invalid method class");


  /* Determine a "good" step size, h */
  if (ctx->need_h) {

    /* Use Jorge's method to compute h */
    if (ctx->jorge) {
      ierr = DiffParameterCompute_More(snes,ctx->data,U,a,&noise,&h);CHKERRQ(ierr);

    /* Use the Brown/Saad method to compute h */
    } else { 
      /* Compute error if desired */
      ierr = SNESGetIterationNumber(snes,&iter);CHKERRQ(ierr);
      if ((ctx->need_err) || ((ctx->compute_err_freq) && (ctx->compute_err_iter != iter) && (!((iter-1)%ctx->compute_err_freq)))) {
        /* Use Jorge's method to compute noise */
        ierr = DiffParameterCompute_More(snes,ctx->data,U,a,&noise,&h);CHKERRQ(ierr);
        ctx->error_rel = sqrt(noise);
        PetscLogInfo(snes,"SNESMatrixFreeMult2_Private: Using Jorge's noise: noise=%g, sqrt(noise)=%g, h_more=%g\n",noise,ctx->error_rel,h);
        ctx->compute_err_iter = iter;
        ctx->need_err = 0;
      }

      ierr = VecDotBegin(U,a,&dot);CHKERRQ(ierr);
      ierr = VecNormBegin(a,NORM_1,&sum);CHKERRQ(ierr);
      ierr = VecNormBegin(a,NORM_2,&norm);CHKERRQ(ierr);
      ierr = VecDotEnd(U,a,&dot);CHKERRQ(ierr);
      ierr = VecNormEnd(a,NORM_1,&sum);CHKERRQ(ierr);
      ierr = VecNormEnd(a,NORM_2,&norm);CHKERRQ(ierr);


      /* Safeguard for step sizes too small */
      if (sum == 0.0) {dot = 1.0; norm = 1.0;}
#if defined(PETSC_USE_COMPLEX)
      else if (PetscAbsScalar(dot) < umin*sum && PetscRealPart(dot) >= 0.0) dot = umin*sum;
      else if (PetscAbsScalar(dot) < 0.0 && PetscRealPart(dot) > -umin*sum) dot = -umin*sum;
#else
      else if (dot < umin*sum && dot >= 0.0) dot = umin*sum;
      else if (dot < 0.0 && dot > -umin*sum) dot = -umin*sum;
#endif
      h = PetscRealPart(ctx->error_rel*dot/(norm*norm));
    }
  } else {
    h = ctx->h;
  }
  if (!ctx->jorge || !ctx->need_h) PetscLogInfo(snes,"SNESMatrixFreeMult2_Private: h = %g\n",h);

  /* Evaluate function at F(u + ha) */
  hs = h;
  ierr = VecWAXPY(&hs,a,U,w);CHKERRQ(ierr);
  ierr = eval_fct(snes,w,y);CHKERRQ(ierr);
  ierr = VecAXPY(&mone,F,y);CHKERRQ(ierr);
  hs = 1.0/hs;
  ierr = VecScale(&hs,y);CHKERRQ(ierr);
  if (ctx->sp) {ierr = MatNullSpaceRemove(ctx->sp,y,PETSC_NULL);CHKERRQ(ierr);}

  ierr = PetscLogEventEnd(MAT_MultMatrixFree,a,y,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESMatrixFreeMatCreate2"
/*@C
   SNESMatrixFreeMatCreate2 - Creates a matrix-free matrix
   context for use with a SNES solver.  This matrix can be used as
   the Jacobian argument for the routine SNESSetJacobian().

   Input Parameters:
.  snes - the SNES context
.  x - vector where SNES solution is to be stored.

   Output Parameter:
.  J - the matrix-free matrix

   Notes:
   The matrix-free matrix context merely contains the function pointers
   and work space for performing finite difference approximations of
   Jacobian-vector products, J(u)*a, via

$       J(u)*a = [J(u+h*a) - J(u)]/h,
$   where by default
$        h = error_rel*u'a/||a||^2                        if  |u'a| > umin*||a||_{1}
$          = error_rel*umin*sign(u'a)*||a||_{1}/||a||^2   otherwise
$   where
$        error_rel = square root of relative error in
$                    function evaluation
$        umin = minimum iterate parameter
$   Alternatively, the differencing parameter, h, can be set using
$   Jorge's nifty new strategy if one specifies the option 
$          -snes_mf_jorge

   The user can set these parameters via MatSNESMFSetFunctionError().
   See the nonlinear solvers chapter of the users manual for details.

   The user should call MatDestroy() when finished with the matrix-free
   matrix context.

   Options Database Keys:
$  -snes_mf_err <error_rel>
$  -snes_mf_unim <umin>
$  -snes_mf_compute_err
$  -snes_mf_freq_err <freq>
$  -snes_mf_jorge

.keywords: SNES, default, matrix-free, create, matrix

.seealso: MatDestroy(), MatSNESMFSetFunctionError()
@*/
int SNESDefaultMatrixFreeCreate2(SNES snes,Vec x,Mat *J)
{
  MPI_Comm      comm;
  MFCtx_Private *mfctx;
  int           n,nloc,ierr;
  PetscTruth    flg;
  char          p[64];

  PetscFunctionBegin;
  ierr = PetscNew(MFCtx_Private,&mfctx);CHKERRQ(ierr);
  ierr  = PetscMemzero(mfctx,sizeof(MFCtx_Private));CHKERRQ(ierr);
  PetscLogObjectMemory(snes,sizeof(MFCtx_Private));
  mfctx->sp   = 0;
  mfctx->snes = snes;
  mfctx->error_rel        = PETSC_SQRT_MACHINE_EPSILON;
  mfctx->umin             = 1.e-6;
  mfctx->h                = 0.0;
  mfctx->need_h           = 1;
  mfctx->need_err         = 0;
  mfctx->compute_err      = 0;
  mfctx->compute_err_freq = 0;
  mfctx->compute_err_iter = -1;
  ierr = PetscOptionsGetReal(snes->prefix,"-snes_mf_err",&mfctx->error_rel,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(snes->prefix,"-snes_mf_umin",&mfctx->umin,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(snes->prefix,"-snes_mf_jorge",(PetscTruth*)&mfctx->jorge);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(snes->prefix,"-snes_mf_compute_err",(PetscTruth*)&mfctx->compute_err);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(snes->prefix,"-snes_mf_freq_err",&mfctx->compute_err_freq,&flg);CHKERRQ(ierr);
  if (flg) {
    if (mfctx->compute_err_freq < 0) mfctx->compute_err_freq = 0;
    mfctx->compute_err = 1; 
  }
  if (mfctx->compute_err == 1) mfctx->need_err = 1;
  if (mfctx->jorge || mfctx->compute_err) {
    ierr = DiffParameterCreate_More(snes,x,&mfctx->data);CHKERRQ(ierr);
  } else mfctx->data = 0;

  ierr = PetscOptionsHasName(PETSC_NULL,"-help",&flg);CHKERRQ(ierr);
  ierr = PetscStrcpy(p,"-");CHKERRQ(ierr);
  if (snes->prefix) PetscStrcat(p,snes->prefix);
  if (flg) {
    ierr = PetscPrintf(snes->comm," Matrix-free Options (via SNES):\n");CHKERRQ(ierr);
    ierr = PetscPrintf(snes->comm,"   %ssnes_mf_err <err>: set sqrt of relative error in function (default %g)\n",p,mfctx->error_rel);CHKERRQ(ierr);
    ierr = PetscPrintf(snes->comm,"   %ssnes_mf_umin <umin>: see users manual (default %g)\n",p,mfctx->umin);CHKERRQ(ierr);
    ierr = PetscPrintf(snes->comm,"   %ssnes_mf_jorge: use Jorge More's method\n",p);CHKERRQ(ierr);
    ierr = PetscPrintf(snes->comm,"   %ssnes_mf_compute_err: compute sqrt or relative error in function\n",p);CHKERRQ(ierr);
    ierr = PetscPrintf(snes->comm,"   %ssnes_mf_freq_err <freq>: frequency to recompute this (default only once)\n",p);CHKERRQ(ierr);
    ierr = PetscPrintf(snes->comm,"   %ssnes_mf_noise_file <file>: set file for printing noise info\n",p);CHKERRQ(ierr);
  }
  ierr = VecDuplicate(x,&mfctx->w);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)x,&comm);CHKERRQ(ierr);
  ierr = VecGetSize(x,&n);CHKERRQ(ierr);
  ierr = VecGetLocalSize(x,&nloc);CHKERRQ(ierr);
  ierr = MatCreateShell(comm,nloc,n,n,n,mfctx,J);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*J,MATOP_MULT,(void(*)(void))SNESMatrixFreeMult2_Private);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*J,MATOP_DESTROY,(void(*)(void))SNESMatrixFreeDestroy2_Private);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*J,MATOP_VIEW,(void(*)(void))SNESMatrixFreeView2_Private);CHKERRQ(ierr);

  PetscLogObjectParent(*J,mfctx->w);
  PetscLogObjectParent(snes,*J);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESDefaultMatrixFreeSetParameters2"
/*@
   SNESDefaultMatrixFreeSetParameters2 - Sets the parameters for the approximation of
   matrix-vector products using finite differences.

$       J(u)*a = [J(u+h*a) - J(u)]/h where

   either the user sets h directly here, or this parameter is computed via

$        h = error_rel*u'a/||a||^2                        if  |u'a| > umin*||a||_{1}
$          = error_rel*umin*sign(u'a)*||a||_{1}/||a||^2   else
$

   Input Parameters:
+  mat - the matrix
.  error_rel - relative error (should be set to the square root of
     the relative error in the function evaluations)
.  umin - minimum allowable u-value
-  h - differencing parameter

   Notes:
   If the user sets the parameter h directly, then this value will be used
   instead of the default computation indicated above.

.keywords: SNES, matrix-free, parameters

.seealso: MatCreateSNESMF()
@*/
int SNESDefaultMatrixFreeSetParameters2(Mat mat,PetscReal error,PetscReal umin,PetscReal h)
{
  MFCtx_Private *ctx;
  int           ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(mat,(void **)&ctx);CHKERRQ(ierr);
  if (ctx) {
    if (error != PETSC_DEFAULT) ctx->error_rel = error;
    if (umin  != PETSC_DEFAULT) ctx->umin = umin;
    if (h     != PETSC_DEFAULT) {
      ctx->h = h;
      ctx->need_h = 0;
    }
  }
  PetscFunctionReturn(0);
}

int SNESUnSetMatrixFreeParameter(SNES snes)
{
  MFCtx_Private *ctx;
  int           ierr;
  Mat           mat;

  PetscFunctionBegin;
  ierr = SNESGetJacobian(snes,&mat,PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatShellGetContext(mat,(void **)&ctx);CHKERRQ(ierr);
  if (ctx) ctx->need_h = 1;
  PetscFunctionReturn(0);
}
     
