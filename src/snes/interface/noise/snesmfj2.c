#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: snesmfj2.c,v 1.19 1999/09/27 21:31:40 bsmith Exp bsmith $";
#endif

#include "src/snes/snesimpl.h"   /*I  "snes.h"   I*/

extern int DiffParameterCreate_More(SNES,Vec,void**);
extern int DiffParameterCompute_More(SNES,void*,Vec,Vec,double*,double*);
extern int DiffParameterDestroy_More(void*);

typedef struct {  /* default context for matrix-free SNES */
  SNES        snes;             /* SNES context */
  Vec         w;                /* work vector */
  PCNullSpace sp;               /* null space context */
  double      error_rel;        /* square root of relative error in computing function */
  double      umin;             /* minimum allowable u'a value relative to |u|_1 */
  int         jorge;            /* flag indicating use of Jorge's method for determining
                                   the differencing parameter */
  double      h;                /* differencing parameter */
  int         need_h;           /* flag indicating whether we must compute h */
  int         need_err;         /* flag indicating whether we must currently compute error_rel */
  int         compute_err;      /* flag indicating whether we must ever compute error_rel */
  int         compute_err_iter; /* last iter where we've computer error_rel */
  int         compute_err_freq; /* frequency of computing error_rel */
  void        *data;            /* implementation-specific data */
} MFCtx_Private;

#undef __FUNC__  
#define __FUNC__ "SNESMatrixFreeDestroy2_Private" /* ADIC Ignore */
int SNESMatrixFreeDestroy2_Private(Mat mat)
{
  int           ierr;
  MFCtx_Private *ctx;

  PetscFunctionBegin;
  ierr = MatShellGetContext(mat,(void **)&ctx);
  ierr = VecDestroy(ctx->w);CHKERRQ(ierr);
  if (ctx->sp) {ierr = PCNullSpaceDestroy(ctx->sp);CHKERRQ(ierr);}
  if (ctx->jorge || ctx->compute_err) {ierr = DiffParameterDestroy_More(ctx->data);CHKERRQ(ierr);}
  ierr = PetscFree(ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "SNESMatrixFreeView2_Private" /* ADIC Ignore */
/*
   SNESMatrixFreeView2_Private - Views matrix-free parameters.
 */
int SNESMatrixFreeView2_Private(Mat J,Viewer viewer)
{
  int           ierr;
  MFCtx_Private *ctx;
  MPI_Comm      comm;
  FILE          *fd;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)J,&comm);CHKERRQ(ierr);
  ierr = MatShellGetContext(J,(void **)&ctx);CHKERRQ(ierr);
  ierr = ViewerASCIIGetPointer(viewer,&fd);CHKERRQ(ierr);
  if (PetscTypeCompare(viewer,ASCII_VIEWER)) {
     ierr = PetscFPrintf(comm,fd,"  SNES matrix-free approximation:\n");CHKERRQ(ierr);
     if (ctx->jorge) {
       ierr = PetscFPrintf(comm,fd,"    using Jorge's method of determining differencing parameter\n");CHKERRQ(ierr);
     }
     ierr = PetscFPrintf(comm,fd,"    err=%g (relative error in function evaluation)\n",ctx->error_rel);CHKERRQ(ierr);
     ierr = PetscFPrintf(comm,fd,"    umin=%g (minimum iterate parameter)\n",ctx->umin);CHKERRQ(ierr);
     if (ctx->compute_err) {
       ierr = PetscFPrintf(comm,fd,"    freq_err=%d (frequency for computing err)\n",ctx->compute_err_freq);CHKERRQ(ierr);
     }
  } else {
    SETERRQ(1,1,"Viewer type not supported by PETSc object");
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "SNESMatrixFreeMult2_Private"
/*
  SNESMatrixFreeMult2_Private - Default matrix-free form for Jacobian-vector
  product, y = F'(u)*a:
        y = ( F(u + ha) - F(u) ) /h, 
  where F = nonlinear function, as set by SNESSetFunction()
        u = current iterate
        h = difference interval
*/
int SNESMatrixFreeMult2_Private(Mat mat,Vec a,Vec y)
{
  MFCtx_Private *ctx;
  SNES          snes;
  double        h, norm, sum, umin, noise;
  Scalar        hs, dot, mone = -1.0;
  Vec           w,U,F;
  int           ierr, iter, (*eval_fct)(SNES,Vec,Vec);
  MPI_Comm      comm;

  PetscFunctionBegin;

  /* We log matrix-free matrix-vector products separately, so that we can
     separate the performance monitoring from the cases that use conventional
     storage.  We may eventually modify event logging to associate events
     with particular objects, hence alleviating the more general problem. */
  PLogEventBegin(MAT_MatrixFreeMult,a,y,0,0);

  ierr = PetscObjectGetComm((PetscObject)mat,&comm);CHKERRQ(ierr);
  ierr = MatShellGetContext(mat,(void **)&ctx);CHKERRQ(ierr);
  snes = ctx->snes;
  w    = ctx->w;
  umin = ctx->umin;

  ierr = SNESGetSolution(snes,&U);CHKERRQ(ierr);
  if (snes->method_class == SNES_NONLINEAR_EQUATIONS) {
    eval_fct = SNESComputeFunction;
    ierr = SNESGetFunction(snes,&F,PETSC_NULL);CHKERRQ(ierr);
  }
  else if (snes->method_class == SNES_UNCONSTRAINED_MINIMIZATION) {
    eval_fct = SNESComputeGradient;
    ierr = SNESGetGradient(snes,&F,PETSC_NULL);CHKERRQ(ierr);
  }
  else SETERRQ(1,0,"Invalid method class");


  /* Determine a "good" step size, h */
  if (ctx->need_h) {

    /* Use Jorge's method to compute h */
    if (ctx->jorge) {
      ierr = DiffParameterCompute_More(snes,ctx->data,U,a,&noise,&h);CHKERRQ(ierr);

    /* Use the Brown/Saad method to compute h */
    } else { 
      /* Compute error if desired */
      ierr = SNESGetIterationNumber(snes,&iter);CHKERRQ(ierr);
      if ((ctx->need_err) ||
          ((ctx->compute_err_freq) && (ctx->compute_err_iter != iter) && (!((iter-1)%ctx->compute_err_freq)))) {
        /* Use Jorge's method to compute noise */
        ierr = DiffParameterCompute_More(snes,ctx->data,U,a,&noise,&h);CHKERRQ(ierr);
        ctx->error_rel = sqrt(noise);
        PLogInfo(snes,"SNESMatrixFreeMult2_Private: Using Jorge's noise: noise=%g, sqrt(noise)=%g, h_more=%g\n",
            noise,ctx->error_rel,h);
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
      else if (PetscAbsScalar(dot) < umin*sum && PetscReal(dot) >= 0.0) dot = umin*sum;
      else if (PetscAbsScalar(dot) < 0.0 && PetscReal(dot) > -umin*sum) dot = -umin*sum;
#else
      else if (dot < umin*sum && dot >= 0.0) dot = umin*sum;
      else if (dot < 0.0 && dot > -umin*sum) dot = -umin*sum;
#endif
      h = PetscReal(ctx->error_rel*dot/(norm*norm));
    }
  } else {
    h = ctx->h;
  }
  if (!ctx->jorge || !ctx->need_h) PLogInfo(snes,"SNESMatrixFreeMult2_Private: h = %g\n",h);

  /* Evaluate function at F(u + ha) */
  hs = h;
  ierr = VecWAXPY(&hs,a,U,w);CHKERRQ(ierr);
  ierr = eval_fct(snes,w,y);CHKERRQ(ierr);
  ierr = VecAXPY(&mone,F,y);CHKERRQ(ierr);
  hs = 1.0/hs;
  ierr = VecScale(&hs,y);CHKERRQ(ierr);
  if (ctx->sp) {ierr = PCNullSpaceRemove(ctx->sp,y);CHKERRQ(ierr);}

  PLogEventEnd(MAT_MatrixFreeMult,a,y,0,0);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "SNESMatrixFreeMatCreate2"
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
int SNESDefaultMatrixFreeCreate2(SNES snes,Vec x, Mat *J)
{
  MPI_Comm      comm;
  MFCtx_Private *mfctx;
  int           n, nloc, ierr, flg;
  char          p[64];

  PetscFunctionBegin;
  mfctx = (MFCtx_Private *) PetscMalloc(sizeof(MFCtx_Private));CHKPTRQ(mfctx);
  ierr  = PetscMemzero(mfctx,sizeof(MFCtx_Private));CHKERRQ(ierr);
  PLogObjectMemory(snes,sizeof(MFCtx_Private));
  mfctx->sp   = 0;
  mfctx->snes = snes;
  mfctx->error_rel        = 1.e-8; /* assumes double precision */
  mfctx->umin             = 1.e-6;
  mfctx->h                = 0.0;
  mfctx->need_h           = 1;
  mfctx->need_err         = 0;
  mfctx->compute_err      = 0;
  mfctx->compute_err_freq = 0;
  mfctx->compute_err_iter = -1;
  ierr = OptionsGetDouble(snes->prefix,"-snes_mf_err",&mfctx->error_rel,&flg);CHKERRQ(ierr);
  ierr = OptionsGetDouble(snes->prefix,"-snes_mf_umin",&mfctx->umin,&flg);CHKERRQ(ierr);
  ierr = OptionsHasName(snes->prefix,"-snes_mf_jorge",&mfctx->jorge);CHKERRQ(ierr);
  ierr = OptionsHasName(snes->prefix,"-snes_mf_compute_err",&mfctx->compute_err);CHKERRQ(ierr);
  ierr = OptionsGetInt(snes->prefix,"-snes_mf_freq_err",&mfctx->compute_err_freq,&flg);CHKERRQ(ierr);
  if (flg) {
    if (mfctx->compute_err_freq < 0) mfctx->compute_err_freq = 0;
    mfctx->compute_err = 1; 
  }
  if (mfctx->compute_err == 1) mfctx->need_err = 1;
  if (mfctx->jorge || mfctx->compute_err) {
    ierr = DiffParameterCreate_More(snes,x,&mfctx->data);CHKERRQ(ierr);
  } else mfctx->data = 0;

  ierr = OptionsHasName(PETSC_NULL,"-help",&flg);CHKERRQ(ierr);
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
  ierr = MatShellSetOperation(*J,MATOP_MULT,(void*)SNESMatrixFreeMult2_Private);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*J,MATOP_DESTROY,(void *)SNESMatrixFreeDestroy2_Private);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*J,MATOP_VIEW,(void *)SNESMatrixFreeView2_Private);CHKERRQ(ierr);

  PLogObjectParent(*J,mfctx->w);
  PLogObjectParent(snes,*J);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "SNESDefaultMatrixFreeSetParameters2"
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
int SNESDefaultMatrixFreeSetParameters2(Mat mat,double error,double umin,double h)
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
  ierr = SNESGetJacobian(snes,&mat,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatShellGetContext(mat,(void **)&ctx);CHKERRQ(ierr);
  if (ctx) ctx->need_h = 1;
  PetscFunctionReturn(0);
}
     
