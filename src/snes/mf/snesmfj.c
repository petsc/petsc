#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: snesmfj.c,v 1.69 1998/10/26 03:26:40 bsmith Exp bsmith $";
#endif

#include "src/snes/snesimpl.h"   /*I  "snes.h"   I*/
#include "pinclude/pviewer.h"
#include <math.h>

typedef struct {  /* default context for matrix-free SNES */
  SNES        snes;      /* SNES context */
  Vec         w;         /* work vector */
  PCNullSpace sp;        /* null space context */
  double      error_rel; /* square root of relative error in computing function */
  double      umin;      /* minimum allowable u'a value relative to |u|_1 */
  Scalar      currenth;  /* last differencing parameter used */
  Scalar      *historyh; /* history of h */
  int         ncurrenth,maxcurrenth; 
} MFCtx_Private;

#undef __FUNC__  
#define __FUNC__ "SNESMatrixFreeDestroy_Private"
int SNESMatrixFreeDestroy_Private(Mat mat)
{
  int           ierr;
  MFCtx_Private *ctx;

  PetscFunctionBegin;
  ierr = MatShellGetContext(mat,(void **)&ctx);CHKERRQ(ierr);
  ierr = VecDestroy(ctx->w); CHKERRQ(ierr);
  if (ctx->sp) {ierr = PCNullSpaceDestroy(ctx->sp);CHKERRQ(ierr);}
  PetscFree(ctx);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "SNESMatrixFreeView_Private"
/*
   SNESMatrixFreeView_Private - Views matrix-free parameters.

*/
int SNESMatrixFreeView_Private(Mat J,Viewer viewer)
{
  int           ierr;
  MFCtx_Private *ctx;
  MPI_Comm      comm;
  FILE          *fd;
  ViewerType    vtype;

  PetscFunctionBegin;
  PetscObjectGetComm((PetscObject)J,&comm);
  ierr = MatShellGetContext(J,(void **)&ctx); CHKERRQ(ierr);
  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
  if (vtype == ASCII_FILE_VIEWER || vtype == ASCII_FILES_VIEWER) {
     PetscFPrintf(comm,fd,"  SNES matrix-free approximation:\n");
     PetscFPrintf(comm,fd,"    err=%g (relative error in function evaluation)\n",ctx->error_rel);
     PetscFPrintf(comm,fd,"    umin=%g (minimum iterate parameter)\n",ctx->umin);
  } else {
    SETERRQ(1,1,"Viewer type not supported for this object");
  }
  PetscFunctionReturn(0);
}

extern int VecDot_Seq(Vec,Vec,Scalar *);
extern int VecNorm_Seq(Vec,NormType,double *);

#undef __FUNC__  
#define __FUNC__ "SNESMatrixFreeMult_Private"
/*
  SNESMatrixFreeMult_Private - Default matrix-free form for Jacobian-vector
  product, y = F'(u)*a:
        y = ( F(u + ha) - F(u) ) /h, 
  where F = nonlinear function, as set by SNESSetFunction()
        u = current iterate
        h = difference interval
*/
int SNESMatrixFreeMult_Private(Mat mat,Vec a,Vec y)
{
  MFCtx_Private *ctx;
  SNES          snes;
  double        ovalues[3],norm, sum, umin;
  Scalar        h, dot, mone = -1.0;
  Vec           w,U,F;
  int           ierr, (*eval_fct)(SNES,Vec,Vec);
  MPI_Comm      comm;
#if !defined(USE_PETSC_COMPLEX)
  double        values[3];
#endif

  PetscFunctionBegin;
  PLogEventBegin(MAT_MatrixFreeMult,a,y,0,0);

  PetscObjectGetComm((PetscObject)mat,&comm);
  ierr = MatShellGetContext(mat,(void **)&ctx); CHKERRQ(ierr);
  snes = ctx->snes;
  w    = ctx->w;
  umin = ctx->umin;

  /* We log matrix-free matrix-vector products separately, so that we can
     separate the performance monitoring from the cases that use conventional
     storage.  We may eventually modify event logging to associate events
     with particular objects, hence alleviating the more general problem. */

  ierr = SNESGetSolution(snes,&U); CHKERRQ(ierr);
  if (snes->method_class == SNES_NONLINEAR_EQUATIONS) {
    eval_fct = SNESComputeFunction;
    ierr = SNESGetFunction(snes,&F); CHKERRQ(ierr);
  }
  else if (snes->method_class == SNES_UNCONSTRAINED_MINIMIZATION) {
    eval_fct = SNESComputeGradient;
    ierr = SNESGetGradient(snes,&F); CHKERRQ(ierr);
  }
  else SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Invalid method class");

  /* Determine a "good" step size, h */

  /*
    ierr = VecDot(U,a,&dot); CHKERRQ(ierr);
    ierr = VecNorm(a,NORM_1,&sum); CHKERRQ(ierr);
    ierr = VecNorm(a,NORM_2,&norm); CHKERRQ(ierr);
  */

  /*
     Call the Seq Vector routines and then do a single reduction 
     to reduce the number of communications required
  */

#if !defined(USE_PETSC_COMPLEX)
  PLogEventBegin(VEC_Dot,U,a,0,0);
  ierr = VecDot_Seq(U,a,ovalues); CHKERRQ(ierr);
  PLogEventEnd(VEC_Dot,U,a,0,0);
  PLogEventBegin(VEC_Norm,a,0,0,0);
  ierr = VecNorm_Seq(a,NORM_1,ovalues+1); CHKERRQ(ierr);
  ierr = VecNorm_Seq(a,NORM_2,ovalues+2); CHKERRQ(ierr);
  ovalues[2] = ovalues[2]*ovalues[2];
  PLogEventBarrierBegin(VEC_NormBarrier,0,0,0,0,comm);
  ierr = MPI_Allreduce(ovalues,values,3,MPI_DOUBLE,MPI_SUM,comm );CHKERRQ(ierr);
  PLogEventBarrierEnd(VEC_NormBarrier,0,0,0,0,comm);
  dot = values[0]; sum = values[1]; norm = sqrt(values[2]);
  PLogEventEnd(VEC_Norm,a,0,0,0);
#else
  {
    Scalar cvalues[3],covalues[3];

    PLogEventBegin(VEC_Dot,U,a,0,0);
    ierr = VecDot_Seq(U,a,covalues); CHKERRQ(ierr);
    PLogEventEnd(VEC_Dot,U,a,0,0);
    PLogEventBegin(VEC_Norm,a,0,0,0);
    ierr = VecNorm_Seq(a,NORM_1,ovalues+1); CHKERRQ(ierr);
    ierr = VecNorm_Seq(a,NORM_2,ovalues+2); CHKERRQ(ierr);
    covalues[1] = ovalues[1];
    covalues[2] = ovalues[2]*ovalues[2];
    PLogEventBarrierBegin(VEC_NormBarrier,0,0,0,0,comm);
    ierr = MPI_Allreduce(covalues,cvalues,6,MPI_DOUBLE,MPI_SUM,comm);CHKERRQ(ierr);
    PLogEventBarrierBegin(VEC_NormBarrier,0,0,0,0,comm);
    dot = cvalues[0]; sum = PetscReal(cvalues[1]); norm = sqrt(PetscReal(cvalues[2]));
    PLogEventEnd(VEC_Norm,a,0,0,0);
  }
#endif


  /* Safeguard for step sizes too small */
  if (sum == 0.0) {dot = 1.0; norm = 1.0;}
#if defined(USE_PETSC_COMPLEX)
  else if (PetscAbsScalar(dot) < umin*sum && PetscReal(dot) >= 0.0) dot = umin*sum;
  else if (PetscAbsScalar(dot) < 0.0 && PetscReal(dot) > -umin*sum) dot = -umin*sum;
#else
  else if (dot < umin*sum && dot >= 0.0) dot = umin*sum;
  else if (dot < 0.0 && dot > -umin*sum) dot = -umin*sum;
#endif
  h = ctx->error_rel*dot/(norm*norm);

  /* keep a record of the current differencing parameter h */  
  ctx->currenth = h;
#if defined(USE_PETSC_COMPLEX)
  PLogInfo(mat,"Current differencing parameter: %g + %g i\n",PetscReal(h),PetscImaginary(h));
#else
  PLogInfo(mat,"Current differencing parameter: %g\n",h);
#endif
  if (ctx->historyh && ctx->ncurrenth < ctx->maxcurrenth) {
    ctx->historyh[ctx->ncurrenth++] = h;
  }

  /* Evaluate function at F(u + ha) */
  ierr = VecWAXPY(&h,a,U,w); CHKERRQ(ierr);
  ierr = eval_fct(snes,w,y); CHKERRQ(ierr);

  ierr = VecAXPY(&mone,F,y); CHKERRQ(ierr);
  h = 1.0/h;
  ierr = VecScale(&h,y); CHKERRQ(ierr);
  if (ctx->sp) {ierr = PCNullSpaceRemove(ctx->sp,y); CHKERRQ(ierr);}

  PLogEventEnd(MAT_MatrixFreeMult,a,y,0,0);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "SNESDefaultMatrixFreeCreate"
/*@C
   SNESDefaultMatrixFreeCreate - Creates a matrix-free matrix
   context for use with a SNES solver.  This matrix can be used as
   the Jacobian argument for the routine SNESSetJacobian().

   Collective on SNES and Vec

   Input Parameters:
+  snes - the SNES context
-  x - vector where SNES solution is to be stored.

   Output Parameter:
.  J - the matrix-free matrix

   Notes:
   The matrix-free matrix context merely contains the function pointers
   and work space for performing finite difference approximations of
   Jacobian-vector products, J(u)*a, via

.vb
     J(u)*a = [J(u+h*a) - J(u)]/h where
     h = error_rel*u'a/||a||^2                        if  |u'a| > umin*||a||_{1}
       = error_rel*umin*sign(u'a)*||a||_{1}/||a||^2   otherwise
 where
     error_rel = square root of relative error in function evaluation
     umin = minimum iterate parameter
.ve

   The user can set these parameters via SNESDefaultMatrixFreeSetParameters().
   See the nonlinear solvers chapter of the users manual for details.

   The user should call MatDestroy() when finished with the matrix-free
   matrix context.

   Options Database Keys:
+  -snes_mf_err <error_rel> - Sets error_rel
-  -snes_mf_unim <umin> - Sets umin

.keywords: SNES, default, matrix-free, create, matrix

.seealso: MatDestroy(), SNESDefaultMatrixFreeSetParameters()
@*/
int SNESDefaultMatrixFreeCreate(SNES snes,Vec x, Mat *J)
{
  MPI_Comm      comm;
  MFCtx_Private *mfctx;
  int           n, nloc, ierr, flg;
  char          p[64];

  PetscFunctionBegin;
  mfctx = (MFCtx_Private *) PetscMalloc(sizeof(MFCtx_Private)); CHKPTRQ(mfctx);
  PLogObjectMemory(snes,sizeof(MFCtx_Private));
  mfctx->sp          = 0;
  mfctx->snes        = snes;
  mfctx->error_rel   = 1.e-8; /* assumes double precision */
  mfctx->umin        = 1.e-6;
  mfctx->currenth    = 0.0;
  mfctx->historyh    = PETSC_NULL;
  mfctx->ncurrenth   = 0;
  mfctx->maxcurrenth = 0;

  ierr = OptionsGetDouble(snes->prefix,"-snes_mf_err",&mfctx->error_rel,&flg); CHKERRQ(ierr);
  ierr = OptionsGetDouble(snes->prefix,"-snes_mf_umin",&mfctx->umin,&flg); CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-help",&flg); CHKERRQ(ierr);
  PetscStrcpy(p,"-");
  if (snes->prefix) PetscStrcat(p,snes->prefix);
  if (flg) {
    (*PetscHelpPrintf)(snes->comm,"   %ssnes_mf_err <err>: set sqrt rel error in function (default %g)\n",p,mfctx->error_rel);
    (*PetscHelpPrintf)(snes->comm,"   %ssnes_mf_umin <umin> see users manual (default %g)\n",p,mfctx->umin);
  }
  ierr = VecDuplicate(x,&mfctx->w); CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)x,&comm); CHKERRQ(ierr);
  ierr = VecGetSize(x,&n); CHKERRQ(ierr);
  ierr = VecGetLocalSize(x,&nloc); CHKERRQ(ierr);
  ierr = MatCreateShell(comm,nloc,n,n,n,mfctx,J); CHKERRQ(ierr);
  ierr = MatShellSetOperation(*J,MATOP_MULT,(void*)SNESMatrixFreeMult_Private);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*J,MATOP_DESTROY,(void *)SNESMatrixFreeDestroy_Private);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*J,MATOP_VIEW,(void *)SNESMatrixFreeView_Private); CHKERRQ(ierr);
  PLogObjectParent(*J,mfctx->w);
  PLogObjectParent(snes,*J);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "SNESDefaultMatrixFreeGetH"
/*@
   SNESDefaultMatrixFreeGetH - Gets the last h that was used as the differencing 
     parameter.

   Not Collective

   Input Parameters:
.   mat - the matrix obtained with SNESDefaultMatrixFreeCreate()

   Output Paramter:
.  h - the differencing step size

.keywords: SNES, matrix-free, parameters

.seealso: SNESDefaultMatrixFreeCreate()
@*/
int SNESDefaultMatrixFreeGetH(Mat mat,Scalar *h)
{
  MFCtx_Private *ctx;
  int           ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(mat,(void **)&ctx); CHKERRQ(ierr);
  if (ctx) {
    *h = ctx->currenth;
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "SNESDefaultMatrixFreeKSPMonitor"
/*
   SNESDefaultMatrixFreeKSPMonitor - A KSP monitor for use with the default PETSc
      SNES matrix free routines. Prints the h differencing parameter used at each
      timestep.

*/
int SNESDefaultMatrixFreeKSPMonitor(KSP ksp,int n,double rnorm,void *dummy)
{
  PC            pc;
  MFCtx_Private *ctx;
  int           ierr;
  Mat           mat;
  MPI_Comm      comm;
  PetscTruth    nonzeroinitialguess;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)ksp,&comm);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc); CHKERRQ(ierr);
  ierr = KSPGetInitialGuessNonzero(ksp,&nonzeroinitialguess);CHKERRQ(ierr);
  ierr = PCGetOperators(pc,&mat,PETSC_NULL,PETSC_NULL); CHKERRQ(ierr);
  ierr = MatShellGetContext(mat,(void **)&ctx); CHKERRQ(ierr);
  if (n > 0 || nonzeroinitialguess) {
#if defined(USE_PETSC_COMPLEX)
    PetscPrintf(comm,"%d KSP Residual norm %14.12e h %g + %g i\n",n,rnorm,
                PetscReal(ctx->currenth),PetscImaginary(ctx->currenth)); 
#else
    PetscPrintf(comm,"%d KSP Residual norm %14.12e h %g \n",n,rnorm,ctx->currenth); 
#endif
  } else {
    PetscPrintf(comm,"%d KSP Residual norm %14.12e\n",n,rnorm); 
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "SNESDefaultMatrixFreeSetParameters"
/*@
   SNESDefaultMatrixFreeSetParameters - Sets the parameters for the approximation of
   matrix-vector products using finite differences.

   Collective on Mat

   Input Parameters:
+  mat - the matrix free matrix created via SNESDefaultMatrixFreeCreate()
.  error_rel - relative error (should be set to the square root of
               the relative error in the function evaluations)
-  umin - minimum allowable u-value

   Notes:
   The default matrix-free matrix-vector product routine computes
.vb
     J(u)*a = [J(u+h*a) - J(u)]/h where
     h = error_rel*u'a/||a||^2                        if  |u'a| > umin*||a||_{1}
       = error_rel*umin*sign(u'a)*||a||_{1}/||a||^2   else
.ve

   Options Database Keys:
+  -snes_mf_err <error_rel> - Sets error_rel
-  -snes_mf_unim <umin> - Sets umin

.keywords: SNES, matrix-free, parameters

.seealso: SNESDefaultMatrixFreeCreate()
@*/
int SNESDefaultMatrixFreeSetParameters(Mat mat,double error,double umin)
{
  MFCtx_Private *ctx;
  int           ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(mat,(void **)&ctx); CHKERRQ(ierr);
  if (ctx) {
    if (error != PETSC_DEFAULT) ctx->error_rel = error;
    if (umin != PETSC_DEFAULT)  ctx->umin = umin;
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "SNESDefaultMatrixFreeAddNullSpace"
/*@
   SNESDefaultMatrixFreeAddNullSpace - Provides a null space that 
   an operator is supposed to have.  Since roundoff will create a 
   small component in the null space, if you know the null space 
   you may have it automatically removed.

   Collective on Mat 

   Input Parameters:
+  J - the matrix-free matrix context
.  has_cnst - PETSC_TRUE or PETSC_FALSE, indicating if null space has constants
.  n - number of vectors (excluding constant vector) in null space
-  vecs - the vectors that span the null space (excluding the constant vector);
          these vectors must be orthonormal

.keywords: SNES, matrix-free, null space
@*/
int SNESDefaultMatrixFreeAddNullSpace(Mat J,int has_cnst,int n,Vec *vecs)
{
  int           ierr;
  MFCtx_Private *ctx;
  MPI_Comm      comm;

  PetscFunctionBegin;
  PetscObjectGetComm((PetscObject)J,&comm);

  ierr = MatShellGetContext(J,(void **)&ctx); CHKERRQ(ierr);
  /* no context indicates that it is not the "matrix free" matrix type */
  if (!ctx) PetscFunctionReturn(0);
  ierr = PCNullSpaceCreate(comm,has_cnst,n,vecs,&ctx->sp); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}




