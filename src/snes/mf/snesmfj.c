
#ifndef lint
static char vcid[] = "$Id: snesmfj.c,v 1.34 1996/09/05 23:56:28 curfman Exp curfman $";
#endif

#include "draw.h"       /*I  "draw.h"   I*/
#include "src/snes/snesimpl.h"   /*I  "snes.h"   I*/
#include "pinclude/pviewer.h"

typedef struct {  /* default context for matrix-free SNES */
  SNES        snes;      /* SNES context */
  Vec         w;         /* work vector */
  PCNullSpace sp;        /* null space context */
  double      error_rel; /* square root of relative error in computing function */
  double      umin;      /* minimum allowable u value */
} MFCtx_Private;

int SNESMatrixFreeDestroy_Private(PetscObject obj)
{
  int           ierr;
  Mat           mat = (Mat) obj;
  MFCtx_Private *ctx;

  ierr = MatShellGetContext(mat,(void **)&ctx);
  ierr = VecDestroy(ctx->w); CHKERRQ(ierr);
  if (ctx->sp) {ierr = PCNullSpaceDestroy(ctx->sp); CHKERRQ(ierr);}
  PetscFree(ctx);
  return 0;
}

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

  PetscObjectGetComm((PetscObject)J,&comm);
  ierr = MatShellGetContext(J,(void **)&ctx); CHKERRQ(ierr);
  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
  if (vtype == ASCII_FILE_VIEWER || vtype == ASCII_FILES_VIEWER) {
     PetscFPrintf(comm,fd,"  SNES matrix-free approximation:\n");
     PetscFPrintf(comm,fd,"    err=%g (relative error in function evaluation)\n",ctx->error_rel);
     PetscFPrintf(comm,fd,"    umin=%g (minimum iterate parameter)\n",ctx->umin);
  }
  return 0;
}

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
  double        norm, sum, umin;
  Scalar        h, dot, mone = -1.0;
  Vec           w,U,F;
  int           ierr, (*eval_fct)(SNES,Vec,Vec);

  MatShellGetContext(mat,(void **)&ctx);
  snes = ctx->snes;
  w    = ctx->w;
  umin = ctx->umin;

  /* We log matrix-free matrix-vector products separately, so that we can
     separate the performance monitoring from the cases that use conventional
     storage.  We may eventually modify event logging to associate events
     with particular objects, hence alleviating the more general problem. */
  PLogEventBegin(MAT_MatrixFreeMult,a,y,0,0);

  ierr = SNESGetSolution(snes,&U); CHKERRQ(ierr);
  if (snes->method_class == SNES_NONLINEAR_EQUATIONS) {
    eval_fct = SNESComputeFunction;
    ierr = SNESGetFunction(snes,&F); CHKERRQ(ierr);
  }
  else if (snes->method_class == SNES_UNCONSTRAINED_MINIMIZATION) {
    eval_fct = SNESComputeGradient;
    ierr = SNESGetGradient(snes,&F); CHKERRQ(ierr);
  }
  else SETERRQ(1,"SNESMatrixFreeMult_Private: Invalid method class");

  /* Determine a "good" step size, h */
  ierr = VecDot(U,a,&dot); CHKERRQ(ierr);
  ierr = VecNorm(a,NORM_1,&sum); CHKERRQ(ierr);
  ierr = VecNorm(a,NORM_2,&norm); CHKERRQ(ierr);

  /* Safeguard for step sizes too small */
  if (sum == 0.0) {dot = 1.0; norm = 1.0;}
#if defined(PETSC_COMPLEX)
  else if (abs(dot) < umin*sum && real(dot) >= 0.0) dot = umin*sum;
  else if (abs(dot) < 0.0 && real(dot) > -umin*sum) dot = -umin*sum;
#else
  else if (dot < umin*sum && dot >= 0.0) dot = umin*sum;
  else if (dot < 0.0 && dot > -umin*sum) dot = -umin*sum;
#endif
  h = ctx->error_rel*dot/(norm*norm);
  
  /* Evaluate function at F(u + ha) */
  ierr = VecWAXPY(&h,a,U,w); CHKERRQ(ierr);
  ierr = eval_fct(snes,w,y); CHKERRQ(ierr);
  ierr = VecAXPY(&mone,F,y); CHKERRQ(ierr);
  h = 1.0/h;
  ierr = VecScale(&h,y); CHKERRQ(ierr);
  if (ctx->sp) {ierr = PCNullSpaceRemove(ctx->sp,y); CHKERRQ(ierr);}

  PLogEventEnd(MAT_MatrixFreeMult,a,y,0,0);
  return 0;
}

/*@C
   SNESDefaultMatrixFreeMatCreate - Creates a matrix-free matrix
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
   matrix operations such as matrix-vector products.

   The user should call MatDestroy() when finished with the matrix-free
   matrix context.

.keywords: SNES, default, matrix-free, create, matrix

.seealso: MatDestroy()
@*/
int SNESDefaultMatrixFreeMatCreate(SNES snes,Vec x, Mat *J)
{
  MPI_Comm      comm;
  MFCtx_Private *mfctx;
  int           n, nloc, ierr, flg;

  mfctx = (MFCtx_Private *) PetscMalloc(sizeof(MFCtx_Private)); CHKPTRQ(mfctx);
  PLogObjectMemory(snes,sizeof(MFCtx_Private));
  mfctx->sp   = 0;
  mfctx->snes = snes;
  mfctx->error_rel = 1.e-8; /* assumes double precision */
  mfctx->umin      = 1.e-8;
  ierr = OptionsGetDouble(snes->prefix,"-snes_mf_err",&mfctx->error_rel,&flg); CHKERRQ(ierr);
  ierr = OptionsGetDouble(snes->prefix,"-snes_mf_umin",&mfctx->umin,&flg); CHKERRQ(ierr);
  ierr = VecDuplicate(x,&mfctx->w); CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)x,&comm); CHKERRQ(ierr);
  ierr = VecGetSize(x,&n); CHKERRQ(ierr);
  ierr = VecGetLocalSize(x,&nloc); CHKERRQ(ierr);
  ierr = MatCreateShell(comm,nloc,n,n,n,(void*)mfctx,J); CHKERRQ(ierr);
  ierr = MatShellSetOperation(*J,MAT_MULT,(void*)SNESMatrixFreeMult_Private); CHKERRQ(ierr);
  ierr = MatShellSetOperation(*J,MAT_DESTROY,(void *)SNESMatrixFreeDestroy_Private); CHKERRQ(ierr);
  ierr = MatShellSetOperation(*J,MAT_VIEW,(void *)SNESMatrixFreeView_Private); CHKERRQ(ierr);
  PLogObjectParent(*J,mfctx->w);
  PLogObjectParent(snes,*J);
  return 0;
}

/*@
   SNESSetMatrixFreeParameters - Sets the parameters for the approximation of
   matrix-vector products using finite differences.

   Input Parameters:
.  snes - the SNES context
.  error_rel - relative error
.  umin - minimum allowable u-value

.keywords: SNES, matrix-free, parameters
@*/
int SNESSetMatrixFreeParameters(SNES snes,double error,double umin)
{
  MFCtx_Private *ctx;
  int           ierr;
  Mat           mat;

  ierr = SNESGetJacobian(snes,&mat,PETSC_NULL,PETSC_NULL); CHKERRQ(ierr);
  ierr = MatShellGetContext(mat,(void **)&ctx); CHKERRQ(ierr);
  if (ctx) {
    if (error != PETSC_DEFAULT) ctx->error_rel = error;
    if (umin != PETSC_DEFAULT)  ctx->umin = umin;
  }
  return 0;
}

/*@
   SNESDefaultMatrixFreeMatAddNullSpace - Provides a null space that 
   an operator is suppose to have.  Since roundoff will create a 
   small component in the null space, if you know the null space 
   you may have it automatically removed.

   Input Parameters:
.  J - the matrix-free matrix context
.  has_cnst - PETSC_TRUE or PETSC_FALSE, indicating if null space has constants
.  n - number of vectors (excluding constant vector) in null space
.  vecs - the vectors that span the null space (excluding the constant vector);
.         these vectors must be orthonormal

.keywords: SNES, matrix-free, null space
@*/
int SNESDefaultMatrixFreeMatAddNullSpace(Mat J,int has_cnst,int n,Vec *vecs)
{
  int           ierr;
  MFCtx_Private *ctx;
  MPI_Comm      comm;

  PetscObjectGetComm((PetscObject)J,&comm);

  ierr = MatShellGetContext(J,(void **)&ctx); CHKERRQ(ierr);
  /* no context indicates that it is not the "matrix free" matrix type */
  if (!ctx) return 0;
  ierr = PCNullSpaceCreate(comm,has_cnst,n,vecs,&ctx->sp); CHKERRQ(ierr);
  return 0;
}




