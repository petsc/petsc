
#ifndef lint
static char vcid[] = "$Id: snesmfj.c,v 1.23 1995/12/21 18:33:54 bsmith Exp bsmith $";
#endif

#include "draw.h"   /*I  "draw.h"   I*/
#include "snes.h"   /*I  "snes.h"   I*/

typedef struct {  /* default context for matrix-free SNES */
  SNES        snes;
  Vec         w;
  PCNullSpace sp;
} MFCtx_Private;

int SNESMatrixFreeDestroy_Private(void *ptr)
{
  int           ierr;
  MFCtx_Private *ctx = (MFCtx_Private* ) ptr;
  ierr = VecDestroy(ctx->w); CHKERRQ(ierr);
  if (ctx->sp) {ierr = PCNullSpaceDestroy(ctx->sp); CHKERRQ(ierr);}
  PetscFree(ptr);
  return 0;
}

/*
  SNESMatrixFreeMult_Private - Default matrix free form of A*u.
*/
int SNESMatrixFreeMult_Private(void *ptr,Vec dx,Vec y)
{
  MFCtx_Private *ctx = (MFCtx_Private* ) ptr;
  SNES          snes = ctx->snes;
  double        norm,sum,epsilon = 1.e-8; /* assumes double precision */
  Scalar        h,dot,mone = -1.0;
  Vec           w = ctx->w,U,F;
  int           ierr;

  ierr = SNESGetSolution(snes,&U); CHKERRQ(ierr);
  ierr = SNESGetFunction(snes,&F); CHKERRQ(ierr);

  /* Determine a "good" step size */
  ierr = VecDot(U,dx,&dot); CHKERRQ(ierr);
  ierr = VecNorm(dx,NORM_1,&sum); CHKERRQ(ierr);
  ierr = VecNorm(dx,NORM_2,&norm); CHKERRQ(ierr);
  if (sum == 0.0) {dot = 1.0; norm = 1.0;}
#if defined(PETSC_COMPLEX)
  else if (abs(dot) < 1.e-16*sum && real(dot) >= 0.0) dot = 1.e-16*sum;
  else if (abs(dot) < 0.0 && real(dot) > 1.e-16*sum) dot = -1.e-16*sum;
#else
  else if (dot < 1.e-16*sum && dot >= 0.0) dot = 1.e-16*sum;
  else if (dot < 0.0 && dot > 1.e-16*sum) dot = -1.e-16*sum;
#endif
  h = epsilon*dot/(norm*norm);
  
  /* Evaluate function at F(x + dx) */
  ierr = VecWAXPY(&h,dx,U,w); CHKERRQ(ierr);
  ierr = SNESComputeFunction(snes,w,y); CHKERRQ(ierr);
  ierr = VecAXPY(&mone,F,y); CHKERRQ(ierr);
  h = 1.0/h;
  ierr = VecScale(&h,y); CHKERRQ(ierr);
  if (ctx->sp) { ierr = PCNullSpaceRemove(ctx->sp,y); CHKERRQ(ierr);}
  return 0;
}
/*@C
   SNESDefaultMatrixFreeMatCreate - Creates a matrix-free matrix
   context for use with a SNES solver.  This matrix can be used as
   the Jacobian argument for the routine SNESSetJacobian().

   Input Parameters:
.  x - vector where SNES solution is to be stored.

   Output Parameters:
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
  int           n,ierr;

  mfctx = (MFCtx_Private *) PetscMalloc(sizeof(MFCtx_Private)); CHKPTRQ(mfctx);
  PLogObjectMemory(snes,sizeof(MFCtx_Private));
  mfctx->sp   = 0;
  mfctx->snes = snes;
  ierr = VecDuplicate(x,&mfctx->w); CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)x,&comm); CHKERRQ(ierr);
  ierr = VecGetSize(x,&n); CHKERRQ(ierr);
  ierr = MatShellCreate(comm,n,n,(void*)mfctx,J); CHKERRQ(ierr);
  ierr = MatShellSetMult(*J,SNESMatrixFreeMult_Private); CHKERRQ(ierr);
  ierr = MatShellSetDestroy(*J,SNESMatrixFreeDestroy_Private); CHKERRQ(ierr);
  PLogObjectParent(*J,mfctx->w);
  PLogObjectParent(snes,*J);
  return 0;
}

/*@
    SNESDefaultMatrixFreeMatAddNullSpace - Provide a null space that 
        an operator is suppose to have. Since round off will create a 
        small component in the null space, if you know the null space 
        you may have it automatically removed.

  Input Parameters:
.  J - the matrix free matrix
.  has_cnst - PETSC_TRUE or PETSC_FALSE indicating if null space has constants
.  n - number of vectors (excluding constant vector) in null space
.  vecs - the vectors that span the null space (excluding the constant vector)
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




