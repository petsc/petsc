
#ifndef lint
static char vcid[] = "$Id: snesmfj.c,v 1.8 1995/05/16 00:37:02 curfman Exp bsmith $";
#endif

#include "draw.h"
#include "snes.h"

typedef struct {
  SNES snes;
  Vec  w;
} MFCtx_Private;

/*
    SNESMatrixFreeMult_Private - Default matrix free form of A*u.

*/
int SNESMatrixFreeMult_Private(void *ptr,Vec dx,Vec y)
{
  MFCtx_Private *ctx = (MFCtx_Private* ) ptr;
  SNES          snes = ctx->snes;
  double        norm,epsilon = 1.e-8; /* assumes double precision */
  Scalar        h,dot,sum;
  Scalar        mone = -1.0;
  Vec           w = ctx->w,U,F;
  int           ierr;

  ierr = SNESGetSolution(snes,&U); CHKERRQ(ierr);
  ierr = SNESGetFunction(snes,&F); CHKERRQ(ierr);
  /* determine a "good" step size */
  VecDot(U,dx,&dot); VecASum(dx,&sum); VecNorm(dx,&norm);
  if (sum == 0.0) {dot = 1.0; norm = 1.0;}
  else if (dot < 1.e-16*sum && dot >= 0.0) dot = 1.e-16*sum;
  else if (dot < 0.0 && dot > 1.e-16*sum) dot = -1.e-16*sum;
  h = epsilon*dot/(norm*norm);
  
  /* evaluate function at F(x + dx) */
  VecWAXPY(&h,dx,U,w); 
  ierr = SNESComputeFunction(snes,w,y); CHKERRQ(ierr);
  VecAXPY(&mone,F,y);
  h = -1.0/h;
  VecScale(&h,y);
  return 0;
}
/*@
   SNESDefaultMatrixFreeComputeJacobian - Computes Jacobian using finite 
   differences, matrix-free style.

   Input Parameters:
.  x - compute Jacobian at this point
.  ctx - application's function context, as set with SNESSetFunction()

   Output Parameters:
.  J - Jacobian
.  B - preconditioner, same as Jacobian
.  flag - matrix flag

   Options Database Key:
$  -snes_mf

.keywords: SNES, finite differences, Jacobian

.seealso: SNESSetJacobian(), SNESTestJacobian()
@*/
int SNESDefaultMatrixFreeComputeJacobian(SNES snes, Vec x1,Mat *J,Mat *B,
                                         MatStructure *flag,void *ctx)
{
  int         n,ierr;
  MatType     type;

  VecGetSize(x1,&n);
  if (*J) MatGetType(*J,&type);
  if (!*J || type != MATSHELL) {
    MPI_Comm    comm;
    MFCtx_Private *mfctx;
    /* first time in, therefore build datastructures */
    mfctx = (MFCtx_Private *) PETSCMALLOC(sizeof(MFCtx_Private)); CHKPTRQ(mfctx);
    mfctx->snes = snes;
    ierr = VecDuplicate(x1,&mfctx->w); CHKERRQ(ierr);
    PetscObjectGetComm((PetscObject)x1,&comm);
    ierr = MatShellCreate(comm,n,n,(void*)mfctx,J); CHKERRQ(ierr);
    MatShellSetMult(*J,SNESMatrixFreeMult_Private);
    *B = *J;
  }
  return 0;
}

