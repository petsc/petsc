
#ifndef lint
static char vcid[] = "$Id: snesmfj.c,v 1.3 1995/05/05 11:44:39 bsmith Exp bsmith $";
#endif

#include "draw.h"
#include "snes.h"
#include "options.h"

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

  SNESGetSolution(snes,&U); SNESGetFunction(snes,&F);
  /* determine a "good" step size */
  VecDot(U,dx,&dot); VecASum(dx,&sum); VecNorm(dx,&norm);
  if (dot < 1.e-16*sum && dot >= 0.0) dot = 1.e-16*sum;
  else if (dot < 0.0 && dot > 1.e-16*sum) dot = -1.e-16*sum;
  h = epsilon*dot/(norm*norm);
  
  /* evaluate function at F(x + dx) */
  VecWAXPY(&h,dx,U,w); 
  ierr = SNESComputeFunction(snes,w,y);
  VecAXPY(&mone,F,y);
  h = -1.0/h;
  VecScale(&h,y);
  return 0;
}
/*@
   SNESDefaultMatrixFreeComputeJacobian - Computes Jacobian using finite 
       differences, matrix free style.

 Input Parameters:
.  x - compute Jacobian at this point
.  ctx - applications Function context

  Output Parameters:
.  J - Jacobian
.  B - preconditioner, same as Jacobian

.keywords: finite differences, Jacobian

.seealso: SNESSetJacobian, SNESTestJacobian
@*/
int SNESDefaultMatrixFreeComputeJacobian(SNES snes, Vec x1,Mat *J,Mat *B,
                                         int *flag,void *ctx)
{
  int         n,ierr;
  MatType     type;

  VecGetSize(x1,&n);
  if (*J) MatGetType(*J,&type);
  if (!*J || type != MATSHELL) {
    MPI_Comm    comm;
    MFCtx_Private *mfctx;
    /* first time in, therefore build datastructures */
    mfctx = NEW(MFCtx_Private); CHKPTR(mfctx);
    mfctx->snes = snes;
    ierr = VecDuplicate(x1,&mfctx->w); CHKERR(ierr);
    PetscObjectGetComm((PetscObject)x1,&comm);
    ierr = MatShellCreate(comm,n,n,(void*)mfctx,J); CHKERR(ierr);
    MatShellSetMult(*J,SNESMatrixFreeMult_Private);
    *B = *J;
  }
  return 0;
}

