/* 
   Routines for matrix-free Jacobian computations in the Julianne code.
   This is essentially just the default SNES matrix-free routines, with
   provisions for the pseudo-transient continuation term.
 */

#include "snes.h"
#include "user.h"

/* Application-defined context for matrix-free SNES */
typedef struct {
  SNES        snes;      /* SNES context */
  Vec         w;         /* work vector */
  double      error_rel; /* square root of relative error in computing function */
  double      umin;      /* minimum allowable u value */
  Euler       *user;     /* user-defined application context */
} MFCtxEuler_Private;

#undef __FUNC__
#define __FUNC__ "UserMatrixFreeDestroy"
int UserMatrixFreeDestroy(PetscObject obj)
{
  int                ierr;
  Mat                mat = (Mat) obj;
  MFCtxEuler_Private *ctx;

  ierr = MatShellGetContext(mat,(void **)&ctx);
  ierr = VecDestroy(ctx->w); CHKERRQ(ierr);
  PetscFree(ctx);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "UserSetMatrixFreeParameters"
/*@
   UserSetMatrixFreeParameters - Sets the parameters for the approximation of
   matrix-vector products using finite differences.

$       J(u)*a = [J(u+h*a) - J(u)]/h where
$        h = error_rel*u'a/||a||^2                        if  |u'a| > umin*||a||_{1}
$          = error_rel*umin*sign(u'a)*||a||_{1}/||a||^2   else
$
   Input Parameters:
.  snes - the SNES context
.  error_rel - relative error (should be set to the square root of
     the relative error in the function evaluations)
.  umin - minimum allowable u-value

.keywords: SNES, matrix-free, parameters
@*/
int UserSetMatrixFreeParameters(SNES snes,double error,double umin)
{
  MFCtxEuler_Private *ctx;
  int                ierr;
  Mat                mat;

  ierr = SNESGetJacobian(snes,&mat,PETSC_NULL,PETSC_NULL); CHKERRQ(ierr);
  ierr = MatShellGetContext(mat,(void **)&ctx); CHKERRQ(ierr);
  if (ctx) {
    if (error != PETSC_DEFAULT) ctx->error_rel = error;
    if (umin != PETSC_DEFAULT)  ctx->umin = umin;
  }
  return 0;
}

#undef __FUNC__
#define __FUNC__ "UserMatrixFreeView"
/*
   UserMatrixFreeView - Viewer parameters in user-defined matrix-free context.
 */
int UserMatrixFreeView(Mat J,Viewer viewer)
{
  int                ierr;
  MFCtxEuler_Private *ctx;
  MPI_Comm           comm;
  FILE               *fd;
  ViewerType         vtype;

  PetscObjectGetComm((PetscObject)J,&comm);
  ierr = MatShellGetContext(J,(void **)&ctx); CHKERRQ(ierr);
  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
  if (vtype == ASCII_FILE_VIEWER || vtype == ASCII_FILES_VIEWER) {
     PetscFPrintf(comm,fd,"  user-defined SNES matrix-free approximation:\n");
     PetscFPrintf(comm,fd,"    err=%g (relative error in function evaluation)\n",ctx->error_rel);
     PetscFPrintf(comm,fd,"    umin=%g (minimum iterate parameter)\n",ctx->umin);
  }
  return 0;
}
#undef __FUNC__
#define __FUNC__ "UserMatrixFreeMult"
/*
  UserMatrixFreeMult - User-defined matrix-free form for Jacobian-vector
  product y = F'(u)*a:
        y = ( F(u + ha) - F(u) ) /h, 
  where F = nonlinear function, as set by SNESSetFunction()
        u = current iterate
        h = difference interval

 This is the default SNES routine (SNESMatrixFreeMult_Private) + the
 pseudo-transient continuation term.
*/
int UserMatrixFreeMult(Mat mat,Vec a,Vec y)
{
  MFCtxEuler_Private *ctx;
  SNES          snes;
  double        norm, sum, umin;
  Scalar        h, dot, mone = -1.0, *ya, *aa, *dt, one = 1.0, dti;
  Vec           w,U,F;
  int           ierr, i, j, k, ijkv, ijkx, nc, jkx, dim, ikx;
  int           xsi, xei, ysi, yei, zsi, zei, mx, my, mz;
  int           xm, ym, zm, xs, ys, zs, xe, ye, ze, ijx;
  Euler         *user;

  MatShellGetContext(mat,(void **)&ctx);
  snes = ctx->snes;
  w    = ctx->w;
  umin = ctx->umin;
  user = (Euler *)ctx->user;
  dt   = user->dt;
  nc   = user->nc;
  if (user->bctype != IMPLICIT) 
    SETERRQ(1,1,"Only bctype = IMPLICIT supports matrix-free methods!");

  /* starting and ending grid points (including boundary points) */
  xs = user->xs;  ys = user->ys;  zs = user->zs;
  xe = user->xe;  ye = user->ye;  ze = user->ze;
  xm = user->xm;  ym = user->ym;  zm = user->zm;

  /* starting and ending interior grid points */
  xsi = user->xsi; ysi = user->ysi; zsi = user->zsi;
  xei = user->xei; yei = user->yei; zei = user->zei;

  /* edges of grid */
  mx = user->mx; my = user->my; mz = user->mz;

  /* We're doing a matrix-vector product now; set flag accordingly */
  user->matrix_free_mult = 1;

  /* We log matrix-free matrix-vector products separately, so that we can
     separate the performance monitoring from the cases that use conventional
     storage.  We may eventually modify event logging to associate events
     with particular objects, hence alleviating the more general problem. */
  PLogEventBegin(MAT_MatrixFreeMult,a,y,0,0);

  ierr = SNESGetSolution(snes,&U); CHKERRQ(ierr);
  ierr = SNESGetFunction(snes,&F); CHKERRQ(ierr);
  /* F = user->F_low; */  /* use lower order function */

  /* Determine a "good" step size */
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
  ierr = SNESComputeFunction(snes,w,y); CHKERRQ(ierr);

  /* Compute base mv-product */
  ierr = VecAXPY(&mone,F,y); CHKERRQ(ierr);
  h = 1.0/h;
  ierr = VecScale(&h,y); CHKERRQ(ierr);

  if (user->sctype == DT_MULT) {
    /* Handle interior grid points and boundary points the same way 
                   J_diag -> J_diag + 1 
    */
    ierr = VecPointwiseMult(a,y,y); CHKERRQ(ierr);
  } 
  else if (user->sctype == DT_DIV) {
    
    ierr = VecGetArray(y,&ya); CHKERRQ(ierr);
    ierr = VecGetArray(a,&aa); CHKERRQ(ierr);

    /* Interior grid points ... Add pseudo-transient continuation term:
          y += a/dt, where dt != 0
       Evaluate dt.  Really need to recalculate dt only when iterates change.
       Then compute y += a/dt, where dt != 0.
    */
    for (k=zsi; k<zei; k++) {
      for (j=ysi; j<yei; j++) {
        jkx = (j-ys)*xm + (k-zs)*xm*ym;
        for (i=xsi; i<xei; i++) {
          ijkv = jkx + i-xs;
          ijkx = nc * ijkv;
          dti = one/dt[ijkv];
	  ya[ijkx]   += aa[ijkx]   * dti;
          ya[ijkx+1] += aa[ijkx+1] * dti;
          ya[ijkx+2] += aa[ijkx+2] * dti;
          ya[ijkx+3] += aa[ijkx+3] * dti;
          ya[ijkx+4] += aa[ijkx+4] * dti;
        }
      }
    }

    /* Boundary edges of brick: rows of Jacobian are identity;
       corresponding residual values are 0.  So, we modify the
       product accordingly. 
     */
    for (k=zs; k<ze; k+=zm-1) {
      if (k == 0 || k == mz-1) {
        for (j=ys; j<ye; j+=ym-1) {
          if (j == 0 || j == my-1) {
            jkx = (j-ys)*xm + (k-zs)*xm*ym;
            for (i=xs; i<xe; i++) {
              ijkx = nc * (jkx + i-xs);
              ya[ijkx]   = aa[ijkx];
              ya[ijkx+1] = aa[ijkx+1];
              ya[ijkx+2] = aa[ijkx+2];
              ya[ijkx+3] = aa[ijkx+3];
              ya[ijkx+4] = aa[ijkx+4];
            }
          }
        }
        for (i=xs; i<xe; i+=xm-1) {
          if (i == 0 || i == mx-1) {
            ikx = i-xs + (k-zs)*xm*ym;
            for (j=ys; j<ye; j++) {
              ijkx = nc * (ikx + (j-ys)*xm);
              ya[ijkx]   = aa[ijkx];
              ya[ijkx+1] = aa[ijkx+1];
              ya[ijkx+2] = aa[ijkx+2];
              ya[ijkx+3] = aa[ijkx+3];
              ya[ijkx+4] = aa[ijkx+4];
            }
          }
        }
      }
    }
    for (j=ys; j<ye; j+=ym-1) {
      if (j == 0 || j == my-1) {
        for (i=xs; i<xe; i+=xm-1) {
          if (i == 0 || i == mx-1) {
            ijx = (j-ys)*xm + i-xs;
            for (k=zs; k<ze; k++) {
              ijkx = nc * (ijx + (k-zs)*xm*ym);
              ya[ijkx]   = aa[ijkx];
              ya[ijkx+1] = aa[ijkx+1];
              ya[ijkx+2] = aa[ijkx+2];
              ya[ijkx+3] = aa[ijkx+3];
              ya[ijkx+4] = aa[ijkx+4];
            }
          }
        }
      }
    }
    ierr = VecRestoreArray(y,&ya); CHKERRQ(ierr);
    ierr = VecRestoreArray(a,&aa); CHKERRQ(ierr);
  } 
  else SETERRQ(1,1,"Unsupported sctype!");

  ierr = VecGetLocalSize(y,&dim); CHKERRQ(ierr);
  PLogFlops(2*dim);

  /* We're done with matrix-vector product */
  user->matrix_free_mult = 0;

  PLogEventEnd(MAT_MatrixFreeMult,a,y,0,0);
  return 0;
}

#undef __FUNC__
#define __FUNC__ "UserMatrixFreeMatCreate"
/*
   UserMatrixFreeMatCreate - Creates a matrix-free matrix
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
*/
int UserMatrixFreeMatCreate(SNES snes,Euler *user,Vec x,Mat *J)
{
  MPI_Comm          comm;
  MFCtxEuler_Private *mfctx;
  int                n, nloc, ierr, flg;

  mfctx = (MFCtxEuler_Private *) PetscMalloc(sizeof(MFCtxEuler_Private)); CHKPTRQ(mfctx);
  PLogObjectMemory(snes,sizeof(MFCtxEuler_Private));
  mfctx->snes = snes;
  mfctx->user = user;
  mfctx->error_rel = 1.e-8; /* assumes double precision */
  mfctx->umin      = 1.e-8;
  ierr = OptionsGetDouble(PETSC_NULL,"-snes_mf_err",&mfctx->error_rel,&flg); CHKERRQ(ierr);
  ierr = OptionsGetDouble(PETSC_NULL,"-snes_mf_umin",&mfctx->umin,&flg); CHKERRQ(ierr);
  ierr = VecDuplicate(x,&mfctx->w); CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)x,&comm); CHKERRQ(ierr);
  ierr = VecGetSize(x,&n); CHKERRQ(ierr);
  ierr = VecGetLocalSize(x,&nloc); CHKERRQ(ierr);
  ierr = MatCreateShell(comm,nloc,n,n,n,(void*)mfctx,J); CHKERRQ(ierr);
  ierr = MatShellSetOperation(*J,MATOP_MULT,(void*)UserMatrixFreeMult); CHKERRQ(ierr);
  ierr = MatShellSetOperation(*J,MATOP_DESTROY,(void *)UserMatrixFreeDestroy); CHKERRQ(ierr);
  ierr = MatShellSetOperation(*J,MATOP_VIEW,(void *)UserMatrixFreeView); CHKERRQ(ierr);
  PLogObjectParent(*J,mfctx->w);
  PLogObjectParent(snes,*J);
  return 0;
}
