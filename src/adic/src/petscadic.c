#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: petscadic.c,v 1.6 1998/05/30 05:23:16 balay Exp $";
#endif

#include "petscadic.h"
#include "src/adic/src/adpetsc.h"

extern int ad_PetscADICFunctionCreate(PetscADICFunction);
extern int ad_PetscADICFunctionInitialize(PetscADICFunction);
extern int ad_PetscADICFunctionEvaluateGradient(PetscADICFunction,double *,double *,double *);
extern int ad_PetscADICFunctionApplyGradientInitialize(PetscADICFunction,double *);
extern int ad_PetscADICFunctionApplyGradient(PetscADICFunction,double *,double *);

#undef __FUNC__  
#define __FUNC__ "ad_PetscADICFunctionSetFunction"
/*
   PetscADICFunctionSetFunction - Creates a data structure to manage the evaluate
                             of a PETSc function and its derivative.
*/
int PETSC_DLLEXPORT PetscADICFunctionSetFunction(PetscADICFunction ctx,int (*Function)(Vec,Vec),
                            int (*FunctionInitialize)(void **))
{
  (ctx)->Function           = Function;
  (ctx)->FunctionInitialize = FunctionInitialize;

  PetscFunctionReturn(0);
}

/*
   PetscADICFunctionCreate - Creates a data structure to manage the evaluate
                             of a PETSc function and its derivative.
*/
int PETSC_DLLEXPORT PetscADICFunctionCreate(Vec in,Vec out,int (*ad_Function)(Vec,Vec),
                            int (*ad_FunctionInitialize)(void **),PetscADICFunction*ctx)
{
  int ierr;
 
  *ctx = PetscNew(struct _p_PetscADICFunction); CHKPTRQ(*ctx);
  (*ctx)->Function              = 0;
  (*ctx)->FunctionInitialize    = 0;
  (*ctx)->ad_Function           = ad_Function;
  (*ctx)->ad_FunctionInitialize = ad_FunctionInitialize;

  PetscObjectGetComm((PetscObject)in,&(*ctx)->comm);
  /*
      Create active vectors to hold the input and output
  */
  ierr = VecGetSize(in,&(*ctx)->m); CHKERRQ(ierr);
  ierr = VecGetSize(out,&(*ctx)->n); CHKERRQ(ierr);
  ierr = ad_PetscADICFunctionCreate(*ctx); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscADICFunctionEvaluateGradient"
/*
    PetscADICFunctionEvaluateGradient - Evaluates a given PETSc function and its derivative
*/
int PETSC_DLLEXPORT PetscADICFunctionEvaluateGradient(PetscADICFunction ctx,Vec in,Vec out,Mat grad)
{
  int    ierr,flg;
  Scalar *inx,*outx,*gradarray;

  ierr = OptionsHasName(0,"-adic_fd",&flg); CHKERRQ(ierr);
  if (flg) {
    ierr = PetscADICFunctionEvaluateGradientFD(ctx,in,out,grad);CHKERRQ(ierr);
  } else {
    ierr = VecGetArray(in,&inx); CHKERRQ(ierr);
    ierr = VecGetArray(out,&outx); CHKERRQ(ierr);
    ierr = MatGetArray(grad,&gradarray); CHKERRQ(ierr);

    ierr = ad_PetscADICFunctionEvaluateGradient(ctx,inx,outx,gradarray); CHKERRQ(ierr);

    ierr = VecRestoreArray(in,&inx); CHKERRQ(ierr);
    ierr = VecRestoreArray(out,&outx); CHKERRQ(ierr);
    ierr = MatRestoreArray(grad,&gradarray); CHKERRQ(ierr);
    ierr = MatAssemblyBegin(grad,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(grad,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

int DefaultComputeJacobian(int (*Function)(Vec,Vec),Vec x1,Mat J)
{
  Vec      j1,j2,x2;
  int      i,ierr,N,start,end,j;
  Scalar   dx, mone = -1.0,*y,scale,*xx,wscale;
  double   amax, epsilon = 1.e-8; /* assumes double precision */
  double   dx_min = 1.e-16, dx_par = 1.e-1;
  MPI_Comm comm;

  PetscObjectGetComm((PetscObject)x1,&comm);
  MatZeroEntries(J);

  ierr = VecDuplicate(x1,&j1); CHKERRQ(ierr);
  ierr = VecDuplicate(x1,&j2); CHKERRQ(ierr);
  ierr = VecDuplicate(x1,&x2); CHKERRQ(ierr);

  ierr = VecGetSize(x1,&N); CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(x1,&start,&end); CHKERRQ(ierr);
  VecGetArray(x1,&xx);
  ierr = (*Function)(x1,j1); CHKERRQ(ierr);

  /* Compute Jacobian approximation, 1 column at a time. 
      x1 = current iterate, j1 = F(x1)
      x2 = perturbed iterate, j2 = F(x2)
   */
  for ( i=0; i<N; i++ ) {
    ierr = VecCopy(x1,x2); CHKERRQ(ierr);
    if ( i>= start && i<end) {
      dx = xx[i-start];
#if !defined(USE_PETSC_COMPLEX)
      if (dx < dx_min && dx >= 0.0) dx = dx_par;
      else if (dx < 0.0 && dx > -dx_min) dx = -dx_par;
#else
      if (PetscAbsScalar(dx) < dx_min && PetscReal(dx) >= 0.0) dx = dx_par;
      else if (PetscReal(dx) < 0.0 && PetscAbsScalar(dx) < dx_min) dx = -dx_par;
#endif
      dx *= epsilon;
      wscale = 1.0/dx;
      VecSetValues(x2,1,&i,&dx,ADD_VALUES); 
    } 
    else {
      wscale = 0.0;
    }
    ierr = (*Function)(x2,j2); CHKERRQ(ierr);
    ierr = VecAXPY(&mone,j1,j2); CHKERRQ(ierr);
    /* Communicate scale to all processors */
#if !defined(USE_PETSC_COMPLEX)
    ierr = MPI_Allreduce(&wscale,&scale,1,MPI_DOUBLE,MPI_SUM,comm);CHKERRQ(ierr);
#else
    ierr = MPI_Allreduce(&wscale,&scale,2,MPI_DOUBLE,MPI_SUM,comm);CHKERRQ(ierr);
#endif
    VecScale(&scale,j2);
    VecGetArray(j2,&y);
    VecNorm(j2,NORM_INFINITY,&amax); amax *= 1.e-14;
    for ( j=start; j<end; j++ ) {
      if (PetscAbsScalar(y[j-start]) > amax) {
        ierr = MatSetValues(J,1,&j,1,&i,y+j-start,INSERT_VALUES); CHKERRQ(ierr);
      }
    }
    VecRestoreArray(j2,&y);
  }
  ierr = VecDestroy(j1); CHKERRQ(ierr);
  ierr = VecDestroy(j2); CHKERRQ(ierr);
  ierr = VecDestroy(x2); CHKERRQ(ierr);

  ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscADICFunctionEvaluateGradientFD"
/*
    PetscADICFunctionEvaluateGradientFD - Evaluates a given PETSc function's derivative
*/
int PETSC_DLLEXPORT PetscADICFunctionEvaluateGradientFD(PetscADICFunction ctx,Vec in,Vec out,Mat grad)
{
  int    ierr;

  if (!ctx->Function) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,1,"Must provide function via PETScADICFunctionSetFunction()\n\
            before using finite differences\n");
  }
  ierr = DefaultComputeJacobian(ctx->Function,in,grad); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------------------*/

#undef __FUNC__  
#define __FUNC__ "PetscADICFunctionApplyGradient"
/*
    PetscADICFunctionApplyGradient - Applys a given PETSc function and its derivative
*/
int PetscADICFunctionApplyGradient(Mat mat,Vec in,Vec grad)
{
  int               ierr;
  Scalar            *inx,*gradarray;
  PetscADICFunction ctx;

  ierr = MatShellGetContext(mat,(void **)&ctx); CHKERRQ(ierr);

  ierr = VecGetArray(in,&inx); CHKERRQ(ierr);
  ierr = VecGetArray(grad,&gradarray); CHKERRQ(ierr);

  ierr = ad_PetscADICFunctionApplyGradient(ctx,inx,gradarray); CHKERRQ(ierr);

  ierr = VecRestoreArray(in,&inx); CHKERRQ(ierr);
  ierr = VecRestoreArray(grad,&gradarray); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscADICFunctionApplyGradientReset"
/*
    PetscADICFunctionApplyReset - Applys a given PETSc function and its derivative
*/
int PETSC_DLLEXPORT PetscADICFunctionApplyGradientReset(Mat mat,Vec in)
{
  int               ierr;
  Scalar            *inx;
  PetscADICFunction ctx;

  ierr = MatShellGetContext(mat,(void **)&ctx); CHKERRQ(ierr);

  ierr = VecGetArray(in,&inx); CHKERRQ(ierr);

  ierr = ad_PetscADICFunctionApplyGradientInitialize(ctx,inx); CHKERRQ(ierr);

  ierr = VecRestoreArray(in,&inx); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscADICFunctionApplyGradientInitialize"
/*
    PetscADICFunctionApplyInitialize - Applys a given PETSc function and its derivative
*/
int PETSC_DLLEXPORT PetscADICFunctionApplyGradientInitialize(PetscADICFunction ctx,Vec in,Mat *mat)
{
  int    n,nloc,ierr;
  Scalar *inx;

  ierr = VecGetSize(in,&n); CHKERRQ(ierr);
  ierr = VecGetLocalSize(in,&nloc); CHKERRQ(ierr);
  ierr = MatCreateShell(ctx->comm,nloc,n,n,n,ctx,mat); CHKERRQ(ierr);
  ierr = MatShellSetOperation(*mat,MATOP_MULT,(void*)PetscADICFunctionApplyGradient);CHKERRQ(ierr);

  ierr = VecGetArray(in,&inx); CHKERRQ(ierr);

  ierr = ad_PetscADICFunctionApplyGradientInitialize(ctx,inx); CHKERRQ(ierr);

  ierr = VecRestoreArray(in,&inx); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------------*/

#undef __FUNC__  
#define __FUNC__ "PetscADICFunctionInitialize"
/*
    PetscADICFunctionInitialize - Initializes the data structure for PetscADICFunction.
*/
int PETSC_DLLEXPORT PetscADICFunctionIntialize(PetscADICFunction ctx)
{
  int    ierr;

  ierr = ad_PetscADICFunctionInitialize(ctx); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

