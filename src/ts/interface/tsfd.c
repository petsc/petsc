
#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: tsfd.c,v 1.10 1999/02/01 03:43:18 curfman Exp bsmith $";
#endif

#include "src/mat/matimpl.h"      /*I  "mat.h"  I*/
#include "src/ts/tsimpl.h"        /*I  "ts.h"  I*/

#undef __FUNC__  
#define __FUNC__ "TSDefaultComputeJacobianColor"
/*@C
    TSDefaultComputeJacobianColor - Computes the Jacobian using
    finite differences and coloring to exploit matrix sparsity.  
  
    Collective on TS, Vec and Mat

    Input Parameters:
+   ts - nonlinear solver object
.   t - current time
.   x1 - location at which to evaluate Jacobian
-   ctx - coloring context, where ctx must have type MatFDColoring, 
          as created via MatFDColoringCreate()

    Output Parameters:
+   J - Jacobian matrix (not altered in this routine)
.   B - newly computed Jacobian matrix to use with preconditioner (generally the same as J)
-   flag - flag indicating whether the matrix sparsity structure has changed

   Options Database Keys:
$  -mat_fd_coloring_freq <freq> 

   Level: intermediate

.keywords: TS, finite differences, Jacobian, coloring, sparse

.seealso: TSSetJacobian(), , MatFDColoringCreate(), MatFDColoringSetFunction()
@*/
int TSDefaultComputeJacobianColor(TS ts,double t,Vec x1,Mat *J,Mat *B,MatStructure *flag,void *ctx)
{
  MatFDColoring color = (MatFDColoring) ctx;
  SNES          snes;
  int           ierr,freq,it;

  PetscFunctionBegin;
  /*
       If we are not using SNES we have no way to know the current iteration.
  */
  ierr = TSGetSNES(ts,&snes); CHKERRQ(ierr);
  if (snes) {
    ierr = MatFDColoringGetFrequency(color,&freq);CHKERRQ(ierr);
    ierr = SNESGetIterationNumber(snes,&it); CHKERRQ(ierr);

    if ((freq > 1) && ((it % freq) != 1)) {
      PLogInfo(color,"TSDefaultComputeJacobianColor:Skipping Jacobian, it %d, freq %d\n",it,freq);
      *flag = SAME_PRECONDITIONER;
      PetscFunctionReturn(0);
    } else {
      PLogInfo(color,"TSDefaultComputeJacobianColor:Computing Jacobian, it %d, freq %d\n",it,freq);
      *flag = SAME_NONZERO_PATTERN;
    }
  }
  ierr = MatFDColoringApplyTS(*B,color,t,x1,flag,ts); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "TSSDefaultComputeJacobian"
/*
   TSDefaultComputeJacobian - Computes the Jacobian using finite differences.

   Input Parameters:
+  ts - TS context
.  xx1 - compute Jacobian at this point
-  ctx - application's function context, as set with SNESSetFunction()

   Output Parameters:
+  J - Jacobian
.  B - preconditioner, same as Jacobian
-  flag - matrix flag

   Notes:
   This routine is slow and expensive, and is not optimized.

   Sparse approximations using colorings are also available and
   would be a much better alternative!

   Level: intermediate

.seealso: TSDefaultComputeJacobianColor()
*/
int TSDefaultComputeJacobian(TS ts,double t,Vec xx1,Mat *J,Mat *B,MatStructure *flag,void *ctx)
{
  Vec      jj1,jj2,xx2;
  int      i,ierr,N,start,end,j;
  Scalar   dx, mone = -1.0,*y,scale,*xx,wscale;
  double   amax, epsilon = 1.e-8; /* assumes double precision */
  double   dx_min = 1.e-16, dx_par = 1.e-1;
  MPI_Comm comm;

  PetscFunctionBegin;
  ierr = VecDuplicate(xx1,&jj1); CHKERRQ(ierr);
  ierr = VecDuplicate(xx1,&jj2); CHKERRQ(ierr);
  ierr = VecDuplicate(xx1,&xx2); CHKERRQ(ierr);

  ierr = PetscObjectGetComm((PetscObject)xx1,&comm);CHKERRQ(ierr);
  ierr = MatZeroEntries(*J);CHKERRQ(ierr);

  ierr = VecGetSize(xx1,&N); CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(xx1,&start,&end); CHKERRQ(ierr);
  ierr = TSComputeRHSFunction(ts,ts->ptime,xx1,jj1); CHKERRQ(ierr);

  /* Compute Jacobian approximation, 1 column at a time.
      xx1 = current iterate, jj1 = F(xx1)
      xx2 = perturbed iterate, jj2 = F(xx2)
   */
  for ( i=0; i<N; i++ ) {
    ierr = VecCopy(xx1,xx2); CHKERRQ(ierr);
    if ( i>= start && i<end) {
      ierr =  VecGetArray(xx1,&xx);CHKERRQ(ierr);
      dx   = xx[i-start];
      ierr =  VecRestoreArray(xx1,&xx);CHKERRQ(ierr);
#if !defined(USE_PETSC_COMPLEX)
      if (dx < dx_min && dx >= 0.0) dx = dx_par;
      else if (dx < 0.0 && dx > -dx_min) dx = -dx_par;
#else
      if (PetscAbsScalar(dx) < dx_min && PetscReal(dx) >= 0.0) dx = dx_par;
      else if (PetscReal(dx) < 0.0 && PetscAbsScalar(dx) < dx_min) dx = -dx_par;
#endif
      dx *= epsilon;
      wscale = 1.0/dx;
      ierr =  VecSetValues(xx2,1,&i,&dx,ADD_VALUES);CHKERRQ(ierr);
    } else {
      wscale = 0.0;
    }
    ierr = TSComputeRHSFunction(ts,t,xx2,jj2); CHKERRQ(ierr);
    ierr = VecAXPY(&mone,jj1,jj2); CHKERRQ(ierr);
    /* Communicate scale to all processors */
#if !defined(USE_PETSC_COMPLEX)
    ierr = MPI_Allreduce(&wscale,&scale,1,MPI_DOUBLE,MPI_SUM,comm);CHKERRQ(ierr);
#else
#endif
    ierr = VecScale(&scale,jj2);CHKERRQ(ierr);
    ierr = VecNorm(jj2,NORM_INFINITY,&amax);CHKERRQ(ierr); amax *= 1.e-14;
    ierr = VecGetArray(jj2,&y);CHKERRQ(ierr);
    for ( j=start; j<end; j++ ) {
      if (PetscAbsScalar(y[j-start]) > amax) {
        ierr = MatSetValues(*J,1,&j,1,&i,y+j-start,INSERT_VALUES); CHKERRQ(ierr);
      }
    }
    ierr = VecRestoreArray(jj2,&y);
  }
  ierr = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  *flag =  DIFFERENT_NONZERO_PATTERN;

  ierr = VecDestroy(jj1); CHKERRQ(ierr);
  ierr = VecDestroy(jj2); CHKERRQ(ierr);
  ierr = VecDestroy(xx2); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


