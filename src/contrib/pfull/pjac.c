


#ifndef lint
static char vcid[] = "$Id: formjac.c,v 1.1 1996/08/20 21:28:06 curfman Exp bsmith $";
#endif

#include "puser.h"

/*
       This is the routine that PETSc calls that up-dates the Jacobians for all the 
    grid levels. In this code it calls the routine ApplicationJacobianOnAGrid() once
    for each level. 
*/


static int ApplicationJacobianOnAGrid(Vec,Mat,AppCtx *,GridCtx *);

int ApplicationJacobian(SNES snes,Vec x1,Mat *J,Mat *B,MatStructure *flag,void *ctx)
{
  AppCtx   *user = (AppCtx *) ctx;
  GridCtx  *grid;
  Mat      jacobian;
  int      i;

  for ( i=0; i<user->Nlevels; i++ ) {
    grid = &user->grids[i]; 
    ApplicationJacobianOnAGrid(x1,jacobian,user,grid); CHKERRQ(ierr);

  }

  /*
     We do not set J in this application because either J == B or J is matrix free
  */
  *B    = jacobian;
  *flag = DIFFERENT_NONZERO_PATTERN; 
  return 0;
}

int ApplicationJacobianOnAGrid(Vec x1,Mat jacobian,AppCtx *ctx,GridCtx *grid)
{
  return 0;
}

/*
   ComputeJacobian - Computes an approximation of the Jacobian matrix using
                     finite differences. 

   Input Parameters:
.  x1 - compute Jacobian at this point
.  ctx - application's function context, as set with SNESSetFunction()

   Output Parameters:
.  J - Jacobian
.  B - preconditioner
.  flag - matrix flag

   Notes:
   We always assemble the preconditioner matrix in this routine.
   Hence, we satisfy both possible cases within the this application code:
     - conventional mode: Jacobian matrix = preconditioner matrix (J=B)
     - matrix-free mode:  do not explicitly form Jacobian matrix, only
       preconditioning matrix, B.

   For a simple, column-by-column finite difference approximation of the
   Jacobian, see the routine SNESDefaultComputeJacobian(), which can be
   activated instead of this sparse variant with the option -snes_fd.
*/
int Jacobian_PotentialFlow(SNES snes,Vec x1,Mat *J,Mat *B,MatStructure *flag,void *ctx)
{
  Vec      localX, localF, localXbak, localFbak, jj1, jj2, x2;
  Scalar   dx, mone = -1.0,*y,scale=0.0,*xx, wscale=0.0;
  double   epsilon = 1.e-8, amax; 
  double   *xxx, *x;
  AppCtx   *user = (AppCtx *) ctx;
  GridCtx  *grid = &user->grids[user->Nlevels - 1];
  MPI_Comm comm;
  int      nloc, *ltog, grow, current, gcol, iter;
  int      mx, my, xs, xe, ys, ye, Xs, Xm, Ys;
  int      i, ierr, N, start, end, j, p, q, row;

  /* Compute Jacobian approximation: x1 = current iterate, j1 = F(x1)
                                     x2 = perturbed iterate, j2 = F(x2)
   */
  localX    = grid->localX;
  localF    = grid->localF;
  localXbak = grid->localXbak;
  localFbak = grid->localFbak;

  /* We cannot change jj1, it holds the current best function */
  ierr = SNESGetFunction(snes,&jj1); CHKERRQ(ierr);

  jj2  = grid->jj2;
  x2   = grid->x2;
  Xs   = grid->Xs;
  Xm   = grid->Xm;
  Ys   = grid->Ys;
  xs   = grid->xs;
  xe   = grid->xe; 
  ys   = grid->ys; 
  ye   = grid->ye;
  mx   = grid->mx; 
  my   = grid->my;

  ierr = SNESGetIterationNumber(snes,&iter); CHKERRQ(ierr);
  if ((iter != 1) && (iter % user->jfreq)) { /* reuse matrix from last iteration */
    *flag = SAME_PRECONDITIONER;
    PLogInfo(snes,"ComputeJacobian: iter=%d: Using old matrix\n",iter);
    return 0;
  }
  PLogInfo(snes,"ComputeJacobian: iter=%d: Forming new matrix\n",iter);
  PetscObjectGetComm((PetscObject)x1,&comm);

  ierr = MatZeroEntries(*B); CHKERRQ(ierr);

  ierr = VecGetSize(x1,&N); CHKERRQ(ierr);
  ierr = VecGetArray(x1,&xx); CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(x1,&start,&end); CHKERRQ(ierr);

  /* Form ghosted local vectors for x1 and F(x1); then copy */
  ierr = DAGlobalToLocalBegin(grid->da,jj1,INSERT_VALUES,localF); CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(grid->da,jj1,INSERT_VALUES,localF); CHKERRQ(ierr);
  ierr = DAGlobalToLocalBegin(grid->da,x1,INSERT_VALUES,localX); CHKERRQ(ierr);
  ierr = VecCopy(localF,localFbak); CHKERRQ(ierr); CHKERRQ(ierr);
  ierr = DAGetGlobalIndices(grid->da,&nloc,&ltog); CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(grid->da,x1,INSERT_VALUES,localX); CHKERRQ(ierr);
  ierr = VecCopy(localX,localXbak); CHKERRQ(ierr); CHKERRQ(ierr);

  /* Loop over columns, doing the usual approx if this column corresponds to
     a grid point on a processor's edge */
  for ( i=0; i<N; i++ ) {
    ierr = VecCopy(x1,x2); CHKERRQ(ierr);
    ierr = VecGetArray(x2,&xxx); CHKERRQ(ierr);
    wscale = 0.0;

    current = 0;
    while (current < nloc && ltog[current] != i) current++;
    q = current / Xm + Ys;   /* local grid point number in y-direction */
    p = current % Xm + Xs;   /* local grid point number in x-direction */

    /* If this is a processor edge point, do the usual approx */
    if (p==xs || p==xe-1 || q==ys || q==ye-1) {
       if (i>= start && i<end) {
          dx = xxx[i-start];
          if (dx < 1.e-16 && dx >= 0.0) dx = 1.e-1;
          else 
            if (dx < 0.0 && dx > -1.e-16) dx = -1.e-1;
          dx *= epsilon;
          ierr = VecSetValues(x2,1,&i,&dx,ADD_VALUES); CHKERRQ(ierr);
          wscale = 1.0/dx; 
       } else wscale = 0.0;
    } 
 
    /* Communicate scale to all processors */
#if !defined(PETSC_COMPLEX)
    MPI_Allreduce(&wscale,&scale,1,MPI_DOUBLE,MPI_SUM,comm);
#else
    MPI_Allreduce(&wscale,&scale,2,MPI_DOUBLE,MPI_SUM,comm);
#endif

    /* Note: scale !=0 only if above: (p==xs || p==xe-1 || q==ys || q==ye-1) */
    if (scale != 0.0) {
      ierr = VecCopy(localFbak,localF); CHKERRQ(ierr);
      ierr = VecCopy(jj1,jj2); CHKERRQ(ierr);
      ierr = MySparseFunction(x2,i,user,jj2); CHKERRQ(ierr);
      ierr = VecAXPY(&mone,jj1,jj2); CHKERRQ(ierr);
      ierr = VecScale(&scale,jj2); CHKERRQ(ierr);
      ierr = VecGetArray(jj2,&y); CHKERRQ(ierr);
      ierr = VecNorm(jj2,NORM_INFINITY,&amax);  CHKERRQ(ierr);
      amax *= 1.e-14;          /* 1.e-14 */

      for ( j=start; j<end; j++ ) {
        if (PetscAbsScalar(y[j-start]) > amax) {
          ierr = MatSetValues(*B,1,&j,1,&i,y+j-start,INSERT_VALUES); CHKERRQ(ierr);
        }
      }
      ierr = VecRestoreArray(jj2,&y); CHKERRQ(ierr);
    }
  }

  /* Compute Jacobian approximation for each processor's interior grid points.
     We handle this part of the computation separately from the above since
     the computations are local and no scatters are needed for the perturbed
     iterates. */
  ierr = VecCopy(localXbak,localX); CHKERRQ(ierr);
  for (q=ys+1; q<ye-1; q++) {
    for (p=xs+1; p<xe-1; p++) {
      ierr = VecCopy(localXbak, localX); CHKERRQ(ierr);
      ierr = VecGetArray(localX, &x); CHKERRQ(ierr);
      row = p - Xs + (q - Ys)*Xm;
      grow = ltog[row];
      dx = x[row];
      if (dx < 1.e-16 && dx >= 0.0) dx = 1.e-1;
      else if (dx < 0.0 && dx > -1.e-16) dx = -1.e-1;
      dx *= epsilon;
      scale = 1.0/dx; 
      x[row] += dx;
      ierr = VecRestoreArray(localX,&x); CHKERRQ(ierr);
      ierr = VecCopy(localFbak,localF); CHKERRQ(ierr);
      ierr = VecCopy(jj1,jj2); CHKERRQ(ierr);
      ierr = InnerSparseFunction(p,q,localX,grow,user,jj2); CHKERRQ(ierr);
      ierr = VecAXPY(&mone,jj1,jj2); CHKERRQ(ierr);
      ierr = VecScale(&scale,jj2); CHKERRQ(ierr);
      ierr = VecGetArray(jj2,&y); CHKERRQ(ierr);
      ierr = VecNorm(jj2,NORM_INFINITY,&amax); CHKERRQ(ierr);
      amax *= 1.e-14; 

      gcol = grow; 
      for ( j=start; j<end; j++ ) {
        if (PetscAbsScalar(y[j-start]) > amax) {
          ierr = MatSetValues(*B,1,&j,1,&gcol,y+j-start,INSERT_VALUES); CHKERRQ(ierr);
        }
      }
      ierr = VecRestoreArray(jj2,&y); CHKERRQ(ierr);
    }
  }

  /* Assemble matrix; set flag to indicate a change in sparsity pattern */
  ierr = MatAssemblyBegin(*B,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  *flag = DIFFERENT_NONZERO_PATTERN; 
  ierr = MatAssemblyEnd(*B,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  return 0;
}

/* --------------------------------------------------------------- */
/* 
   MySparseFunction - Evaluates function at the local grid point (p,q),
   which corresponds to the current column number (index).  This routine
   is called from ComputeJacobian() for a sparse finite difference 
   approximation of the Jacobian matrix.

   Input Parameters:
.  x2 - perturbed iterate
.  index - current global column number
.  user - user-defined application context

   Output Parameter:
.  jj2 - F(X)
 */
int MySparseFunction(Vec x2,int index,AppCtx *user,Vec jj2)
{
  double   *f, *x;
  GridCtx  *grid = &user->grids[user->Nlevels - 1];
  Vec      localX = grid->localX, localF = grid->localF;
  int      N, *gindex, current, inputflag, ja, jb, ia, ib;
  int      xs, xe, ys, ye, Xs, Xm, Ys, ierr, p, q,  i, j;

  xs = grid->xs;
  xe = grid->xe; 
  ys = grid->ys; 
  ye = grid->ye;
  Ys = grid->Ys;
  Xs = grid->Xs;
  Xm = grid->Xm;

  /* Scatter permuted iterate vector to local work vector; get pointers
     to local vector data */
  ierr = DAGlobalToLocalBegin(grid->da,x2,INSERT_VALUES,localX);
  ierr = DAGetGlobalIndices(grid->da,&N,&gindex); CHKERRQ(ierr);
  ierr = VecGetArray(localF,&f); CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(grid->da,x2,INSERT_VALUES,localX);
  ierr = VecGetArray(localX,&x); CHKERRQ(ierr);
  
  current = 0;
  while (current < N && gindex[current] != index) current++;
  q = current / Xm + Ys;  /* local y coordinate of the index */
  p = current % Xm + Xs;  /* local x coordinate of the index */

  ierr = OptionsHasName(PETSC_NULL,"-dense",&inputflag); CHKERRQ(ierr);
  if (inputflag) {  /* dense */
    ja=ys; jb=ye; ia=xs; ib=xe;
  } else {          /* sparse: 9 point stencil */
    ja=PetscMax(q-1,ys); jb=PetscMin(q+2,ye);
    ia=PetscMax(p-1,xs); ib=PetscMin(p+2,xe);
  }

  for (j=ja; j<jb; j++) {
    for (i=ia; i<ib; i++) {
      /* Evaluate function at the grid point (i,j) */
      ierr = EvaluateFunction(user,x,i,j,PETSC_NULL,f); CHKERRQ(ierr);
    }
  }

  /* Restore vectors so that they are ready for later use */
  ierr = VecRestoreArray(localX,&x); CHKERRQ(ierr);
  ierr = VecRestoreArray(localF,&f); CHKERRQ(ierr);

  /* Place newly computed local function vectors in global vector */
  ierr = DALocalToGlobal(grid->da,localF,INSERT_VALUES,jj2); CHKERRQ(ierr);

  return 0;
}
/* --------------------------------------------------------------- */
/* 
   InnerSparseFunction - Computes 9-point stencil function for interior points
   on each processor.  We write this part of the computation separately from
   the general function evaluation since this part can be computed completely
   locally and no scatters are needed for the perturbed iterates for finite
   difference Jacobian approximation.

   Input Parameters:
.  p,q - local grid points in the x- and y-directions
.  x2 - perturbed iterate (LOCAL vector!)
.  index - current index
.  user - user-defined application context

   Output Parameter:
.  jj2 - F(x2)
*/
int InnerSparseFunction(int p,int q,Vec x2,int index,AppCtx *user,Vec jj2)
{
  double   *f, *x;
  GridCtx  *grid = &user->grids[user->Nlevels - 1];
  Vec      localF = grid->localF;
  int      ja, jb, ia, ib, ierr, i, j;

  /* Get pointers to local vector data */
  ierr = VecGetArray(localF,&f); CHKERRQ(ierr);
  ierr = VecGetArray(x2,&x); CHKERRQ(ierr);

  /* Set grid points for sparse, 9-point stencil for point (p,q) */
  ja=q-1; jb=q+2; ia=p-1; ib=p+2;
  if (ia<grid->xs || ib>grid->xe) SETERRQ(1,"InnerSparseFunction: bad p value!");
  if (ja<grid->ys || jb>grid->ye) SETERRQ(1,"InnerSparseFunction: bad q value!");
  for (j=ja; j<jb; j++) {
    for (i=ia; i<ib; i++) {
      /* Evaluate function at grid point (i,j) */
      ierr = EvaluateFunction(user,x,i,j,PETSC_NULL,f); CHKERRQ(ierr);
    }
  }
  /* Restore vectors so that they are ready for later use */
  ierr = VecRestoreArray(x2,&x); CHKERRQ(ierr);
  ierr = VecRestoreArray(localF,&f); CHKERRQ(ierr);

  /* Place newly computed local function vectors in global vector */
  ierr = DALocalToGlobal(grid->da,localF,INSERT_VALUES,jj2); CHKERRQ(ierr);

  return 0;
}
