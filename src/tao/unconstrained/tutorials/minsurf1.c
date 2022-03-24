/* Program usage: mpiexec -n 1 minsurf1 [-help] [all TAO options] */

/*  Include "petsctao.h" so we can use TAO solvers.  */
#include <petsctao.h>

static char  help[] =
"This example demonstrates use of the TAO package to \n\
solve an unconstrained minimization problem.  This example is based on a \n\
problem from the MINPACK-2 test suite.  Given a rectangular 2-D domain and \n\
boundary values along the edges of the domain, the objective is to find the\n\
surface with the minimal area that satisfies the boundary conditions.\n\
The command line options are:\n\
  -mx <xg>, where <xg> = number of grid points in the 1st coordinate direction\n\
  -my <yg>, where <yg> = number of grid points in the 2nd coordinate direction\n\
  -start <st>, where <st> =0 for zero vector, and an average of the boundary conditions otherwise \n\n";

/*T
   Concepts: TAO^Solving an unconstrained minimization problem
   Routines: TaoCreate(); TaoSetType();
   Routines: TaoSetSolution();
   Routines: TaoSetObjectiveAndGradient();
   Routines: TaoSetHessian(); TaoSetFromOptions();
   Routines: TaoGetKSP(); TaoSolve();
   Routines: TaoDestroy();
   Processors: 1
T*/

/*
   User-defined application context - contains data needed by the
   application-provided call-back routines, FormFunctionGradient()
   and FormHessian().
*/
typedef struct {
  PetscInt    mx, my;                 /* discretization in x, y directions */
  PetscReal   *bottom, *top, *left, *right;             /* boundary values */
  Mat         H;
} AppCtx;

/* -------- User-defined Routines --------- */

static PetscErrorCode MSA_BoundaryConditions(AppCtx*);
static PetscErrorCode MSA_InitialPoint(AppCtx*,Vec);
static PetscErrorCode QuadraticH(AppCtx*,Vec,Mat);
PetscErrorCode FormFunctionGradient(Tao,Vec,PetscReal*,Vec,void*);
PetscErrorCode FormHessian(Tao,Vec,Mat,Mat,void*);

int main(int argc, char **argv)
{
  PetscInt           N;                 /* Size of vector */
  PetscMPIInt        size;              /* Number of processors */
  Vec                x;                 /* solution, gradient vectors */
  KSP                ksp;               /*  PETSc Krylov subspace method */
  PetscBool          flg;               /* A return value when checking for user options */
  Tao                tao;               /* Tao solver context */
  AppCtx             user;              /* user-defined work context */

  /* Initialize TAO,PETSc */
  CHKERRQ(PetscInitialize(&argc, &argv,(char *)0,help));

  CHKERRMPI(MPI_Comm_size(MPI_COMM_WORLD,&size));
  PetscCheck(size == 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"Incorrect number of processors");

  /* Specify default dimension of the problem */
  user.mx = 4; user.my = 4;

  /* Check for any command line arguments that override defaults */
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-mx",&user.mx,&flg));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-my",&user.my,&flg));

  CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"\n---- Minimum Surface Area Problem -----\n"));
  CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"mx: %D     my: %D   \n\n",user.mx,user.my));

  /* Calculate any derived values from parameters */
  N    = user.mx*user.my;

  /* Create TAO solver and set desired solution method  */
  CHKERRQ(TaoCreate(PETSC_COMM_SELF,&tao));
  CHKERRQ(TaoSetType(tao,TAOLMVM));

  /* Initialize minsurf application data structure for use in the function evaluations  */
  CHKERRQ(MSA_BoundaryConditions(&user));            /* Application specific routine */

  /*
     Create a vector to hold the variables.  Compute an initial solution.
     Set this vector, which will be used by TAO.
  */
  CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,N,&x));
  CHKERRQ(MSA_InitialPoint(&user,x));                /* Application specific routine */
  CHKERRQ(TaoSetSolution(tao,x));   /* A TAO routine                */

  /* Provide TAO routines for function, gradient, and Hessian evaluation */
  CHKERRQ(TaoSetObjectiveAndGradient(tao,NULL,FormFunctionGradient,(void *)&user));

  /* Create a matrix data structure to store the Hessian.  This structure will be used by TAO */
  CHKERRQ(MatCreateSeqAIJ(PETSC_COMM_SELF,N,N,7,NULL,&(user.H)));
  CHKERRQ(MatSetOption(user.H,MAT_SYMMETRIC,PETSC_TRUE));
  CHKERRQ(TaoSetHessian(tao,user.H,user.H,FormHessian,(void *)&user));

  /* Check for any TAO command line options */
  CHKERRQ(TaoSetFromOptions(tao));

  /* Limit the number of iterations in the KSP linear solver */
  CHKERRQ(TaoGetKSP(tao,&ksp));
  if (ksp) {
    CHKERRQ(KSPSetTolerances(ksp,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,user.mx*user.my));
  }

  /* SOLVE THE APPLICATION */
  CHKERRQ(TaoSolve(tao));

  CHKERRQ(TaoDestroy(&tao));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(MatDestroy(&user.H));
  CHKERRQ(PetscFree(user.bottom));
  CHKERRQ(PetscFree(user.top));
  CHKERRQ(PetscFree(user.left));
  CHKERRQ(PetscFree(user.right));

  CHKERRQ(PetscFinalize());
  return 0;
}

/* -------------------------------------------------------------------- */

/*  FormFunctionGradient - Evaluates function and corresponding gradient.

    Input Parameters:
.   tao     - the Tao context
.   X       - input vector
.   userCtx - optional user-defined context, as set by TaoSetFunctionGradient()

    Output Parameters:
.   fcn     - the newly evaluated function
.   G       - vector containing the newly evaluated gradient
*/
PetscErrorCode FormFunctionGradient(Tao tao,Vec X,PetscReal *fcn,Vec G,void *userCtx)
{
  AppCtx            *user = (AppCtx *) userCtx;
  PetscInt          i,j,row;
  PetscInt          mx=user->mx, my=user->my;
  PetscReal         rhx=mx+1, rhy=my+1;
  PetscReal         hx=1.0/(mx+1),hy=1.0/(my+1), hydhx=hy/hx, hxdhy=hx/hy, area=0.5*hx*hy, ft=0;
  PetscReal         f1,f2,f3,f4,f5,f6,d1,d2,d3,d4,d5,d6,d7,d8,xc,xl,xr,xt,xb,xlt,xrb;
  PetscReal         df1dxc,df2dxc,df3dxc,df4dxc,df5dxc,df6dxc;
  PetscReal         zero=0.0;
  PetscScalar       *g;
  const PetscScalar *x;

  PetscFunctionBeginUser;
  CHKERRQ(VecSet(G, zero));

  CHKERRQ(VecGetArrayRead(X,&x));
  CHKERRQ(VecGetArray(G,&g));

  /* Compute function over the locally owned part of the mesh */
  for (j=0; j<my; j++) {
    for (i=0; i< mx; i++) {
      row=(j)*mx + (i);
      xc = x[row];
      xlt=xrb=xl=xr=xb=xt=xc;
      if (i==0) { /* left side */
        xl  = user->left[j+1];
        xlt = user->left[j+2];
      } else {
        xl = x[row-1];
      }

      if (j==0) { /* bottom side */
        xb  = user->bottom[i+1];
        xrb = user->bottom[i+2];
      } else {
        xb = x[row-mx];
      }

      if (i+1 == mx) { /* right side */
        xr  = user->right[j+1];
        xrb = user->right[j];
      } else {
        xr = x[row+1];
      }

      if (j+1==0+my) { /* top side */
        xt  = user->top[i+1];
        xlt = user->top[i];
      }else {
        xt = x[row+mx];
      }

      if (i>0 && j+1<my) {
        xlt = x[row-1+mx];
      }
      if (j>0 && i+1<mx) {
        xrb = x[row+1-mx];
      }

      d1 = (xc-xl);
      d2 = (xc-xr);
      d3 = (xc-xt);
      d4 = (xc-xb);
      d5 = (xr-xrb);
      d6 = (xrb-xb);
      d7 = (xlt-xl);
      d8 = (xt-xlt);

      df1dxc = d1*hydhx;
      df2dxc = (d1*hydhx + d4*hxdhy);
      df3dxc = d3*hxdhy;
      df4dxc = (d2*hydhx + d3*hxdhy);
      df5dxc = d2*hydhx;
      df6dxc = d4*hxdhy;

      d1 *= rhx;
      d2 *= rhx;
      d3 *= rhy;
      d4 *= rhy;
      d5 *= rhy;
      d6 *= rhx;
      d7 *= rhy;
      d8 *= rhx;

      f1 = PetscSqrtScalar(1.0 + d1*d1 + d7*d7);
      f2 = PetscSqrtScalar(1.0 + d1*d1 + d4*d4);
      f3 = PetscSqrtScalar(1.0 + d3*d3 + d8*d8);
      f4 = PetscSqrtScalar(1.0 + d3*d3 + d2*d2);
      f5 = PetscSqrtScalar(1.0 + d2*d2 + d5*d5);
      f6 = PetscSqrtScalar(1.0 + d4*d4 + d6*d6);

      ft = ft + (f2 + f4);

      df1dxc /= f1;
      df2dxc /= f2;
      df3dxc /= f3;
      df4dxc /= f4;
      df5dxc /= f5;
      df6dxc /= f6;

      g[row] = (df1dxc+df2dxc+df3dxc+df4dxc+df5dxc+df6dxc)/2.0;
    }
  }

  for (j=0; j<my; j++) {   /* left side */
    d3 = (user->left[j+1] - user->left[j+2])*rhy;
    d2 = (user->left[j+1] - x[j*mx])*rhx;
    ft = ft+PetscSqrtScalar(1.0 + d3*d3 + d2*d2);
  }

  for (i=0; i<mx; i++) { /* bottom */
    d2 = (user->bottom[i+1]-user->bottom[i+2])*rhx;
    d3 = (user->bottom[i+1]-x[i])*rhy;
    ft = ft+PetscSqrtScalar(1.0 + d3*d3 + d2*d2);
  }

  for (j=0; j< my; j++) { /* right side */
    d1 = (x[(j+1)*mx-1]-user->right[j+1])*rhx;
    d4 = (user->right[j]-user->right[j+1])*rhy;
    ft = ft+PetscSqrtScalar(1.0 + d1*d1 + d4*d4);
  }

  for (i=0; i<mx; i++) { /* top side */
    d1 = (x[(my-1)*mx + i] - user->top[i+1])*rhy;
    d4 = (user->top[i+1] - user->top[i])*rhx;
    ft = ft+PetscSqrtScalar(1.0 + d1*d1 + d4*d4);
  }

  /* Bottom left corner */
  d1  = (user->left[0]-user->left[1])*rhy;
  d2  = (user->bottom[0]-user->bottom[1])*rhx;
  ft += PetscSqrtScalar(1.0 + d1*d1 + d2*d2);

  /* Top right corner */
  d1  = (user->right[my+1] - user->right[my])*rhy;
  d2  = (user->top[mx+1] - user->top[mx])*rhx;
  ft += PetscSqrtScalar(1.0 + d1*d1 + d2*d2);

  (*fcn)=ft*area;

  /* Restore vectors */
  CHKERRQ(VecRestoreArrayRead(X,&x));
  CHKERRQ(VecRestoreArray(G,&g));
  CHKERRQ(PetscLogFlops(67.0*mx*my));
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
/*
   FormHessian - Evaluates the Hessian matrix.

   Input Parameters:
.  tao  - the Tao context
.  x    - input vector
.  ptr  - optional user-defined context, as set by TaoSetHessian()

   Output Parameters:
.  H    - Hessian matrix
.  Hpre - optionally different preconditioning matrix
.  flg  - flag indicating matrix structure

*/
PetscErrorCode FormHessian(Tao tao,Vec X,Mat H, Mat Hpre, void *ptr)
{
  AppCtx         *user = (AppCtx *) ptr;

  PetscFunctionBeginUser;
  /* Evaluate the Hessian entries*/
  CHKERRQ(QuadraticH(user,X,H));
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
/*
   QuadraticH - Evaluates the Hessian matrix.

   Input Parameters:
.  user - user-defined context, as set by TaoSetHessian()
.  X    - input vector

   Output Parameter:
.  H    - Hessian matrix
*/
PetscErrorCode QuadraticH(AppCtx *user, Vec X, Mat Hessian)
{
  PetscInt          i,j,k,row;
  PetscInt          mx=user->mx, my=user->my;
  PetscInt          col[7];
  PetscReal         hx=1.0/(mx+1), hy=1.0/(my+1), hydhx=hy/hx, hxdhy=hx/hy;
  PetscReal         rhx=mx+1, rhy=my+1;
  PetscReal         f1,f2,f3,f4,f5,f6,d1,d2,d3,d4,d5,d6,d7,d8,xc,xl,xr,xt,xb,xlt,xrb;
  PetscReal         hl,hr,ht,hb,hc,htl,hbr;
  const PetscScalar *x;
  PetscReal         v[7];

  PetscFunctionBeginUser;
  /* Get pointers to vector data */
  CHKERRQ(VecGetArrayRead(X,&x));

  /* Initialize matrix entries to zero */
  CHKERRQ(MatZeroEntries(Hessian));

  /* Set various matrix options */
  CHKERRQ(MatSetOption(Hessian,MAT_IGNORE_OFF_PROC_ENTRIES,PETSC_TRUE));

  /* Compute Hessian over the locally owned part of the mesh */
  for (i=0; i< mx; i++) {
    for (j=0; j<my; j++) {

      row=(j)*mx + (i);

      xc = x[row];
      xlt=xrb=xl=xr=xb=xt=xc;

      /* Left side */
      if (i==0) {
        xl= user->left[j+1];
        xlt = user->left[j+2];
      } else {
        xl = x[row-1];
      }

      if (j==0) {
        xb=user->bottom[i+1];
        xrb = user->bottom[i+2];
      } else {
        xb = x[row-mx];
      }

      if (i+1 == mx) {
        xr=user->right[j+1];
        xrb = user->right[j];
      } else {
        xr = x[row+1];
      }

      if (j+1==my) {
        xt=user->top[i+1];
        xlt = user->top[i];
      }else {
        xt = x[row+mx];
      }

      if (i>0 && j+1<my) {
        xlt = x[row-1+mx];
      }
      if (j>0 && i+1<mx) {
        xrb = x[row+1-mx];
      }

      d1 = (xc-xl)*rhx;
      d2 = (xc-xr)*rhx;
      d3 = (xc-xt)*rhy;
      d4 = (xc-xb)*rhy;
      d5 = (xrb-xr)*rhy;
      d6 = (xrb-xb)*rhx;
      d7 = (xlt-xl)*rhy;
      d8 = (xlt-xt)*rhx;

      f1 = PetscSqrtScalar(1.0 + d1*d1 + d7*d7);
      f2 = PetscSqrtScalar(1.0 + d1*d1 + d4*d4);
      f3 = PetscSqrtScalar(1.0 + d3*d3 + d8*d8);
      f4 = PetscSqrtScalar(1.0 + d3*d3 + d2*d2);
      f5 = PetscSqrtScalar(1.0 + d2*d2 + d5*d5);
      f6 = PetscSqrtScalar(1.0 + d4*d4 + d6*d6);

      hl = (-hydhx*(1.0+d7*d7)+d1*d7)/(f1*f1*f1)+(-hydhx*(1.0+d4*d4)+d1*d4)/(f2*f2*f2);
      hr = (-hydhx*(1.0+d5*d5)+d2*d5)/(f5*f5*f5)+(-hydhx*(1.0+d3*d3)+d2*d3)/(f4*f4*f4);
      ht = (-hxdhy*(1.0+d8*d8)+d3*d8)/(f3*f3*f3)+(-hxdhy*(1.0+d2*d2)+d2*d3)/(f4*f4*f4);
      hb = (-hxdhy*(1.0+d6*d6)+d4*d6)/(f6*f6*f6)+(-hxdhy*(1.0+d1*d1)+d1*d4)/(f2*f2*f2);

      hbr = -d2*d5/(f5*f5*f5) - d4*d6/(f6*f6*f6);
      htl = -d1*d7/(f1*f1*f1) - d3*d8/(f3*f3*f3);

      hc = hydhx*(1.0+d7*d7)/(f1*f1*f1) + hxdhy*(1.0+d8*d8)/(f3*f3*f3) + hydhx*(1.0+d5*d5)/(f5*f5*f5) + hxdhy*(1.0+d6*d6)/(f6*f6*f6) +
           (hxdhy*(1.0+d1*d1)+hydhx*(1.0+d4*d4)-2*d1*d4)/(f2*f2*f2) +  (hxdhy*(1.0+d2*d2)+hydhx*(1.0+d3*d3)-2*d2*d3)/(f4*f4*f4);

      hl*=0.5; hr*=0.5; ht*=0.5; hb*=0.5; hbr*=0.5; htl*=0.5;  hc*=0.5;

      k=0;
      if (j>0) {
        v[k]=hb; col[k]=row - mx; k++;
      }

      if (j>0 && i < mx -1) {
        v[k]=hbr; col[k]=row - mx+1; k++;
      }

      if (i>0) {
        v[k]= hl; col[k]=row - 1; k++;
      }

      v[k]= hc; col[k]=row; k++;

      if (i < mx-1) {
        v[k]= hr; col[k]=row+1; k++;
      }

      if (i>0 && j < my-1) {
        v[k]= htl; col[k] = row+mx-1; k++;
      }

      if (j < my-1) {
        v[k]= ht; col[k] = row+mx; k++;
      }

      /*
         Set matrix values using local numbering, which was defined
         earlier, in the main routine.
      */
      CHKERRQ(MatSetValues(Hessian,1,&row,k,col,v,INSERT_VALUES));
    }
  }

  /* Restore vectors */
  CHKERRQ(VecRestoreArrayRead(X,&x));

  /* Assemble the matrix */
  CHKERRQ(MatAssemblyBegin(Hessian,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(Hessian,MAT_FINAL_ASSEMBLY));

  CHKERRQ(PetscLogFlops(199.0*mx*my));
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
/*
   MSA_BoundaryConditions -  Calculates the boundary conditions for
   the region.

   Input Parameter:
.  user - user-defined application context

   Output Parameter:
.  user - user-defined application context
*/
static PetscErrorCode MSA_BoundaryConditions(AppCtx * user)
{
  PetscInt       i,j,k,limit=0;
  PetscInt       maxits=5;
  PetscInt       mx=user->mx,my=user->my;
  PetscInt       bsize=0, lsize=0, tsize=0, rsize=0;
  PetscReal      one=1.0, two=2.0, three=3.0, tol=1e-10;
  PetscReal      fnorm,det,hx,hy,xt=0,yt=0;
  PetscReal      u1,u2,nf1,nf2,njac11,njac12,njac21,njac22;
  PetscReal      b=-0.5, t=0.5, l=-0.5, r=0.5;
  PetscReal      *boundary;

  PetscFunctionBeginUser;
  bsize=mx+2; lsize=my+2; rsize=my+2; tsize=mx+2;

  CHKERRQ(PetscMalloc1(bsize,&user->bottom));
  CHKERRQ(PetscMalloc1(tsize,&user->top));
  CHKERRQ(PetscMalloc1(lsize,&user->left));
  CHKERRQ(PetscMalloc1(rsize,&user->right));

  hx= (r-l)/(mx+1); hy=(t-b)/(my+1);

  for (j=0; j<4; j++) {
    if (j==0) {
      yt=b;
      xt=l;
      limit=bsize;
      boundary=user->bottom;
    } else if (j==1) {
      yt=t;
      xt=l;
      limit=tsize;
      boundary=user->top;
    } else if (j==2) {
      yt=b;
      xt=l;
      limit=lsize;
      boundary=user->left;
    } else {  /*  if (j==3) */
      yt=b;
      xt=r;
      limit=rsize;
      boundary=user->right;
    }

    for (i=0; i<limit; i++) {
      u1=xt;
      u2=-yt;
      for (k=0; k<maxits; k++) {
        nf1=u1 + u1*u2*u2 - u1*u1*u1/three-xt;
        nf2=-u2 - u1*u1*u2 + u2*u2*u2/three-yt;
        fnorm=PetscSqrtScalar(nf1*nf1+nf2*nf2);
        if (fnorm <= tol) break;
        njac11=one+u2*u2-u1*u1;
        njac12=two*u1*u2;
        njac21=-two*u1*u2;
        njac22=-one - u1*u1 + u2*u2;
        det = njac11*njac22-njac21*njac12;
        u1 = u1-(njac22*nf1-njac12*nf2)/det;
        u2 = u2-(njac11*nf2-njac21*nf1)/det;
      }

      boundary[i]=u1*u1-u2*u2;
      if (j==0 || j==1) {
        xt=xt+hx;
      } else { /*  if (j==2 || j==3) */
        yt=yt+hy;
      }
    }
  }
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
/*
   MSA_InitialPoint - Calculates the initial guess in one of three ways.

   Input Parameters:
.  user - user-defined application context
.  X - vector for initial guess

   Output Parameters:
.  X - newly computed initial guess
*/
static PetscErrorCode MSA_InitialPoint(AppCtx * user, Vec X)
{
  PetscInt       start=-1,i,j;
  PetscReal      zero=0.0;
  PetscBool      flg;

  PetscFunctionBeginUser;
  CHKERRQ(VecSet(X, zero));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-start",&start,&flg));

  if (flg && start==0) { /* The zero vector is reasonable */
     CHKERRQ(VecSet(X, zero));
   } else { /* Take an average of the boundary conditions */
    PetscInt    row;
    PetscInt    mx=user->mx,my=user->my;
    PetscReal *x;

    /* Get pointers to vector data */
    CHKERRQ(VecGetArray(X,&x));
    /* Perform local computations */
    for (j=0; j<my; j++) {
      for (i=0; i< mx; i++) {
        row=(j)*mx + (i);
        x[row] = (((j+1)*user->bottom[i+1]+(my-j+1)*user->top[i+1])/(my+2)+ ((i+1)*user->left[j+1]+(mx-i+1)*user->right[j+1])/(mx+2))/2.0;
      }
    }
    /* Restore vectors */
    CHKERRQ(VecRestoreArray(X,&x));
  }
  PetscFunctionReturn(0);
}

/*TEST

   build:
      requires: !complex

   test:
      args: -tao_smonitor -tao_type nls -mx 10 -my 8 -tao_gatol 1.e-4
      requires: !single

   test:
      suffix: 2
      args: -tao_smonitor -tao_type bmrm -mx 10 -my 8 -tao_gatol 1.e-3
      requires: !single

   test:
      suffix: 3
      args: -tao_smonitor -tao_type lmvm -mx 10 -my 8 -tao_gatol 1.e-3
      requires: !single

   test:
      suffix: 4
      args: -tao_smonitor -tao_type bntr -mx 10 -my 8 -tao_gatol 1.e-4
      requires: !single

   test:
      suffix: 5
      args: -tao_smonitor -tao_type bntl -mx 10 -my 8 -tao_gatol 1.e-4
      requires: !single

   test:
      suffix: 6
      args: -tao_smonitor -tao_type bnls -mx 10 -my 8 -tao_gatol 1.e-4
      requires: !single

   test:
      suffix: 7
      args: -tao_smonitor -tao_type bntr -mx 10 -my 8 -tao_bnk_max_cg_its 3 -tao_gatol 1.e-4
      requires: !single

   test:
      suffix: 8
      args: -tao_smonitor -tao_type bntl -mx 10 -my 8 -tao_bnk_max_cg_its 3 -tao_gatol 1.e-4
      requires: !single

   test:
      suffix: 9
      args: -tao_smonitor -tao_type bnls -mx 10 -my 8 -tao_bnk_max_cg_its 3 -tao_gatol 1.e-4
      requires: !single

   test:
      suffix: 10
      args: -tao_smonitor -tao_type bnls -mx 10 -my 8 -tao_bnk_max_cg_its 3 -tao_gatol 1.e-4 -tao_mf_hessian

   test:
      suffix: 11
      args: -tao_smonitor -tao_type bntr -mx 10 -my 8 -tao_gatol 1.e-4 -tao_mf_hessian
      requires: !single

   test:
      suffix: 12
      args: -tao_smonitor -tao_type bntl -mx 10 -my 8 -tao_gatol 1.e-4 -tao_mf_hessian
      requires: !single

TEST*/
