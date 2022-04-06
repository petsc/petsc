/* Program usage: mpiexec -n <proc> minsurf2 [-help] [all TAO options] */

/*
  Include "petsctao.h" so we can use TAO solvers.
  petscdmda.h for distributed array
*/
#include <petsctao.h>
#include <petscdmda.h>

static  char help[] =
"This example demonstrates use of the TAO package to \n\
solve an unconstrained minimization problem.  This example is based on a \n\
problem from the MINPACK-2 test suite.  Given a rectangular 2-D domain and \n\
boundary values along the edges of the domain, the objective is to find the\n\
surface with the minimal area that satisfies the boundary conditions.\n\
The command line options are:\n\
  -mx <xg>, where <xg> = number of grid points in the 1st coordinate direction\n\
  -my <yg>, where <yg> = number of grid points in the 2nd coordinate direction\n\
  -start <st>, where <st> =0 for zero vector, <st> >0 for random start, and <st> <0 \n\
               for an average of the boundary conditions\n\n";

/*
   User-defined application context - contains data needed by the
   application-provided call-back routines, FormFunctionGradient()
   and FormHessian().
*/
typedef struct {
  PetscInt    mx, my;                 /* discretization in x, y directions */
  PetscReal   *bottom, *top, *left, *right;             /* boundary values */
  DM          dm;                      /* distributed array data structure */
  Mat         H;                       /* Hessian */
} AppCtx;

/* -------- User-defined Routines --------- */

static PetscErrorCode MSA_BoundaryConditions(AppCtx*);
static PetscErrorCode MSA_InitialPoint(AppCtx*,Vec);
PetscErrorCode QuadraticH(AppCtx*,Vec,Mat);
PetscErrorCode FormFunctionGradient(Tao,Vec,PetscReal *,Vec,void*);
PetscErrorCode FormGradient(Tao,Vec,Vec,void*);
PetscErrorCode FormHessian(Tao,Vec,Mat,Mat,void*);
PetscErrorCode My_Monitor(Tao, void *);

int main(int argc, char **argv)
{
  PetscInt           Nx, Ny;              /* number of processors in x- and y- directions */
  Vec                x;                   /* solution, gradient vectors */
  PetscBool          flg, viewmat;        /* flags */
  PetscBool          fddefault, fdcoloring;   /* flags */
  Tao                tao;                 /* TAO solver context */
  AppCtx             user;                /* user-defined work context */
  ISColoring         iscoloring;
  MatFDColoring      matfdcoloring;

  /* Initialize TAO */
  PetscCall(PetscInitialize(&argc, &argv,(char *)0,help));

  /* Specify dimension of the problem */
  user.mx = 10; user.my = 10;

  /* Check for any command line arguments that override defaults */
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-mx",&user.mx,&flg));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-my",&user.my,&flg));

  PetscCall(PetscPrintf(MPI_COMM_WORLD,"\n---- Minimum Surface Area Problem -----\n"));
  PetscCall(PetscPrintf(MPI_COMM_WORLD,"mx: %D     my: %D   \n\n",user.mx,user.my));

  /* Let PETSc determine the vector distribution */
  Nx = PETSC_DECIDE; Ny = PETSC_DECIDE;

  /* Create distributed array (DM) to manage parallel grid and vectors  */
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,user.mx, user.my,Nx,Ny,1,1,NULL,NULL,&user.dm));
  PetscCall(DMSetFromOptions(user.dm));
  PetscCall(DMSetUp(user.dm));

  /* Create TAO solver and set desired solution method.*/
  PetscCall(TaoCreate(PETSC_COMM_WORLD,&tao));
  PetscCall(TaoSetType(tao,TAOCG));

  /*
     Extract global vector from DA for the vector of variables --  PETSC routine
     Compute the initial solution                              --  application specific, see below
     Set this vector for use by TAO                            --  TAO routine
  */
  PetscCall(DMCreateGlobalVector(user.dm,&x));
  PetscCall(MSA_BoundaryConditions(&user));
  PetscCall(MSA_InitialPoint(&user,x));
  PetscCall(TaoSetSolution(tao,x));

  /*
     Initialize the Application context for use in function evaluations  --  application specific, see below.
     Set routines for function and gradient evaluation
  */
  PetscCall(TaoSetObjectiveAndGradient(tao,NULL,FormFunctionGradient,(void *)&user));

  /*
     Given the command line arguments, calculate the hessian with either the user-
     provided function FormHessian, or the default finite-difference driven Hessian
     functions
  */
  PetscCall(PetscOptionsHasName(NULL,NULL,"-tao_fddefault",&fddefault));
  PetscCall(PetscOptionsHasName(NULL,NULL,"-tao_fdcoloring",&fdcoloring));

  /*
     Create a matrix data structure to store the Hessian and set
     the Hessian evalution routine.
     Set the matrix structure to be used for Hessian evalutions
  */
  PetscCall(DMCreateMatrix(user.dm,&user.H));
  PetscCall(MatSetOption(user.H,MAT_SYMMETRIC,PETSC_TRUE));

  if (fdcoloring) {
    PetscCall(DMCreateColoring(user.dm,IS_COLORING_GLOBAL,&iscoloring));
    PetscCall(MatFDColoringCreate(user.H,iscoloring,&matfdcoloring));
    PetscCall(ISColoringDestroy(&iscoloring));
    PetscCall(MatFDColoringSetFunction(matfdcoloring,(PetscErrorCode (*)(void))FormGradient,(void*)&user));
    PetscCall(MatFDColoringSetFromOptions(matfdcoloring));
    PetscCall(TaoSetHessian(tao,user.H,user.H,TaoDefaultComputeHessianColor,(void *)matfdcoloring));
  } else if (fddefault) {
    PetscCall(TaoSetHessian(tao,user.H,user.H,TaoDefaultComputeHessian,(void *)NULL));
  } else {
    PetscCall(TaoSetHessian(tao,user.H,user.H,FormHessian,(void *)&user));
  }

  /*
     If my_monitor option is in command line, then use the user-provided
     monitoring function
  */
  PetscCall(PetscOptionsHasName(NULL,NULL,"-my_monitor",&viewmat));
  if (viewmat) {
    PetscCall(TaoSetMonitor(tao,My_Monitor,NULL,NULL));
  }

  /* Check for any tao command line options */
  PetscCall(TaoSetFromOptions(tao));

  /* SOLVE THE APPLICATION */
  PetscCall(TaoSolve(tao));

  PetscCall(TaoView(tao,PETSC_VIEWER_STDOUT_WORLD));

  /* Free TAO data structures */
  PetscCall(TaoDestroy(&tao));

  /* Free PETSc data structures */
  PetscCall(VecDestroy(&x));
  PetscCall(MatDestroy(&user.H));
  if (fdcoloring) {
    PetscCall(MatFDColoringDestroy(&matfdcoloring));
  }
  PetscCall(PetscFree(user.bottom));
  PetscCall(PetscFree(user.top));
  PetscCall(PetscFree(user.left));
  PetscCall(PetscFree(user.right));
  PetscCall(DMDestroy(&user.dm));
  PetscCall(PetscFinalize());
  return 0;
}

PetscErrorCode FormGradient(Tao tao, Vec X, Vec G,void *userCtx)
{
  PetscReal      fcn;

  PetscFunctionBegin;
  PetscCall(FormFunctionGradient(tao,X,&fcn,G,userCtx));
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------- */
/*  FormFunctionGradient - Evaluates the function and corresponding gradient.

    Input Parameters:
.   tao     - the Tao context
.   XX      - input vector
.   userCtx - optional user-defined context, as set by TaoSetObjectiveAndGradient()

    Output Parameters:
.   fcn     - the newly evaluated function
.   GG       - vector containing the newly evaluated gradient
*/
PetscErrorCode FormFunctionGradient(Tao tao, Vec X, PetscReal *fcn,Vec G,void *userCtx)
{
  AppCtx         *user = (AppCtx *) userCtx;
  PetscInt       i,j;
  PetscInt       mx=user->mx, my=user->my;
  PetscInt       xs,xm,gxs,gxm,ys,ym,gys,gym;
  PetscReal      ft=0.0;
  PetscReal      hx=1.0/(mx+1),hy=1.0/(my+1), hydhx=hy/hx, hxdhy=hx/hy, area=0.5*hx*hy;
  PetscReal      rhx=mx+1, rhy=my+1;
  PetscReal      f1,f2,f3,f4,f5,f6,d1,d2,d3,d4,d5,d6,d7,d8,xc,xl,xr,xt,xb,xlt,xrb;
  PetscReal      df1dxc,df2dxc,df3dxc,df4dxc,df5dxc,df6dxc;
  PetscReal      **g, **x;
  Vec            localX;

  PetscFunctionBegin;
  /* Get local mesh boundaries */
  PetscCall(DMGetLocalVector(user->dm,&localX));
  PetscCall(DMDAGetCorners(user->dm,&xs,&ys,NULL,&xm,&ym,NULL));
  PetscCall(DMDAGetGhostCorners(user->dm,&gxs,&gys,NULL,&gxm,&gym,NULL));

  /* Scatter ghost points to local vector */
  PetscCall(DMGlobalToLocalBegin(user->dm,X,INSERT_VALUES,localX));
  PetscCall(DMGlobalToLocalEnd(user->dm,X,INSERT_VALUES,localX));

  /* Get pointers to vector data */
  PetscCall(DMDAVecGetArray(user->dm,localX,(void**)&x));
  PetscCall(DMDAVecGetArray(user->dm,G,(void**)&g));

  /* Compute function and gradient over the locally owned part of the mesh */
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i< xs+xm; i++) {

      xc = x[j][i];
      xlt=xrb=xl=xr=xb=xt=xc;

      if (i==0) { /* left side */
        xl= user->left[j-ys+1];
        xlt = user->left[j-ys+2];
      } else {
        xl = x[j][i-1];
      }

      if (j==0) { /* bottom side */
        xb=user->bottom[i-xs+1];
        xrb =user->bottom[i-xs+2];
      } else {
        xb = x[j-1][i];
      }

      if (i+1 == gxs+gxm) { /* right side */
        xr=user->right[j-ys+1];
        xrb = user->right[j-ys];
      } else {
        xr = x[j][i+1];
      }

      if (j+1==gys+gym) { /* top side */
        xt=user->top[i-xs+1];
        xlt = user->top[i-xs];
      }else {
        xt = x[j+1][i];
      }

      if (i>gxs && j+1<gys+gym) {
        xlt = x[j+1][i-1];
      }
      if (j>gys && i+1<gxs+gxm) {
        xrb = x[j-1][i+1];
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

      f1 = PetscSqrtReal(1.0 + d1*d1 + d7*d7);
      f2 = PetscSqrtReal(1.0 + d1*d1 + d4*d4);
      f3 = PetscSqrtReal(1.0 + d3*d3 + d8*d8);
      f4 = PetscSqrtReal(1.0 + d3*d3 + d2*d2);
      f5 = PetscSqrtReal(1.0 + d2*d2 + d5*d5);
      f6 = PetscSqrtReal(1.0 + d4*d4 + d6*d6);

      ft = ft + (f2 + f4);

      df1dxc /= f1;
      df2dxc /= f2;
      df3dxc /= f3;
      df4dxc /= f4;
      df5dxc /= f5;
      df6dxc /= f6;

      g[j][i] = (df1dxc+df2dxc+df3dxc+df4dxc+df5dxc+df6dxc) * 0.5;

    }
  }

  /* Compute triangular areas along the border of the domain. */
  if (xs==0) { /* left side */
    for (j=ys; j<ys+ym; j++) {
      d3=(user->left[j-ys+1] - user->left[j-ys+2])*rhy;
      d2=(user->left[j-ys+1] - x[j][0]) *rhx;
      ft = ft+PetscSqrtReal(1.0 + d3*d3 + d2*d2);
    }
  }
  if (ys==0) { /* bottom side */
    for (i=xs; i<xs+xm; i++) {
      d2=(user->bottom[i+1-xs]-user->bottom[i-xs+2])*rhx;
      d3=(user->bottom[i-xs+1]-x[0][i])*rhy;
      ft = ft+PetscSqrtReal(1.0 + d3*d3 + d2*d2);
    }
  }

  if (xs+xm==mx) { /* right side */
    for (j=ys; j< ys+ym; j++) {
      d1=(x[j][mx-1] - user->right[j-ys+1])*rhx;
      d4=(user->right[j-ys]-user->right[j-ys+1])*rhy;
      ft = ft+PetscSqrtReal(1.0 + d1*d1 + d4*d4);
    }
  }
  if (ys+ym==my) { /* top side */
    for (i=xs; i<xs+xm; i++) {
      d1=(x[my-1][i] - user->top[i-xs+1])*rhy;
      d4=(user->top[i-xs+1] - user->top[i-xs])*rhx;
      ft = ft+PetscSqrtReal(1.0 + d1*d1 + d4*d4);
    }
  }

  if (ys==0 && xs==0) {
    d1=(user->left[0]-user->left[1])/hy;
    d2=(user->bottom[0]-user->bottom[1])*rhx;
    ft +=PetscSqrtReal(1.0 + d1*d1 + d2*d2);
  }
  if (ys+ym == my && xs+xm == mx) {
    d1=(user->right[ym+1] - user->right[ym])*rhy;
    d2=(user->top[xm+1] - user->top[xm])*rhx;
    ft +=PetscSqrtReal(1.0 + d1*d1 + d2*d2);
  }

  ft=ft*area;
  PetscCallMPI(MPI_Allreduce(&ft,fcn,1,MPIU_REAL,MPIU_SUM,MPI_COMM_WORLD));

  /* Restore vectors */
  PetscCall(DMDAVecRestoreArray(user->dm,localX,(void**)&x));
  PetscCall(DMDAVecRestoreArray(user->dm,G,(void**)&g));

  /* Scatter values to global vector */
  PetscCall(DMRestoreLocalVector(user->dm,&localX));
  PetscCall(PetscLogFlops(67.0*xm*ym));
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
/*
   FormHessian - Evaluates Hessian matrix.

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

  PetscFunctionBegin;
  /* Evaluate the Hessian entries*/
  PetscCall(QuadraticH(user,X,H));
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
/*
   QuadraticH - Evaluates Hessian matrix.

   Input Parameters:
.  user - user-defined context, as set by TaoSetHessian()
.  X    - input vector

   Output Parameter:
.  H    - Hessian matrix
*/
PetscErrorCode QuadraticH(AppCtx *user, Vec X, Mat Hessian)
{
  PetscInt i,j,k;
  PetscInt mx=user->mx, my=user->my;
  PetscInt xs,xm,gxs,gxm,ys,ym,gys,gym;
  PetscReal hx=1.0/(mx+1), hy=1.0/(my+1), hydhx=hy/hx, hxdhy=hx/hy;
  PetscReal f1,f2,f3,f4,f5,f6,d1,d2,d3,d4,d5,d6,d7,d8,xc,xl,xr,xt,xb,xlt,xrb;
  PetscReal hl,hr,ht,hb,hc,htl,hbr;
  PetscReal **x, v[7];
  MatStencil col[7],row;
  Vec    localX;
  PetscBool assembled;

  PetscFunctionBegin;
  /* Get local mesh boundaries */
  PetscCall(DMGetLocalVector(user->dm,&localX));

  PetscCall(DMDAGetCorners(user->dm,&xs,&ys,NULL,&xm,&ym,NULL));
  PetscCall(DMDAGetGhostCorners(user->dm,&gxs,&gys,NULL,&gxm,&gym,NULL));

  /* Scatter ghost points to local vector */
  PetscCall(DMGlobalToLocalBegin(user->dm,X,INSERT_VALUES,localX));
  PetscCall(DMGlobalToLocalEnd(user->dm,X,INSERT_VALUES,localX));

  /* Get pointers to vector data */
  PetscCall(DMDAVecGetArray(user->dm,localX,(void**)&x));

  /* Initialize matrix entries to zero */
  PetscCall(MatAssembled(Hessian,&assembled));
  if (assembled) PetscCall(MatZeroEntries(Hessian));

  /* Set various matrix options */
  PetscCall(MatSetOption(Hessian,MAT_IGNORE_OFF_PROC_ENTRIES,PETSC_TRUE));

  /* Compute Hessian over the locally owned part of the mesh */

  for (j=ys; j<ys+ym; j++) {

    for (i=xs; i< xs+xm; i++) {

      xc = x[j][i];
      xlt=xrb=xl=xr=xb=xt=xc;

      /* Left side */
      if (i==0) {
        xl  = user->left[j-ys+1];
        xlt = user->left[j-ys+2];
      } else {
        xl  = x[j][i-1];
      }

      if (j==0) {
        xb  = user->bottom[i-xs+1];
        xrb = user->bottom[i-xs+2];
      } else {
        xb  = x[j-1][i];
      }

      if (i+1 == mx) {
        xr  = user->right[j-ys+1];
        xrb = user->right[j-ys];
      } else {
        xr  = x[j][i+1];
      }

      if (j+1==my) {
        xt  = user->top[i-xs+1];
        xlt = user->top[i-xs];
      }else {
        xt  = x[j+1][i];
      }

      if (i>0 && j+1<my) {
        xlt = x[j+1][i-1];
      }
      if (j>0 && i+1<mx) {
        xrb = x[j-1][i+1];
      }

      d1 = (xc-xl)/hx;
      d2 = (xc-xr)/hx;
      d3 = (xc-xt)/hy;
      d4 = (xc-xb)/hy;
      d5 = (xrb-xr)/hy;
      d6 = (xrb-xb)/hx;
      d7 = (xlt-xl)/hy;
      d8 = (xlt-xt)/hx;

      f1 = PetscSqrtReal(1.0 + d1*d1 + d7*d7);
      f2 = PetscSqrtReal(1.0 + d1*d1 + d4*d4);
      f3 = PetscSqrtReal(1.0 + d3*d3 + d8*d8);
      f4 = PetscSqrtReal(1.0 + d3*d3 + d2*d2);
      f5 = PetscSqrtReal(1.0 + d2*d2 + d5*d5);
      f6 = PetscSqrtReal(1.0 + d4*d4 + d6*d6);

      hl = (-hydhx*(1.0+d7*d7)+d1*d7)/(f1*f1*f1)+
        (-hydhx*(1.0+d4*d4)+d1*d4)/(f2*f2*f2);
      hr = (-hydhx*(1.0+d5*d5)+d2*d5)/(f5*f5*f5)+
        (-hydhx*(1.0+d3*d3)+d2*d3)/(f4*f4*f4);
      ht = (-hxdhy*(1.0+d8*d8)+d3*d8)/(f3*f3*f3)+
        (-hxdhy*(1.0+d2*d2)+d2*d3)/(f4*f4*f4);
      hb = (-hxdhy*(1.0+d6*d6)+d4*d6)/(f6*f6*f6)+
        (-hxdhy*(1.0+d1*d1)+d1*d4)/(f2*f2*f2);

      hbr = -d2*d5/(f5*f5*f5) - d4*d6/(f6*f6*f6);
      htl = -d1*d7/(f1*f1*f1) - d3*d8/(f3*f3*f3);

      hc = hydhx*(1.0+d7*d7)/(f1*f1*f1) + hxdhy*(1.0+d8*d8)/(f3*f3*f3) +
        hydhx*(1.0+d5*d5)/(f5*f5*f5) + hxdhy*(1.0+d6*d6)/(f6*f6*f6) +
        (hxdhy*(1.0+d1*d1)+hydhx*(1.0+d4*d4)-2*d1*d4)/(f2*f2*f2) +
        (hxdhy*(1.0+d2*d2)+hydhx*(1.0+d3*d3)-2*d2*d3)/(f4*f4*f4);

      hl/=2.0; hr/=2.0; ht/=2.0; hb/=2.0; hbr/=2.0; htl/=2.0;  hc/=2.0;

      row.j = j; row.i = i;
      k=0;
      if (j>0) {
        v[k]=hb;
        col[k].j = j - 1; col[k].i = i;
        k++;
      }

      if (j>0 && i < mx -1) {
        v[k]=hbr;
        col[k].j = j - 1; col[k].i = i+1;
        k++;
      }

      if (i>0) {
        v[k]= hl;
        col[k].j = j; col[k].i = i-1;
        k++;
      }

      v[k]= hc;
      col[k].j = j; col[k].i = i;
      k++;

      if (i < mx-1) {
        v[k]= hr;
        col[k].j = j; col[k].i = i+1;
        k++;
      }

      if (i>0 && j < my-1) {
        v[k]= htl;
        col[k].j = j+1; col[k].i = i-1;
        k++;
      }

      if (j < my-1) {
        v[k]= ht;
        col[k].j = j+1; col[k].i = i;
        k++;
      }

      /*
         Set matrix values using local numbering, which was defined
         earlier, in the main routine.
      */
      PetscCall(MatSetValuesStencil(Hessian,1,&row,k,col,v,INSERT_VALUES));
    }
  }

  PetscCall(DMDAVecRestoreArray(user->dm,localX,(void**)&x));
  PetscCall(DMRestoreLocalVector(user->dm,&localX));

  PetscCall(MatAssemblyBegin(Hessian,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Hessian,MAT_FINAL_ASSEMBLY));

  PetscCall(PetscLogFlops(199.0*xm*ym));
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
  PetscInt       i,j,k,limit=0,maxits=5;
  PetscInt       xs,ys,xm,ym,gxs,gys,gxm,gym;
  PetscInt       mx=user->mx,my=user->my;
  PetscInt       bsize=0, lsize=0, tsize=0, rsize=0;
  PetscReal      one=1.0, two=2.0, three=3.0, tol=1e-10;
  PetscReal      fnorm,det,hx,hy,xt=0,yt=0;
  PetscReal      u1,u2,nf1,nf2,njac11,njac12,njac21,njac22;
  PetscReal      b=-0.5, t=0.5, l=-0.5, r=0.5;
  PetscReal      *boundary;
  PetscBool      flg;

  PetscFunctionBegin;
  /* Get local mesh boundaries */
  PetscCall(DMDAGetCorners(user->dm,&xs,&ys,NULL,&xm,&ym,NULL));
  PetscCall(DMDAGetGhostCorners(user->dm,&gxs,&gys,NULL,&gxm,&gym,NULL));

  bsize=xm+2;
  lsize=ym+2;
  rsize=ym+2;
  tsize=xm+2;

  PetscCall(PetscMalloc1(bsize,&user->bottom));
  PetscCall(PetscMalloc1(tsize,&user->top));
  PetscCall(PetscMalloc1(lsize,&user->left));
  PetscCall(PetscMalloc1(rsize,&user->right));

  hx= (r-l)/(mx+1); hy=(t-b)/(my+1);

  for (j=0; j<4; j++) {
    if (j==0) {
      yt=b;
      xt=l+hx*xs;
      limit=bsize;
      boundary=user->bottom;
    } else if (j==1) {
      yt=t;
      xt=l+hx*xs;
      limit=tsize;
      boundary=user->top;
    } else if (j==2) {
      yt=b+hy*ys;
      xt=l;
      limit=lsize;
      boundary=user->left;
    } else { /* if (j==3) */
      yt=b+hy*ys;
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
        fnorm=PetscSqrtReal(nf1*nf1+nf2*nf2);
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

  /* Scale the boundary if desired */
  if (1==1) {
    PetscReal scl = 1.0;

    PetscCall(PetscOptionsGetReal(NULL,NULL,"-bottom",&scl,&flg));
    if (flg) {
      for (i=0;i<bsize;i++) user->bottom[i]*=scl;
    }

    PetscCall(PetscOptionsGetReal(NULL,NULL,"-top",&scl,&flg));
    if (flg) {
      for (i=0;i<tsize;i++) user->top[i]*=scl;
    }

    PetscCall(PetscOptionsGetReal(NULL,NULL,"-right",&scl,&flg));
    if (flg) {
      for (i=0;i<rsize;i++) user->right[i]*=scl;
    }

    PetscCall(PetscOptionsGetReal(NULL,NULL,"-left",&scl,&flg));
    if (flg) {
      for (i=0;i<lsize;i++) user->left[i]*=scl;
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
  PetscInt       start2=-1,i,j;
  PetscReal      start1=0;
  PetscBool      flg1,flg2;

  PetscFunctionBegin;
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-start",&start1,&flg1));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-random",&start2,&flg2));

  if (flg1) { /* The zero vector is reasonable */

    PetscCall(VecSet(X, start1));

  } else if (flg2 && start2>0) { /* Try a random start between -0.5 and 0.5 */
    PetscRandom rctx;  PetscReal np5=-0.5;

    PetscCall(PetscRandomCreate(PETSC_COMM_WORLD,&rctx));
    PetscCall(VecSetRandom(X, rctx));
    PetscCall(PetscRandomDestroy(&rctx));
    PetscCall(VecShift(X, np5));

  } else { /* Take an average of the boundary conditions */
    PetscInt  xs,xm,ys,ym;
    PetscInt  mx=user->mx,my=user->my;
    PetscReal **x;

    /* Get local mesh boundaries */
    PetscCall(DMDAGetCorners(user->dm,&xs,&ys,NULL,&xm,&ym,NULL));

    /* Get pointers to vector data */
    PetscCall(DMDAVecGetArray(user->dm,X,(void**)&x));

    /* Perform local computations */
    for (j=ys; j<ys+ym; j++) {
      for (i=xs; i< xs+xm; i++) {
        x[j][i] = (((j+1)*user->bottom[i-xs+1]+(my-j+1)*user->top[i-xs+1])/(my+2)+((i+1)*user->left[j-ys+1]+(mx-i+1)*user->right[j-ys+1])/(mx+2))/2.0;
      }
    }
    PetscCall(DMDAVecRestoreArray(user->dm,X,(void**)&x));
    PetscCall(PetscLogFlops(9.0*xm*ym));
  }
  PetscFunctionReturn(0);
}

/*-----------------------------------------------------------------------*/
PetscErrorCode My_Monitor(Tao tao, void *ctx)
{
  Vec            X;

  PetscFunctionBegin;
  PetscCall(TaoGetSolution(tao,&X));
  PetscCall(VecView(X,PETSC_VIEWER_STDOUT_WORLD));
  PetscFunctionReturn(0);
}

/*TEST

   build:
      requires: !complex

   test:
      args: -tao_smonitor -tao_type lmvm -mx 10 -my 8 -tao_gatol 1.e-3
      requires: !single

   test:
      suffix: 2
      nsize: 2
      args: -tao_smonitor -tao_type nls -tao_nls_ksp_max_it 15 -tao_gatol 1.e-4
      filter: grep -v "nls ksp"
      requires: !single

   test:
      suffix: 3
      nsize: 3
      args: -tao_smonitor -tao_type cg -tao_cg_type fr -mx 10 -my 10 -tao_gatol 1.e-3
      requires: !single

   test:
      suffix: 5
      nsize: 2
      args: -tao_smonitor -tao_type bmrm -mx 10 -my 8 -tao_gatol 1.e-3
      requires: !single

TEST*/
