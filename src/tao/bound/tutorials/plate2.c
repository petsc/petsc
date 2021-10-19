#include <petscdmda.h>
#include <petsctao.h>

static  char help[] =
"This example demonstrates use of the TAO package to \n\
solve an unconstrained minimization problem.  This example is based on a \n\
problem from the MINPACK-2 test suite.  Given a rectangular 2-D domain, \n\
boundary values along the edges of the domain, and a plate represented by \n\
lower boundary conditions, the objective is to find the\n\
surface with the minimal area that satisfies the boundary conditions.\n\
The command line options are:\n\
  -mx <xg>, where <xg> = number of grid points in the 1st coordinate direction\n\
  -my <yg>, where <yg> = number of grid points in the 2nd coordinate direction\n\
  -bmx <bxg>, where <bxg> = number of grid points under plate in 1st direction\n\
  -bmy <byg>, where <byg> = number of grid points under plate in 2nd direction\n\
  -bheight <ht>, where <ht> = height of the plate\n\
  -start <st>, where <st> =0 for zero vector, <st> >0 for random start, and <st> <0 \n\
               for an average of the boundary conditions\n\n";

/*T
   Concepts: TAO^Solving a bound constrained minimization problem
   Routines: TaoCreate();
   Routines: TaoSetType(); TaoSetObjectiveAndGradientRoutine();
   Routines: TaoSetHessianRoutine();
   Routines: TaoSetInitialVector();
   Routines: TaoSetVariableBounds();
   Routines: TaoSetFromOptions();
   Routines: TaoSolve(); TaoView();
   Routines: TaoDestroy();
   Processors: n
T*/

/*
   User-defined application context - contains data needed by the
   application-provided call-back routines, FormFunctionGradient(),
   FormHessian().
*/
typedef struct {
  /* problem parameters */
  PetscReal      bheight;                  /* Height of plate under the surface */
  PetscInt       mx, my;                   /* discretization in x, y directions */
  PetscInt       bmx,bmy;                  /* Size of plate under the surface */
  Vec            Bottom, Top, Left, Right; /* boundary values */

  /* Working space */
  Vec         localX, localV;           /* ghosted local vector */
  DM          dm;                       /* distributed array data structure */
  Mat         H;
} AppCtx;

/* -------- User-defined Routines --------- */

static PetscErrorCode MSA_BoundaryConditions(AppCtx*);
static PetscErrorCode MSA_InitialPoint(AppCtx*,Vec);
static PetscErrorCode MSA_Plate(Vec,Vec,void*);
PetscErrorCode FormFunctionGradient(Tao,Vec,PetscReal*,Vec,void*);
PetscErrorCode FormHessian(Tao,Vec,Mat,Mat,void*);

/* For testing matrix free submatrices */
PetscErrorCode MatrixFreeHessian(Tao,Vec,Mat, Mat,void*);
PetscErrorCode MyMatMult(Mat,Vec,Vec);

int main(int argc, char **argv)
{
  PetscErrorCode         ierr;                 /* used to check for functions returning nonzeros */
  PetscInt               Nx, Ny;               /* number of processors in x- and y- directions */
  PetscInt               m, N;                 /* number of local and global elements in vectors */
  Vec                    x,xl,xu;               /* solution vector  and bounds*/
  PetscBool              flg;                /* A return variable when checking for user options */
  Tao                    tao;                  /* Tao solver context */
  ISLocalToGlobalMapping isltog;   /* local-to-global mapping object */
  Mat                    H_shell;                  /* to test matrix-free submatrices */
  AppCtx                 user;                 /* user-defined work context */

  /* Initialize PETSc, TAO */
  ierr = PetscInitialize(&argc, &argv,(char *)0,help);if (ierr) return ierr;

  /* Specify default dimension of the problem */
  user.mx = 10; user.my = 10; user.bheight=0.1;

  /* Check for any command line arguments that override defaults */
  ierr = PetscOptionsGetInt(NULL,NULL,"-mx",&user.mx,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-my",&user.my,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-bheight",&user.bheight,&flg);CHKERRQ(ierr);

  user.bmx = user.mx/2; user.bmy = user.my/2;
  ierr = PetscOptionsGetInt(NULL,NULL,"-bmx",&user.bmx,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-bmy",&user.bmy,&flg);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n---- Minimum Surface Area With Plate Problem -----\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"mx:%D, my:%D, bmx:%D, bmy:%D, height:%g\n",user.mx,user.my,user.bmx,user.bmy,(double)user.bheight);CHKERRQ(ierr);

  /* Calculate any derived values from parameters */
  N    = user.mx*user.my;

  /* Let Petsc determine the dimensions of the local vectors */
  Nx = PETSC_DECIDE; Ny = PETSC_DECIDE;

  /*
     A two dimensional distributed array will help define this problem,
     which derives from an elliptic PDE on two dimensional domain.  From
     the distributed array, Create the vectors.
  */
  ierr = DMDACreate2d(MPI_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,user.mx,user.my,Nx,Ny,1,1,NULL,NULL,&user.dm);CHKERRQ(ierr);
  ierr = DMSetFromOptions(user.dm);CHKERRQ(ierr);
  ierr = DMSetUp(user.dm);CHKERRQ(ierr);
  /*
     Extract global and local vectors from DM; The local vectors are
     used solely as work space for the evaluation of the function,
     gradient, and Hessian.  Duplicate for remaining vectors that are
     the same types.
  */
  ierr = DMCreateGlobalVector(user.dm,&x);CHKERRQ(ierr); /* Solution */
  ierr = DMCreateLocalVector(user.dm,&user.localX);CHKERRQ(ierr);
  ierr = VecDuplicate(user.localX,&user.localV);CHKERRQ(ierr);

  ierr = VecDuplicate(x,&xl);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&xu);CHKERRQ(ierr);

  /* The TAO code begins here */

  /*
     Create TAO solver and set desired solution method
     The method must either be TAOTRON or TAOBLMVM
     If TAOBLMVM is used, then hessian function is not called.
  */
  ierr = TaoCreate(PETSC_COMM_WORLD,&tao);CHKERRQ(ierr);
  ierr = TaoSetType(tao,TAOBLMVM);CHKERRQ(ierr);

  /* Set initial solution guess; */
  ierr = MSA_BoundaryConditions(&user);CHKERRQ(ierr);
  ierr = MSA_InitialPoint(&user,x);CHKERRQ(ierr);
  ierr = TaoSetInitialVector(tao,x);CHKERRQ(ierr);

  /* Set routines for function, gradient and hessian evaluation */
  ierr = TaoSetObjectiveAndGradientRoutine(tao,FormFunctionGradient,(void*) &user);CHKERRQ(ierr);

  ierr = VecGetLocalSize(x,&m);CHKERRQ(ierr);
  ierr = MatCreateAIJ(MPI_COMM_WORLD,m,m,N,N,7,NULL,3,NULL,&(user.H));CHKERRQ(ierr);
  ierr = MatSetOption(user.H,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);

  ierr = DMGetLocalToGlobalMapping(user.dm,&isltog);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMapping(user.H,isltog,isltog);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,NULL,"-matrixfree",&flg);CHKERRQ(ierr);
  if (flg) {
      ierr = MatCreateShell(PETSC_COMM_WORLD,m,m,N,N,(void*)&user,&H_shell);CHKERRQ(ierr);
      ierr = MatShellSetOperation(H_shell,MATOP_MULT,(void(*)(void))MyMatMult);CHKERRQ(ierr);
      ierr = MatSetOption(H_shell,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
      ierr = TaoSetHessianRoutine(tao,H_shell,H_shell,MatrixFreeHessian,(void*)&user);CHKERRQ(ierr);
  } else {
      ierr = TaoSetHessianRoutine(tao,user.H,user.H,FormHessian,(void*)&user);CHKERRQ(ierr);
  }

  /* Set Variable bounds */
  ierr = MSA_Plate(xl,xu,(void*)&user);CHKERRQ(ierr);
  ierr = TaoSetVariableBounds(tao,xl,xu);CHKERRQ(ierr);

  /* Check for any tao command line options */
  ierr = TaoSetFromOptions(tao);CHKERRQ(ierr);

  /* SOLVE THE APPLICATION */
  ierr = TaoSolve(tao);CHKERRQ(ierr);

  ierr = TaoView(tao,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* Free TAO data structures */
  ierr = TaoDestroy(&tao);CHKERRQ(ierr);

  /* Free PETSc data structures */
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&xl);CHKERRQ(ierr);
  ierr = VecDestroy(&xu);CHKERRQ(ierr);
  ierr = MatDestroy(&user.H);CHKERRQ(ierr);
  ierr = VecDestroy(&user.localX);CHKERRQ(ierr);
  ierr = VecDestroy(&user.localV);CHKERRQ(ierr);
  ierr = VecDestroy(&user.Bottom);CHKERRQ(ierr);
  ierr = VecDestroy(&user.Top);CHKERRQ(ierr);
  ierr = VecDestroy(&user.Left);CHKERRQ(ierr);
  ierr = VecDestroy(&user.Right);CHKERRQ(ierr);
  ierr = DMDestroy(&user.dm);CHKERRQ(ierr);
  if (flg) {
    ierr = MatDestroy(&H_shell);CHKERRQ(ierr);
  }
  ierr = PetscFinalize();
  return ierr;
}

/*  FormFunctionGradient - Evaluates f(x) and gradient g(x).

    Input Parameters:
.   tao     - the Tao context
.   X      - input vector
.   userCtx - optional user-defined context, as set by TaoSetObjectiveAndGradientRoutine()

    Output Parameters:
.   fcn     - the function value
.   G      - vector containing the newly evaluated gradient

   Notes:
   In this case, we discretize the domain and Create triangles. The
   surface of each triangle is planar, whose surface area can be easily
   computed.  The total surface area is found by sweeping through the grid
   and computing the surface area of the two triangles that have their
   right angle at the grid point.  The diagonal line segments on the
   grid that define the triangles run from top left to lower right.
   The numbering of points starts at the lower left and runs left to
   right, then bottom to top.
*/
PetscErrorCode FormFunctionGradient(Tao tao, Vec X, PetscReal *fcn, Vec G,void *userCtx)
{
  AppCtx         *user = (AppCtx *) userCtx;
  PetscErrorCode ierr;
  PetscInt       i,j,row;
  PetscInt       mx=user->mx, my=user->my;
  PetscInt       xs,xm,gxs,gxm,ys,ym,gys,gym;
  PetscReal      ft=0;
  PetscReal      zero=0.0;
  PetscReal      hx=1.0/(mx+1),hy=1.0/(my+1), hydhx=hy/hx, hxdhy=hx/hy, area=0.5*hx*hy;
  PetscReal      rhx=mx+1, rhy=my+1;
  PetscReal      f1,f2,f3,f4,f5,f6,d1,d2,d3,d4,d5,d6,d7,d8,xc,xl,xr,xt,xb,xlt,xrb;
  PetscReal      df1dxc,df2dxc,df3dxc,df4dxc,df5dxc,df6dxc;
  PetscReal      *g, *x,*left,*right,*bottom,*top;
  Vec            localX = user->localX, localG = user->localV;

  /* Get local mesh boundaries */
  ierr = DMDAGetCorners(user->dm,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(user->dm,&gxs,&gys,NULL,&gxm,&gym,NULL);CHKERRQ(ierr);

  /* Scatter ghost points to local vector */
  ierr = DMGlobalToLocalBegin(user->dm,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(user->dm,X,INSERT_VALUES,localX);CHKERRQ(ierr);

  /* Initialize vector to zero */
  ierr = VecSet(localG, zero);CHKERRQ(ierr);

  /* Get pointers to vector data */
  ierr = VecGetArray(localX,&x);CHKERRQ(ierr);
  ierr = VecGetArray(localG,&g);CHKERRQ(ierr);
  ierr = VecGetArray(user->Top,&top);CHKERRQ(ierr);
  ierr = VecGetArray(user->Bottom,&bottom);CHKERRQ(ierr);
  ierr = VecGetArray(user->Left,&left);CHKERRQ(ierr);
  ierr = VecGetArray(user->Right,&right);CHKERRQ(ierr);

  /* Compute function over the locally owned part of the mesh */
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i< xs+xm; i++) {
      row=(j-gys)*gxm + (i-gxs);

      xc = x[row];
      xlt=xrb=xl=xr=xb=xt=xc;

      if (i==0) { /* left side */
        xl= left[j-ys+1];
        xlt = left[j-ys+2];
      } else {
        xl = x[row-1];
      }

      if (j==0) { /* bottom side */
        xb=bottom[i-xs+1];
        xrb = bottom[i-xs+2];
      } else {
        xb = x[row-gxm];
      }

      if (i+1 == gxs+gxm) { /* right side */
        xr=right[j-ys+1];
        xrb = right[j-ys];
      } else {
        xr = x[row+1];
      }

      if (j+1==gys+gym) { /* top side */
        xt=top[i-xs+1];
        xlt = top[i-xs];
      }else {
        xt = x[row+gxm];
      }

      if (i>gxs && j+1<gys+gym) {
        xlt = x[row-1+gxm];
      }
      if (j>gys && i+1<gxs+gxm) {
        xrb = x[row+1-gxm];
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

      g[row] = (df1dxc+df2dxc+df3dxc+df4dxc+df5dxc+df6dxc) * 0.5;

    }
  }

  /* Compute triangular areas along the border of the domain. */
  if (xs==0) { /* left side */
    for (j=ys; j<ys+ym; j++) {
      d3=(left[j-ys+1] - left[j-ys+2])*rhy;
      d2=(left[j-ys+1] - x[(j-gys)*gxm])*rhx;
      ft = ft+PetscSqrtScalar(1.0 + d3*d3 + d2*d2);
    }
  }
  if (ys==0) { /* bottom side */
    for (i=xs; i<xs+xm; i++) {
      d2=(bottom[i+1-xs]-bottom[i-xs+2])*rhx;
      d3=(bottom[i-xs+1]-x[i-gxs])*rhy;
      ft = ft+PetscSqrtScalar(1.0 + d3*d3 + d2*d2);
    }
  }

  if (xs+xm==mx) { /* right side */
    for (j=ys; j< ys+ym; j++) {
      d1=(x[(j+1-gys)*gxm-1]-right[j-ys+1])*rhx;
      d4=(right[j-ys]-right[j-ys+1])*rhy;
      ft = ft+PetscSqrtScalar(1.0 + d1*d1 + d4*d4);
    }
  }
  if (ys+ym==my) { /* top side */
    for (i=xs; i<xs+xm; i++) {
      d1=(x[(gym-1)*gxm + i-gxs] - top[i-xs+1])*rhy;
      d4=(top[i-xs+1] - top[i-xs])*rhx;
      ft = ft+PetscSqrtScalar(1.0 + d1*d1 + d4*d4);
    }
  }

  if (ys==0 && xs==0) {
    d1=(left[0]-left[1])*rhy;
    d2=(bottom[0]-bottom[1])*rhx;
    ft +=PetscSqrtScalar(1.0 + d1*d1 + d2*d2);
  }
  if (ys+ym == my && xs+xm == mx) {
    d1=(right[ym+1] - right[ym])*rhy;
    d2=(top[xm+1] - top[xm])*rhx;
    ft +=PetscSqrtScalar(1.0 + d1*d1 + d2*d2);
  }

  ft=ft*area;
  ierr = MPI_Allreduce(&ft,fcn,1,MPIU_REAL,MPIU_SUM,MPI_COMM_WORLD);CHKERRMPI(ierr);

  /* Restore vectors */
  ierr = VecRestoreArray(localX,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(localG,&g);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->Left,&left);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->Top,&top);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->Bottom,&bottom);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->Right,&right);CHKERRQ(ierr);

  /* Scatter values to global vector */
  ierr = DMLocalToGlobalBegin(user->dm,localG,INSERT_VALUES,G);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(user->dm,localG,INSERT_VALUES,G);CHKERRQ(ierr);

  ierr = PetscLogFlops(70.0*xm*ym);CHKERRQ(ierr);

  return 0;
}

/* ------------------------------------------------------------------- */
/*
   FormHessian - Evaluates Hessian matrix.

   Input Parameters:
.  tao  - the Tao context
.  x    - input vector
.  ptr  - optional user-defined context, as set by TaoSetHessianRoutine()

   Output Parameters:
.  A    - Hessian matrix
.  B    - optionally different preconditioning matrix

   Notes:
   Due to mesh point reordering with DMs, we must always work
   with the local mesh points, and then transform them to the new
   global numbering with the local-to-global mapping.  We cannot work
   directly with the global numbers for the original uniprocessor mesh!

   Two methods are available for imposing this transformation
   when setting matrix entries:
     (A) MatSetValuesLocal(), using the local ordering (including
         ghost points!)
         - Do the following two steps once, before calling TaoSolve()
           - Use DMGetISLocalToGlobalMapping() to extract the
             local-to-global map from the DM
           - Associate this map with the matrix by calling
             MatSetLocalToGlobalMapping()
         - Then set matrix entries using the local ordering
           by calling MatSetValuesLocal()
     (B) MatSetValues(), using the global ordering
         - Use DMGetGlobalIndices() to extract the local-to-global map
         - Then apply this map explicitly yourself
         - Set matrix entries using the global ordering by calling
           MatSetValues()
   Option (A) seems cleaner/easier in many cases, and is the procedure
   used in this example.
*/
PetscErrorCode FormHessian(Tao tao,Vec X,Mat Hptr, Mat Hessian, void *ptr)
{
  PetscErrorCode ierr;
  AppCtx         *user = (AppCtx *) ptr;
  PetscInt       i,j,k,row;
  PetscInt       mx=user->mx, my=user->my;
  PetscInt       xs,xm,gxs,gxm,ys,ym,gys,gym,col[7];
  PetscReal      hx=1.0/(mx+1), hy=1.0/(my+1), hydhx=hy/hx, hxdhy=hx/hy;
  PetscReal      rhx=mx+1, rhy=my+1;
  PetscReal      f1,f2,f3,f4,f5,f6,d1,d2,d3,d4,d5,d6,d7,d8,xc,xl,xr,xt,xb,xlt,xrb;
  PetscReal      hl,hr,ht,hb,hc,htl,hbr;
  PetscReal      *x,*left,*right,*bottom,*top;
  PetscReal      v[7];
  Vec            localX = user->localX;
  PetscBool      assembled;

  /* Set various matrix options */
  ierr = MatSetOption(Hessian,MAT_IGNORE_OFF_PROC_ENTRIES,PETSC_TRUE);CHKERRQ(ierr);

  /* Initialize matrix entries to zero */
  ierr = MatAssembled(Hessian,&assembled);CHKERRQ(ierr);
  if (assembled) {ierr = MatZeroEntries(Hessian);CHKERRQ(ierr);}

  /* Get local mesh boundaries */
  ierr = DMDAGetCorners(user->dm,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(user->dm,&gxs,&gys,NULL,&gxm,&gym,NULL);CHKERRQ(ierr);

  /* Scatter ghost points to local vector */
  ierr = DMGlobalToLocalBegin(user->dm,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(user->dm,X,INSERT_VALUES,localX);CHKERRQ(ierr);

  /* Get pointers to vector data */
  ierr = VecGetArray(localX,&x);CHKERRQ(ierr);
  ierr = VecGetArray(user->Top,&top);CHKERRQ(ierr);
  ierr = VecGetArray(user->Bottom,&bottom);CHKERRQ(ierr);
  ierr = VecGetArray(user->Left,&left);CHKERRQ(ierr);
  ierr = VecGetArray(user->Right,&right);CHKERRQ(ierr);

  /* Compute Hessian over the locally owned part of the mesh */

  for (i=xs; i< xs+xm; i++) {

    for (j=ys; j<ys+ym; j++) {

      row=(j-gys)*gxm + (i-gxs);

      xc = x[row];
      xlt=xrb=xl=xr=xb=xt=xc;

      /* Left side */
      if (i==gxs) {
        xl= left[j-ys+1];
        xlt = left[j-ys+2];
      } else {
        xl = x[row-1];
      }

      if (j==gys) {
        xb=bottom[i-xs+1];
        xrb = bottom[i-xs+2];
      } else {
        xb = x[row-gxm];
      }

      if (i+1 == gxs+gxm) {
        xr=right[j-ys+1];
        xrb = right[j-ys];
      } else {
        xr = x[row+1];
      }

      if (j+1==gys+gym) {
        xt=top[i-xs+1];
        xlt = top[i-xs];
      }else {
        xt = x[row+gxm];
      }

      if (i>gxs && j+1<gys+gym) {
        xlt = x[row-1+gxm];
      }
      if (j>gys && i+1<gxs+gxm) {
        xrb = x[row+1-gxm];
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

      hl*=0.5; hr*=0.5; ht*=0.5; hb*=0.5; hbr*=0.5; htl*=0.5;  hc*=0.5;

      k=0;
      if (j>0) {
        v[k]=hb; col[k]=row - gxm; k++;
      }

      if (j>0 && i < mx -1) {
        v[k]=hbr; col[k]=row - gxm+1; k++;
      }

      if (i>0) {
        v[k]= hl; col[k]=row - 1; k++;
      }

      v[k]= hc; col[k]=row; k++;

      if (i < mx-1) {
        v[k]= hr; col[k]=row+1; k++;
      }

      if (i>0 && j < my-1) {
        v[k]= htl; col[k] = row+gxm-1; k++;
      }

      if (j < my-1) {
        v[k]= ht; col[k] = row+gxm; k++;
      }

      /*
         Set matrix values using local numbering, which was defined
         earlier, in the main routine.
      */
      ierr = MatSetValuesLocal(Hessian,1,&row,k,col,v,INSERT_VALUES);CHKERRQ(ierr);

    }
  }

  /* Restore vectors */
  ierr = VecRestoreArray(localX,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->Left,&left);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->Top,&top);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->Bottom,&bottom);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->Right,&right);CHKERRQ(ierr);

  /* Assemble the matrix */
  ierr = MatAssemblyBegin(Hessian,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Hessian,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = PetscLogFlops(199.0*xm*ym);CHKERRQ(ierr);
  return 0;
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
  int        ierr;
  PetscInt   i,j,k,maxits=5,limit=0;
  PetscInt   xs,ys,xm,ym,gxs,gys,gxm,gym;
  PetscInt   mx=user->mx,my=user->my;
  PetscInt   bsize=0, lsize=0, tsize=0, rsize=0;
  PetscReal  one=1.0, two=2.0, three=3.0, scl=1.0, tol=1e-10;
  PetscReal  fnorm,det,hx,hy,xt=0,yt=0;
  PetscReal  u1,u2,nf1,nf2,njac11,njac12,njac21,njac22;
  PetscReal  b=-0.5, t=0.5, l=-0.5, r=0.5;
  PetscReal  *boundary;
  PetscBool  flg;
  Vec        Bottom,Top,Right,Left;

  /* Get local mesh boundaries */
  ierr = DMDAGetCorners(user->dm,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(user->dm,&gxs,&gys,NULL,&gxm,&gym,NULL);CHKERRQ(ierr);

  bsize=xm+2;
  lsize=ym+2;
  rsize=ym+2;
  tsize=xm+2;

  ierr = VecCreateMPI(MPI_COMM_WORLD,bsize,PETSC_DECIDE,&Bottom);CHKERRQ(ierr);
  ierr = VecCreateMPI(MPI_COMM_WORLD,tsize,PETSC_DECIDE,&Top);CHKERRQ(ierr);
  ierr = VecCreateMPI(MPI_COMM_WORLD,lsize,PETSC_DECIDE,&Left);CHKERRQ(ierr);
  ierr = VecCreateMPI(MPI_COMM_WORLD,rsize,PETSC_DECIDE,&Right);CHKERRQ(ierr);

  user->Top=Top;
  user->Left=Left;
  user->Bottom=Bottom;
  user->Right=Right;

  hx= (r-l)/(mx+1); hy=(t-b)/(my+1);

  for (j=0; j<4; j++) {
    if (j==0) {
      yt=b;
      xt=l+hx*xs;
      limit=bsize;
      VecGetArray(Bottom,&boundary);
    } else if (j==1) {
      yt=t;
      xt=l+hx*xs;
      limit=tsize;
      VecGetArray(Top,&boundary);
    } else if (j==2) {
      yt=b+hy*ys;
      xt=l;
      limit=lsize;
      VecGetArray(Left,&boundary);
    } else if (j==3) {
      yt=b+hy*ys;
      xt=r;
      limit=rsize;
      VecGetArray(Right,&boundary);
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
      } else if (j==2 || j==3) {
        yt=yt+hy;
      }
    }
    if (j==0) {
      ierr = VecRestoreArray(Bottom,&boundary);CHKERRQ(ierr);
    } else if (j==1) {
      ierr = VecRestoreArray(Top,&boundary);CHKERRQ(ierr);
    } else if (j==2) {
      ierr = VecRestoreArray(Left,&boundary);CHKERRQ(ierr);
    } else if (j==3) {
      ierr = VecRestoreArray(Right,&boundary);CHKERRQ(ierr);
    }
  }

  /* Scale the boundary if desired */

  ierr = PetscOptionsGetReal(NULL,NULL,"-bottom",&scl,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = VecScale(Bottom, scl);CHKERRQ(ierr);
  }
  ierr = PetscOptionsGetReal(NULL,NULL,"-top",&scl,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = VecScale(Top, scl);CHKERRQ(ierr);
  }
  ierr = PetscOptionsGetReal(NULL,NULL,"-right",&scl,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = VecScale(Right, scl);CHKERRQ(ierr);
  }

  ierr = PetscOptionsGetReal(NULL,NULL,"-left",&scl,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = VecScale(Left, scl);CHKERRQ(ierr);
  }
  return 0;
}

/* ------------------------------------------------------------------- */
/*
   MSA_Plate -  Calculates an obstacle for surface to stretch over.

   Input Parameter:
.  user - user-defined application context

   Output Parameter:
.  user - user-defined application context
*/
static PetscErrorCode MSA_Plate(Vec XL,Vec XU,void *ctx)
{
  AppCtx         *user=(AppCtx *)ctx;
  PetscErrorCode ierr;
  PetscInt       i,j,row;
  PetscInt       xs,ys,xm,ym;
  PetscInt       mx=user->mx, my=user->my, bmy, bmx;
  PetscReal      t1,t2,t3;
  PetscReal      *xl, lb=PETSC_NINFINITY, ub=PETSC_INFINITY;
  PetscBool      cylinder;

  user->bmy = PetscMax(0,user->bmy);user->bmy = PetscMin(my,user->bmy);
  user->bmx = PetscMax(0,user->bmx);user->bmx = PetscMin(mx,user->bmx);
  bmy=user->bmy; bmx=user->bmx;

  ierr = DMDAGetCorners(user->dm,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);

  ierr = VecSet(XL, lb);CHKERRQ(ierr);
  ierr = VecSet(XU, ub);CHKERRQ(ierr);

  ierr = VecGetArray(XL,&xl);CHKERRQ(ierr);

  ierr = PetscOptionsHasName(NULL,NULL,"-cylinder",&cylinder);CHKERRQ(ierr);
  /* Compute the optional lower box */
  if (cylinder) {
    for (i=xs; i< xs+xm; i++) {
      for (j=ys; j<ys+ym; j++) {
        row=(j-ys)*xm + (i-xs);
        t1=(2.0*i-mx)*bmy;
        t2=(2.0*j-my)*bmx;
        t3=bmx*bmx*bmy*bmy;
        if (t1*t1 + t2*t2 <= t3) {
          xl[row] = user->bheight;
        }
      }
    }
  } else {
    /* Compute the optional lower box */
    for (i=xs; i< xs+xm; i++) {
      for (j=ys; j<ys+ym; j++) {
        row=(j-ys)*xm + (i-xs);
        if (i>=(mx-bmx)/2 && i<mx-(mx-bmx)/2 &&
            j>=(my-bmy)/2 && j<my-(my-bmy)/2) {
          xl[row] = user->bheight;
        }
      }
    }
  }
    ierr = VecRestoreArray(XL,&xl);CHKERRQ(ierr);

  return 0;
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
  PetscErrorCode ierr;
  PetscInt       start=-1,i,j;
  PetscReal      zero=0.0;
  PetscBool      flg;

  ierr = PetscOptionsGetInt(NULL,NULL,"-start",&start,&flg);CHKERRQ(ierr);
  if (flg && start==0) { /* The zero vector is reasonable */
    ierr = VecSet(X, zero);CHKERRQ(ierr);
  } else if (flg && start>0) { /* Try a random start between -0.5 and 0.5 */
    PetscRandom rctx;  PetscReal np5=-0.5;

    ierr = PetscRandomCreate(MPI_COMM_WORLD,&rctx);CHKERRQ(ierr);
    for (i=0; i<start; i++) {
      ierr = VecSetRandom(X, rctx);CHKERRQ(ierr);
    }
    ierr = PetscRandomDestroy(&rctx);CHKERRQ(ierr);
    ierr = VecShift(X, np5);CHKERRQ(ierr);

  } else { /* Take an average of the boundary conditions */

    PetscInt row,xs,xm,gxs,gxm,ys,ym,gys,gym;
    PetscInt mx=user->mx,my=user->my;
    PetscReal *x,*left,*right,*bottom,*top;
    Vec    localX = user->localX;

    /* Get local mesh boundaries */
    ierr = DMDAGetCorners(user->dm,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);
    ierr = DMDAGetGhostCorners(user->dm,&gxs,&gys,NULL,&gxm,&gym,NULL);CHKERRQ(ierr);

    /* Get pointers to vector data */
    ierr = VecGetArray(user->Top,&top);CHKERRQ(ierr);
    ierr = VecGetArray(user->Bottom,&bottom);CHKERRQ(ierr);
    ierr = VecGetArray(user->Left,&left);CHKERRQ(ierr);
    ierr = VecGetArray(user->Right,&right);CHKERRQ(ierr);

    ierr = VecGetArray(localX,&x);CHKERRQ(ierr);
    /* Perform local computations */
    for (j=ys; j<ys+ym; j++) {
      for (i=xs; i< xs+xm; i++) {
        row=(j-gys)*gxm + (i-gxs);
        x[row] = ((j+1)*bottom[i-xs+1]/my + (my-j+1)*top[i-xs+1]/(my+2)+(i+1)*left[j-ys+1]/mx + (mx-i+1)*right[j-ys+1]/(mx+2))/2.0;
      }
    }

    /* Restore vectors */
    ierr = VecRestoreArray(localX,&x);CHKERRQ(ierr);

    ierr = VecRestoreArray(user->Left,&left);CHKERRQ(ierr);
    ierr = VecRestoreArray(user->Top,&top);CHKERRQ(ierr);
    ierr = VecRestoreArray(user->Bottom,&bottom);CHKERRQ(ierr);
    ierr = VecRestoreArray(user->Right,&right);CHKERRQ(ierr);

    /* Scatter values into global vector */
    ierr = DMLocalToGlobalBegin(user->dm,localX,INSERT_VALUES,X);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(user->dm,localX,INSERT_VALUES,X);CHKERRQ(ierr);

  }
  return 0;
}

/* For testing matrix free submatrices */
PetscErrorCode MatrixFreeHessian(Tao tao, Vec x, Mat H, Mat Hpre, void *ptr)
{
  PetscErrorCode ierr;
  AppCtx         *user = (AppCtx*)ptr;
  PetscFunctionBegin;
  ierr = FormHessian(tao,x,user->H,user->H,ptr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
PetscErrorCode MyMatMult(Mat H_shell, Vec X, Vec Y)
{
  PetscErrorCode ierr;
  void           *ptr;
  AppCtx         *user;
  PetscFunctionBegin;
  ierr = MatShellGetContext(H_shell,&ptr);CHKERRQ(ierr);
  user = (AppCtx*)ptr;
  ierr = MatMult(user->H,X,Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*TEST

   build:
      requires: !complex

   test:
      args: -tao_smonitor -mx 8 -my 6 -bmx 3 -bmy 3 -bheight 0.2 -tao_type tron -tao_gatol 1.e-5
      requires: !single

   test:
      suffix: 2
      nsize: 2
      args: -tao_smonitor -mx 8 -my 8 -bmx 2 -bmy 5 -bheight 0.3 -tao_type blmvm -tao_gatol 1.e-4
      requires: !single

   test:
      suffix: 3
      nsize: 3
      args: -tao_smonitor -mx 8 -my 12 -bmx 4 -bmy 10 -bheight 0.1 -tao_type tron -tao_gatol 1.e-5
      requires: !single

   test:
      suffix: 4
      nsize: 3
      args: -tao_smonitor -mx 8 -my 12 -bmx 4 -bmy 10 -bheight 0.1 -tao_subset_type mask -tao_type tron -tao_gatol 1.e-5
      requires: !single

   test:
      suffix: 5
      nsize: 3
      args: -tao_smonitor -mx 8 -my 12 -bmx 4 -bmy 10 -bheight 0.1 -tao_subset_type matrixfree -matrixfree -pc_type none -tao_type tron -tao_gatol 1.e-5
      requires: !single

   test:
      suffix: 6
      nsize: 3
      args: -tao_smonitor -mx 8 -my 12 -bmx 4 -bmy 10 -bheight 0.1 -tao_subset_type matrixfree -matrixfree -tao_type blmvm -tao_gatol 1.e-4
      requires: !single

   test:
      suffix: 7
      nsize: 3
      args: -tao_smonitor -mx 8 -my 12 -bmx 4 -bmy 10 -bheight 0.1 -tao_subset_type matrixfree -pc_type none -tao_type gpcg -tao_gatol 1.e-5
      requires: !single

   test:
      suffix: 8
      args: -tao_smonitor -mx 8 -my 6 -bmx 3 -bmy 3 -bheight 0.2 -tao_type bncg -tao_bncg_type gd -tao_gatol 1e-4
      requires: !single

   test:
      suffix: 9
      args: -tao_smonitor -mx 8 -my 6 -bmx 3 -bmy 3 -bheight 0.2 -tao_type bncg -tao_gatol 1e-4
      requires: !single

   test:
      suffix: 10
      args: -tao_smonitor -mx 8 -my 6 -bmx 3 -bmy 3 -bheight 0.2 -tao_type bnls -tao_gatol 1e-5
      requires: !single

   test:
      suffix: 11
      args: -tao_smonitor -mx 8 -my 6 -bmx 3 -bmy 3 -bheight 0.2 -tao_type bntr -tao_gatol 1e-5
      requires: !single

   test:
      suffix: 12
      args: -tao_smonitor -mx 8 -my 6 -bmx 3 -bmy 3 -bheight 0.2 -tao_type bntl -tao_gatol 1e-5
      requires: !single

   test:
      suffix: 13
      args: -tao_smonitor -mx 8 -my 6 -bmx 3 -bmy 3 -bheight 0.2 -tao_type bnls -tao_gatol 1e-5 -tao_bnk_max_cg_its 3
      requires: !single

   test:
      suffix: 14
      args: -tao_smonitor -mx 8 -my 6 -bmx 3 -bmy 3 -bheight 0.2 -tao_type bntr -tao_gatol 1e-5 -tao_bnk_max_cg_its 3
      requires: !single

   test:
      suffix: 15
      args: -tao_smonitor -mx 8 -my 6 -bmx 3 -bmy 3 -bheight 0.2 -tao_type bntl -tao_gatol 1e-5 -tao_bnk_max_cg_its 3
      requires: !single

   test:
     suffix: 16
     args: -tao_smonitor -mx 8 -my 8 -bmx 2 -bmy 5 -bheight 0.3 -tao_gatol 1e-4 -tao_type bqnls
     requires: !single

   test:
     suffix: 17
     args: -tao_smonitor -mx 8 -my 8 -bmx 2 -bmy 5 -bheight 0.3 -tao_gatol 1e-4 -tao_type bqnkls -tao_bqnk_mat_type lmvmbfgs
     requires: !single

   test:
     suffix: 18
     args: -tao_smonitor -mx 8 -my 6 -bmx 3 -bmy 3 -bheight 0.2 -tao_type bnls -tao_gatol 1e-5 -tao_mf_hessian
     requires: !single

   test:
     suffix: 19
     args: -tao_smonitor -mx 8 -my 6 -bmx 3 -bmy 3 -bheight 0.2 -tao_type bntr -tao_gatol 1e-5 -tao_mf_hessian
     requires: !single

   test:
     suffix: 20
     args: -tao_smonitor -mx 8 -my 6 -bmx 3 -bmy 3 -bheight 0.2 -tao_type bntl -tao_gatol 1e-5 -tao_mf_hessian

TEST*/
