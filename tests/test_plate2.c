/*$Id$*/
#include "stdlib.h"
#include "petscda.h"
#include "taosolver.h"

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
   Concepts: TAO - Solving a bound constrained minimization problem
   Routines: TaoInitialize(); TaoFinalize();
   Routines: TaoApplicationCreate(); TaoAppDestroy();
   Routines: TaoAppSetObjectiveAndGradientRoutine(); TaoAppSetHessianRoutine(); 
   Routines: TaoAppSetInitialSolutionVec(); TaoAppSetHessianMat();
   Routines: TaoAppSetVariableBounds();
   Routines: TaoCreate();  TaoDestroy(); 
   Routines: TaoSetOptions();
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
  Vec         Bottom, Top, Left, Right; /* boundary values */
  
  /* Working space */
  Vec         localX, localV;           /* ghosted local vector */
  DA          da;                       /* distributed array data structure */
  Mat         H;
} AppCtx;

/* -------- User-defined Routines --------- */

static PetscErrorCode MSA_BoundaryConditions(AppCtx*);
static PetscErrorCode MSA_InitialPoint(AppCtx*,Vec);
static PetscErrorCode MSA_Plate(Vec,Vec,void*);
PetscErrorCode FormFunctionGradient(TaoSolver,Vec,PetscReal*,Vec,void*);
PetscErrorCode FormHessian(TaoSolver,Vec,Mat*,Mat*,MatStructure*,void*);

#undef __FUNCT__
#define __FUNCT__ "main"
int main( int argc, char **argv )
{
  PetscErrorCode info;                 /* used to check for functions returning nonzeros */
  PetscInt   Nx, Ny;               /* number of processors in x- and y- directions */
  PetscInt   m, N;                 /* number of local and global elements in vectors */
  Vec        x,xl,xu;               /* solution vector  and bounds*/
  PetscBool   flg;                /* A return variable when checking for user options */
  TaoSolver  tao;                  /* TAO_SOLVER solver context */
  PetscReal  ff,gnorm,cnorm;       /* iteration information */
  PetscInt   iter;
  ISLocalToGlobalMapping isltog;   /* local-to-global mapping object */
  TaoSolverTerminationReason reason;
  AppCtx     user;                 /* user-defined work context */

  /* Initialize PETSc, TAO */
  PetscInitialize( &argc, &argv,(char *)0,help );
  TaoInitialize( &argc, &argv,(char *)0,help );

  /* Specify default dimension of the problem */
  user.mx = 10; user.my = 10; user.bheight=0.1;

  /* Check for any command line arguments that override defaults */
  info = PetscOptionsGetInt(PETSC_NULL,"-mx",&user.mx,&flg); CHKERRQ(info);
  info = PetscOptionsGetInt(PETSC_NULL,"-my",&user.my,&flg); CHKERRQ(info);
  info = PetscOptionsGetReal(PETSC_NULL,"-bheight",&user.bheight,&flg); CHKERRQ(info);

  user.bmx = user.mx/2; user.bmy = user.my/2;
  info = PetscOptionsGetInt(PETSC_NULL,"-bmx",&user.bmx,&flg); CHKERRQ(info);
  info = PetscOptionsGetInt(PETSC_NULL,"-bmy",&user.bmy,&flg); CHKERRQ(info);

  PetscPrintf(PETSC_COMM_WORLD,"\n---- Minimum Surface Area With Plate Problem -----\n");
  PetscPrintf(PETSC_COMM_WORLD,"mx:%d, my:%d, bmx:%d, bmy:%d, height:%4.2f\n",
	      user.mx,user.my,user.bmx,user.bmy,user.bheight);

  /* Calculate any derived values from parameters */
  N    = user.mx*user.my;

  /* Let Petsc determine the dimensions of the local vectors */
  Nx = PETSC_DECIDE; Ny = PETSC_DECIDE;

  /*
     A two dimensional distributed array will help define this problem,
     which derives from an elliptic PDE on two dimensional domain.  From
     the distributed array, Create the vectors.
  */
  info = DACreate2d(MPI_COMM_WORLD,DA_NONPERIODIC,DA_STENCIL_BOX,user.mx,
                    user.my,Nx,Ny,1,1,PETSC_NULL,PETSC_NULL,&user.da); CHKERRQ(info);

  /*
     Extract global and local vectors from DA; The local vectors are
     used solely as work space for the evaluation of the function, 
     gradient, and Hessian.  Duplicate for remaining vectors that are 
     the same types.
  */
  info = DACreateGlobalVector(user.da,&x); CHKERRQ(info); /* Solution */
  info = DACreateLocalVector(user.da,&user.localX); CHKERRQ(info);
  info = VecDuplicate(user.localX,&user.localV); CHKERRQ(info);

  info = VecDuplicate(x,&xl); CHKERRQ(info);
  info = VecDuplicate(x,&xu); CHKERRQ(info);

  /* 
     Create a matrix data structure to store the Hessian.
     Here we (optionally) also
     associate the local numbering scheme with the matrix so that
     later we can use local indices for matrix assembly.  We could
     alternatively use global indices for matrix assembly.
  */
  info = VecGetLocalSize(x,&m); CHKERRQ(info);
  info = MatCreateMPIAIJ(MPI_COMM_WORLD,m,m,N,N,7,PETSC_NULL,
                         3,PETSC_NULL,&(user.H)); CHKERRQ(info);
  info = MatSetOption(user.H,MAT_SYMMETRIC,PETSC_TRUE); CHKERRQ(info);

  info = DAGetISLocalToGlobalMapping(user.da,&isltog); CHKERRQ(info);
  info = MatSetLocalToGlobalMapping(user.H,isltog); CHKERRQ(info);


  /* The TAO code begins here */

  /* 
     Create TAO solver and set desired solution method 
     The method must either be 'tao_tron' or 'tao_blmvm'
     If blmvm is used, then hessian function is not called.
  */
  info = TaoSolverCreate(PETSC_COMM_WORLD,&tao); CHKERRQ(info);
  info = TaoSolverSetType(tao,"tao_blmvm"); CHKERRQ(info);

  /* Set initial solution guess; */
  info = MSA_BoundaryConditions(&user); CHKERRQ(info);
  info = MSA_InitialPoint(&user,x); CHKERRQ(info);
  info = TaoSolverSetInitialVector(tao,x); CHKERRQ(info);
  
  /* Set routines for function, gradient and hessian evaluation */
  info = TaoSolverSetObjectiveAndGradientRoutine(tao,FormFunctionGradient,(void*) &user); 
  CHKERRQ(info);
  
  info = TaoSolverSetHessianRoutine(tao,user.H,user.H,FormHessian,(void*)&user); 
  CHKERRQ(info);

  /* Set Variable bounds */
  info = MSA_Plate(xl,xu,(void*)&user); CHKERRQ(info);
  info = TaoSolverSetVariableBounds(tao,xl,xu); CHKERRQ(info);

  /* Check for any tao command line options */
  info = TaoSolverSetFromOptions(tao); CHKERRQ(info);

  /* SOLVE THE APPLICATION */
  info = TaoSolverSolve(tao);  CHKERRQ(info);

  /* Get information on termination */
  info = TaoSolverGetConvergedReason(tao,&reason); CHKERRQ(info);
  info = TaoSolverView(tao,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(info);
  if (reason <= 0){
    PetscPrintf(PETSC_COMM_WORLD,"Try a different TAO method, adjust some parameters, or check the function evaluation routines\n");
    PetscPrintf(PETSC_COMM_WORLD,"Iteration: %d, f: %4.2e, Residual: %4.2e, Infeas: %4.2e\n",iter,ff,gnorm,cnorm);
  }
  /* Free TAO data structures */
  info = TaoSolverDestroy(tao); CHKERRQ(info);

  /* Free PETSc data structures */
  info = VecDestroy(x); CHKERRQ(info);
  info = VecDestroy(xl); CHKERRQ(info);
  info = VecDestroy(xu); CHKERRQ(info);
  info = MatDestroy(user.H); CHKERRQ(info);
  info = VecDestroy(user.localX); CHKERRQ(info); 
  info = VecDestroy(user.localV); CHKERRQ(info);
  info = VecDestroy(user.Bottom); CHKERRQ(info);
  info = VecDestroy(user.Top); CHKERRQ(info);
  info = VecDestroy(user.Left); CHKERRQ(info);
  info = VecDestroy(user.Right); CHKERRQ(info);
  info = DADestroy(user.da); CHKERRQ(info);

  /* Finalize TAO and PETSc */
  TaoFinalize();
  PetscFinalize();
  
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "FormFunctionGradient"
/*  FormFunctionGradient - Evaluates f(x) and gradient g(x).             

    Input Parameters:
.   taoapp     - the TAO_APPLICATION context
.   X      - input vector
.   userCtx - optional user-defined context, as set by TaoSetPetscFunctionGradient()
    
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
PetscErrorCode FormFunctionGradient(TaoSolver tao, Vec X, PetscReal *fcn, Vec G,void *userCtx)
{
  AppCtx * user = (AppCtx *) userCtx;
  PetscErrorCode    info;
  PetscInt i,j,row;
  PetscInt mx=user->mx, my=user->my;
  PetscInt xs,xm,gxs,gxm,ys,ym,gys,gym;
  PetscReal ft=0;
  PetscReal zero=0.0;
  PetscReal hx=1.0/(mx+1),hy=1.0/(my+1), hydhx=hy/hx, hxdhy=hx/hy, area=0.5*hx*hy;
  PetscReal rhx=mx+1, rhy=my+1;
  PetscReal f1,f2,f3,f4,f5,f6,d1,d2,d3,d4,d5,d6,d7,d8,xc,xl,xr,xt,xb,xlt,xrb;
  PetscReal df1dxc,df2dxc,df3dxc,df4dxc,df5dxc,df6dxc;
  PetscReal *g, *x,*left,*right,*bottom,*top;
  Vec    localX = user->localX, localG = user->localV;

  /* Get local mesh boundaries */
  info = DAGetCorners(user->da,&xs,&ys,PETSC_NULL,&xm,&ym,PETSC_NULL); CHKERRQ(info);
  info = DAGetGhostCorners(user->da,&gxs,&gys,PETSC_NULL,&gxm,&gym,PETSC_NULL); CHKERRQ(info);

  /* Scatter ghost points to local vector */
  info = DAGlobalToLocalBegin(user->da,X,INSERT_VALUES,localX); CHKERRQ(info);
  info = DAGlobalToLocalEnd(user->da,X,INSERT_VALUES,localX); CHKERRQ(info);

  /* Initialize vector to zero */
  info = VecSet(localG, zero); CHKERRQ(info);

  /* Get pointers to vector data */
  info = VecGetArray(localX,&x); CHKERRQ(info);
  info = VecGetArray(localG,&g); CHKERRQ(info);
  info = VecGetArray(user->Top,&top); CHKERRQ(info);
  info = VecGetArray(user->Bottom,&bottom); CHKERRQ(info);
  info = VecGetArray(user->Left,&left); CHKERRQ(info);
  info = VecGetArray(user->Right,&right); CHKERRQ(info);

  /* Compute function over the locally owned part of the mesh */
  for (j=ys; j<ys+ym; j++){
    for (i=xs; i< xs+xm; i++){
      row=(j-gys)*gxm + (i-gxs);
      
      xc = x[row];
      xlt=xrb=xl=xr=xb=xt=xc;
      
      if (i==0){ /* left side */
        xl= left[j-ys+1];
        xlt = left[j-ys+2];
      } else {
        xl = x[row-1];
      }

      if (j==0){ /* bottom side */
        xb=bottom[i-xs+1];
        xrb = bottom[i-xs+2];
      } else {
        xb = x[row-gxm];
      }
      
      if (i+1 == gxs+gxm){ /* right side */
        xr=right[j-ys+1];
        xrb = right[j-ys];
      } else {
        xr = x[row+1];
      }

      if (j+1==gys+gym){ /* top side */
        xt=top[i-xs+1];
        xlt = top[i-xs];
      }else {
        xt = x[row+gxm];
      }

      if (i>gxs && j+1<gys+gym){
        xlt = x[row-1+gxm];
      }
      if (j>gys && i+1<gxs+gxm){
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
      df2dxc = ( d1*hydhx + d4*hxdhy );
      df3dxc = d3*hxdhy;
      df4dxc = ( d2*hydhx + d3*hxdhy );
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

      f1 = sqrt( 1.0 + d1*d1 + d7*d7);
      f2 = sqrt( 1.0 + d1*d1 + d4*d4);
      f3 = sqrt( 1.0 + d3*d3 + d8*d8);
      f4 = sqrt( 1.0 + d3*d3 + d2*d2);
      f5 = sqrt( 1.0 + d2*d2 + d5*d5);
      f6 = sqrt( 1.0 + d4*d4 + d6*d6);
      
      ft = ft + (f2 + f4);

      df1dxc /= f1;
      df2dxc /= f2;
      df3dxc /= f3;
      df4dxc /= f4;
      df5dxc /= f5;
      df6dxc /= f6;

      g[row] = (df1dxc+df2dxc+df3dxc+df4dxc+df5dxc+df6dxc ) * 0.5;
      
    }
  }


  /* Compute triangular areas along the border of the domain. */
  if (xs==0){ /* left side */
    for (j=ys; j<ys+ym; j++){
      d3=(left[j-ys+1] - left[j-ys+2])*rhy;
      d2=(left[j-ys+1] - x[(j-gys)*gxm])*rhx;
      ft = ft+sqrt( 1.0 + d3*d3 + d2*d2);
    }
  }
  if (ys==0){ /* bottom side */
    for (i=xs; i<xs+xm; i++){
      d2=(bottom[i+1-xs]-bottom[i-xs+2])*rhx;
      d3=(bottom[i-xs+1]-x[i-gxs])*rhy;
      ft = ft+sqrt( 1.0 + d3*d3 + d2*d2);
    }
  }

  if (xs+xm==mx){ /* right side */
    for (j=ys; j< ys+ym; j++){
      d1=(x[(j+1-gys)*gxm-1]-right[j-ys+1])*rhx;
      d4=(right[j-ys]-right[j-ys+1])*rhy;
      ft = ft+sqrt( 1.0 + d1*d1 + d4*d4);
    }
  }
  if (ys+ym==my){ /* top side */
    for (i=xs; i<xs+xm; i++){
      d1=(x[(gym-1)*gxm + i-gxs] - top[i-xs+1])*rhy;
      d4=(top[i-xs+1] - top[i-xs])*rhx;
      ft = ft+sqrt( 1.0 + d1*d1 + d4*d4);
    }
  }

  if (ys==0 && xs==0){
    d1=(left[0]-left[1])*rhy;
    d2=(bottom[0]-bottom[1])*rhx;
    ft +=sqrt( 1.0 + d1*d1 + d2*d2);
  }
  if (ys+ym == my && xs+xm == mx){
    d1=(right[ym+1] - right[ym])*rhy;
    d2=(top[xm+1] - top[xm])*rhx;
    ft +=sqrt( 1.0 + d1*d1 + d2*d2);
  }

  ft=ft*area;
  info = MPI_Allreduce(&ft,fcn,1,MPIU_REAL,MPI_SUM,MPI_COMM_WORLD);CHKERRQ(info);

  
  /* Restore vectors */
  info = VecRestoreArray(localX,&x); CHKERRQ(info);
  info = VecRestoreArray(localG,&g); CHKERRQ(info);
  info = VecRestoreArray(user->Left,&left); CHKERRQ(info);
  info = VecRestoreArray(user->Top,&top); CHKERRQ(info);
  info = VecRestoreArray(user->Bottom,&bottom); CHKERRQ(info);
  info = VecRestoreArray(user->Right,&right); CHKERRQ(info);

  /* Scatter values to global vector */
  info = DALocalToGlobal(user->da,localG,INSERT_VALUES,G); CHKERRQ(info);

  info = PetscLogFlops(70*xm*ym); CHKERRQ(info);

  return 0;
}

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "FormHessian"
/*
   FormHessian - Evaluates Hessian matrix.

   Input Parameters:
.  taoapp  - the TAO_APPLICATION context
.  x    - input vector
.  ptr  - optional user-defined context, as set by TaoSetHessian()

   Output Parameters:
.  A    - Hessian matrix
.  B    - optionally different preconditioning matrix
.  flag - flag indicating matrix structure

   Notes:
   Due to mesh point reordering with DAs, we must always work
   with the local mesh points, and then transform them to the new
   global numbering with the local-to-global mapping.  We cannot work
   directly with the global numbers for the original uniprocessor mesh!  

   Two methods are available for imposing this transformation
   when setting matrix entries:
     (A) MatSetValuesLocal(), using the local ordering (including
         ghost points!)
         - Do the following two steps once, before calling TaoSolve()
           - Use DAGetISLocalToGlobalMapping() to extract the
             local-to-global map from the DA
           - Associate this map with the matrix by calling
             MatSetLocalToGlobalMapping() 
         - Then set matrix entries using the local ordering
           by calling MatSetValuesLocal()
     (B) MatSetValues(), using the global ordering 
         - Use DAGetGlobalIndices() to extract the local-to-global map
         - Then apply this map explicitly yourself
         - Set matrix entries using the global ordering by calling
           MatSetValues()
   Option (A) seems cleaner/easier in many cases, and is the procedure
   used in this example.
*/
PetscErrorCode FormHessian(TaoSolver tao,Vec X,Mat *Hptr, Mat *Hpc, MatStructure *flag, void *ptr)
{ 
  PetscErrorCode    info;
  AppCtx *user = (AppCtx *) ptr;
  Mat Hessian = *Hpc;
  PetscInt   i,j,k,row;
  PetscInt   mx=user->mx, my=user->my;
  PetscInt   xs,xm,gxs,gxm,ys,ym,gys,gym,col[7];
  PetscReal hx=1.0/(mx+1), hy=1.0/(my+1), hydhx=hy/hx, hxdhy=hx/hy;
  PetscReal rhx=mx+1, rhy=my+1;
  PetscReal f1,f2,f3,f4,f5,f6,d1,d2,d3,d4,d5,d6,d7,d8,xc,xl,xr,xt,xb,xlt,xrb;
  PetscReal hl,hr,ht,hb,hc,htl,hbr;
  PetscReal *x,*left,*right,*bottom,*top;
  PetscReal v[7];
  Vec    localX = user->localX;
  PetscBool assembled;


  /* Set various matrix options */
  info = MatSetOption(Hessian,MAT_IGNORE_OFF_PROC_ENTRIES,PETSC_TRUE); CHKERRQ(info);

  /* Initialize matrix entries to zero */
  info = MatAssembled(Hessian,&assembled); CHKERRQ(info);
  if (assembled){info = MatZeroEntries(Hessian);  CHKERRQ(info);}

  /* Get local mesh boundaries */
  info = DAGetCorners(user->da,&xs,&ys,PETSC_NULL,&xm,&ym,PETSC_NULL); CHKERRQ(info);
  info = DAGetGhostCorners(user->da,&gxs,&gys,PETSC_NULL,&gxm,&gym,PETSC_NULL); CHKERRQ(info);

  /* Scatter ghost points to local vector */
  info = DAGlobalToLocalBegin(user->da,X,INSERT_VALUES,localX); CHKERRQ(info);
  info = DAGlobalToLocalEnd(user->da,X,INSERT_VALUES,localX); CHKERRQ(info);

  /* Get pointers to vector data */
  info = VecGetArray(localX,&x); CHKERRQ(info);
  info = VecGetArray(user->Top,&top); CHKERRQ(info);
  info = VecGetArray(user->Bottom,&bottom); CHKERRQ(info);
  info = VecGetArray(user->Left,&left); CHKERRQ(info);
  info = VecGetArray(user->Right,&right); CHKERRQ(info);

  /* Compute Hessian over the locally owned part of the mesh */

  for (i=xs; i< xs+xm; i++){

    for (j=ys; j<ys+ym; j++){

      row=(j-gys)*gxm + (i-gxs);
      
      xc = x[row]; 
      xlt=xrb=xl=xr=xb=xt=xc;

      /* Left side */
      if (i==gxs){
        xl= left[j-ys+1];
        xlt = left[j-ys+2];
      } else {
        xl = x[row-1];
      }
      
      if (j==gys){
        xb=bottom[i-xs+1];
        xrb = bottom[i-xs+2];
      } else {
        xb = x[row-gxm];
      }
      
      if (i+1 == gxs+gxm){
        xr=right[j-ys+1];
        xrb = right[j-ys];
      } else {
        xr = x[row+1];
      }

      if (j+1==gys+gym){
        xt=top[i-xs+1];
        xlt = top[i-xs];
      }else {
        xt = x[row+gxm];
      }

      if (i>gxs && j+1<gys+gym){
        xlt = x[row-1+gxm];
      }
      if (j>gys && i+1<gxs+gxm){
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
      
      f1 = sqrt( 1.0 + d1*d1 + d7*d7);
      f2 = sqrt( 1.0 + d1*d1 + d4*d4);
      f3 = sqrt( 1.0 + d3*d3 + d8*d8);
      f4 = sqrt( 1.0 + d3*d3 + d2*d2);
      f5 = sqrt( 1.0 + d2*d2 + d5*d5);
      f6 = sqrt( 1.0 + d4*d4 + d6*d6);


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
      if (j>0){ 
	v[k]=hb; col[k]=row - gxm; k++;
      }
      
      if (j>0 && i < mx -1){
	v[k]=hbr; col[k]=row - gxm+1; k++;
      }
      
      if (i>0){
	v[k]= hl; col[k]=row - 1; k++;
      }
      
      v[k]= hc; col[k]=row; k++;
      
      if (i < mx-1 ){
	v[k]= hr; col[k]=row+1; k++;
      }
      
      if (i>0 && j < my-1 ){
	v[k]= htl; col[k] = row+gxm-1; k++;
      }
      
      if (j < my-1 ){
	v[k]= ht; col[k] = row+gxm; k++;
      }
      
      /* 
	 Set matrix values using local numbering, which was defined
	 earlier, in the main routine.
      */
      info = MatSetValuesLocal(Hessian,1,&row,k,col,v,INSERT_VALUES); 
      CHKERRQ(info);
      
    }
  }
  
  /* Restore vectors */
  info = VecRestoreArray(localX,&x); CHKERRQ(info);
  info = VecRestoreArray(user->Left,&left); CHKERRQ(info);
  info = VecRestoreArray(user->Top,&top); CHKERRQ(info);
  info = VecRestoreArray(user->Bottom,&bottom); CHKERRQ(info);
  info = VecRestoreArray(user->Right,&right); CHKERRQ(info);

  /* Assemble the matrix */
  info = MatAssemblyBegin(Hessian,MAT_FINAL_ASSEMBLY); CHKERRQ(info);
  info = MatAssemblyEnd(Hessian,MAT_FINAL_ASSEMBLY); CHKERRQ(info);

  info = PetscLogFlops(199*xm*ym); CHKERRQ(info);
  return 0;
}

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "MSA_BoundaryConditions"
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
  int        info;
  PetscInt   i,j,k,maxits=5,limit=0;
  PetscInt   xs,ys,xm,ym,gxs,gys,gxm,gym;
  PetscInt   mx=user->mx,my=user->my;
  PetscInt   bsize=0, lsize=0, tsize=0, rsize=0;
  PetscReal     one=1.0, two=2.0, three=3.0, scl=1.0, tol=1e-10;
  PetscReal     fnorm,det,hx,hy,xt=0,yt=0;
  PetscReal     u1,u2,nf1,nf2,njac11,njac12,njac21,njac22;
  PetscReal     b=-0.5, t=0.5, l=-0.5, r=0.5;
  PetscReal     *boundary;
  PetscBool   flg;
  Vec        Bottom,Top,Right,Left;

  /* Get local mesh boundaries */
  info = DAGetCorners(user->da,&xs,&ys,PETSC_NULL,&xm,&ym,PETSC_NULL); CHKERRQ(info);
  info = DAGetGhostCorners(user->da,&gxs,&gys,PETSC_NULL,&gxm,&gym,PETSC_NULL); CHKERRQ(info);


  bsize=xm+2;
  lsize=ym+2;
  rsize=ym+2;
  tsize=xm+2;

  info = VecCreateMPI(MPI_COMM_WORLD,bsize,PETSC_DECIDE,&Bottom); CHKERRQ(info);
  info = VecCreateMPI(MPI_COMM_WORLD,tsize,PETSC_DECIDE,&Top); CHKERRQ(info);
  info = VecCreateMPI(MPI_COMM_WORLD,lsize,PETSC_DECIDE,&Left); CHKERRQ(info);
  info = VecCreateMPI(MPI_COMM_WORLD,rsize,PETSC_DECIDE,&Right); CHKERRQ(info);

  user->Top=Top;
  user->Left=Left;
  user->Bottom=Bottom;
  user->Right=Right;

  hx= (r-l)/(mx+1); hy=(t-b)/(my+1);

  for (j=0; j<4; j++){
    if (j==0){
      yt=b;
      xt=l+hx*xs;
      limit=bsize;
      VecGetArray(Bottom,&boundary);
    } else if (j==1){
      yt=t;
      xt=l+hx*xs;
      limit=tsize;
      VecGetArray(Top,&boundary);
    } else if (j==2){
      yt=b+hy*ys;
      xt=l;
      limit=lsize;
      VecGetArray(Left,&boundary);
    } else if (j==3){
      yt=b+hy*ys;
      xt=r;
      limit=rsize;
      VecGetArray(Right,&boundary);
    }

    for (i=0; i<limit; i++){
      u1=xt;
      u2=-yt;
      for (k=0; k<maxits; k++){
	nf1=u1 + u1*u2*u2 - u1*u1*u1/three-xt;
	nf2=-u2 - u1*u1*u2 + u2*u2*u2/three-yt;
	fnorm=sqrt(nf1*nf1+nf2*nf2);
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
      } else if (j==2 || j==3){
	yt=yt+hy;
      }
      
    }
    
    if (j==0){
      info = VecRestoreArray(Bottom,&boundary); CHKERRQ(info);
    } else if (j==1){
      info = VecRestoreArray(Top,&boundary); CHKERRQ(info);
    } else if (j==2){
      info = VecRestoreArray(Left,&boundary); CHKERRQ(info);
    } else if (j==3){
      info = VecRestoreArray(Right,&boundary); CHKERRQ(info);
    }

  }

  /* Scale the boundary if desired */

  info = PetscOptionsGetReal(PETSC_NULL,"-bottom",&scl,&flg); 
  CHKERRQ(info);
  if (flg){
    info = VecScale(Bottom, scl); CHKERRQ(info);
  }
  
  info = PetscOptionsGetReal(PETSC_NULL,"-top",&scl,&flg); 
  CHKERRQ(info);
  if (flg){
    info = VecScale(Top, scl); CHKERRQ(info);
  }
  
  info = PetscOptionsGetReal(PETSC_NULL,"-right",&scl,&flg); 
  CHKERRQ(info);
  if (flg){
    info = VecScale(Right, scl); CHKERRQ(info);
  }
  
  info = PetscOptionsGetReal(PETSC_NULL,"-left",&scl,&flg); 
  CHKERRQ(info);
  if (flg){
    info = VecScale(Left, scl); CHKERRQ(info);
  }

  return 0;
}


/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "MSA_Plate"
/* 
   MSA_Plate -  Calculates an obstacle for surface to stretch over.

   Input Parameter:
.  user - user-defined application context

   Output Parameter:
.  user - user-defined application context
*/
static PetscErrorCode MSA_Plate(Vec XL,Vec XU,void *ctx){

  AppCtx *user=(AppCtx *)ctx;
  PetscErrorCode info;
  PetscInt i,j,row;
  PetscInt xs,ys,xm,ym;
  PetscInt mx=user->mx, my=user->my, bmy, bmx;
  PetscReal t1,t2,t3;
  PetscReal *xl, lb=TAO_NINFINITY, ub=TAO_INFINITY;
  PetscBool cylinder;

  user->bmy = PetscMax(0,user->bmy);user->bmy = PetscMin(my,user->bmy);
  user->bmx = PetscMax(0,user->bmx);user->bmx = PetscMin(mx,user->bmx);
  bmy=user->bmy, bmx=user->bmx;

  info = DAGetCorners(user->da,&xs,&ys,PETSC_NULL,&xm,&ym,PETSC_NULL); CHKERRQ(info);

  info = VecSet(XL, lb); CHKERRQ(info);
  info = VecSet(XU, ub); CHKERRQ(info);

  info = VecGetArray(XL,&xl); CHKERRQ(info);

  info = PetscOptionsHasName(PETSC_NULL,"-cylinder",&cylinder); CHKERRQ(info);
  /* Compute the optional lower box */
  if (cylinder){
    for (i=xs; i< xs+xm; i++){    
      for (j=ys; j<ys+ym; j++){
	row=(j-ys)*xm + (i-xs);
	t1=(2.0*i-mx)*bmy;
	t2=(2.0*j-my)*bmx;
	t3=bmx*bmx*bmy*bmy;
	if ( t1*t1 + t2*t2 <= t3 ){
	  xl[row] = user->bheight;
	}
      }
    }
  } else {
    /* Compute the optional lower box */
    for (i=xs; i< xs+xm; i++){    
      for (j=ys; j<ys+ym; j++){
	row=(j-ys)*xm + (i-xs);
	if (i>=(mx-bmx)/2 && i<mx-(mx-bmx)/2 && 
	    j>=(my-bmy)/2 && j<my-(my-bmy)/2 ){
	  xl[row] = user->bheight;
	}
      }
    }
  }
    info = VecRestoreArray(XL,&xl); CHKERRQ(info);

  return 0;
}


/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "MSA_InitialPoint"
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
  PetscErrorCode      info;
  PetscInt start=-1,i,j;
  PetscReal   zero=0.0;
  PetscBool flg;

  info = PetscOptionsGetInt(PETSC_NULL,"-start",&start,&flg); CHKERRQ(info);

  if (flg && start==0){ /* The zero vector is reasonable */
 
    info = VecSet(X, zero); CHKERRQ(info);

  } else if (flg && start>0){ /* Try a random start between -0.5 and 0.5 */

    PetscRandom rctx;  PetscReal np5=-0.5;

    info = PetscRandomCreate(MPI_COMM_WORLD,&rctx); 
    CHKERRQ(info);
    for (i=0; i<start; i++){
      info = VecSetRandom(X, rctx); CHKERRQ(info);
    }
    info = PetscRandomDestroy(rctx); CHKERRQ(info);
    info = VecShift(X, np5);

  } else { /* Take an average of the boundary conditions */

    PetscInt row,xs,xm,gxs,gxm,ys,ym,gys,gym;
    PetscInt mx=user->mx,my=user->my;
    PetscReal *x,*left,*right,*bottom,*top;
    Vec    localX = user->localX;
    
    /* Get local mesh boundaries */
    info = DAGetCorners(user->da,&xs,&ys,PETSC_NULL,&xm,&ym,PETSC_NULL); CHKERRQ(info);
    info = DAGetGhostCorners(user->da,&gxs,&gys,PETSC_NULL,&gxm,&gym,PETSC_NULL); CHKERRQ(info);
    
    /* Get pointers to vector data */
    info = VecGetArray(user->Top,&top); CHKERRQ(info);
    info = VecGetArray(user->Bottom,&bottom); CHKERRQ(info);
    info = VecGetArray(user->Left,&left); CHKERRQ(info);
    info = VecGetArray(user->Right,&right); CHKERRQ(info);

    info = VecGetArray(localX,&x); CHKERRQ(info);
    /* Perform local computations */    
    for (j=ys; j<ys+ym; j++){
      for (i=xs; i< xs+xm; i++){
	row=(j-gys)*gxm + (i-gxs);
	x[row] = ( (j+1)*bottom[i-xs+1]/my + (my-j+1)*top[i-xs+1]/(my+2)+
		   (i+1)*left[j-ys+1]/mx + (mx-i+1)*right[j-ys+1]/(mx+2))/2.0; 
      }
    }
    
    /* Restore vectors */
    info = VecRestoreArray(localX,&x); CHKERRQ(info);

    info = VecRestoreArray(user->Left,&left); CHKERRQ(info);
    info = VecRestoreArray(user->Top,&top); CHKERRQ(info);
    info = VecRestoreArray(user->Bottom,&bottom); CHKERRQ(info);
    info = VecRestoreArray(user->Right,&right); CHKERRQ(info);
    
    /* Scatter values into global vector */
    info = DALocalToGlobal(user->da,localX,INSERT_VALUES,X); CHKERRQ(info);
    
  }
  return 0;
}







