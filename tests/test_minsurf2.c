/* Program usage: mpirun -np <proc> minsurf2 [-help] [all TAO options] */

/*
  Include "taosolver.h" so we can use TAO solvers.
  petscdm.h for distributed array
*/
#include "taosolver.h"
#include "petscdm.h"

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

/*T
   Concepts: TAO - Solving an unconstrained minimization problem
   Routines: TaoInitialize(); TaoFinalize();
   Routines: TaoCreate(); TaoDestroy();
   Routines: TaoApplicationCreate(); TaoAppDestroy();
   Routines: TaoAppSetInitialSolutionVec();
   Routines: TaoAppSetObjectiveAndGradientRoutine();
   Routines: TaoAppSetHessianMat(); TaoAppSetHessianRoutine();
   Routines: TaoSetOptions();
   Routines: TaoAppGetKSP(); TaoSolveApplication();
   Routines: TaoAppSetMonitor(); TaoView();
   Routines: TaoAppGetSolutionVec();
   Processors: 1
T*/

/* 
   User-defined application context - contains data needed by the 
   application-provided call-back routines, FormFunctionGradient() 
   and FormHessian().
*/
typedef struct {
  PetscInt      mx, my;                 /* discretization in x, y directions */
  PetscReal      *bottom, *top, *left, *right;             /* boundary values */
  DM          dm;                      /* distributed array data structure */
  Mat         H;                       /* Hessian */
} AppCtx;


/* -------- User-defined Routines --------- */

static PetscErrorCode MSA_BoundaryConditions(AppCtx*);
static PetscErrorCode MSA_InitialPoint(AppCtx*,Vec);
PetscErrorCode QuadraticH(AppCtx*,Vec,Mat);
PetscErrorCode FormFunctionGradient(TaoSolver,Vec,PetscReal *,Vec,void*);
PetscErrorCode FormGradient(TaoSolver,Vec,Vec,void*);
PetscErrorCode FormHessian(TaoSolver,Vec,Mat*,Mat*,MatStructure *,void*);
PetscErrorCode My_Monitor(TaoSolver, void *);

#undef __FUNCT__
#define __FUNCT__ "main"
int main( int argc, char **argv )
{
  PetscErrorCode    ierr;                /* used to check for functions returning nonzeros */
  PetscInt          Nx, Ny;              /* number of processors in x- and y- directions */
  Vec             x;                   /* solution, gradient vectors */
  PetscBool      flg, viewmat;        /* flags */
  PetscBool      fddefault, fdcoloring;   /* flags */
  TaoSolverTerminationReason reason;           
  TaoSolver       tao;                 /* TAO solver context */
  AppCtx          user;                /* user-defined work context */
  ISColoring     iscoloring;
  MatFDColoring  matfdcoloring;

  /* Initialize TAO */
  PetscInitialize( &argc, &argv,(char *)0,help );
  TaoInitialize( &argc, &argv,(char *)0,help );

  /* Specify dimension of the problem */
  user.mx = 10; user.my = 10;

  /* Check for any command line arguments that override defaults */
  ierr = PetscOptionsGetInt(PETSC_NULL,"-mx",&user.mx,&flg); CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-my",&user.my,&flg); CHKERRQ(ierr);

  PetscPrintf(MPI_COMM_WORLD,"\n---- Minimum Surface Area Problem -----\n");
  PetscPrintf(MPI_COMM_WORLD,"mx: %d     my: %d   \n\n",user.mx,user.my);


  /* Let PETSc determine the vector distribution */
  Nx = PETSC_DECIDE; Ny = PETSC_DECIDE;

  /* Create distributed array (DM) to manage parallel grid and vectors  */
  ierr = DMDACreate2d(PETSC_COMM_WORLD,DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE,
		      DMDA_STENCIL_BOX,
		      user.mx, user.my,Nx,Ny,1,1,PETSC_NULL,PETSC_NULL,
		      &user.dm); CHKERRQ(ierr);
  

  /* Create TAO solver and set desired solution method.*/
  ierr = TaoSolverCreate(PETSC_COMM_WORLD,&tao); CHKERRQ(ierr);
  ierr = TaoSolverSetType(tao,"tao_cg"); CHKERRQ(ierr);

  /*
     Extract global vector from DA for the vector of variables --  PETSC routine
     Compute the initial solution                              --  application specific, see below
     Set this vector for use by TAO                            --  TAO routine
  */
  ierr = DMCreateGlobalVector(user.dm,&x); CHKERRQ(ierr);
  ierr = MSA_BoundaryConditions(&user); CHKERRQ(ierr);         
  ierr = MSA_InitialPoint(&user,x); CHKERRQ(ierr);
  ierr = TaoSolverSetInitialVector(tao,x); CHKERRQ(ierr);

  /* 
     Initialize the Application context for use in function evaluations  --  application specific, see below.
     Set routines for function and gradient evaluation 
  */
  ierr = TaoSolverSetObjectiveAndGradientRoutine(tao,FormFunctionGradient,(void *)&user); CHKERRQ(ierr);

  /* 
     Given the command line arguments, calculate the hessian with either the user-
     provided function FormHessian, or the default finite-difference driven Hessian
     functions 
  */
  ierr = PetscOptionsHasName(PETSC_NULL,"-tao_fddefault",&fddefault);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL,"-tao_fdcoloring",&fdcoloring);CHKERRQ(ierr);


  /* 
     Create a matrix data structure to store the Hessian and set 
     the Hessian evalution routine.
     Set the matrix structure to be used for Hessian evalutions
  */
  ierr = DMGetMatrix(user.dm,MATAIJ,&user.H);CHKERRQ(ierr);
  ierr = MatSetOption(user.H,MAT_SYMMETRIC,PETSC_TRUE); CHKERRQ(ierr);


  if (fdcoloring) {
      ierr = DMGetColoring(user.dm,IS_COLORING_GLOBAL,MATAIJ,&iscoloring); 
      CHKERRQ(ierr);

      ierr = MatFDColoringCreate(user.H,iscoloring,&matfdcoloring);
      CHKERRQ(ierr);

      ierr = ISColoringDestroy(&iscoloring);CHKERRQ(ierr);
      ierr = MatFDColoringSetFunction(matfdcoloring,(PetscErrorCode (*)(void))FormGradient,(void*)&user); CHKERRQ(ierr);
      ierr = MatFDColoringSetFromOptions(matfdcoloring); CHKERRQ(ierr);
      
      ierr = TaoSolverSetHessianRoutine(tao,user.H,user.H,TaoSolverDefaultComputeHessianColor,(void *)matfdcoloring); CHKERRQ(ierr);

  } else if (fddefault){
      ierr = TaoSolverSetHessianRoutine(tao,user.H,user.H,TaoSolverDefaultComputeHessian,(void *)PETSC_NULL); CHKERRQ(ierr);

  } else { 
      ierr = TaoSolverSetHessianRoutine(tao,user.H,user.H,FormHessian,(void *)&user); CHKERRQ(ierr);
  }


  /* 
     If my_monitor option is in command line, then use the user-provided
     monitoring function
  */
  ierr = PetscOptionsHasName(PETSC_NULL,"-my_monitor",&viewmat); CHKERRQ(ierr);
  if (viewmat){
    ierr = TaoSolverSetMonitor(tao,My_Monitor,PETSC_NULL,PETSC_NULL); CHKERRQ(ierr);
  }

  /* Check for any tao command line options */
  ierr = TaoSolverSetFromOptions(tao); CHKERRQ(ierr);

  /* Limit the number of iterations in the KSP linear solver */
  //ierr = TaoAppGetKSP(minsurfapp,&ksp); CHKERRQ(ierr);
  //if (ksp) {                                              /* Modify the PETSc KSP structure */
//    ierr = KSPSetTolerances(ksp,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,user.mx*user.my);
//    CHKERRQ(ierr);
//  }

  /* SOLVE THE APPLICATION */
  ierr = TaoSolverSolve(tao); CHKERRQ(ierr);

  ierr = TaoSolverView(tao,PETSC_VIEWER_STDOUT_WORLD);

  /* Get information on termination */
  ierr = TaoSolverGetTerminationReason(tao,&reason); CHKERRQ(ierr);
  if (reason <= 0 ){
      PetscPrintf(MPI_COMM_WORLD,"Try a different TAO method \n");//, adjust some parameters, or check the function evaluation routines\n");
  }


  /* Free TAO data structures */
  ierr = TaoSolverDestroy(&tao); CHKERRQ(ierr);

  /* Free PETSc data structures */
  ierr = VecDestroy(&x); CHKERRQ(ierr);
  ierr = MatDestroy(&user.H); CHKERRQ(ierr);
  if (fdcoloring) {
      ierr = MatFDColoringDestroy(&matfdcoloring); CHKERRQ(ierr);
  }
  ierr = PetscFree(user.bottom); CHKERRQ(ierr);
  ierr = PetscFree(user.top); CHKERRQ(ierr);
  ierr = PetscFree(user.left); CHKERRQ(ierr);
  ierr = PetscFree(user.right); CHKERRQ(ierr);
  ierr = DMDestroy(&user.dm); CHKERRQ(ierr);

  /* Finalize TAO */
  TaoFinalize();
  PetscFinalize();
  
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "FormGradient"
PetscErrorCode FormGradient(TaoSolver tao, Vec X, Vec G,void *userCtx){
  PetscErrorCode ierr;
  PetscReal fcn;
  PetscFunctionBegin;
  ierr = FormFunctionGradient(tao,X,&fcn,G,userCtx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "FormFunctionGradient"
/*  FormFunctionGradient - Evaluates the function and corresponding gradient.

    Input Parameters:
.   taoapp     - the TAO_APPLICATION context
.   XX      - input vector
.   userCtx - optional user-defined context, as set by TaoSetFunctionGradient()
    
    Output Parameters:
.   fcn     - the newly evaluated function
.   GG       - vector containing the newly evaluated gradient
*/
PetscErrorCode FormFunctionGradient(TaoSolver tao, Vec X, PetscReal *fcn,Vec G,void *userCtx){

  AppCtx * user = (AppCtx *) userCtx;
  PetscErrorCode    ierr;
  PetscInt i,j;
  PetscInt mx=user->mx, my=user->my;
  PetscInt xs,xm,gxs,gxm,ys,ym,gys,gym;
  PetscReal ft=0;
  PetscReal hx=1.0/(mx+1),hy=1.0/(my+1), hydhx=hy/hx, hxdhy=hx/hy, area=0.5*hx*hy;
  PetscReal rhx=mx+1, rhy=my+1;
  PetscReal f1,f2,f3,f4,f5,f6,d1,d2,d3,d4,d5,d6,d7,d8,xc,xl,xr,xt,xb,xlt,xrb;
  PetscReal df1dxc,df2dxc,df3dxc,df4dxc,df5dxc,df6dxc;
  PetscReal **g, **x;
  Vec    localX;

  PetscFunctionBegin;
  /* Get local mesh boundaries */
  ierr = DMGetLocalVector(user->dm,&localX);CHKERRQ(ierr);

  ierr = DMDAGetCorners(user->dm,&xs,&ys,PETSC_NULL,&xm,&ym,PETSC_NULL); CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(user->dm,&gxs,&gys,PETSC_NULL,&gxm,&gym,PETSC_NULL); CHKERRQ(ierr);

  /* Scatter ghost points to local vector */
  ierr = DMGlobalToLocalBegin(user->dm,X,INSERT_VALUES,localX); CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(user->dm,X,INSERT_VALUES,localX); CHKERRQ(ierr);

  /* Get pointers to vector data */
  ierr = DMDAVecGetArray(user->dm,localX,(void**)&x);
  ierr = DMDAVecGetArray(user->dm,G,(void**)&g);

  /* Compute function and gradient over the locally owned part of the mesh */
  for (j=ys; j<ys+ym; j++){
    for (i=xs; i< xs+xm; i++){
      
      xc = x[j][i];
      xlt=xrb=xl=xr=xb=xt=xc;
      
      if (i==0){ /* left side */
        xl= user->left[j-ys+1];
        xlt = user->left[j-ys+2];
      } else {
        xl = x[j][i-1];
      }

      if (j==0){ /* bottom side */
        xb=user->bottom[i-xs+1];
        xrb =user->bottom[i-xs+2];
      } else {
        xb = x[j-1][i];
      }
      
      if (i+1 == gxs+gxm){ /* right side */
        xr=user->right[j-ys+1];
        xrb = user->right[j-ys];
      } else {
        xr = x[j][i+1];
      }

      if (j+1==gys+gym){ /* top side */
        xt=user->top[i-xs+1];
        xlt = user->top[i-xs];
      }else {
        xt = x[j+1][i];
      }

      if (i>gxs && j+1<gys+gym){
        xlt = x[j+1][i-1];
      }
      if (j>gys && i+1<gxs+gxm){
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
      
      f2 = sqrt( 1.0 + d1*d1 + d4*d4);
      f4 = sqrt( 1.0 + d3*d3 + d2*d2);

      ft = ft + (f2 + f4);

      df1dxc /= f1;
      df2dxc /= f2;
      df3dxc /= f3;
      df4dxc /= f4;
      df5dxc /= f5;
      df6dxc /= f6;

      g[j][i] = (df1dxc+df2dxc+df3dxc+df4dxc+df5dxc+df6dxc ) * 0.5;
      
    }
  }

  /* Compute triangular areas along the border of the domain. */
  if (xs==0){ /* left side */
    for (j=ys; j<ys+ym; j++){
      d3=(user->left[j-ys+1] - user->left[j-ys+2])*rhy;
      d2=(user->left[j-ys+1] - x[j][0]) *rhx;
      ft = ft+sqrt( 1.0 + d3*d3 + d2*d2);
    }
  }
  if (ys==0){ /* bottom side */
    for (i=xs; i<xs+xm; i++){
      d2=(user->bottom[i+1-xs]-user->bottom[i-xs+2])*rhx;
      d3=(user->bottom[i-xs+1]-x[0][i])*rhy;
      ft = ft+sqrt( 1.0 + d3*d3 + d2*d2);
    }
  }

  if (xs+xm==mx){ /* right side */
    for (j=ys; j< ys+ym; j++){
      d1=(x[j][mx-1] - user->right[j-ys+1])*rhx;
      d4=(user->right[j-ys]-user->right[j-ys+1])*rhy;
      ft = ft+sqrt( 1.0 + d1*d1 + d4*d4);
    }
  }
  if (ys+ym==my){ /* top side */
    for (i=xs; i<xs+xm; i++){
      d1=(x[my-1][i] - user->top[i-xs+1])*rhy;
      d4=(user->top[i-xs+1] - user->top[i-xs])*rhx;
      ft = ft+sqrt( 1.0 + d1*d1 + d4*d4);
    }
  }

  if (ys==0 && xs==0){
    d1=(user->left[0]-user->left[1])/hy;
    d2=(user->bottom[0]-user->bottom[1])*rhx;
    ft +=sqrt( 1.0 + d1*d1 + d2*d2);
  }
  if (ys+ym == my && xs+xm == mx){
    d1=(user->right[ym+1] - user->right[ym])*rhy;
    d2=(user->top[xm+1] - user->top[xm])*rhx;
    ft +=sqrt( 1.0 + d1*d1 + d2*d2);
  }

  ft=ft*area;
  ierr = MPI_Allreduce(&ft,fcn,1,MPIU_REAL,MPI_SUM,MPI_COMM_WORLD);CHKERRQ(ierr);

  /* Restore vectors */
  ierr = DMDAVecRestoreArray(user->dm,localX,(void**)&x);
  ierr = DMDAVecRestoreArray(user->dm,G,(void**)&g);

  /* Scatter values to global vector */
  ierr = DMRestoreLocalVector(user->dm,&localX); CHKERRQ(ierr);

  ierr = PetscLogFlops(67*xm*ym); CHKERRQ(ierr);

  PetscFunctionReturn(0);
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
.  H    - Hessian matrix
.  Hpre - optionally different preconditioning matrix
.  flg  - flag indicating matrix structure

*/
PetscErrorCode FormHessian(TaoSolver tao,Vec X,Mat *H, Mat *Hpre, MatStructure *flg, void *ptr)
{ 
  PetscErrorCode    ierr;
  AppCtx *user = (AppCtx *) ptr;

  PetscFunctionBegin;
  /* Evaluate the Hessian entries*/
  ierr = QuadraticH(user,X,*H); CHKERRQ(ierr);


  /* Indicate that this matrix has the same sparsity pattern during
     successive iterations; setting this flag can save significant work
     in computing the preconditioner for some methods. */
  *flg=SAME_NONZERO_PATTERN;
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "QuadraticH"
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
  PetscErrorCode    ierr;
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
  ierr = DMGetLocalVector(user->dm,&localX);CHKERRQ(ierr);

  ierr = DMDAGetCorners(user->dm,&xs,&ys,PETSC_NULL,&xm,&ym,PETSC_NULL); CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(user->dm,&gxs,&gys,PETSC_NULL,&gxm,&gym,PETSC_NULL); CHKERRQ(ierr);

  /* Scatter ghost points to local vector */
  ierr = DMGlobalToLocalBegin(user->dm,X,INSERT_VALUES,localX); CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(user->dm,X,INSERT_VALUES,localX); CHKERRQ(ierr);

  /* Get pointers to vector data */
  ierr = DMDAVecGetArray(user->dm,localX,(void**)&x);

  /* Initialize matrix entries to zero */
  ierr = MatAssembled(Hessian,&assembled); CHKERRQ(ierr);
  if (assembled){ierr = MatZeroEntries(Hessian);  CHKERRQ(ierr);}


  /* Set various matrix options */
  ierr = MatSetOption(Hessian,MAT_IGNORE_OFF_PROC_ENTRIES,PETSC_TRUE); CHKERRQ(ierr);

  /* Compute Hessian over the locally owned part of the mesh */

  for (j=ys; j<ys+ym; j++){
      
    for (i=xs; i< xs+xm; i++){

      xc = x[j][i];
      xlt=xrb=xl=xr=xb=xt=xc;

      /* Left side */
      if (i==0){
        xl  = user->left[j-ys+1];
        xlt = user->left[j-ys+2];
      } else {
        xl  = x[j][i-1];
      }
      
      if (j==0){
        xb  = user->bottom[i-xs+1];
        xrb = user->bottom[i-xs+2];
      } else {
        xb  = x[j-1][i];
      }
      
      if (i+1 == mx){
        xr  = user->right[j-ys+1];
        xrb = user->right[j-ys];
      } else {
        xr  = x[j][i+1];
      }

      if (j+1==my){
        xt  = user->top[i-xs+1];
        xlt = user->top[i-xs];
      }else {
        xt  = x[j+1][i];
      }

      if (i>0 && j+1<my){
        xlt = x[j+1][i-1];
      }
      if (j>0 && i+1<mx){
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

      hl/=2.0; hr/=2.0; ht/=2.0; hb/=2.0; hbr/=2.0; htl/=2.0;  hc/=2.0; 

      row.j = j; row.i = i;
      k=0;
      if (j>0){ 
	v[k]=hb;
	col[k].j = j - 1; col[k].i = i;
	k++;
      }
      
      if (j>0 && i < mx -1){
	v[k]=hbr;
	col[k].j = j - 1; col[k].i = i+1;
	k++;
      }
      
      if (i>0){
	v[k]= hl;
	col[k].j = j; col[k].i = i-1;
	k++;
      }
      
      v[k]= hc;
      col[k].j = j; col[k].i = i;
      k++;
      
      if (i < mx-1 ){
	v[k]= hr; 
	col[k].j = j; col[k].i = i+1;
	k++;
      }
      
      if (i>0 && j < my-1 ){
	v[k]= htl;
	col[k].j = j+1; col[k].i = i-1;
	k++;
      }
      
      if (j < my-1 ){
	v[k]= ht; 
	col[k].j = j+1; col[k].i = i;
	k++;
      }
      
      /* 
	 Set matrix values using local numbering, which was defined
	 earlier, in the main routine.
      */
      ierr = MatSetValuesStencil(Hessian,1,&row,k,col,v,INSERT_VALUES);
      CHKERRQ(ierr);
      
    }
  }
  
  /* Restore vectors */
  ierr = DMDAVecRestoreArray(user->dm,localX,(void**)&x);

  ierr = DMRestoreLocalVector(user->dm,&localX); CHKERRQ(ierr);

  /* Assemble the matrix */
  ierr = MatAssemblyBegin(Hessian,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Hessian,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  ierr = PetscLogFlops(199*xm*ym); CHKERRQ(ierr);
  PetscFunctionReturn(0);
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
  PetscErrorCode        ierr;
  PetscInt     i,j,k,limit=0,maxits=5;
  PetscInt     xs,ys,xm,ym,gxs,gys,gxm,gym;
  PetscInt     mx=user->mx,my=user->my;
  PetscInt     bsize=0, lsize=0, tsize=0, rsize=0;
  PetscReal     one=1.0, two=2.0, three=3.0, tol=1e-10;
  PetscReal     fnorm,det,hx,hy,xt=0,yt=0;
  PetscReal     u1,u2,nf1,nf2,njac11,njac12,njac21,njac22;
  PetscReal     b=-0.5, t=0.5, l=-0.5, r=0.5;
  PetscReal     *boundary;
  PetscBool   flg;

  PetscFunctionBegin;
  /* Get local mesh boundaries */
  ierr = DMDAGetCorners(user->dm,&xs,&ys,PETSC_NULL,&xm,&ym,PETSC_NULL); CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(user->dm,&gxs,&gys,PETSC_NULL,&gxm,&gym,PETSC_NULL); CHKERRQ(ierr);

  bsize=xm+2;
  lsize=ym+2;
  rsize=ym+2;
  tsize=xm+2;

  ierr = PetscMalloc(bsize*sizeof(PetscReal),&user->bottom); CHKERRQ(ierr);
  ierr = PetscMalloc(tsize*sizeof(PetscReal),&user->top); CHKERRQ(ierr);
  ierr = PetscMalloc(lsize*sizeof(PetscReal),&user->left); CHKERRQ(ierr);
  ierr = PetscMalloc(rsize*sizeof(PetscReal),&user->right); CHKERRQ(ierr);

  hx= (r-l)/(mx+1); hy=(t-b)/(my+1);

  for (j=0; j<4; j++){
    if (j==0){
      yt=b;
      xt=l+hx*xs;
      limit=bsize;
      boundary=user->bottom;
    } else if (j==1){
      yt=t;
      xt=l+hx*xs;
      limit=tsize;
      boundary=user->top;
    } else if (j==2){
      yt=b+hy*ys;
      xt=l;
      limit=lsize;
      boundary=user->left;
    } else { //if (j==3)
      yt=b+hy*ys;
      xt=r;
      limit=rsize;
      boundary=user->right;
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
      } else { // if (j==2 || j==3)
	yt=yt+hy;
      }
      
    }

  }

  /* Scale the boundary if desired */
  if (1==1){ 
    PetscReal scl = 1.0;

    ierr = PetscOptionsGetReal(PETSC_NULL,"-bottom",&scl,&flg); 
    CHKERRQ(ierr);
    if (flg){
      for (i=0;i<bsize;i++) user->bottom[i]*=scl;
    }

    ierr = PetscOptionsGetReal(PETSC_NULL,"-top",&scl,&flg); 
    CHKERRQ(ierr);
    if (flg){
      for (i=0;i<tsize;i++) user->top[i]*=scl;
    }

    ierr = PetscOptionsGetReal(PETSC_NULL,"-right",&scl,&flg); 
    CHKERRQ(ierr);
    if (flg){
      for (i=0;i<rsize;i++) user->right[i]*=scl;
    }

    ierr = PetscOptionsGetReal(PETSC_NULL,"-left",&scl,&flg); 
    CHKERRQ(ierr);
    if (flg){
      for (i=0;i<lsize;i++) user->left[i]*=scl;
    }
  }
  
  PetscFunctionReturn(0);
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
  PetscErrorCode      ierr;
  PetscInt   start2=-1,i,j;
  PetscReal   start1=0;
  PetscBool flg1,flg2;
  PetscFunctionBegin;

  ierr = PetscOptionsGetReal(PETSC_NULL,"-start",&start1,&flg1); CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-random",&start2,&flg2); CHKERRQ(ierr);

  if (flg1){ /* The zero vector is reasonable */
 
    ierr = VecSet(X, start1); CHKERRQ(ierr);

  } else if (flg2 && start2>0){ /* Try a random start between -0.5 and 0.5 */

    PetscRandom rctx;  PetscReal np5=-0.5;

    ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rctx); CHKERRQ(ierr);
    ierr = PetscRandomSetType(rctx,PETSCRAND); 
    CHKERRQ(ierr);
    for (i=0; i<start2; i++){
      ierr = VecSetRandom(X, rctx); CHKERRQ(ierr);
    }
    ierr = PetscRandomDestroy(&rctx); CHKERRQ(ierr);
    ierr = VecShift(X, np5); CHKERRQ(ierr);

  } else { /* Take an average of the boundary conditions */

    PetscInt xs,xm,ys,ym;
    PetscInt mx=user->mx,my=user->my;
    PetscReal **x;
    
    /* Get local mesh boundaries */
    ierr = DMDAGetCorners(user->dm,&xs,&ys,PETSC_NULL,&xm,&ym,PETSC_NULL); CHKERRQ(ierr);
    
    /* Get pointers to vector data */
    ierr = DMDAVecGetArray(user->dm,X,(void**)&x);

    /* Perform local computations */    
    for (j=ys; j<ys+ym; j++){
      for (i=xs; i< xs+xm; i++){
	x[j][i] = ( ((j+1)*user->bottom[i-xs+1]+(my-j+1)*user->top[i-xs+1])/(my+2)+
		   ((i+1)*user->left[j-ys+1]+(mx-i+1)*user->right[j-ys+1])/(mx+2))/2.0; 
      }
    }
    
    /* Restore vectors */
    ierr = DMDAVecRestoreArray(user->dm,X,(void**)&x);  CHKERRQ(ierr);

    ierr = PetscLogFlops(9*xm*ym); CHKERRQ(ierr);
    
  }
  PetscFunctionReturn(0);
}

/*-----------------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "My_Monitor"
PetscErrorCode My_Monitor(TaoSolver tao, void *ctx){
  PetscErrorCode ierr;
  Vec X;
  PetscFunctionBegin;
  ierr = TaoSolverGetSolutionVector(tao,&X); CHKERRQ(ierr);
  ierr = VecView(X,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
