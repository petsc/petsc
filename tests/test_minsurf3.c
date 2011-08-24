/* Program usage: mpirun -np 1 minsurf1 [-help] [all TAO options] */

/*  Include "tao.h" so we can use TAO solvers.  */
#include "taosolver.h"

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
   Concepts: TAO - Solving an unconstrained minimization problem
   Routines: TaoInitialize(); TaoFinalize();
   Routines: TaoCreate(); TaoDestroy();
   Routines: TaoApplicationCreate(); TaoAppDestroy();
   Routines: TaoAppSetObjectiveAndGradientRoutine(); TaoAppSetInitialSolutionVec();
   Routines: TaoAppSetHessianMat(); TaoSetHessianRoutine();
   Routines: TaoSetFromOptions(); TaoAppGetKSP();
   Routines: TaoSolve(); TaoView();
   Processors: 1
T*/

/* 
   User-defined application context - contains data needed by the 
   application-provided call-back routines, FormFunctionGradient() 
   and FormHessian().
*/
typedef struct {
  PetscInt    mx, my;                 /* discretization in x, y directions */
  PetscReal *bottom, *top, *left, *right;             /* boundary values */
  Mat         H;
} AppCtx;

/* -------- User-defined Routines --------- */

static PetscErrorCode MSA_BoundaryConditions(AppCtx*);
static PetscErrorCode MSA_InitialPoint(AppCtx*,Vec);
//static PetscErrorCode QuadraticH(AppCtx*,Vec,Mat);
PetscErrorCode FormSeparableFunction(TaoSolver,Vec,Vec,void*);
//PetscErrorCode FormHessian(TaoSolver,Vec,Mat*,Mat*,MatStructure *,void*);

#undef __FUNCT__
#define __FUNCT__ "main"
int main( int argc, char **argv )
{
  PetscErrorCode    ierr;              /* used to check for functions returning nonzeros */
  PetscInt          N;                 /* Size of vector */
  PetscMPIInt       size;              /* Number of processors */
  Vec             F,x;                 /* solution, gradient vectors */
  PetscBool       flg;               /* A return value when checking for user options */
  TaoSolverTerminationReason reason;
  TaoSolver       tao;               /* TaoSolver solver context */
  AppCtx          user;              /* user-defined work context */

  /* Initialize TAO,PETSc */
  PetscInitialize( &argc, &argv,(char *)0,help );
  TaoInitialize( &argc, &argv,(char *)0,help );

  MPI_Comm_size(MPI_COMM_WORLD,&size);
  if (size >1) {
    PetscPrintf(PETSC_COMM_SELF,"This example is intended for single processor use!\n");
    PetscPrintf(PETSC_COMM_SELF,"Try the example minsurf2!\n");
    SETERRQ(PETSC_COMM_SELF,1,"Incorrect number of processors");
  }

  /* Specify default dimension of the problem */
  user.mx = 4; user.my = 4;

  /* Check for any command line arguments that override defaults */
  ierr = PetscOptionsGetInt(PETSC_NULL,"-mx",&user.mx,&flg); CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-my",&user.my,&flg); CHKERRQ(ierr);

  PetscPrintf(PETSC_COMM_SELF,"\n---- Minimum Surface Area Problem -----\n");
  PetscPrintf(PETSC_COMM_SELF,"mx: %d     my: %d   \n\n",user.mx,user.my);

  /* Calculate any derived values from parameters */
  N    = user.mx*user.my;

  /* Create TAO solver and set desired solution method  */
  ierr = TaoSolverCreate(PETSC_COMM_SELF,&tao); CHKERRQ(ierr);
  ierr = TaoSolverSetType(tao,"tao_pounders"); CHKERRQ(ierr);

  /* Initialize minsurf application data structure for use in the function evaluations  */
  ierr = MSA_BoundaryConditions(&user); CHKERRQ(ierr);            /* Application specific routine */

  /*
     Create a vector to hold the variables.  Compute an initial solution.
     Set this vector, which will be used by TAO.
  */
  ierr = VecCreateSeq(PETSC_COMM_SELF,N,&x);                      /* PETSc routine                */
  ierr = MSA_InitialPoint(&user,x); CHKERRQ(ierr);                /* Application specific routine */
  ierr = TaoSolverSetInitialVector(tao,x); CHKERRQ(ierr);   /* A TAO routine                */

  /* Provide TAO routines for function, gradient, and Hessian evaluation */
  ierr = VecCreateSeq(PETSC_COMM_SELF,1,&F); CHKERRQ(ierr);
  ierr = TaoSolverSetSeparableObjectiveRoutine(tao,F,FormSeparableFunction,(void *)&user); CHKERRQ(ierr);

  /* Check for any TAO command line options */
  ierr = TaoSolverSetFromOptions(tao); CHKERRQ(ierr);

  /* Limit the number of iterations in the KSP linear solver */
/*  ierr = TaoAppGetKSP(minsurfapp,&ksp); CHKERRQ(ierr);
  if (ksp) {                                            
    ierr = KSPSetTolerances(ksp,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,user.mx*user.my);
    CHKERRQ(ierr);
    }*/

  /* SOLVE THE APPLICATION */
  ierr = TaoSolverSolve(tao); CHKERRQ(ierr);

  /* Get information on termination */
  ierr = TaoSolverGetTerminationReason(tao,&reason); CHKERRQ(ierr);
  if (reason <= 0)
    PetscPrintf(MPI_COMM_WORLD,"Try a different TAO method, adjust some parameters, or check the function evaluation routines\n");

  /*
    To View TAO solver information use
     ierr = TaoView(tao); CHKERRQ(ierr); 
  */

  /* Free TAO data structures */
  ierr = TaoSolverDestroy(&tao); CHKERRQ(ierr);

  /* Free PETSc data structures */
  ierr = VecDestroy(&x); CHKERRQ(ierr);
  ierr = VecDestroy(&F); CHKERRQ(ierr);
  ierr = PetscFree(user.bottom); CHKERRQ(ierr);
  ierr = PetscFree(user.top); CHKERRQ(ierr);
  ierr = PetscFree(user.left); CHKERRQ(ierr);
  ierr = PetscFree(user.right); CHKERRQ(ierr);

  /* Finalize TAO */
  TaoFinalize();
  PetscFinalize();
  
  return 0;
}


/* -------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "FormSeparableFunction"

/*  FormFunctionGradient - Evaluates function and corresponding gradient.             

    Input Parameters:
.   tao     - the TaoSolver context
.   X       - input vector
.   userCtx - optional user-defined context, as set by TaoSetFunctionGradient()
    
    Output Parameters:
.   fcn     - the newly evaluated function
.   G       - vector containing the newly evaluated gradient
*/
PetscErrorCode FormSeparableFunction(TaoSolver tao,Vec X,Vec F,void *userCtx) {

  AppCtx * user = (AppCtx *) userCtx;
  PetscErrorCode ierr;
  PetscInt i,j,row;
  PetscInt mx=user->mx, my=user->my;
  PetscReal rhx=mx+1, rhy=my+1,fcn;
  PetscReal hx=1.0/(mx+1),hy=1.0/(my+1), hydhx=hy/hx, hxdhy=hx/hy, area=0.5*hx*hy, ft=0;
  PetscReal f1,f2,f3,f4,f5,f6,d1,d2,d3,d4,d5,d6,d7,d8,xc,xl,xr,xt,xb,xlt,xrb;
  PetscReal df1dxc,df2dxc,df3dxc,df4dxc,df5dxc,df6dxc;
  PetscReal *x;

  /* Get pointers to vector data */
  ierr = VecGetArray(X,&x); CHKERRQ(ierr);

  /* Compute function over the locally owned part of the mesh */
  for (j=0; j<my; j++){
    for (i=0; i< mx; i++){
      row=(j)*mx + (i);
      
      xc = x[row];
      xlt=xrb=xl=xr=xb=xt=xc;
      
      if (i==0){ /* left side */
        xl= user->left[j+1];
        xlt = user->left[j+2];
      } else {
        xl = x[row-1];
      }

      if (j==0){ /* bottom side */
        xb=user->bottom[i+1];
        xrb = user->bottom[i+2];
      } else {
        xb = x[row-mx];
      }
      
      if (i+1 == mx){ /* right side */
        xr=user->right[j+1];
        xrb = user->right[j];
      } else {
        xr = x[row+1];
      }

      if (j+1==0+my){ /* top side */
        xt=user->top[i+1];
        xlt = user->top[i];
      }else {
        xt = x[row+mx];
      }

      if (i>0 && j+1<my){
        xlt = x[row-1+mx];
      }
      if (j>0 && i+1<mx){
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

   }
  }
  
  for (j=0; j<my; j++){   /* left side */
    d3=(user->left[j+1] - user->left[j+2])*rhy;
    d2=(user->left[j+1] - x[j*mx])*rhx;
    ft = ft+sqrt( 1.0 + d3*d3 + d2*d2);
  }
  
  for (i=0; i<mx; i++){ /* bottom */
    d2=(user->bottom[i+1]-user->bottom[i+2])*rhx;
    d3=(user->bottom[i+1]-x[i])*rhy;
    ft = ft+sqrt( 1.0 + d3*d3 + d2*d2);
  }
  
  for (j=0; j< my; j++){ /* right side */
    d1=(x[(j+1)*mx-1]-user->right[j+1])*rhx;
    d4=(user->right[j]-user->right[j+1])*rhy;
    ft = ft+sqrt( 1.0 + d1*d1 + d4*d4);
  }
  
  for (i=0; i<mx; i++){ /* top side */
    d1=(x[(my-1)*mx + i] - user->top[i+1])*rhy;
    d4=(user->top[i+1] - user->top[i])*rhx;
    ft = ft+sqrt( 1.0 + d1*d1 + d4*d4);
  }
  
  /* Bottom left corner */
  d1=(user->left[0]-user->left[1])*rhy;
  d2=(user->bottom[0]-user->bottom[1])*rhx;
  ft +=sqrt( 1.0 + d1*d1 + d2*d2);

  /* Top right corner */
  d1=(user->right[my+1] - user->right[my])*rhy;
  d2=(user->top[mx+1] - user->top[mx])*rhx;
  ft +=sqrt( 1.0 + d1*d1 + d2*d2);

  (fcn)=ft*area;

  /* Restore vectors */
  ierr = VecRestoreArray(X,&x); CHKERRQ(ierr);

  ierr = VecSetValue(F,0,fcn,INSERT_VALUES); CHKERRQ(ierr);
  ierr = VecAssemblyBegin(F); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(F); CHKERRQ(ierr);
  ierr = PetscLogFlops(67*mx*my); CHKERRQ(ierr);

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
  PetscErrorCode ierr;
  PetscInt     i,j,k,limit=0;
  PetscInt     maxits=5;
  PetscInt     mx=user->mx,my=user->my;
  PetscInt     bsize=0, lsize=0, tsize=0, rsize=0;
  PetscReal     one=1.0, two=2.0, three=3.0, tol=1e-10;
  PetscReal     fnorm,det,hx,hy,xt=0,yt=0;
  PetscReal     u1,u2,nf1,nf2,njac11,njac12,njac21,njac22;
  PetscReal     b=-0.5, t=0.5, l=-0.5, r=0.5;
  PetscReal     *boundary;

  bsize=mx+2; lsize=my+2; rsize=my+2; tsize=mx+2;

  ierr = PetscMalloc(bsize*sizeof(PetscReal),&user->bottom); CHKERRQ(ierr);
  ierr = PetscMalloc(tsize*sizeof(PetscReal),&user->top); CHKERRQ(ierr);
  ierr = PetscMalloc(lsize*sizeof(PetscReal),&user->left); CHKERRQ(ierr);
  ierr = PetscMalloc(rsize*sizeof(PetscReal),&user->right); CHKERRQ(ierr);

  hx= (r-l)/(mx+1); hy=(t-b)/(my+1);

  for (j=0; j<4; j++){
    if (j==0){
      yt=b;
      xt=l;
      limit=bsize;
      boundary=user->bottom;
    } else if (j==1){
      yt=t;
      xt=l;
      limit=tsize;
      boundary=user->top;
    } else if (j==2){
      yt=b;
      xt=l;
      limit=lsize;
      boundary=user->left;
    } else {  // if (j==3)
      yt=b;
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
  PetscInt      start=-1,i,j;
  PetscErrorCode ierr;
  PetscReal   zero=0.0;
  PetscBool flg;

  ierr = VecSet(X, zero); CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-start",&start,&flg); CHKERRQ(ierr);

  if (flg && start==0){ /* The zero vector is reasonable */
 
    ierr = VecSet(X, zero); CHKERRQ(ierr);
    /* PetscLogInfo((user,"Min. Surface Area Problem: Start with 0 vector \n")); */


  } else { /* Take an average of the boundary conditions */

    PetscInt    row;
    PetscInt    mx=user->mx,my=user->my;
    PetscReal *x;
    
    /* Get pointers to vector data */
    ierr = VecGetArray(X,&x); CHKERRQ(ierr);

    /* Perform local computations */    
    for (j=0; j<my; j++){
      for (i=0; i< mx; i++){
	row=(j)*mx + (i);
	x[row] = ( ((j+1)*user->bottom[i+1]+(my-j+1)*user->top[i+1])/(my+2)+
		   ((i+1)*user->left[j+1]+(mx-i+1)*user->right[j+1])/(mx+2))/2.0;
      }
    }
    
    /* Restore vectors */
    ierr = VecRestoreArray(X,&x); CHKERRQ(ierr);
    
  }
  return 0;
}
