/*$Id$*/

/*
  Include "tao.h" so we can use TAO solvers with PETSc support.  
  Include "petscdm.h" so that we can use distributed arrays (DMs) for managing
  the parallel mesh.
*/

#include "petscdm.h"
#include "petscksp.h"
#include "taosolver.h"
#include <math.h>  /* for cos() sin(0), and atan() */

static  char help[]=
"This example demonstrates use of the TAO package to \n\
solve a bound constrained minimization problem.  This example is based on \n\
the problem DPJB from the MINPACK-2 test suite.  This pressure journal \n\
bearing problem is an example of elliptic variational problem defined over \n\
a two dimensional rectangle.  By discretizing the domain into triangular \n\
elements, the pressure surrounding the journal bearing is defined as the \n\
minimum of a quadratic function whose variables are bounded below by zero.\n\
The command line options are:\n\
  -mx <xg>, where <xg> = number of grid points in the 1st coordinate direction\n\
  -my <yg>, where <yg> = number of grid points in the 2nd coordinate direction\n\
 \n";

/*T
   Concepts: TAO - Solving a bound constrained minimization problem
   Routines: TaoInitialize(); TaoFinalize();
   Routines: TaoApplicationCreate();  TaoAppDestroy();
   Routines: TaoAppSetObjectiveAndGradientRoutine(); TaoAppSetHessianRoutine();
   Routines: TaoAppSetInitialSolutionVec(); TaoAppSetHessianMat();
   Routines: TaoCreate(); TaoDestroy();
   Routines: TaoSetOptions(); TaoApplyGetKSP()
   Routines: TaoSolveApplication(); TaoGetTerminationReason();
   Processors: n
T*/

/* 
   User-defined application context - contains data needed by the 
   application-provided call-back routines, FormFunctionGradient(),
   FormHessian().
*/
typedef struct {
  /* problem parameters */
  PetscReal      ecc;          /* test problem parameter */
  PetscReal      b;            /* A dimension of journal bearing */
  int         nx,ny;        /* discretization in x, y directions */

  /* Working space */
  DM          dm;           /* distributed array data structure */
  Mat         A;            /* Quadratic Objective term */
  Vec         B;            /* Linear Objective term */
} AppCtx;

/* User-defined routines */
static PetscReal p(PetscReal xi, PetscReal ecc);
static PetscErrorCode FormFunctionGradient(TaoSolver, Vec, PetscReal *,Vec,void *);
static PetscErrorCode FormHessian(TaoSolver,Vec,Mat *, Mat *, MatStructure *, void *);
static PetscErrorCode ComputeB(AppCtx*);


#undef __FUNCT__
#define __FUNCT__ "main"
int main( int argc, char **argv )
{
  PetscErrorCode        info;               /* used to check for functions returning nonzeros */
  PetscInt        Nx, Ny;             /* number of processors in x- and y- directions */
  PetscInt        m, N;               /* number of local and global elements in vectors */
  Vec        x;                  /* variables vector */
  Vec        xl,xu;                  /* bounds vectors */
  PetscReal d1000 = 1000;
  PetscBool   flg;              /* A return variable when checking for user options */
  TaoSolver tao;                /* TAO_SOLVER solver context */

  ISLocalToGlobalMapping isltog; /* local-to-global mapping object */
  PetscInt nloc;                      /* The number of local elements */
  PetscInt *ltog;                     /* mapping of local elements to global elements */
  TaoSolverTerminationReason reason;
  AppCtx     user;               /* user-defined work context */
  PetscReal     zero=0.0;           /* lower bound on all variables */


  
  /* Initialize PETSC and TAO */
  PetscInitialize( &argc, &argv,(char *)0,help );
  TaoInitialize( &argc, &argv,(char *)0,help );

  /* Set the default values for the problem parameters */
  user.nx = 50; user.ny = 50; user.ecc = 0.1; user.b = 10.0;

  /* Check for any command line arguments that override defaults */
  info = PetscOptionsGetInt(TAO_NULL,"-mx",&user.nx,&flg); CHKERRQ(info);
  info = PetscOptionsGetInt(TAO_NULL,"-my",&user.ny,&flg); CHKERRQ(info);
  info = PetscOptionsGetReal(TAO_NULL,"-ecc",&user.ecc,&flg); CHKERRQ(info);
  info = PetscOptionsGetReal(TAO_NULL,"-b",&user.b,&flg); CHKERRQ(info);


  PetscPrintf(PETSC_COMM_WORLD,"\n---- Journal Bearing Problem SHB-----\n");
  PetscPrintf(PETSC_COMM_WORLD,"mx: %d,  my: %d,  ecc: %4.3f \n\n",
	      user.nx,user.ny,user.ecc);

  /* Calculate any derived values from parameters */
  N = user.nx*user.ny; 

  /* Let Petsc determine the grid division */
  Nx = PETSC_DECIDE; Ny = PETSC_DECIDE;

  /*
     A two dimensional distributed array will help define this problem,
     which derives from an elliptic PDE on two dimensional domain.  From
     the distributed array, Create the vectors.
  */
  info = DMDACreate2d(PETSC_COMM_WORLD,DMDA_NONPERIODIC,DMDA_STENCIL_STAR,
		      user.nx,user.ny,Nx,Ny,1,1,PETSC_NULL,PETSC_NULL,
		      &user.dm); CHKERRQ(info);

  /*
     Extract global and local vectors from DM; the vector user.B is
     used solely as work space for the evaluation of the function, 
     gradient, and Hessian.  Duplicate for remaining vectors that are 
     the same types.
  */
  info = DMCreateGlobalVector(user.dm,&x); CHKERRQ(info); /* Solution */
  info = VecDuplicate(x,&user.B); CHKERRQ(info); /* Linear objective */


  /*  Create matrix user.A to store quadratic, Create a local ordering scheme. */
  info = VecGetLocalSize(x,&m); CHKERRQ(info);
  info = MatCreateMPIAIJ(PETSC_COMM_WORLD,m,m,N,N,5,TAO_NULL,3,TAO_NULL,&user.A); CHKERRQ(info);



  info = DMDAGetGlobalIndices(user.dm,&nloc,&ltog); CHKERRQ(info);
  info = ISLocalToGlobalMappingCreate(PETSC_COMM_SELF,nloc,ltog,PETSC_COPY_VALUES,&isltog); 
  CHKERRQ(info);
  info = MatSetLocalToGlobalMapping(user.A,isltog,isltog); CHKERRQ(info);
  info = ISLocalToGlobalMappingDestroy(isltog); CHKERRQ(info);


  /* User defined function -- compute linear term of quadratic */
  info = ComputeB(&user); CHKERRQ(info);


  /* The TAO code begins here */

  /* 
     Create the optimization solver, Petsc application 
     Suitable methods: "tao_gpcg","tao_bqpip","tao_tron","tao_blmvm" 
  */
  info = TaoSolverCreate(PETSC_COMM_WORLD,&tao); CHKERRQ(info);
  info = TaoSolverSetType(tao,"tao_blmvm"); CHKERRQ(info);


  /* Set the initial vector */
  info = VecSet(x, zero); CHKERRQ(info);
  info = TaoSolverSetInitialVector(tao,x); CHKERRQ(info);

  /* Set the user function, gradient, hessian evaluation routines and data structures */
  info = TaoSolverSetObjectiveAndGradientRoutine(tao,FormFunctionGradient,(void*) &user); 
  CHKERRQ(info);
  
  info = TaoSolverSetHessianRoutine(tao,user.A,user.A,FormHessian,(void*)&user); CHKERRQ(info);

  /* Set a routine that defines the bounds */
  info = VecDuplicate(x,&xl); CHKERRQ(info);
  info = VecDuplicate(x,&xu); CHKERRQ(info);
  info = VecSet(xl, zero); CHKERRQ(info);
  info = VecSet(xu, d1000); CHKERRQ(info);
  info = TaoSolverSetVariableBounds(tao,xl,xu); CHKERRQ(info);

/*  info = TaoAppGetKSP(jbearingapp,&ksp); CHKERRQ(info);
  if (ksp) {                                         
    info = KSPSetType(ksp,KSPCG); CHKERRQ(info);
  }
*/


  /* Check for any tao command line options */
  info = TaoSolverSetFromOptions(tao); CHKERRQ(info);

  /* Solve the bound constrained problem */
  info = TaoSolverSolve(tao); CHKERRQ(info);

  //info = TaoSolverView(tao,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(info);

  info = TaoSolverGetConvergedReason(tao,&reason); CHKERRQ(info);
  if (reason <= 0)
    PetscPrintf(PETSC_COMM_WORLD,"Try a different TAO method, adjust some parameters, or check the function evaluation routines\n");

  /* Free TAO data structures */
  info = TaoSolverDestroy(tao); CHKERRQ(info);

  /* Free PETSc data structures */
  info = VecDestroy(x); CHKERRQ(info); 
  info = VecDestroy(xl); CHKERRQ(info); 
  info = VecDestroy(xu); CHKERRQ(info); 
  info = MatDestroy(user.A); CHKERRQ(info);
  info = VecDestroy(user.B); CHKERRQ(info); 
  info = DMDestroy(user.dm); CHKERRQ(info);

  TaoFinalize();
  PetscFinalize();

  return 0;
}


static PetscReal p(PetscReal xi, PetscReal ecc)
{ 
  PetscReal t=1.0+ecc*cos(xi); 
  return (t*t*t); 
}

#undef __FUNCT__
#define __FUNCT__ "ComputeB"
PetscErrorCode ComputeB(AppCtx* user)
{
  PetscErrorCode info;
  PetscInt i,j,k;
  PetscInt nx,ny,xs,xm,gxs,gxm,ys,ym,gys,gym;
  PetscReal two=2.0, pi=4.0*atan(1.0);
  PetscReal hx,hy,ehxhy;
  PetscReal temp,*b;
  PetscReal ecc=user->ecc;

  nx=user->nx;
  ny=user->ny;
  hx=two*pi/(nx+1.0);
  hy=two*user->b/(ny+1.0);
  ehxhy = ecc*hx*hy;


  /*
     Get local grid boundaries
  */
  info = DMDAGetCorners(user->dm,&xs,&ys,TAO_NULL,&xm,&ym,TAO_NULL); CHKERRQ(info);
  info = DMDAGetGhostCorners(user->dm,&gxs,&gys,TAO_NULL,&gxm,&gym,TAO_NULL); CHKERRQ(info);
  

  /* Compute the linear term in the objective function */  
  info = VecGetArray(user->B,&b); CHKERRQ(info);
  for (i=xs; i<xs+xm; i++){
    temp=sin((i+1)*hx);
    for (j=ys; j<ys+ym; j++){
      k=xm*(j-ys)+(i-xs);
      b[k]=  - ehxhy*temp;
    }
  }
  info = VecRestoreArray(user->B,&b); CHKERRQ(info);
  info = PetscLogFlops(5*xm*ym+3*xm); CHKERRQ(info);

  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "FormFunctionGradient"
PetscErrorCode FormFunctionGradient(TaoSolver tao, Vec X, PetscReal *fcn,Vec G,void *ptr)
{
  AppCtx* user=(AppCtx*)ptr;
  PetscErrorCode info;
  PetscInt i,j,k,kk;
  PetscInt col[5],row,nx,ny,xs,xm,gxs,gxm,ys,ym,gys,gym;
  PetscReal one=1.0, two=2.0, six=6.0,pi=4.0*atan(1.0);
  PetscReal hx,hy,hxhy,hxhx,hyhy;
  PetscReal xi,v[5];
  PetscReal ecc=user->ecc, trule1,trule2,trule3,trule4,trule5,trule6;
  PetscReal vmiddle, vup, vdown, vleft, vright;
  PetscReal tt,f1,f2;
  PetscReal *x,*g,zero=0.0;
  Vec localX;

  nx=user->nx;
  ny=user->ny;
  hx=two*pi/(nx+1.0);
  hy=two*user->b/(ny+1.0);
  hxhy=hx*hy;
  hxhx=one/(hx*hx);
  hyhy=one/(hy*hy);

  info = DMGetLocalVector(user->dm,&localX);CHKERRQ(info);

  info = DMGlobalToLocalBegin(user->dm,X,INSERT_VALUES,localX); CHKERRQ(info);
  info = DMGlobalToLocalEnd(user->dm,X,INSERT_VALUES,localX); CHKERRQ(info);

  info = VecSet(G, zero); CHKERRQ(info);
  /*
    Get local grid boundaries
  */
  info = DMDAGetCorners(user->dm,&xs,&ys,TAO_NULL,&xm,&ym,TAO_NULL); CHKERRQ(info);
  info = DMDAGetGhostCorners(user->dm,&gxs,&gys,TAO_NULL,&gxm,&gym,TAO_NULL); CHKERRQ(info);
  
  info = VecGetArray(localX,&x); CHKERRQ(info);
  info = VecGetArray(G,&g); CHKERRQ(info);

  for (i=xs; i< xs+xm; i++){
    xi=(i+1)*hx;
    trule1=hxhy*( p(xi,ecc) + p(xi+hx,ecc) + p(xi,ecc) ) / six; /* L(i,j) */
    trule2=hxhy*( p(xi,ecc) + p(xi-hx,ecc) + p(xi,ecc) ) / six; /* U(i,j) */
    trule3=hxhy*( p(xi,ecc) + p(xi+hx,ecc) + p(xi+hx,ecc) ) / six; /* U(i+1,j) */
    trule4=hxhy*( p(xi,ecc) + p(xi-hx,ecc) + p(xi-hx,ecc) ) / six; /* L(i-1,j) */
    trule5=trule1; /* L(i,j-1) */
    trule6=trule2; /* U(i,j+1) */

    vdown=-(trule5+trule2)*hyhy;
    vleft=-hxhx*(trule2+trule4);
    vright= -hxhx*(trule1+trule3);
    vup=-hyhy*(trule1+trule6);
    vmiddle=(hxhx)*(trule1+trule2+trule3+trule4)+hyhy*(trule1+trule2+trule5+trule6);

    for (j=ys; j<ys+ym; j++){
      
      row=(j-gys)*gxm + (i-gxs);
       v[0]=0; v[1]=0; v[2]=0; v[3]=0; v[4]=0;
       
       k=0;
       if (j>gys){ 
	 v[k]=vdown; col[k]=row - gxm; k++;
       }
       
       if (i>gxs){
	 v[k]= vleft; col[k]=row - 1; k++;
       }

       v[k]= vmiddle; col[k]=row; k++;
       
       if (i+1 < gxs+gxm){
	 v[k]= vright; col[k]=row+1; k++;
       }
       
       if (j+1 <gys+gym){
	 v[k]= vup; col[k] = row+gxm; k++;
       }
       tt=0;
       for (kk=0;kk<k;kk++){
	 tt+=v[kk]*x[col[kk]];
       }
       row=(j-ys)*xm + (i-xs);
       g[row]=tt;

     }

  }

  info = VecRestoreArray(localX,&x); CHKERRQ(info);
  info = VecRestoreArray(G,&g); CHKERRQ(info);

  info = DMRestoreLocalVector(user->dm,&localX); CHKERRQ(info);

  info = VecDot(X,G,&f1); CHKERRQ(info);
  info = VecDot(user->B,X,&f2); CHKERRQ(info);
  info = VecAXPY(G, one, user->B); CHKERRQ(info);
  *fcn = f1/2.0 + f2;
  VecNorm(G,NORM_2,&f2);
  PetscPrintf(PETSC_COMM_SELF,"fcn=%f\t gnorm=%f\n",*fcn,f2);

  info = PetscLogFlops((91 + 10*ym) * xm); CHKERRQ(info);
  return 0;

}


#undef __FUNCT__
#define __FUNCT__ "FormHessian"
/* 
   FormHessian computes the quadratic term in the quadratic objective function 
   Notice that the objective function in this problem is quadratic (therefore a constant
   hessian).  If using a nonquadratic solver, then you might want to reconsider this function
*/
PetscErrorCode FormHessian(TaoSolver tao,Vec X,Mat *H, Mat *Hpre, MatStructure *flg, void *ptr)
{
  AppCtx* user=(AppCtx*)ptr;
  PetscErrorCode info;
  PetscInt i,j,k;
  PetscInt col[5],row,nx,ny,xs,xm,gxs,gxm,ys,ym,gys,gym;
  PetscReal one=1.0, two=2.0, six=6.0,pi=4.0*atan(1.0);
  PetscReal hx,hy,hxhy,hxhx,hyhy;
  PetscReal xi,v[5];
  PetscReal ecc=user->ecc, trule1,trule2,trule3,trule4,trule5,trule6;
  PetscReal vmiddle, vup, vdown, vleft, vright;
  Mat hes=*H;
  PetscBool assembled;

  nx=user->nx;
  ny=user->ny;
  hx=two*pi/(nx+1.0);
  hy=two*user->b/(ny+1.0);
  hxhy=hx*hy;
  hxhx=one/(hx*hx);
  hyhy=one/(hy*hy);

  *flg=SAME_NONZERO_PATTERN;
  /*
    Get local grid boundaries
  */
  info = DMDAGetCorners(user->dm,&xs,&ys,TAO_NULL,&xm,&ym,TAO_NULL); CHKERRQ(info);
  info = DMDAGetGhostCorners(user->dm,&gxs,&gys,TAO_NULL,&gxm,&gym,TAO_NULL); CHKERRQ(info);
  
  info = MatAssembled(hes,&assembled); CHKERRQ(info);
  if (assembled){info = MatZeroEntries(hes);  CHKERRQ(info);}

  for (i=xs; i< xs+xm; i++){
    xi=(i+1)*hx;
    trule1=hxhy*( p(xi,ecc) + p(xi+hx,ecc) + p(xi,ecc) ) / six; /* L(i,j) */
    trule2=hxhy*( p(xi,ecc) + p(xi-hx,ecc) + p(xi,ecc) ) / six; /* U(i,j) */
    trule3=hxhy*( p(xi,ecc) + p(xi+hx,ecc) + p(xi+hx,ecc) ) / six; /* U(i+1,j) */
    trule4=hxhy*( p(xi,ecc) + p(xi-hx,ecc) + p(xi-hx,ecc) ) / six; /* L(i-1,j) */
    trule5=trule1; /* L(i,j-1) */
    trule6=trule2; /* U(i,j+1) */

    vdown=-(trule5+trule2)*hyhy;
    vleft=-hxhx*(trule2+trule4);
    vright= -hxhx*(trule1+trule3);
    vup=-hyhy*(trule1+trule6);
    vmiddle=(hxhx)*(trule1+trule2+trule3+trule4)+hyhy*(trule1+trule2+trule5+trule6);
    v[0]=0; v[1]=0; v[2]=0; v[3]=0; v[4]=0;

    for (j=ys; j<ys+ym; j++){
      row=(j-gys)*gxm + (i-gxs);
       
      k=0;
      if (j>gys){ 
	v[k]=vdown; col[k]=row - gxm; k++;
      }
       
      if (i>gxs){
	v[k]= vleft; col[k]=row - 1; k++;
      }

      v[k]= vmiddle; col[k]=row; k++;
       
      if (i+1 < gxs+gxm){
	v[k]= vright; col[k]=row+1; k++;
      }
       
      if (j+1 <gys+gym){
	v[k]= vup; col[k] = row+gxm; k++;
      }
      info = MatSetValuesLocal(hes,1,&row,k,col,v,INSERT_VALUES); CHKERRQ(info);
       
    }

  }

  /* 
     Assemble matrix, using the 2-step process:
     MatAssemblyBegin(), MatAssemblyEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.
  */
  info = MatAssemblyBegin(hes,MAT_FINAL_ASSEMBLY); CHKERRQ(info);
  info = MatAssemblyEnd(hes,MAT_FINAL_ASSEMBLY); CHKERRQ(info);

  /*
    Tell the matrix we will never add a new nonzero location to the
    matrix. If we do it will generate an error.
  */
  info = MatSetOption(hes,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE); CHKERRQ(info);
  info = MatSetOption(hes,MAT_SYMMETRIC,PETSC_TRUE); CHKERRQ(info);

  info = PetscLogFlops(9*xm*ym+49*xm); CHKERRQ(info);
  info = MatNorm(hes,NORM_1,&hx); CHKERRQ(info);
  return 0;
}
