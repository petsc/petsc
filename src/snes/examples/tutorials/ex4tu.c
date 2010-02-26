/* Program usage:  mpiexec -n <procs> ex5 [-help] [all PETSc options] */

static char help[] = "Nonlinear PDE in 2d.\n\
We solve the Lane-Emden equation in a 2D rectangular\n\
domain, using distributed arrays (DAs) to partition the parallel grid.\n\n";

/*T
   Concepts: SNES^parallel Lane-Emden example
   Concepts: DA^using distributed arrays;
   Processors: n
T*/

/* ------------------------------------------------------------------------

    The Lane-Emden equation is given by the partial differential equation
  
            -alpha*Laplacian u - lambda*u^3 = 0,  0 < x,y < 1,
  
    with boundary conditions
   
             u = 0  for  x = 0, x = 1, y = 0, y = 1.
  
    A bilinear finite element approximation is used to discretize the boundary
    value problem to obtain a nonlinear system of equations.

  ------------------------------------------------------------------------- */

/* 
   Include "petscda.h" so that we can use distributed arrays (DAs).
   Include "petscsnes.h" so that we can use SNES solvers.  Note that this
   file automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h - vectors
     petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
     petscksp.h   - linear solvers
*/
#include "petscda.h"
#include "petscdmmg.h"
#include "petscsnes.h"
#include "../src/snes/impls/ls/ls.h"
/* 
   User-defined application context - contains data needed by the 
   application-provided call-back routines, FormJacobianLocal() and
   FormFunctionLocal().
*/
typedef struct {
  //  DA        da;             /* distributed array data structure */
   PetscReal alpha;          /* parameter controlling linearity */
   PetscReal lambda;         /* parameter controlling nonlinearity */
  PetscTruth     draw_contours;                /* flag - 1 indicates drawing contours */
} AppCtx;


/* 
   User-defined routines
*/
extern PetscErrorCode FormInitialGuess(DMMG,Vec);
extern PetscErrorCode FormFunctionLocal(DALocalInfo*,PetscScalar**,PetscScalar**,AppCtx*);
extern PetscErrorCode FormFunctionLocali(DALocalInfo*,MatStencil*,PetscScalar**,PetscScalar*,AppCtx*);
extern PetscErrorCode FormFunctionLocali4(DALocalInfo*,MatStencil*,PetscScalar**,PetscScalar*,AppCtx*);
extern PetscErrorCode FormJacobianLocal(DALocalInfo*,PetscScalar**,Mat,AppCtx*);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  DMMG                   *dmmg;                /* multilevel grid structure */
  SNES                   snes;                 /* nonlinear solver */
  //Vec                    x,r;                  /* solution, residual vectors */
  //Mat                    J;                    /* Jacobian matrix */
  AppCtx                 user;                 /* user-defined work context */
  PetscInt               its;                  /* iterations for convergence */
  SNESConvergedReason    reason;
  PetscErrorCode         ierr;
  PetscReal              lambda_max = 6.81, lambda_min = 0.0;
  MPI_Comm       comm;
   PetscInt       mx,my;
   DA                    da;
/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscInitialize(&argc,&argv,(char *)0,help);
comm = PETSC_COMM_WORLD;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize problem parameters
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  user.alpha = 1.0;
  user.lambda = 6.0;
  ierr = PetscOptionsGetReal(PETSC_NULL,"-alpha",&user.alpha,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(PETSC_NULL,"-lambda",&user.lambda,PETSC_NULL);CHKERRQ(ierr);
  if (user.lambda > lambda_max || user.lambda < lambda_min) {
    SETERRQ3(1,"Lambda %g is out of range [%g, %g]", user.lambda, lambda_min, lambda_max);
  }


  // in order only run once, I block it: PreLoadBegin(PETSC_TRUE,"SetUp");
    ierr = DMMGCreate(comm,2,&user,&dmmg);CHKERRQ(ierr);


    /*
      Create distributed array multigrid object (DMMG) to manage parallel grid and vectors
      for principal unknowns (x) and governing residuals (f)
    */
    ierr = DACreate2d(PETSC_COMM_WORLD,DA_NONPERIODIC,DA_STENCIL_BOX,-4,-4,PETSC_DECIDE,PETSC_DECIDE,1,1,PETSC_NULL,PETSC_NULL,&da);CHKERRQ(ierr);
    
    ierr = DMMGSetDM(dmmg,(DM)da);CHKERRQ(ierr);
    ierr = DADestroy(da);CHKERRQ(ierr);

    ierr = DAGetInfo(DMMGGetDA(dmmg),0,&mx,&my,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,
                     PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"mx = %d, my= %d\n",
		       mx,my);CHKERRQ(ierr);
 
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Create user context, set problem data, create vector data structures.
       Also, compute the initial guess.
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Create nonlinear solver context

      
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
     ierr = DMMGSetSNESLocal(dmmg,FormFunctionLocal,0,ad_FormFunctionLocal,admf_FormFunctionLocal);CHKERRQ(ierr);
     ierr = DMMGSetFromOptions(dmmg);CHKERRQ(ierr);
     ierr = DMMGSetSNESLocali(dmmg,FormFunctionLocali,ad_FormFunctionLocali,admf_FormFunctionLocali);CHKERRQ(ierr);
     ierr = DMMGSetSNESLocalib(dmmg,FormFunctionLocali4,ad_FormFunctionLocali4,admf_FormFunctionLocali4);CHKERRQ(ierr); 

 

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Solve the nonlinear system
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ierr = DMMGSetInitialGuess(dmmg,FormInitialGuess);CHKERRQ(ierr);

    //I block it:  PreLoadStage("Solve");
    ierr = DMMGSolve(dmmg);CHKERRQ(ierr); 

    snes = DMMGGetSNES(dmmg);
    ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"Number of Newton iterations = %D\n", its);CHKERRQ(ierr);
     
    /*
      Visualize solution
    */
    ierr = PetscOptionsHasName(PETSC_NULL,"-contours",&user.draw_contours);CHKERRQ(ierr);
    if (user.draw_contours) {
      ierr = VecView(DMMGGetx(dmmg),PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr); 
      //ierr = VecView(DMMGGetx(dmmg),PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Free work space.  All PETSc objects should be destroyed when they
       are no longer needed.
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    ierr = DMMGDestroy(dmmg);CHKERRQ(ierr);
    //PreLoadEnd();

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;

}

#undef __FUNCT__
#define __FUNCT__ "FormInitialGuess"
/* 
   FormInitialGuess - Forms initial approximation.

   Input Parameters:
   user - user-defined application context
   X - vector

   Output Parameter:
   X - vector
 */
PetscErrorCode FormInitialGuess(DMMG dmmg,Vec X)
{
  AppCtx         *user = (AppCtx*)dmmg->user;
  DA             da = (DA)dmmg->dm;
  PetscInt       i,j,Mx,My,xs,ys,xm,ym;
  PetscErrorCode ierr;
  PetscReal      lambda,temp1,temp,hx,hy;
  PetscScalar    **x;

  PetscFunctionBegin;
  ierr = DAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,
                   PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);
 
  lambda = user->lambda;
  hx     = 1.0/(PetscReal)(Mx-1);
  hy     = 1.0/(PetscReal)(My-1);
  if (lambda == 0.0) {
    temp1  = 1.0;
  } else {
    temp1  = lambda/(lambda + 1.0);
  }

  /*
     Get a pointer to vector data.
       - For default PETSc vectors, VecGetArray() returns a pointer to
         the data array.  Otherwise, the routine is implementation dependent.
       - You MUST call VecRestoreArray() when you no longer need access to
         the array.
  */
  ierr = DAVecGetArray(da,X,&x);CHKERRQ(ierr);

  /*
     Get local grid boundaries (for 2-dimensional DA):
       xs, ys   - starting grid indices (no ghost points)
       xm, ym   - widths of local grid (no ghost points)

  */
  ierr = DAGetCorners(da,&xs,&ys,PETSC_NULL,&xm,&ym,PETSC_NULL);CHKERRQ(ierr);

  /*
     Compute initial guess over the locally owned part of the grid
  */
  for (j=ys; j<ys+ym; j++) {
    temp = (PetscReal)(PetscMin(j,My-j-1))*hy;
    for (i=xs; i<xs+xm; i++) {

      if (i == 0 || j == 0 || i == Mx-1 || j == My-1) {
        /* boundary conditions are all zero Dirichlet */
        x[j][i] = 0.0; 
      } else {
        x[j][i] = temp1*sqrt(PetscMin((PetscReal)(PetscMin(i,Mx-i-1))*hx,temp)); 
      }
    }
  }

  /*
     Restore vector
  */
  ierr = DAVecRestoreArray(da,X,&x);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FormFunctionLocal"
/* 
   FormFunctionLocal - Evaluates nonlinear function, F(x).

       Process adiC(36): FormFunctionLocal nonlinearResidual FormFunctionLocali FormFunctionLocali4

 */
PetscErrorCode FormFunctionLocal(DALocalInfo *info,PetscScalar **x,PetscScalar **f,AppCtx *user)

{
  PetscScalar    uLocal[4];
  PetscScalar    rLocal[4];
  PetscScalar    lintx[4],linty[4],lintw[4],intnum=4, int1,int2,int3,int4,p1,p2,p3,p4,p;
  PetscReal      alpha,lambda,hx,hy,hxhy,sc;
  PetscInt       i,j,k,l;
  PetscErrorCode ierr;
 PetscFunctionBegin;
   /* Compute function over the locally owned part of the grid. For each
     vertex (i,j), we consider the element below:

       3         2
     i,j+1 --- i+1,j
       |         |
       |         |
      i,j  --- i+1,j
       0         1

     and therefore we do not loop over the last vertex in each dimension.
  */

  alpha  = user->alpha;
  lambda = user->lambda;
  hx     = 1.0/(PetscReal)(info->mx-1);
  hy     = 1.0/(PetscReal)(info->my-1);
  sc     = hx*hy*lambda;
  hxhy   = hx*hy; 


  // set all function values to be zero, this maybe not good for parallel computing
   for(j = info->ys; j < info->ys+info->ym-1; j++) {
    for(i = info->xs; i < info->xs+info->xm-1; i++) {
      f[j][i]=0.0;
    }
   }

 
   lintx[0] = 0.21132486540518713;//(1.0-sqrt(1.0/3.0))/2.0;
   lintx[1] = 0.78867513459481287;//(1.0+sqrt(1.0/3.0))/2.0;
   lintx[2] = 0.78867513459481287;//(1.0+sqrt(1.0/3.0))/2.0;
   lintx[3] = 0.21132486540518713;//(1.0-sqrt(1.0/3.0))/2.0;
   linty[0] = 0.21132486540518713;//(1.0-sqrt(1.0/3.0))/2.0;
   linty[1] = 0.21132486540518713;//(1.0-sqrt(1.0/3.0))/2.0;
   linty[2] = 0.78867513459481287;//(1.0+sqrt(1.0/3.0))/2.0;
   linty[3] = 0.78867513459481287;//(1.0+sqrt(1.0/3.0))/2.0;

   lintw[0] = 1.0/4.0;
   lintw[1] = 1.0/4.0;
   lintw[2] = 1.0/4.0;
   lintw[3] = 1.0/4.0;
  
       
  for(j = info->ys; j < info->ys+info->ym-1; j++) {
    for(i = info->xs; i < info->xs+info->xm-1; i++) {
      uLocal[0] = x[j][i];
      uLocal[1] = x[j][i+1];
      uLocal[2] = x[j+1][i+1];
      uLocal[3] = x[j+1][i];
      
     
      /* Laplace term */ 
        rLocal[0] =  2.0/3.0*uLocal[0]-1.0/6.0*uLocal[1]-1.0/3.0*uLocal[2]-1.0/6.0*uLocal[3];
        rLocal[0] *= hxhy*alpha;

        rLocal[1] = -1.0/6.0*uLocal[0]+2.0/3.0*uLocal[1]-1.0/6.0*uLocal[2]-1.0/3.0*uLocal[3];
        rLocal[1] *= hxhy*alpha;

        rLocal[2] = -1.0/3.0*uLocal[0]-1.0/6.0*uLocal[1]+2.0/3.0*uLocal[2]-1.0/6.0*uLocal[3];
        rLocal[2] *= hxhy*alpha;

        rLocal[3] = -1.0/6.0*uLocal[0]-1.0/3.0*uLocal[1]-1.0/6.0*uLocal[2]+2.0/3.0*uLocal[3];
        rLocal[3] *= hxhy*alpha;
      
	/* nonlinear term */
        int1 = 0;        int2 = 0;        int3 = 0;        int4 = 0;
        for(k=0; k<intnum; k++){
          p1   = (1.0-lintx[k])*(1.0-linty[k]);
          p2   =  lintx[k]     *(1.0-linty[k]);
          p3   =  lintx[k]     * linty[k];
          p4   = (1.0-lintx[k])* linty[k];
	  p    =  lintw[k]*PetscExpScalar( uLocal[0]*p1+ uLocal[1]*p2 + uLocal[2]*p3+ uLocal[3]*p4 );
	  // p    =  lintw[k]*( uLocal[0]*p1+ uLocal[1]*p2 + uLocal[2]*p3+ uLocal[3]*p4 )*( uLocal[0]*p1+ uLocal[1]*p2 + uLocal[2]*p3+ uLocal[3]*p4 )*( uLocal[0]*p1+ uLocal[1]*p2 + uLocal[2]*p3+ uLocal[3]*p4 );
          int1 = int1 + p*p1;
          int2 = int2 + p*p2;  
    	  int3 = int3 + p*p3;
	  int4 = int4 + p*p4;
        }
     
	f[j][i]     += rLocal[0] + int1*hxhy*(-1.0*sc);
	f[j][i+1]   += rLocal[1] + int2*hxhy*(-1.0*sc);
	f[j+1][i+1] += rLocal[2] + int3*hxhy*(-1.0*sc);
	f[j+1][i]   += rLocal[3] + int4*hxhy*(-1.0*sc);
      
      if (i == 0 || j == 0) {
        f[j][i] = x[j][i];
      }
      if ((i == info->mx-2) || (j == 0)) {
        f[j][i+1] = x[j][i+1];
      }
      if ((i == info->mx-2) || (j == info->my-2)) {
        f[j+1][i+1] = x[j+1][i+1];
      }
      if ((i == 0) || (j == info->my-2)) {
        f[j+1][i] = x[j+1][i];
      }
    }
  }

  ierr = PetscLogFlops(68.0*(info->ym-1)*(info->xm-1));CHKERRQ(ierr);
  PetscFunctionReturn(0); 
} 

#undef __FUNCT__
#define __FUNCT__ "FormFunctionLocali"

PetscErrorCode FormFunctionLocali(DALocalInfo *info,MatStencil *st,PetscScalar **x,PetscScalar *f,AppCtx *user)
{
  PetscScalar    uLocal[4];
  PetscScalar    rLocal[4];
  PetscReal      alpha,lambda,hx,hy,hxhy,sc;
  PetscScalar    lintx[4],linty[4],lintw[4],intnum=4, int1,int2,int3,int4,p1,p2,p3,p4,p;
  PetscInt       i,j,k,l;
  PetscErrorCode ierr;
 PetscFunctionBegin;

  alpha  = user->alpha;
  lambda = user->lambda;
  hx     = 1.0/(PetscReal)(info->mx-1);
  hy     = 1.0/(PetscReal)(info->my-1);
  sc     = -1.0*hx*hy*lambda;
  hxhy   = hx*hy; 

  /* Compute function over the locally owned part of the grid. For each
     vertex (i,j), we consider the element below:

       3         2
     i,j+1 --- i+1,j
       |         |
       |         |
      i,j  --- i+1,j
       0         1

     and therefore we do not loop over the last vertex in each dimension.
  */


   
   lintx[0] = 0.21132486540518713;//(1.0-sqrt(1.0/3.0))/2.0;
   lintx[1] = 0.78867513459481287;//(1.0+sqrt(1.0/3.0))/2.0;
   lintx[2] = 0.78867513459481287;//(1.0+sqrt(1.0/3.0))/2.0;
   lintx[3] = 0.21132486540518713;//(1.0-sqrt(1.0/3.0))/2.0;
   linty[0] = 0.21132486540518713;//(1.0-sqrt(1.0/3.0))/2.0;
   linty[1] = 0.21132486540518713;//(1.0-sqrt(1.0/3.0))/2.0;
   linty[2] = 0.78867513459481287;//(1.0+sqrt(1.0/3.0))/2.0;
   linty[3] = 0.78867513459481287;//(1.0+sqrt(1.0/3.0))/2.0;

   lintw[0] = 1.0/4.0;
   lintw[1] = 1.0/4.0;
   lintw[2] = 1.0/4.0;
   lintw[3] = 1.0/4.0;
  

  i = st->i; j = st->j;

  /*
                              
      i-1,j+1 --- i,j+1 --- i+1,j+1
        |         |           |
	| 2 (1)   |  1(0)     |
      i-1,j   --- i,j  --- i+1,j
        |         |           |
	| 3(2)    |  4(3)     |
      i-1,j-1 --- i,j-1--- i+1,j-1
                              
  */
 

  // boundary treatment 

  if (i == 0 || j == 0 || i == info->mx-1 || j== info->my-1) {
    *f = x[j][i];
    PetscFunctionReturn(0);
  }
 

  /* element 1 */
  uLocal[0] = x[j][i];
  uLocal[1] = x[j][i+1];
  uLocal[2] = x[j+1][i+1];
  uLocal[3] = x[j+1][i];
   
  rLocal[0] =  2.0/3.0*uLocal[0]-1.0/6.0*uLocal[1]-1.0/3.0*uLocal[2]-1.0/6.0*uLocal[3];
  rLocal[0] *= hxhy*alpha;
  
 
   int1 = 0;        
        for(k=0; k<intnum; k++){
          p1   = (1.0-lintx[k])*(1.0-linty[k]);
          p2   =  lintx[k]     *(1.0-linty[k]);
          p3   =  lintx[k]     * linty[k];
          p4   = (1.0-lintx[k])* linty[k];
          p    =  lintw[k]*PetscExpScalar( uLocal[0]*p1+ uLocal[1]*p2 + uLocal[2]*p3+ uLocal[3]*p4 );
	  int1 = int1 + p*p1;
         }
     
	*f      = rLocal[0] + int1*hxhy*(sc);

  /* element 2 */
  uLocal[0] = x[j][i-1];
  uLocal[1] = x[j][i];
  uLocal[2] = x[j+1][i];
  uLocal[3] = x[j+1][i-1];
 
  rLocal[1] = -1.0/6.0*uLocal[0]+2.0/3.0*uLocal[1]-1.0/6.0*uLocal[2]-1.0/3.0*uLocal[3];
  rLocal[1] *= hxhy*alpha; 
  int2 = 0;        
        for(k=0; k<intnum; k++){
          p1   = (1.0-lintx[k])*(1.0-linty[k]);
          p2   =  lintx[k]     *(1.0-linty[k]);
          p3   =  lintx[k]     * linty[k];
          p4   = (1.0-lintx[k])* linty[k];
          p    =  lintw[k]*PetscExpScalar( uLocal[0]*p1+ uLocal[1]*p2 + uLocal[2]*p3+ uLocal[3]*p4 );
	  int2 = int2 + p*p2;
         }
     
	*f     += rLocal[1] + int2*hxhy*(sc);
  
  /* element 3 */
  uLocal[0] = x[j-1][i-1];
  uLocal[1] = x[j-1][i];
  uLocal[2] = x[j][i];
  uLocal[3] = x[j][i-1];
 
  rLocal[2] = -1.0/3.0*uLocal[0]-1.0/6.0*uLocal[1]+2.0/3.0*uLocal[2]-1.0/6.0*uLocal[3];
  rLocal[2] *= hxhy*alpha;
  int3 = 0;        
        for(k=0; k<intnum; k++){
          p1   = (1.0-lintx[k])*(1.0-linty[k]);
          p2   =  lintx[k]     *(1.0-linty[k]);
          p3   =  lintx[k]     * linty[k];
          p4   = (1.0-lintx[k])* linty[k];
          p    =  lintw[k]*PetscExpScalar( uLocal[0]*p1+ uLocal[1]*p2 + uLocal[2]*p3+ uLocal[3]*p4 );
	  int3 = int3 + p*p3;
         }
     
	*f     += rLocal[2] + int3*hxhy*(sc);

 /* element 4 */
  uLocal[0] = x[j-1][i];
  uLocal[1] = x[j-1][i+1];
  uLocal[2] = x[j][i+1];
  uLocal[3] = x[j][i];

  rLocal[3] = -1.0/6.0*uLocal[0]-1.0/3.0*uLocal[1]-1.0/6.0*uLocal[2]+2.0/3.0*uLocal[3];
  rLocal[3] *= hxhy*alpha;
  int4 = 0;        
        for(k=0; k<intnum; k++){
          p1   = (1.0-lintx[k])*(1.0-linty[k]);
          p2   =  lintx[k]     *(1.0-linty[k]);
          p3   =  lintx[k]     * linty[k];
          p4   = (1.0-lintx[k])* linty[k];
          p    =  lintw[k]*PetscExpScalar( uLocal[0]*p1+ uLocal[1]*p2 + uLocal[2]*p3+ uLocal[3]*p4 );
	  int4 = int4 + p*p4;
         }
     
	*f     += rLocal[3] + int4*hxhy*(sc);



  PetscFunctionReturn(0); 
} 





//****************************************************************************



#undef __FUNCT__
#define __FUNCT__ "FormFunctionLocali4"
PetscErrorCode FormFunctionLocali4(DALocalInfo *info,MatStencil *st,PetscScalar **x,PetscScalar *f,AppCtx *user)
{ 
  PetscScalar    uLocal[4],fLocal[5][5];
  PetscScalar    rLocal[4];
  PetscScalar    lintx[4],linty[4],lintw[4],intnum=4, int1,int2,int3,int4,p1,p2,p3,p4,p;
  PetscReal      alpha,lambda,hx,hy,hxhy,sc;
  PetscInt       i,j,k,kk,ll,istar,iend,jstar,jend,imax,jmax,id,jd;
  PetscErrorCode ierr;

 PetscFunctionBegin;

  alpha  = user->alpha;
  lambda = user->lambda;
  hx     = 1.0/(PetscReal)(info->mx-1);
  hy     = 1.0/(PetscReal)(info->my-1);
  sc     = -1.0*hx*hy*lambda;
  hxhy   = hx*hy; 

  /* Compute function over the locally owned part of the grid. For each
     vertex (i,j), we consider the element below:

       3         2
     i,j+1 --- i+1,j
       |         |
       |         |
      i,j  --- i+1,j
       0         1

     and therefore we do not loop over the last vertex in each dimension.
  */


  

  i = st->i; j = st->j;

  /*
                              
      i-1,j+1 --- i,j+1 --- i+1,j+1
        |         |           |
	| 2 (1)   |  1(0)     |
      i-1,j   --- i,j  --- i+1,j
        |         |           |
	| 3(2)    |  4(3)     |
      i-1,j-1 --- i,j-1--- i+1,j-1
                              
  */
 

  // boundary treatment 

  if (i == 0 || j == 0 || i == info->mx-1 || j== info->my-1) {
    f[0] = x[j][i];
    PetscFunctionReturn(0);
  }
  
  //(2) second cases: next to boundary


  if(i==1 || j==1 ||i==info->mx-2 ||j==info->my-2){
 

   lintx[0] = 0.21132486540518713;//(1.0-sqrt(1.0/3.0))/2.0;
   lintx[1] = 0.78867513459481287;//(1.0+sqrt(1.0/3.0))/2.0;
   lintx[2] = 0.78867513459481287;//(1.0+sqrt(1.0/3.0))/2.0;
   lintx[3] = 0.21132486540518713;//(1.0-sqrt(1.0/3.0))/2.0;
   linty[0] = 0.21132486540518713;//(1.0-sqrt(1.0/3.0))/2.0;
   linty[1] = 0.21132486540518713;//(1.0-sqrt(1.0/3.0))/2.0;
   linty[2] = 0.78867513459481287;//(1.0+sqrt(1.0/3.0))/2.0;
   linty[3] = 0.78867513459481287;//(1.0+sqrt(1.0/3.0))/2.0;

   lintw[0] = 1.0/4.0;
   lintw[1] = 1.0/4.0;
   lintw[2] = 1.0/4.0;
   lintw[3] = 1.0/4.0;
  

  /* element 1 */
  uLocal[0] = x[j][i];
  uLocal[1] = x[j][i+1];
  uLocal[2] = x[j+1][i+1];
  uLocal[3] = x[j+1][i];
   
  rLocal[0] =  2.0/3.0*uLocal[0]-1.0/6.0*uLocal[1]-1.0/3.0*uLocal[2]-1.0/6.0*uLocal[3];
  rLocal[0] *= hxhy*alpha;
  
 
   int1 = 0;        
        for(k=0; k<intnum; k++){
          p1   = (1.0-lintx[k])*(1.0-linty[k]);
          p2   =  lintx[k]     *(1.0-linty[k]);
          p3   =  lintx[k]     * linty[k];
          p4   = (1.0-lintx[k])* linty[k];
          p    =  lintw[k]*PetscExpScalar( uLocal[0]*p1+ uLocal[1]*p2 + uLocal[2]*p3+ uLocal[3]*p4 );
	  int1 = int1 + p*p1;
         }
     
	f[0]      = rLocal[0] + int1*hxhy*(sc);

  /* element 2 */
  uLocal[0] = x[j][i-1];
  uLocal[1] = x[j][i];
  uLocal[2] = x[j+1][i];
  uLocal[3] = x[j+1][i-1];
 
  rLocal[1] = -1.0/6.0*uLocal[0]+2.0/3.0*uLocal[1]-1.0/6.0*uLocal[2]-1.0/3.0*uLocal[3];
  rLocal[1] *= hxhy*alpha; 
  int2 = 0;        
        for(k=0; k<intnum; k++){
          p1   = (1.0-lintx[k])*(1.0-linty[k]);
          p2   =  lintx[k]     *(1.0-linty[k]);
          p3   =  lintx[k]     * linty[k];
          p4   = (1.0-lintx[k])* linty[k];
          p    =  lintw[k]*PetscExpScalar( uLocal[0]*p1+ uLocal[1]*p2 + uLocal[2]*p3+ uLocal[3]*p4 );
	  int2 = int2 + p*p2;
         }
     
	f[0]     += rLocal[1] + int2*hxhy*(sc);
  
  /* element 3 */
  uLocal[0] = x[j-1][i-1];
  uLocal[1] = x[j-1][i];
  uLocal[2] = x[j][i];
  uLocal[3] = x[j][i-1];
 
  rLocal[2] = -1.0/3.0*uLocal[0]-1.0/6.0*uLocal[1]+2.0/3.0*uLocal[2]-1.0/6.0*uLocal[3];
  rLocal[2] *= hxhy*alpha;
  int3 = 0;        
        for(k=0; k<intnum; k++){
          p1   = (1.0-lintx[k])*(1.0-linty[k]);
          p2   =  lintx[k]     *(1.0-linty[k]);
          p3   =  lintx[k]     * linty[k];
          p4   = (1.0-lintx[k])* linty[k];
          p    =  lintw[k]*PetscExpScalar( uLocal[0]*p1+ uLocal[1]*p2 + uLocal[2]*p3+ uLocal[3]*p4 );
	  int3 = int3 + p*p3;
         }
     
	f[0]     += rLocal[2] + int3*hxhy*(sc);

 /* element 4 */
  uLocal[0] = x[j-1][i];
  uLocal[1] = x[j-1][i+1];
  uLocal[2] = x[j][i+1];
  uLocal[3] = x[j][i];

  rLocal[3] = -1.0/6.0*uLocal[0]-1.0/3.0*uLocal[1]-1.0/6.0*uLocal[2]+2.0/3.0*uLocal[3];
  rLocal[3] *= hxhy*alpha;
  int4 = 0;        
        for(k=0; k<intnum; k++){
          p1   = (1.0-lintx[k])*(1.0-linty[k]);
          p2   =  lintx[k]     *(1.0-linty[k]);
          p3   =  lintx[k]     * linty[k];
          p4   = (1.0-lintx[k])* linty[k];
          p    =  lintw[k]*PetscExpScalar( uLocal[0]*p1+ uLocal[1]*p2 + uLocal[2]*p3+ uLocal[3]*p4 );
	  int4 = int4 + p*p4;
         }
     
	f[0]     += rLocal[3] + int4*hxhy*(sc);



  PetscFunctionReturn(0); 
  }

/*
  i-2,j+2 --- i-1,j+2 --- i,j+2 --- i+1,j+2 --- i+2,j+2
     |           |         |           |           |
     |    e13    |   e14   |    e15    |     e16   |
  i-2,j+1 --- i-1,j+1 --- i,j+1 --- i+1,j+1 --- i+2;j+1
     |           |(6)      |(7)        |(8)        |
     |    e9     |   e10   |    e11    |     e12   |
  i-2,j   --- i-1,j   --- i,j  ---  i+1,j   --- i+2,j 
     |           |(3)      |(4)        |(5)        |
     |    e5     |   e6    |     e7    |     e8    |
  i-2,j-1 --- i-1,j-1 --- i,j-1--- i+1,j-1  --- i+2,j-1
     |           |(0)      |(1)        |(2)        |
     |    e1     |   e2    |     e3    |     e4    |
  i-2,j-2 --- i-1,j-2 --- i,j-2--- i+1,j-2  --- i+2,j-2
                              
  */
  
  jstar  = j - 2;
  jend   = j + 2;
  istar  = i - 2;
  iend   = i + 2;
 
   
 lintx[0] = 0.21132486540518713;//(1.0-sqrt(1.0/3.0))/2.0;
   lintx[1] = 0.78867513459481287;//(1.0+sqrt(1.0/3.0))/2.0;
   lintx[2] = 0.78867513459481287;//(1.0+sqrt(1.0/3.0))/2.0;
   lintx[3] = 0.21132486540518713;//(1.0-sqrt(1.0/3.0))/2.0;
   linty[0] = 0.21132486540518713;//(1.0-sqrt(1.0/3.0))/2.0;
   linty[1] = 0.21132486540518713;//(1.0-sqrt(1.0/3.0))/2.0;
   linty[2] = 0.78867513459481287;//(1.0+sqrt(1.0/3.0))/2.0;
   linty[3] = 0.78867513459481287;//(1.0+sqrt(1.0/3.0))/2.0;

   lintw[0] = 1.0/4.0;
   lintw[1] = 1.0/4.0;
   lintw[2] = 1.0/4.0;
   lintw[3] = 1.0/4.0;
  

   id=0;
   for(kk=1;kk<4;kk++) {
     for(ll=1; ll<4;ll++) {
       i=istar+ll;
       j=jstar+kk; 

  /* element 1 */
  uLocal[0] = x[j][i];
  uLocal[1] = x[j][i+1];
  uLocal[2] = x[j+1][i+1];
  uLocal[3] = x[j+1][i];
   
  rLocal[0] =  2.0/3.0*uLocal[0]-1.0/6.0*uLocal[1]-1.0/3.0*uLocal[2]-1.0/6.0*uLocal[3];
  rLocal[0] *= hxhy*alpha;
  
 
   int1 = 0;        
        for(k=0; k<intnum; k++){
          p1   = (1.0-lintx[k])*(1.0-linty[k]);
          p2   =  lintx[k]     *(1.0-linty[k]);
          p3   =  lintx[k]     * linty[k];
          p4   = (1.0-lintx[k])* linty[k];
          p    =  lintw[k]*PetscExpScalar( uLocal[0]*p1+ uLocal[1]*p2 + uLocal[2]*p3+ uLocal[3]*p4 );
	  int1 = int1 + p*p1;
         }
     
	f[id]      = rLocal[0] + int1*hxhy*(sc);

  /* element 2 */
  uLocal[0] = x[j][i-1];
  uLocal[1] = x[j][i];
  uLocal[2] = x[j+1][i];
  uLocal[3] = x[j+1][i-1];
 
  rLocal[1] = -1.0/6.0*uLocal[0]+2.0/3.0*uLocal[1]-1.0/6.0*uLocal[2]-1.0/3.0*uLocal[3];
  rLocal[1] *= hxhy*alpha; 
  int2 = 0;        
        for(k=0; k<intnum; k++){
          p1   = (1.0-lintx[k])*(1.0-linty[k]);
          p2   =  lintx[k]     *(1.0-linty[k]);
          p3   =  lintx[k]     * linty[k];
          p4   = (1.0-lintx[k])* linty[k];
          p    =  lintw[k]*PetscExpScalar( uLocal[0]*p1+ uLocal[1]*p2 + uLocal[2]*p3+ uLocal[3]*p4 );
	  int2 = int2 + p*p2;
         }
     
	f[id]     += rLocal[1] + int2*hxhy*(sc);
  
  /* element 3 */
  uLocal[0] = x[j-1][i-1];
  uLocal[1] = x[j-1][i];
  uLocal[2] = x[j][i];
  uLocal[3] = x[j][i-1];
 
  rLocal[2] = -1.0/3.0*uLocal[0]-1.0/6.0*uLocal[1]+2.0/3.0*uLocal[2]-1.0/6.0*uLocal[3];
  rLocal[2] *= hxhy*alpha;
  int3 = 0;        
        for(k=0; k<intnum; k++){
          p1   = (1.0-lintx[k])*(1.0-linty[k]);
          p2   =  lintx[k]     *(1.0-linty[k]);
          p3   =  lintx[k]     * linty[k];
          p4   = (1.0-lintx[k])* linty[k];
          p    =  lintw[k]*PetscExpScalar( uLocal[0]*p1+ uLocal[1]*p2 + uLocal[2]*p3+ uLocal[3]*p4 );
	  int3 = int3 + p*p3;
         }
     
	f[id]     += rLocal[2] + int3*hxhy*(sc);

 /* element 4 */
  uLocal[0] = x[j-1][i];
  uLocal[1] = x[j-1][i+1];
  uLocal[2] = x[j][i+1];
  uLocal[3] = x[j][i];

  rLocal[3] = -1.0/6.0*uLocal[0]-1.0/3.0*uLocal[1]-1.0/6.0*uLocal[2]+2.0/3.0*uLocal[3];
  rLocal[3] *= hxhy*alpha;
  int4 = 0;        
        for(k=0; k<intnum; k++){
          p1   = (1.0-lintx[k])*(1.0-linty[k]);
          p2   =  lintx[k]     *(1.0-linty[k]);
          p3   =  lintx[k]     * linty[k];
          p4   = (1.0-lintx[k])* linty[k];
          p    =  lintw[k]*PetscExpScalar( uLocal[0]*p1+ uLocal[1]*p2 + uLocal[2]*p3+ uLocal[3]*p4 );
	  int4 = int4 + p*p4;
         }
     
	f[id]     += rLocal[3] + int4*hxhy*(sc);
	id++;
     }
   }



  PetscFunctionReturn(0); 
} 

//****************************************************************************

