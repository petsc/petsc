/* "$Id: ex19.c,v 1.3 2000/01/11 21:02:16 bsmith Exp balay $" */

static char help[] ="\
  -mx <xg>, where <xg> = number of grid points in the x-direction\n\
  -my <yg>, where <yg> = number of grid points in the y-direction\n\
  -Nx <npx>, where <npx> = number of processors in the x-direction\n\
  -Ny <npy>, where <npy> = number of processors in the y-direction\n\n";

/*  
    This problem is modeled by
    the partial differential equation
  
            -Laplacian u  = g,  0 < x,y < 1,
  
    with boundary conditions
   
             u = 0  for  x = 0, x = 1, y = 0, y = 1.
  
    A finite difference approximation with the usual 5-point stencil
    is used to discretize the boundary value problem to obtain a nonlinear 
    system of equations.
*/

#include "petscsles.h"
#include "petscda.h"
#include "petscmg.h"

/* User-defined application contexts */

typedef struct {
   int        mx,my;            /* number grid points in x and y direction */
   Vec        localX,localF;    /* local vectors with ghost region */
   DA         da;
   Vec        x,b,r;            /* global vectors */
   Mat        J;                /* Jacobian on grid */
} GridCtx;

typedef struct {
   GridCtx     fine;
   GridCtx     coarse;
   SLES        sles_coarse;
   int         ratio;
   Mat         I;               /* interpolation from coarse to fine */
} AppCtx;

#define COARSE_LEVEL 0
#define FINE_LEVEL   1

extern int FormJacobian_Grid(AppCtx *,GridCtx *,Mat *);

/*
      Mm_ratio - ration of grid lines between fine and coarse grids.
*/
#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  AppCtx        user;                      
  int           ierr,its,N,n,Nx = PETSC_DECIDE,Ny = PETSC_DECIDE;
  int           size,nlocal,Nlocal;
  SLES          sles,sles_fine;
  PC            pc;
  Scalar        one = 1.0;

  PetscInitialize(&argc,&argv,PETSC_NULL,help);

  user.ratio = 2;
  user.coarse.mx = 5; user.coarse.my = 5; 
  ierr = OptionsGetInt(PETSC_NULL,"-Mx",&user.coarse.mx,PETSC_NULL);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-My",&user.coarse.my,PETSC_NULL);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-ratio",&user.ratio,PETSC_NULL);CHKERRA(ierr);
  user.fine.mx = user.ratio*(user.coarse.mx-1)+1; user.fine.my = user.ratio*(user.coarse.my-1)+1;

  PetscPrintf(PETSC_COMM_WORLD,"Coarse grid size %d by %d\n",user.coarse.mx,user.coarse.my);
  PetscPrintf(PETSC_COMM_WORLD,"Fine grid size %d by %d\n",user.fine.mx,user.fine.my);

  n = user.fine.mx*user.fine.my; N = user.coarse.mx*user.coarse.my;

  MPI_Comm_size(PETSC_COMM_WORLD,&size);
  ierr = OptionsGetInt(PETSC_NULL,"-Nx",&Nx,PETSC_NULL);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-Ny",&Ny,PETSC_NULL);CHKERRA(ierr);

  /* Set up distributed array for fine grid */
  ierr = DACreate2d(PETSC_COMM_WORLD,DA_NONPERIODIC,DA_STENCIL_STAR,user.fine.mx,
                    user.fine.my,Nx,Ny,1,1,PETSC_NULL,PETSC_NULL,&user.fine.da);CHKERRA(ierr);
  ierr = DACreateGlobalVector(user.fine.da,&user.fine.x);CHKERRA(ierr);
  ierr = VecDuplicate(user.fine.x,&user.fine.r);CHKERRA(ierr);
  ierr = VecDuplicate(user.fine.x,&user.fine.b);CHKERRA(ierr);
  ierr = VecGetLocalSize(user.fine.x,&nlocal);CHKERRA(ierr);
  ierr = DACreateLocalVector(user.fine.da,&user.fine.localX);CHKERRA(ierr);
  ierr = VecDuplicate(user.fine.localX,&user.fine.localF);CHKERRA(ierr);
  ierr = MatCreateMPIAIJ(PETSC_COMM_WORLD,nlocal,nlocal,n,n,5,PETSC_NULL,3,PETSC_NULL,&user.fine.J);CHKERRA(ierr);

  /* Set up distributed array for coarse grid */
  ierr = DACreate2d(PETSC_COMM_WORLD,DA_NONPERIODIC,DA_STENCIL_STAR,user.coarse.mx,
                    user.coarse.my,Nx,Ny,1,1,PETSC_NULL,PETSC_NULL,&user.coarse.da);CHKERRA(ierr);
  ierr = DACreateGlobalVector(user.coarse.da,&user.coarse.x);CHKERRA(ierr);
  ierr = VecDuplicate(user.coarse.x,&user.coarse.b);CHKERRA(ierr);
  ierr = VecGetLocalSize(user.coarse.x,&Nlocal);CHKERRA(ierr);
  ierr = DACreateLocalVector(user.coarse.da,&user.coarse.localX);CHKERRA(ierr);
  ierr = VecDuplicate(user.coarse.localX,&user.coarse.localF);CHKERRA(ierr);
  ierr = MatCreateMPIAIJ(PETSC_COMM_WORLD,Nlocal,Nlocal,N,N,5,PETSC_NULL,3,PETSC_NULL,&user.coarse.J);CHKERRA(ierr);

  /* Create linear solver */
  ierr = SLESCreate(PETSC_COMM_WORLD,&sles);CHKERRA(ierr);

  /* set two level additive Schwarz preconditioner */
  ierr = SLESGetPC(sles,&pc);CHKERRA(ierr);
  ierr = PCSetType(pc,PCMG);CHKERRA(ierr);
  ierr = MGSetLevels(pc,2);CHKERRA(ierr);
  ierr = MGSetType(pc,MGADDITIVE);CHKERRA(ierr);

  ierr = FormJacobian_Grid(&user,&user.coarse,&user.coarse.J);CHKERRA(ierr);
  ierr = FormJacobian_Grid(&user,&user.fine,&user.fine.J);CHKERRA(ierr);

  /* Create coarse level */
  ierr = MGGetCoarseSolve(pc,&user.sles_coarse);CHKERRA(ierr);
  ierr = SLESSetOptionsPrefix(user.sles_coarse,"coarse_");CHKERRA(ierr);
  ierr = SLESSetFromOptions(user.sles_coarse);CHKERRA(ierr);
  ierr = SLESSetOperators(user.sles_coarse,user.coarse.J,user.coarse.J,DIFFERENT_NONZERO_PATTERN);CHKERRA(ierr);
  ierr = MGSetX(pc,COARSE_LEVEL,user.coarse.x);CHKERRA(ierr); 
  ierr = MGSetRhs(pc,COARSE_LEVEL,user.coarse.b);CHKERRA(ierr); 

  /* Create fine level */
  ierr = MGGetSmoother(pc,FINE_LEVEL,&sles_fine);CHKERRA(ierr);
  ierr = SLESSetOptionsPrefix(sles_fine,"fine_");CHKERRA(ierr);
  ierr = SLESSetFromOptions(sles_fine);CHKERRA(ierr);
  ierr = SLESSetOperators(sles_fine,user.fine.J,user.fine.J,DIFFERENT_NONZERO_PATTERN);CHKERRA(ierr);
  ierr = MGSetR(pc,FINE_LEVEL,user.fine.r);CHKERRA(ierr); 
  ierr = MGSetResidual(pc,FINE_LEVEL,MGDefaultResidual,user.fine.J);CHKERRA(ierr);

  /* Create interpolation between the levels */
  ierr = DAGetInterpolation(user.coarse.da,user.fine.da,&user.I,PETSC_NULL);CHKERRA(ierr);
  ierr = MGSetInterpolate(pc,FINE_LEVEL,user.I);CHKERRA(ierr);
  ierr = MGSetRestriction(pc,FINE_LEVEL,user.I);CHKERRA(ierr);

  ierr = SLESSetOperators(sles,user.fine.J,user.fine.J,DIFFERENT_NONZERO_PATTERN);CHKERRA(ierr);

  ierr = VecSet(&one,user.fine.b);CHKERRQ(ierr);
  {
    PetscRandom rand;
    ierr = PetscRandomCreate(PETSC_COMM_WORLD,RANDOM_DEFAULT,&rand);CHKERRA(ierr);
    ierr = VecSetRandom(rand,user.fine.b);CHKERRA(ierr);
    ierr = PetscRandomDestroy(rand);CHKERRA(ierr);
  }

  /* Set options, then solve nonlinear system */
  ierr = SLESSetFromOptions(sles);CHKERRA(ierr);

  ierr = SLESSolve(sles,user.fine.b,user.fine.x,&its);CHKERRA(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of iterations = %d\n",its);CHKERRA(ierr);

  /* Free data structures */
  ierr = MatDestroy(user.fine.J);CHKERRA(ierr);
  ierr = VecDestroy(user.fine.x);CHKERRA(ierr);
  ierr = VecDestroy(user.fine.r);CHKERRA(ierr);
  ierr = VecDestroy(user.fine.b);CHKERRA(ierr);
  ierr = DADestroy(user.fine.da);CHKERRA(ierr);
  ierr = VecDestroy(user.fine.localX);CHKERRA(ierr);
  ierr = VecDestroy(user.fine.localF);CHKERRA(ierr);

  ierr = MatDestroy(user.coarse.J);CHKERRA(ierr);
  ierr = VecDestroy(user.coarse.x);CHKERRA(ierr);
  ierr = VecDestroy(user.coarse.b);CHKERRA(ierr);
  ierr = DADestroy(user.coarse.da);CHKERRA(ierr);
  ierr = VecDestroy(user.coarse.localX);CHKERRA(ierr);
  ierr = VecDestroy(user.coarse.localF);CHKERRA(ierr);

  ierr = SLESDestroy(sles);CHKERRA(ierr);
  ierr = MatDestroy(user.I);CHKERRA(ierr); 
  PetscFinalize();

  return 0;
}

#undef __FUNC__
#define __FUNC__ "FormJacobian_Grid"
int FormJacobian_Grid(AppCtx *user,GridCtx *grid,Mat *J)
{
  Mat     jac = *J;
  int     ierr,i,j,row,mx,my,xs,ys,xm,ym,Xs,Ys,Xm,Ym,col[5];
  int     nloc,*ltog,grow;
  Scalar  two = 2.0,one = 1.0,v[5],hx,hy,hxdhy,hydhx,value;

  mx = grid->mx;            my = grid->my;            
  hx = one/(double)(mx-1);  hy = one/(double)(my-1);
  hxdhy = hx/hy;            hydhx = hy/hx;

  /* Get ghost points */
  ierr = DAGetCorners(grid->da,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(grid->da,&Xs,&Ys,0,&Xm,&Ym,0);CHKERRQ(ierr);
  ierr = DAGetGlobalIndices(grid->da,&nloc,&ltog);CHKERRQ(ierr);

  /* Evaluate Jacobian of function */
  for (j=ys; j<ys+ym; j++) {
    row = (j - Ys)*Xm + xs - Xs - 1; 
    for (i=xs; i<xs+xm; i++) {
      row++;
      grow = ltog[row];
      if (i > 0 && i < mx-1 && j > 0 && j < my-1) {
        v[0] = -hxdhy; col[0] = ltog[row - Xm];
        v[1] = -hydhx; col[1] = ltog[row - 1];
        v[2] = two*(hydhx + hxdhy); col[2] = grow;
        v[3] = -hydhx; col[3] = ltog[row + 1];
        v[4] = -hxdhy; col[4] = ltog[row + Xm];
        ierr = MatSetValues(jac,1,&grow,5,col,v,INSERT_VALUES);CHKERRQ(ierr);
      } else if ((i > 0 && i < mx-1) || (j > 0 && j < my-1)){
        value = .5*two*(hydhx + hxdhy);
        ierr = MatSetValues(jac,1,&grow,1,&grow,&value,INSERT_VALUES);CHKERRQ(ierr);
      } else {
        value = .25*two*(hydhx + hxdhy);
        ierr = MatSetValues(jac,1,&grow,1,&grow,&value,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }
  ierr = MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  return 0;
}
