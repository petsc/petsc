
static char help[] ="Solvers Laplacian with multigrid, bad way.\n\
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

#include <petscksp.h>
#include <petscdm.h>
#include <petscdmda.h>

/* User-defined application contexts */

typedef struct {
  PetscInt mx,my;               /* number grid points in x and y direction */
  Vec      localX,localF;       /* local vectors with ghost region */
  DM       da;
  Vec      x,b,r;               /* global vectors */
  Mat      J;                   /* Jacobian on grid */
} GridCtx;

typedef struct {
  GridCtx  fine;
  GridCtx  coarse;
  KSP      ksp_coarse;
  PetscInt ratio;
  Mat      Ii;                  /* interpolation from coarse to fine */
} AppCtx;

#define COARSE_LEVEL 0
#define FINE_LEVEL   1

extern PetscErrorCode FormJacobian_Grid(AppCtx*,GridCtx*,Mat*);

/*
      Mm_ratio - ration of grid lines between fine and coarse grids.
*/
int main(int argc,char **argv)
{
  AppCtx         user;
  PetscErrorCode ierr;
  PetscInt       its,N,n,Nx = PETSC_DECIDE,Ny = PETSC_DECIDE,nlocal,Nlocal;
  PetscMPIInt    size;
  KSP            ksp,ksp_fine;
  PC             pc;
  PetscScalar    one = 1.0;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  user.ratio     = 2;
  user.coarse.mx = 5; user.coarse.my = 5;

  ierr = PetscOptionsGetInt(NULL,NULL,"-Mx",&user.coarse.mx,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-My",&user.coarse.my,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-ratio",&user.ratio,NULL);CHKERRQ(ierr);

  user.fine.mx = user.ratio*(user.coarse.mx-1)+1; user.fine.my = user.ratio*(user.coarse.my-1)+1;

  ierr = PetscPrintf(PETSC_COMM_WORLD,"Coarse grid size %D by %D\n",user.coarse.mx,user.coarse.my);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Fine grid size %D by %D\n",user.fine.mx,user.fine.my);CHKERRQ(ierr);

  n = user.fine.mx*user.fine.my; N = user.coarse.mx*user.coarse.my;

  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-Nx",&Nx,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-Ny",&Ny,NULL);CHKERRQ(ierr);

  /* Set up distributed array for fine grid */
  ierr = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,user.fine.mx,user.fine.my,Nx,Ny,1,1,NULL,NULL,&user.fine.da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(user.fine.da);CHKERRQ(ierr);
  ierr = DMSetUp(user.fine.da);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(user.fine.da,&user.fine.x);CHKERRQ(ierr);
  ierr = VecDuplicate(user.fine.x,&user.fine.r);CHKERRQ(ierr);
  ierr = VecDuplicate(user.fine.x,&user.fine.b);CHKERRQ(ierr);
  ierr = VecGetLocalSize(user.fine.x,&nlocal);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(user.fine.da,&user.fine.localX);CHKERRQ(ierr);
  ierr = VecDuplicate(user.fine.localX,&user.fine.localF);CHKERRQ(ierr);
  ierr = MatCreateAIJ(PETSC_COMM_WORLD,nlocal,nlocal,n,n,5,NULL,3,NULL,&user.fine.J);CHKERRQ(ierr);

  /* Set up distributed array for coarse grid */
  ierr = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,user.coarse.mx,user.coarse.my,Nx,Ny,1,1,NULL,NULL,&user.coarse.da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(user.coarse.da);CHKERRQ(ierr);
  ierr = DMSetUp(user.coarse.da);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(user.coarse.da,&user.coarse.x);CHKERRQ(ierr);
  ierr = VecDuplicate(user.coarse.x,&user.coarse.b);CHKERRQ(ierr);
  ierr = VecGetLocalSize(user.coarse.x,&Nlocal);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(user.coarse.da,&user.coarse.localX);CHKERRQ(ierr);
  ierr = VecDuplicate(user.coarse.localX,&user.coarse.localF);CHKERRQ(ierr);
  ierr = MatCreateAIJ(PETSC_COMM_WORLD,Nlocal,Nlocal,N,N,5,NULL,3,NULL,&user.coarse.J);CHKERRQ(ierr);

  /* Create linear solver */
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);

  /* set two level additive Schwarz preconditioner */
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCMG);CHKERRQ(ierr);
  ierr = PCMGSetLevels(pc,2,NULL);CHKERRQ(ierr);
  ierr = PCMGSetType(pc,PC_MG_ADDITIVE);CHKERRQ(ierr);

  ierr = FormJacobian_Grid(&user,&user.coarse,&user.coarse.J);CHKERRQ(ierr);
  ierr = FormJacobian_Grid(&user,&user.fine,&user.fine.J);CHKERRQ(ierr);

  /* Create coarse level */
  ierr = PCMGGetCoarseSolve(pc,&user.ksp_coarse);CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(user.ksp_coarse,"coarse_");CHKERRQ(ierr);
  ierr = KSPSetFromOptions(user.ksp_coarse);CHKERRQ(ierr);
  ierr = KSPSetOperators(user.ksp_coarse,user.coarse.J,user.coarse.J);CHKERRQ(ierr);
  ierr = PCMGSetX(pc,COARSE_LEVEL,user.coarse.x);CHKERRQ(ierr);
  ierr = PCMGSetRhs(pc,COARSE_LEVEL,user.coarse.b);CHKERRQ(ierr);

  /* Create fine level */
  ierr = PCMGGetSmoother(pc,FINE_LEVEL,&ksp_fine);CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(ksp_fine,"fine_");CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp_fine);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp_fine,user.fine.J,user.fine.J);CHKERRQ(ierr);
  ierr = PCMGSetR(pc,FINE_LEVEL,user.fine.r);CHKERRQ(ierr);

  /* Create interpolation between the levels */
  ierr = DMCreateInterpolation(user.coarse.da,user.fine.da,&user.Ii,NULL);CHKERRQ(ierr);
  ierr = PCMGSetInterpolation(pc,FINE_LEVEL,user.Ii);CHKERRQ(ierr);
  ierr = PCMGSetRestriction(pc,FINE_LEVEL,user.Ii);CHKERRQ(ierr);

  ierr = KSPSetOperators(ksp,user.fine.J,user.fine.J);CHKERRQ(ierr);

  ierr = VecSet(user.fine.b,one);CHKERRQ(ierr);

  /* Set options, then solve nonlinear system */
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  ierr = KSPSolve(ksp,user.fine.b,user.fine.x);CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of iterations = %D\n",its);CHKERRQ(ierr);

  /* Free data structures */
  ierr = MatDestroy(&user.fine.J);CHKERRQ(ierr);
  ierr = VecDestroy(&user.fine.x);CHKERRQ(ierr);
  ierr = VecDestroy(&user.fine.r);CHKERRQ(ierr);
  ierr = VecDestroy(&user.fine.b);CHKERRQ(ierr);
  ierr = DMDestroy(&user.fine.da);CHKERRQ(ierr);
  ierr = VecDestroy(&user.fine.localX);CHKERRQ(ierr);
  ierr = VecDestroy(&user.fine.localF);CHKERRQ(ierr);

  ierr = MatDestroy(&user.coarse.J);CHKERRQ(ierr);
  ierr = VecDestroy(&user.coarse.x);CHKERRQ(ierr);
  ierr = VecDestroy(&user.coarse.b);CHKERRQ(ierr);
  ierr = DMDestroy(&user.coarse.da);CHKERRQ(ierr);
  ierr = VecDestroy(&user.coarse.localX);CHKERRQ(ierr);
  ierr = VecDestroy(&user.coarse.localF);CHKERRQ(ierr);

  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&user.Ii);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

PetscErrorCode FormJacobian_Grid(AppCtx *user,GridCtx *grid,Mat *J)
{
  Mat                    jac = *J;
  PetscErrorCode         ierr;
  PetscInt               i,j,row,mx,my,xs,ys,xm,ym,Xs,Ys,Xm,Ym,col[5];
  PetscInt               grow;
  const PetscInt         *ltog;
  PetscScalar            two = 2.0,one = 1.0,v[5],hx,hy,hxdhy,hydhx,value;
  ISLocalToGlobalMapping ltogm;

  mx    = grid->mx;               my = grid->my;
  hx    = one/(PetscReal)(mx-1);  hy = one/(PetscReal)(my-1);
  hxdhy = hx/hy;               hydhx = hy/hx;

  /* Get ghost points */
  ierr = DMDAGetCorners(grid->da,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(grid->da,&Xs,&Ys,0,&Xm,&Ym,0);CHKERRQ(ierr);
  ierr = DMGetLocalToGlobalMapping(grid->da,&ltogm);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetIndices(ltogm,&ltog);CHKERRQ(ierr);

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
      } else if ((i > 0 && i < mx-1) || (j > 0 && j < my-1)) {
        value = .5*two*(hydhx + hxdhy);
        ierr  = MatSetValues(jac,1,&grow,1,&grow,&value,INSERT_VALUES);CHKERRQ(ierr);
      } else {
        value = .25*two*(hydhx + hxdhy);
        ierr  = MatSetValues(jac,1,&grow,1,&grow,&value,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }
  ierr = ISLocalToGlobalMappingRestoreIndices(ltogm,&ltog);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  return 0;
}

/*TEST

    test:
      args: -ksp_gmres_cgs_refinement_type refine_always -pc_type jacobi -ksp_monitor_short -ksp_type gmres 

    test:
      suffix: 2
      nsize: 3
      args: -ksp_monitor_short

TEST*/
