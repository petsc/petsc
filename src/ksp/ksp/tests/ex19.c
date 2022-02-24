
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

  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-Mx",&user.coarse.mx,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-My",&user.coarse.my,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-ratio",&user.ratio,NULL));

  user.fine.mx = user.ratio*(user.coarse.mx-1)+1; user.fine.my = user.ratio*(user.coarse.my-1)+1;

  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Coarse grid size %D by %D\n",user.coarse.mx,user.coarse.my));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Fine grid size %D by %D\n",user.fine.mx,user.fine.my));

  n = user.fine.mx*user.fine.my; N = user.coarse.mx*user.coarse.my;

  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-Nx",&Nx,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-Ny",&Ny,NULL));

  /* Set up distributed array for fine grid */
  CHKERRQ(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,user.fine.mx,user.fine.my,Nx,Ny,1,1,NULL,NULL,&user.fine.da));
  CHKERRQ(DMSetFromOptions(user.fine.da));
  CHKERRQ(DMSetUp(user.fine.da));
  CHKERRQ(DMCreateGlobalVector(user.fine.da,&user.fine.x));
  CHKERRQ(VecDuplicate(user.fine.x,&user.fine.r));
  CHKERRQ(VecDuplicate(user.fine.x,&user.fine.b));
  CHKERRQ(VecGetLocalSize(user.fine.x,&nlocal));
  CHKERRQ(DMCreateLocalVector(user.fine.da,&user.fine.localX));
  CHKERRQ(VecDuplicate(user.fine.localX,&user.fine.localF));
  CHKERRQ(MatCreateAIJ(PETSC_COMM_WORLD,nlocal,nlocal,n,n,5,NULL,3,NULL,&user.fine.J));

  /* Set up distributed array for coarse grid */
  CHKERRQ(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,user.coarse.mx,user.coarse.my,Nx,Ny,1,1,NULL,NULL,&user.coarse.da));
  CHKERRQ(DMSetFromOptions(user.coarse.da));
  CHKERRQ(DMSetUp(user.coarse.da));
  CHKERRQ(DMCreateGlobalVector(user.coarse.da,&user.coarse.x));
  CHKERRQ(VecDuplicate(user.coarse.x,&user.coarse.b));
  CHKERRQ(VecGetLocalSize(user.coarse.x,&Nlocal));
  CHKERRQ(DMCreateLocalVector(user.coarse.da,&user.coarse.localX));
  CHKERRQ(VecDuplicate(user.coarse.localX,&user.coarse.localF));
  CHKERRQ(MatCreateAIJ(PETSC_COMM_WORLD,Nlocal,Nlocal,N,N,5,NULL,3,NULL,&user.coarse.J));

  /* Create linear solver */
  CHKERRQ(KSPCreate(PETSC_COMM_WORLD,&ksp));

  /* set two level additive Schwarz preconditioner */
  CHKERRQ(KSPGetPC(ksp,&pc));
  CHKERRQ(PCSetType(pc,PCMG));
  CHKERRQ(PCMGSetLevels(pc,2,NULL));
  CHKERRQ(PCMGSetType(pc,PC_MG_ADDITIVE));

  CHKERRQ(FormJacobian_Grid(&user,&user.coarse,&user.coarse.J));
  CHKERRQ(FormJacobian_Grid(&user,&user.fine,&user.fine.J));

  /* Create coarse level */
  CHKERRQ(PCMGGetCoarseSolve(pc,&user.ksp_coarse));
  CHKERRQ(KSPSetOptionsPrefix(user.ksp_coarse,"coarse_"));
  CHKERRQ(KSPSetFromOptions(user.ksp_coarse));
  CHKERRQ(KSPSetOperators(user.ksp_coarse,user.coarse.J,user.coarse.J));
  CHKERRQ(PCMGSetX(pc,COARSE_LEVEL,user.coarse.x));
  CHKERRQ(PCMGSetRhs(pc,COARSE_LEVEL,user.coarse.b));

  /* Create fine level */
  CHKERRQ(PCMGGetSmoother(pc,FINE_LEVEL,&ksp_fine));
  CHKERRQ(KSPSetOptionsPrefix(ksp_fine,"fine_"));
  CHKERRQ(KSPSetFromOptions(ksp_fine));
  CHKERRQ(KSPSetOperators(ksp_fine,user.fine.J,user.fine.J));
  CHKERRQ(PCMGSetR(pc,FINE_LEVEL,user.fine.r));

  /* Create interpolation between the levels */
  CHKERRQ(DMCreateInterpolation(user.coarse.da,user.fine.da,&user.Ii,NULL));
  CHKERRQ(PCMGSetInterpolation(pc,FINE_LEVEL,user.Ii));
  CHKERRQ(PCMGSetRestriction(pc,FINE_LEVEL,user.Ii));

  CHKERRQ(KSPSetOperators(ksp,user.fine.J,user.fine.J));

  CHKERRQ(VecSet(user.fine.b,one));

  /* Set options, then solve nonlinear system */
  CHKERRQ(KSPSetFromOptions(ksp));

  CHKERRQ(KSPSolve(ksp,user.fine.b,user.fine.x));
  CHKERRQ(KSPGetIterationNumber(ksp,&its));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Number of iterations = %D\n",its));

  /* Free data structures */
  CHKERRQ(MatDestroy(&user.fine.J));
  CHKERRQ(VecDestroy(&user.fine.x));
  CHKERRQ(VecDestroy(&user.fine.r));
  CHKERRQ(VecDestroy(&user.fine.b));
  CHKERRQ(DMDestroy(&user.fine.da));
  CHKERRQ(VecDestroy(&user.fine.localX));
  CHKERRQ(VecDestroy(&user.fine.localF));

  CHKERRQ(MatDestroy(&user.coarse.J));
  CHKERRQ(VecDestroy(&user.coarse.x));
  CHKERRQ(VecDestroy(&user.coarse.b));
  CHKERRQ(DMDestroy(&user.coarse.da));
  CHKERRQ(VecDestroy(&user.coarse.localX));
  CHKERRQ(VecDestroy(&user.coarse.localF));

  CHKERRQ(KSPDestroy(&ksp));
  CHKERRQ(MatDestroy(&user.Ii));
  ierr = PetscFinalize();
  return ierr;
}

PetscErrorCode FormJacobian_Grid(AppCtx *user,GridCtx *grid,Mat *J)
{
  Mat                    jac = *J;
  PetscInt               i,j,row,mx,my,xs,ys,xm,ym,Xs,Ys,Xm,Ym,col[5];
  PetscInt               grow;
  const PetscInt         *ltog;
  PetscScalar            two = 2.0,one = 1.0,v[5],hx,hy,hxdhy,hydhx,value;
  ISLocalToGlobalMapping ltogm;

  mx    = grid->mx;               my = grid->my;
  hx    = one/(PetscReal)(mx-1);  hy = one/(PetscReal)(my-1);
  hxdhy = hx/hy;               hydhx = hy/hx;

  /* Get ghost points */
  CHKERRQ(DMDAGetCorners(grid->da,&xs,&ys,0,&xm,&ym,0));
  CHKERRQ(DMDAGetGhostCorners(grid->da,&Xs,&Ys,0,&Xm,&Ym,0));
  CHKERRQ(DMGetLocalToGlobalMapping(grid->da,&ltogm));
  CHKERRQ(ISLocalToGlobalMappingGetIndices(ltogm,&ltog));

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
        CHKERRQ(MatSetValues(jac,1,&grow,5,col,v,INSERT_VALUES));
      } else if ((i > 0 && i < mx-1) || (j > 0 && j < my-1)) {
        value = .5*two*(hydhx + hxdhy);
        CHKERRQ(MatSetValues(jac,1,&grow,1,&grow,&value,INSERT_VALUES));
      } else {
        value = .25*two*(hydhx + hxdhy);
        CHKERRQ(MatSetValues(jac,1,&grow,1,&grow,&value,INSERT_VALUES));
      }
    }
  }
  CHKERRQ(ISLocalToGlobalMappingRestoreIndices(ltogm,&ltog));
  CHKERRQ(MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY));

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
