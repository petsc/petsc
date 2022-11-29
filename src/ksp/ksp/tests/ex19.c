
static char help[] = "Solvers Laplacian with multigrid, bad way.\n\
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
  PetscInt mx, my;         /* number grid points in x and y direction */
  Vec      localX, localF; /* local vectors with ghost region */
  DM       da;
  Vec      x, b, r; /* global vectors */
  Mat      J;       /* Jacobian on grid */
} GridCtx;

typedef struct {
  GridCtx  fine;
  GridCtx  coarse;
  KSP      ksp_coarse;
  PetscInt ratio;
  Mat      Ii; /* interpolation from coarse to fine */
} AppCtx;

#define COARSE_LEVEL 0
#define FINE_LEVEL   1

extern PetscErrorCode FormJacobian_Grid(AppCtx *, GridCtx *, Mat *);

/*
      Mm_ratio - ration of grid lines between fine and coarse grids.
*/
int main(int argc, char **argv)
{
  AppCtx      user;
  PetscInt    its, N, n, Nx = PETSC_DECIDE, Ny = PETSC_DECIDE, nlocal, Nlocal;
  PetscMPIInt size;
  KSP         ksp, ksp_fine;
  PC          pc;
  PetscScalar one = 1.0;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  user.ratio     = 2;
  user.coarse.mx = 5;
  user.coarse.my = 5;

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-Mx", &user.coarse.mx, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-My", &user.coarse.my, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-ratio", &user.ratio, NULL));

  user.fine.mx = user.ratio * (user.coarse.mx - 1) + 1;
  user.fine.my = user.ratio * (user.coarse.my - 1) + 1;

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Coarse grid size %" PetscInt_FMT " by %" PetscInt_FMT "\n", user.coarse.mx, user.coarse.my));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Fine grid size %" PetscInt_FMT " by %" PetscInt_FMT "\n", user.fine.mx, user.fine.my));

  n = user.fine.mx * user.fine.my;
  N = user.coarse.mx * user.coarse.my;

  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-Nx", &Nx, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-Ny", &Ny, NULL));

  /* Set up distributed array for fine grid */
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, user.fine.mx, user.fine.my, Nx, Ny, 1, 1, NULL, NULL, &user.fine.da));
  PetscCall(DMSetFromOptions(user.fine.da));
  PetscCall(DMSetUp(user.fine.da));
  PetscCall(DMCreateGlobalVector(user.fine.da, &user.fine.x));
  PetscCall(VecDuplicate(user.fine.x, &user.fine.r));
  PetscCall(VecDuplicate(user.fine.x, &user.fine.b));
  PetscCall(VecGetLocalSize(user.fine.x, &nlocal));
  PetscCall(DMCreateLocalVector(user.fine.da, &user.fine.localX));
  PetscCall(VecDuplicate(user.fine.localX, &user.fine.localF));
  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, nlocal, nlocal, n, n, 5, NULL, 3, NULL, &user.fine.J));

  /* Set up distributed array for coarse grid */
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, user.coarse.mx, user.coarse.my, Nx, Ny, 1, 1, NULL, NULL, &user.coarse.da));
  PetscCall(DMSetFromOptions(user.coarse.da));
  PetscCall(DMSetUp(user.coarse.da));
  PetscCall(DMCreateGlobalVector(user.coarse.da, &user.coarse.x));
  PetscCall(VecDuplicate(user.coarse.x, &user.coarse.b));
  PetscCall(VecGetLocalSize(user.coarse.x, &Nlocal));
  PetscCall(DMCreateLocalVector(user.coarse.da, &user.coarse.localX));
  PetscCall(VecDuplicate(user.coarse.localX, &user.coarse.localF));
  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, Nlocal, Nlocal, N, N, 5, NULL, 3, NULL, &user.coarse.J));

  /* Create linear solver */
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));

  /* set two level additive Schwarz preconditioner */
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCMG));
  PetscCall(PCMGSetLevels(pc, 2, NULL));
  PetscCall(PCMGSetType(pc, PC_MG_ADDITIVE));

  PetscCall(FormJacobian_Grid(&user, &user.coarse, &user.coarse.J));
  PetscCall(FormJacobian_Grid(&user, &user.fine, &user.fine.J));

  /* Create coarse level */
  PetscCall(PCMGGetCoarseSolve(pc, &user.ksp_coarse));
  PetscCall(KSPSetOptionsPrefix(user.ksp_coarse, "coarse_"));
  PetscCall(KSPSetFromOptions(user.ksp_coarse));
  PetscCall(KSPSetOperators(user.ksp_coarse, user.coarse.J, user.coarse.J));
  PetscCall(PCMGSetX(pc, COARSE_LEVEL, user.coarse.x));
  PetscCall(PCMGSetRhs(pc, COARSE_LEVEL, user.coarse.b));

  /* Create fine level */
  PetscCall(PCMGGetSmoother(pc, FINE_LEVEL, &ksp_fine));
  PetscCall(KSPSetOptionsPrefix(ksp_fine, "fine_"));
  PetscCall(KSPSetFromOptions(ksp_fine));
  PetscCall(KSPSetOperators(ksp_fine, user.fine.J, user.fine.J));
  PetscCall(PCMGSetR(pc, FINE_LEVEL, user.fine.r));

  /* Create interpolation between the levels */
  PetscCall(DMCreateInterpolation(user.coarse.da, user.fine.da, &user.Ii, NULL));
  PetscCall(PCMGSetInterpolation(pc, FINE_LEVEL, user.Ii));
  PetscCall(PCMGSetRestriction(pc, FINE_LEVEL, user.Ii));

  PetscCall(KSPSetOperators(ksp, user.fine.J, user.fine.J));

  PetscCall(VecSet(user.fine.b, one));

  /* Set options, then solve nonlinear system */
  PetscCall(KSPSetFromOptions(ksp));

  PetscCall(KSPSolve(ksp, user.fine.b, user.fine.x));
  PetscCall(KSPGetIterationNumber(ksp, &its));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Number of iterations = %" PetscInt_FMT "\n", its));

  /* Free data structures */
  PetscCall(MatDestroy(&user.fine.J));
  PetscCall(VecDestroy(&user.fine.x));
  PetscCall(VecDestroy(&user.fine.r));
  PetscCall(VecDestroy(&user.fine.b));
  PetscCall(DMDestroy(&user.fine.da));
  PetscCall(VecDestroy(&user.fine.localX));
  PetscCall(VecDestroy(&user.fine.localF));

  PetscCall(MatDestroy(&user.coarse.J));
  PetscCall(VecDestroy(&user.coarse.x));
  PetscCall(VecDestroy(&user.coarse.b));
  PetscCall(DMDestroy(&user.coarse.da));
  PetscCall(VecDestroy(&user.coarse.localX));
  PetscCall(VecDestroy(&user.coarse.localF));

  PetscCall(KSPDestroy(&ksp));
  PetscCall(MatDestroy(&user.Ii));
  PetscCall(PetscFinalize());
  return 0;
}

PetscErrorCode FormJacobian_Grid(AppCtx *user, GridCtx *grid, Mat *J)
{
  Mat                    jac = *J;
  PetscInt               i, j, row, mx, my, xs, ys, xm, ym, Xs, Ys, Xm, Ym, col[5];
  PetscInt               grow;
  const PetscInt        *ltog;
  PetscScalar            two = 2.0, one = 1.0, v[5], hx, hy, hxdhy, hydhx, value;
  ISLocalToGlobalMapping ltogm;

  mx    = grid->mx;
  my    = grid->my;
  hx    = one / (PetscReal)(mx - 1);
  hy    = one / (PetscReal)(my - 1);
  hxdhy = hx / hy;
  hydhx = hy / hx;

  /* Get ghost points */
  PetscCall(DMDAGetCorners(grid->da, &xs, &ys, 0, &xm, &ym, 0));
  PetscCall(DMDAGetGhostCorners(grid->da, &Xs, &Ys, 0, &Xm, &Ym, 0));
  PetscCall(DMGetLocalToGlobalMapping(grid->da, &ltogm));
  PetscCall(ISLocalToGlobalMappingGetIndices(ltogm, &ltog));

  /* Evaluate Jacobian of function */
  for (j = ys; j < ys + ym; j++) {
    row = (j - Ys) * Xm + xs - Xs - 1;
    for (i = xs; i < xs + xm; i++) {
      row++;
      grow = ltog[row];
      if (i > 0 && i < mx - 1 && j > 0 && j < my - 1) {
        v[0]   = -hxdhy;
        col[0] = ltog[row - Xm];
        v[1]   = -hydhx;
        col[1] = ltog[row - 1];
        v[2]   = two * (hydhx + hxdhy);
        col[2] = grow;
        v[3]   = -hydhx;
        col[3] = ltog[row + 1];
        v[4]   = -hxdhy;
        col[4] = ltog[row + Xm];
        PetscCall(MatSetValues(jac, 1, &grow, 5, col, v, INSERT_VALUES));
      } else if ((i > 0 && i < mx - 1) || (j > 0 && j < my - 1)) {
        value = .5 * two * (hydhx + hxdhy);
        PetscCall(MatSetValues(jac, 1, &grow, 1, &grow, &value, INSERT_VALUES));
      } else {
        value = .25 * two * (hydhx + hxdhy);
        PetscCall(MatSetValues(jac, 1, &grow, 1, &grow, &value, INSERT_VALUES));
      }
    }
  }
  PetscCall(ISLocalToGlobalMappingRestoreIndices(ltogm, &ltog));
  PetscCall(MatAssemblyBegin(jac, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(jac, MAT_FINAL_ASSEMBLY));

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
