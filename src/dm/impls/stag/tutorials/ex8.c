static char help[] = "Solves the Poisson equation using DMStag, with a single field in 1D,\n"
                     "intended to demonstrate the simplest use of geometric multigrid\n\n";

#include <petscdm.h>
#include <petscksp.h>
#include <petscdmstag.h>

static PetscErrorCode AssembleSystem(DM, Mat *, Vec *);

int main(int argc, char **argv)
{
  Mat      A;
  Vec      x, b;
  KSP      ksp;
  DM       dm;
  PetscInt dim;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));

  dim = 1;

  /* Create a DMStag object with a single degree of freedom for each point
     on a single stratum, either vertices (0-cells) or elements (d-cells in d dimensions) */
  if (dim == 1) {
    PetscCall(DMStagCreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, 8, /* Global element count */
                             1,                                     /* Unknowns per vertex */
                             0,                                     /* Unknowns per element */
                             DMSTAG_STENCIL_BOX, 1,                 /* Elementwise stencil width */
                             NULL, &dm));
  } else SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "Unsupported dimension: %" PetscInt_FMT, dim);
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));

  /* Assemble the discrete system */
  PetscCall(AssembleSystem(dm, &A, &b));

  /* Solve */
  PetscCall(KSPCreate(PetscObjectComm((PetscObject)dm), &ksp));
  PetscCall(KSPSetType(ksp, KSPFGMRES));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetDM(ksp, dm));
  PetscCall(KSPSetDMActive(ksp, PETSC_FALSE));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(VecDuplicate(b, &x));
  PetscCall(KSPSolve(ksp, b, x));
  {
    KSPConvergedReason reason;

    PetscCall(KSPGetConvergedReason(ksp, &reason));
    PetscCheck(reason >= 0, PETSC_COMM_WORLD, PETSC_ERR_CONV_FAILED, "Linear solve failed");
  }

  /* Destroy PETSc objects and finalize */
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

static PetscErrorCode AssembleSystem1DVertexCentered(DM dm, Mat *pA, Vec *pb)
{
  Mat      A;
  Vec      b;
  PetscInt start, n, n_extra, N;

  PetscFunctionBeginUser;

  PetscCall(DMCreateMatrix(dm, pA));
  A = *pA;
  PetscCall(DMCreateGlobalVector(dm, pb));
  b = *pb;
  PetscCall(DMStagGetCorners(dm, &start, NULL, NULL, &n, NULL, NULL, &n_extra, NULL, NULL));
  PetscCall(DMStagGetGlobalSizes(dm, &N, NULL, NULL));

  /* Loop over all elements, including the non-physical, extra one past the right boundary */
  for (PetscInt e = start; e < start + n + n_extra; ++e) {
    DMStagStencil row;

    row.i   = e;
    row.c   = 0;
    row.loc = DMSTAG_LEFT;

    if (e == 0) {
      /* Left bondary conditions (Dirichlet) */
      PetscScalar val;

      val = 1.0;
      PetscCall(DMStagMatSetValuesStencil(dm, A, 1, &row, 1, &row, &val, INSERT_VALUES));
    } else if (e == N) {
      /* Right boundary (Dirichlet Boundary conditions)*/
      DMStagStencil row_extra;
      PetscScalar   val;

      row_extra.i   = e;
      row_extra.c   = 0;
      row_extra.loc = DMSTAG_LEFT;
      val           = 1.0;

      PetscCall(DMStagMatSetValuesStencil(dm, A, 1, &row_extra, 1, &row_extra, &val, INSERT_VALUES));
    } else {
      /* Interior */
      DMStagStencil col[3];
      PetscScalar   val[3];

      col[0].i   = e - 1;
      col[0].c   = 0;
      col[0].loc = DMSTAG_LEFT;
      val[0]     = 1.0;
      col[1].i   = e;
      col[1].c   = 0;
      col[1].loc = DMSTAG_LEFT;
      val[1]     = -2.0;
      col[2].i   = e + 1;
      col[2].c   = 0;
      col[2].loc = DMSTAG_LEFT;
      val[2]     = 1.0;

      PetscCall(DMStagMatSetValuesStencil(dm, A, 1, &row, 3, col, val, INSERT_VALUES));
    }

    /* Forcing */
    {
      PetscScalar x, f, h;

      h = 1.0 / N;              /* Assume a constant spacing instead of accessing coordinates */
      x = e / ((PetscScalar)N); // 0 - 1
      f = (x - 0.5) * h * h;    // Scale by h^2
      PetscCall(DMStagVecSetValuesStencil(dm, b, 1, &row, &f, INSERT_VALUES));
    }
  }

  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(VecAssemblyBegin(b));
  PetscCall(VecAssemblyEnd(b));

  PetscFunctionReturn(0);
}

PetscErrorCode AssembleSystem(DM dm, Mat *pA, Vec *pb)
{
  PetscInt dim;

  PetscFunctionBeginUser;
  PetscCall(DMSetMatrixPreallocateOnly(dm, PETSC_TRUE));
  PetscCall(DMGetDimension(dm, &dim));
  switch (dim) {
  case 1:
    PetscCall(AssembleSystem1DVertexCentered(dm, pA, pb));
    break;
  default:
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "Unsupported dimension: %" PetscInt_FMT, dim);
  }
  PetscFunctionReturn(0);
}

/*TEST
   test:
      args: -pc_type mg -pc_mg_galerkin -pc_mg_levels 7 -stag_grid_x 512 -ksp_converged_reason

TEST*/
