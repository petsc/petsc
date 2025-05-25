static char help[] = "Tests for high order geometry\n\n";

#include <petscdmplex.h>
#include <petscds.h>

typedef struct {
  PetscReal volume; /* Analytical volume of the mesh */
  PetscReal tol;    /* Tolerance for volume check */
} AppCtx;

PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscFunctionBegin;
  options->volume = -1.0;
  options->tol    = PETSC_SMALL;

  PetscOptionsBegin(comm, "", "Meshing Interpolation Test Options", "DMPLEX");
  PetscCall(PetscOptionsReal("-volume", "The analytical volume of the mesh", "ex33.c", options->volume, &options->volume, NULL));
  PetscCall(PetscOptionsReal("-tol", "The tolerance for the volume check", "ex33.c", options->tol, &options->tol, NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *ctx, DM *dm)
{
  PetscFunctionBegin;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static void volume(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar vol[])
{
  vol[0] = 1.;
}

static PetscErrorCode CreateDiscretization(DM dm, AppCtx *ctx)
{
  PetscDS        ds;
  PetscFE        fe;
  DMPolytopeType ct;
  PetscInt       dim, cStart;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, NULL));
  PetscCall(DMPlexGetCellType(dm, cStart, &ct));
  PetscCall(PetscFECreateByCell(PETSC_COMM_SELF, dim, 1, ct, NULL, PETSC_DETERMINE, &fe));
  PetscCall(PetscFESetName(fe, "scalar"));
  PetscCall(DMAddField(dm, NULL, (PetscObject)fe));
  PetscCall(PetscFEDestroy(&fe));
  PetscCall(DMCreateDS(dm));
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(PetscDSSetObjective(ds, 0, volume));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckVolume(DM dm, AppCtx *ctx)
{
  Vec         u;
  PetscScalar result;
  PetscReal   vol, tol = ctx->tol;

  PetscFunctionBeginUser;
  PetscCall(DMGetGlobalVector(dm, &u));
  PetscCall(DMPlexComputeIntegralFEM(dm, u, &result, ctx));
  vol = PetscRealPart(result);
  PetscCall(DMRestoreGlobalVector(dm, &u));
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm), "Volume: %g\n", (double)vol));
  PetscCheck(ctx->volume <= 0.0 || PetscAbsReal(ctx->volume - vol) <= tol, PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Calculated volume %g != %g actual volume (error %g > %g tol)", (double)vol, (double)ctx->volume, (double)PetscAbsReal(ctx->volume - vol), (double)tol);
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM     dm;
  AppCtx user;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &user, &dm));
  PetscCall(CreateDiscretization(dm, &user));
  PetscCall(CheckVolume(dm, &user));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    args: -dm_plex_simplex 0 -dm_plex_box_faces 1,1 -dm_plex_box_lower -1.,-1. -dm_plex_box_upper 1.,1. -volume 4.

    test:
      suffix: square_0
      args: -dm_coord_petscspace_degree 1

    test:
      suffix: square_1
      args: -dm_coord_petscspace_degree 2

    test:
      suffix: square_2
      args: -dm_refine 1 -dm_coord_petscspace_degree 1

    test:
      suffix: square_3
      args: -dm_refine 1 -dm_coord_petscspace_degree 2

  testset:
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_plex_box_faces 1,1,1 -dm_plex_box_lower -1.,-1.,-1. -dm_plex_box_upper 1.,1.,1. -volume 8.

    test:
      suffix: cube_0
      args: -dm_coord_petscspace_degree 1

    test:
      suffix: cube_1
      args: -dm_coord_petscspace_degree 2

    test:
      suffix: cube_2
      args: -dm_refine 1 -dm_coord_petscspace_degree 1

    test:
      suffix: cube_3
      args: -dm_refine 1 -dm_coord_petscspace_degree 2

  testset:
    args: -dm_plex_simplex 0 -dm_plex_box_faces 1,1 -dm_plex_box_lower -1.,-1. -dm_plex_box_upper 1.,1. \
          -volume 4. -dm_coord_remap -dm_coord_map shear -dm_coord_map_params 0.0,0.0,3.0

    test:
      suffix: shear_0
      args: -dm_coord_petscspace_degree 1

    test:
      suffix: shear_1
      args: -dm_coord_petscspace_degree 2

    test:
      suffix: shear_2
      args: -dm_refine 1 -dm_coord_petscspace_degree 1

    test:
      suffix: shear_3
      args: -dm_refine 1 -dm_coord_petscspace_degree 2

  testset:
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_plex_box_faces 1,1,1 -dm_plex_box_lower -1.,-1.,-1. -dm_plex_box_upper 1.,1.,1. \
          -volume 8. -dm_coord_remap -dm_coord_map shear -dm_coord_map_params 0.0,0.0,3.0,4.0

    test:
      suffix: shear_4
      args: -dm_coord_petscspace_degree 1

    test:
      suffix: shear_5
      args: -dm_coord_petscspace_degree 2

    test:
      suffix: shear_6
      args: -dm_refine 1 -dm_coord_petscspace_degree 1

    test:
      suffix: shear_7
      args: -dm_refine 1 -dm_coord_petscspace_degree 2

  testset:
    args: -dm_plex_simplex 0 -dm_plex_box_faces 1,1 -dm_plex_box_lower 1.,-1. -dm_plex_box_upper 3.,1. \
          -dm_coord_petscspace_degree 1 -volume 8. -dm_coord_remap -dm_coord_map flare

    test:
      suffix: flare_0
      args:

    test:
      suffix: flare_1
      args: -dm_refine 2

  testset:
    # Area: 3/4 \pi = 2.3562
    args: -dm_plex_simplex 0 -dm_plex_box_faces 1,1 -petscfe_default_quadrature_order 2 \
          -volume 2.35619449019235 -dm_coord_remap -dm_coord_map annulus

    test:
      # Area: (a+b)/2 h = 3/\sqrt{2} (sqrt{2} - 1/\sqrt{2}) = 3/2
      suffix: annulus_0
      requires: double
      args: -dm_coord_petscspace_degree 1 -volume 1.5

    test:
      suffix: annulus_1
      requires: double
      args: -dm_refine 3 -dm_coord_petscspace_degree 1 -tol .016

    test:
      suffix: annulus_2
      requires: double
      args: -dm_refine 3 -dm_coord_petscspace_degree 2 -tol .0038

    test:
      suffix: annulus_3
      requires: double
      args: -dm_refine 3 -dm_coord_petscspace_degree 3 -tol 2.2e-6

    test:
      suffix: annulus_4
      requires: double
      args: -dm_refine 2 -dm_coord_petscspace_degree 2 -tol .00012

    test:
      suffix: annulus_5
      requires: double
      args: -dm_refine 2 -dm_coord_petscspace_degree 3 -petscfe_default_quadrature_order 3 -tol 1.2e-7

  testset:
    # Volume: 4/3 \pi (8 - 1)/2 = 14/3 \pi = 14.66076571675238
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_plex_box_faces 1,1,1 -dm_plex_box_lower -1.,-1.,-1. -dm_plex_box_upper 1.,1.,1. \
          -volume 14.66076571675238 -dm_coord_remap -dm_coord_map shell

    test:
      suffix: shell_0
      requires: double
      args: -dm_refine 1 -dm_coord_petscspace_degree 1 -petscfe_default_quadrature_order 1 -volume 5.633164922 -tol 1.0e-7

    test:
      suffix: shell_1
      requires: double
      args: -dm_refine 2 -dm_coord_petscspace_degree 1 -petscfe_default_quadrature_order 1 -tol 3.1

    test:
      suffix: shell_2
      requires: double
      args: -dm_refine 2 -dm_coord_petscspace_degree 2 -petscfe_default_quadrature_order 2 -tol .1

    test:
      suffix: shell_3
      requires: double
      args: -dm_refine 2 -dm_coord_petscspace_degree 3 -petscfe_default_quadrature_order 3 -tol .02

    test:
      suffix: shell_4
      requires: double
      args: -dm_refine 2 -dm_coord_petscspace_degree 4 -petscfe_default_quadrature_order 4 -tol .006

  test:
    # Volume: 1.0
    suffix: gmsh_q2
    requires: double
    args: -dm_plex_filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/quads-q2.msh -dm_plex_gmsh_project -volume 1.0 -tol 1e-6

  test:
    # Volume: 1.0
    suffix: gmsh_q3
    requires: double
    nsize: {{1 2}}
    args: -dm_plex_filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/quads-q3.msh -dm_plex_gmsh_project -volume 1.0 -tol 1e-6

  test:
    # Volume: 1.0
    suffix: gmsh_3d_q2
    requires: double
    args: -dm_plex_filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/cube_q2.msh -dm_plex_gmsh_project -volume 1.0 -tol 1e-6

  test:
    # Volume: 1.0
    suffix: gmsh_3d_q3
    requires: double
    nsize: {{1 2}}
    args: -dm_plex_filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/cube_q3.msh -dm_plex_gmsh_project -volume 1.0 -tol 1e-6

TEST*/
