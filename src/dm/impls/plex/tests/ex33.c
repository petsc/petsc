static char help[] = "Tests for high order geometry\n\n";

#include <petscdmplex.h>
#include <petscds.h>

typedef enum {TRANSFORM_NONE, TRANSFORM_SHEAR, TRANSFORM_FLARE, TRANSFORM_ANNULUS, TRANSFORM_SHELL} Transform;
const char * const TransformTypes[] = {"none", "shear", "flare", "annulus", "shell", "Mesh Transform", "TRANSFORM_", NULL};

typedef struct {
  PetscBool   coordSpace;         /* Flag to create coordinate space */
  Transform   meshTransform;      /* Transform for initial box mesh */
  PetscReal   *transformDataReal; /* Parameters for mesh transform */
  PetscScalar *transformData;     /* Parameters for mesh transform */
  PetscReal   volume;             /* Analytical volume of the mesh */
  PetscReal   tol;                /* Tolerance for volume check */
} AppCtx;

PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscInt       n = 0, i;

  PetscFunctionBegin;
  options->coordSpace        = PETSC_TRUE;
  options->meshTransform     = TRANSFORM_NONE;
  options->transformDataReal = NULL;
  options->transformData     = NULL;
  options->volume            = -1.0;
  options->tol               = PETSC_SMALL;

  PetscOptionsBegin(comm, "", "Meshing Interpolation Test Options", "DMPLEX");
  PetscCall(PetscOptionsBool("-coord_space", "Flag to create a coordinate space", "ex33.c", options->coordSpace, &options->coordSpace, NULL));
  PetscCall(PetscOptionsEnum("-mesh_transform", "Method to transform initial box mesh <none, shear, annulus, shell>", "ex33.c", TransformTypes, (PetscEnum) options->meshTransform, (PetscEnum *) &options->meshTransform, NULL));
  switch (options->meshTransform) {
    case TRANSFORM_NONE: break;
    case TRANSFORM_SHEAR:
      n = 2;
      PetscCall(PetscMalloc1(n, &options->transformDataReal));
      for (i = 0; i < n; ++i) options->transformDataReal[i] = 1.0;
      PetscCall(PetscOptionsRealArray("-transform_data", "Parameters for mesh transforms", "ex33.c", options->transformDataReal, &n, NULL));
      break;
    case TRANSFORM_FLARE:
      n = 4;
      PetscCall(PetscMalloc1(n, &options->transformData));
      for (i = 0; i < n; ++i) options->transformData[i] = 1.0;
      options->transformData[0] = (PetscScalar) 0;
      PetscCall(PetscOptionsScalarArray("-transform_data", "Parameters for mesh transforms", "ex33.c", options->transformData, &n, NULL));
      break;
    case TRANSFORM_ANNULUS:
      n = 2;
      PetscCall(PetscMalloc1(n, &options->transformData));
      options->transformData[0] = 1.0;
      options->transformData[1] = 2.0;
      PetscCall(PetscOptionsScalarArray("-transform_data", "Parameters for mesh transforms", "ex33.c", options->transformData, &n, NULL));
      break;
    case TRANSFORM_SHELL:
      n = 2;
      PetscCall(PetscMalloc1(n, &options->transformData));
      options->transformData[0] = 1.0;
      options->transformData[1] = 2.0;
      PetscCall(PetscOptionsScalarArray("-transform_data", "Parameters for mesh transforms", "ex33.c", options->transformData, &n, NULL));
      break;
    default: SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "Unknown mesh transform %d", options->meshTransform);
  }
  PetscCall(PetscOptionsReal("-volume", "The analytical volume of the mesh", "ex33.c", options->volume, &options->volume, NULL));
  PetscCall(PetscOptionsReal("-tol", "The tolerance for the volume check", "ex33.c", options->tol, &options->tol, NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(0);
}

static void identity(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                     const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                     const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                     PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscInt Nc = uOff[1] - uOff[0];
  PetscInt       c;

  for (c = 0; c < Nc; ++c) f0[c] = u[c];
}

/* Flare applies the transformation, assuming we fix x_f,

   x_i = x_i * alpha_i x_f
*/
static void f0_flare(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                     const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                     const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                     PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar coords[])
{
  const PetscInt Nc = uOff[1]-uOff[0];
  const PetscInt cf = (PetscInt) PetscRealPart(constants[0]);
  PetscInt       c;

  for (c = 0; c < Nc; ++c) {
    coords[c] = u[c] * (c == cf ? 1.0 : constants[c+1]*u[cf]);
  }
}

/*
  We would like to map the unit square to a quarter of the annulus between circles of radius 1 and 2. We start by mapping the straight sections, which
  will correspond to the top and bottom of our square. So

    (0,0)--(1,0)  ==>  (1,0)--(2,0)      Just a shift of (1,0)
    (0,1)--(1,1)  ==>  (0,1)--(0,2)      Switch x and y

  So it looks like we want to map each layer in y to a ray, so x is the radius and y is the angle:

    (x, y)  ==>  (x+1, \pi/2 y)                           in (r', \theta') space
            ==>  ((x+1) cos(\pi/2 y), (x+1) sin(\pi/2 y)) in (x', y') space
*/
static void f0_annulus(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                       const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                       const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                       PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar xp[])
{
  const PetscReal ri = PetscRealPart(constants[0]);
  const PetscReal ro = PetscRealPart(constants[1]);

  xp[0] = (x[0] * (ro-ri) + ri) * PetscCosReal(0.5*PETSC_PI*x[1]);
  xp[1] = (x[0] * (ro-ri) + ri) * PetscSinReal(0.5*PETSC_PI*x[1]);
}

/*
  We would like to map the unit cube to a hemisphere of the spherical shell between balls of radius 1 and 2. We want to map the bottom surface onto the
  lower hemisphere and the upper surface onto the top, letting z be the radius.

    (x, y)  ==>  ((z+3)/2, \pi/2 (|x| or |y|), arctan y/x)                                                  in (r', \theta', \phi') space
            ==>  ((z+3)/2 \cos(\theta') cos(\phi'), (z+3)/2 \cos(\theta') sin(\phi'), (z+3)/2 sin(\theta')) in (x', y', z') space
*/
static void f0_shell(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                     const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                     const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                     PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar xp[])
{
  const PetscReal pi4    = PETSC_PI/4.0;
  const PetscReal ri     = PetscRealPart(constants[0]);
  const PetscReal ro     = PetscRealPart(constants[1]);
  const PetscReal rp     = (x[2]+1) * 0.5*(ro-ri) + ri;
  const PetscReal phip   = PetscAtan2Real(x[1], x[0]);
  const PetscReal thetap = 0.5*PETSC_PI * (1.0 - ((((phip <= pi4) && (phip >= -pi4)) || ((phip >= 3.0*pi4) || (phip <= -3.0*pi4))) ? PetscAbsReal(x[0]) : PetscAbsReal(x[1])));

  xp[0] = rp * PetscCosReal(thetap) * PetscCosReal(phip);
  xp[1] = rp * PetscCosReal(thetap) * PetscSinReal(phip);
  xp[2] = rp * PetscSinReal(thetap);
}

static PetscErrorCode DMCreateCoordinateDisc(DM dm)
{
  DM             cdm;
  PetscFE        fe;
  DMPolytopeType ct;
  PetscInt       dim, dE, cStart;
  PetscBool      simplex;

  PetscFunctionBegin;
  PetscCall(DMGetCoordinateDM(dm, &cdm));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetCoordinateDim(dm, &dE));
  PetscCall(DMPlexGetHeightStratum(cdm, 0, &cStart, NULL));
  PetscCall(DMPlexGetCellType(dm, cStart, &ct));
  simplex = DMPolytopeTypeGetNumVertices(ct) == DMPolytopeTypeGetDim(ct)+1 ? PETSC_TRUE : PETSC_FALSE;
  PetscCall(PetscFECreateDefault(PETSC_COMM_SELF, dim, dE, simplex, "dm_coord_", -1, &fe));
  PetscCall(DMProjectCoordinates(dm, fe));
  PetscCall(PetscFEDestroy(&fe));
  PetscFunctionReturn(0);
}

PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *ctx, DM *dm)
{
  DM             cdm;
  PetscDS        cds;

  PetscFunctionBegin;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));

  if (ctx->coordSpace) PetscCall(DMCreateCoordinateDisc(*dm));
  switch (ctx->meshTransform) {
    case TRANSFORM_NONE:
      PetscCall(DMPlexRemapGeometry(*dm, 0.0, identity));
      break;
    case TRANSFORM_SHEAR:
      PetscCall(DMPlexShearGeometry(*dm, DM_X, ctx->transformDataReal));
      break;
    case TRANSFORM_FLARE:
      PetscCall(DMGetCoordinateDM(*dm, &cdm));
      PetscCall(DMGetDS(cdm, &cds));
      PetscCall(PetscDSSetConstants(cds, 4, ctx->transformData));
      PetscCall(DMPlexRemapGeometry(*dm, 0.0, f0_flare));
      break;
    case TRANSFORM_ANNULUS:
      PetscCall(DMGetCoordinateDM(*dm, &cdm));
      PetscCall(DMGetDS(cdm, &cds));
      PetscCall(PetscDSSetConstants(cds, 2, ctx->transformData));
      PetscCall(DMPlexRemapGeometry(*dm, 0.0, f0_annulus));
      break;
    case TRANSFORM_SHELL:
      PetscCall(DMGetCoordinateDM(*dm, &cdm));
      PetscCall(DMGetDS(cdm, &cds));
      PetscCall(PetscDSSetConstants(cds, 2, ctx->transformData));
      PetscCall(DMPlexRemapGeometry(*dm, 0.0, f0_shell));
      break;
    default: SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "Unknown mesh transform %d", ctx->meshTransform);
  }
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(0);
}

static void volume(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                   const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                   const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                   PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar vol[])
{
  vol[0] = 1.;
}

static PetscErrorCode CreateDiscretization(DM dm, AppCtx *ctx)
{
  PetscDS        ds;
  PetscFE        fe;
  DMPolytopeType ct;
  PetscInt       dim, cStart;
  PetscBool      simplex;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, NULL));
  PetscCall(DMPlexGetCellType(dm, cStart, &ct));
  simplex = DMPolytopeTypeGetNumVertices(ct) == DMPolytopeTypeGetDim(ct)+1 ? PETSC_TRUE : PETSC_FALSE;
  PetscCall(PetscFECreateDefault(PETSC_COMM_SELF, dim, 1, simplex, NULL, PETSC_DETERMINE, &fe));
  PetscCall(PetscFESetName(fe, "scalar"));
  PetscCall(DMAddField(dm, NULL, (PetscObject) fe));
  PetscCall(PetscFEDestroy(&fe));
  PetscCall(DMCreateDS(dm));
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(PetscDSSetObjective(ds, 0, volume));
  PetscFunctionReturn(0);
}

static PetscErrorCode CheckVolume(DM dm, AppCtx *ctx)
{
  Vec            u;
  PetscScalar    result;
  PetscReal      vol, tol = ctx->tol;

  PetscFunctionBeginUser;
  PetscCall(DMGetGlobalVector(dm, &u));
  PetscCall(DMPlexComputeIntegralFEM(dm, u, &result, ctx));
  vol  = PetscRealPart(result);
  PetscCall(DMRestoreGlobalVector(dm, &u));
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject) dm), "Volume: %g\n", (double) vol));
  if (ctx->volume > 0.0 && PetscAbsReal(ctx->volume - vol) > tol) {
    SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_PLIB, "Calculated volume %g != %g actual volume (error %g > %g tol)", (double) vol, (double) ctx->volume, (double) PetscAbsReal(ctx->volume - vol), (double) tol);
  }
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;
  AppCtx         user;

  PetscCall(PetscInitialize(&argc, &argv, NULL,help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &user, &dm));
  PetscCall(CreateDiscretization(dm, &user));
  PetscCall(CheckVolume(dm, &user));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFree(user.transformDataReal));
  PetscCall(PetscFree(user.transformData));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    args: -dm_plex_simplex 0 -dm_plex_box_faces 1,1 -dm_plex_box_lower -1.,-1. -dm_plex_box_upper 1.,1. -volume 4. -dm_coord_space 0

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
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_plex_box_faces 1,1,1 -dm_plex_box_lower -1.,-1.,-1. -dm_plex_box_upper 1.,1.,1. -volume 8. -dm_coord_space 0

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
    args: -dm_plex_simplex 0 -dm_plex_box_faces 1,1 -dm_plex_box_lower -1.,-1. -dm_plex_box_upper 1.,1. -volume 4. -dm_coord_space 0

    test:
      suffix: shear_0
      args: -dm_coord_petscspace_degree 1 -mesh_transform shear -transform_data 3.0

    test:
      suffix: shear_1
      args: -dm_coord_petscspace_degree 2 -mesh_transform shear -transform_data 3.0

    test:
      suffix: shear_2
      args: -dm_refine 1 -dm_coord_petscspace_degree 1 -mesh_transform shear -transform_data 3.0

    test:
      suffix: shear_3
      args: -dm_refine 1 -dm_coord_petscspace_degree 2 -mesh_transform shear -transform_data 3.0

  testset:
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_plex_box_faces 1,1,1 -dm_plex_box_lower -1.,-1.,-1. -dm_plex_box_upper 1.,1.,1. -volume 8. -dm_coord_space 0

    test:
      suffix: shear_4
      args: -dm_coord_petscspace_degree 1 -mesh_transform shear -transform_data 3.0

    test:
      suffix: shear_5
      args: -dm_coord_petscspace_degree 2 -mesh_transform shear -transform_data 3.0

    test:
      suffix: shear_6
      args: -dm_refine 1 -dm_coord_petscspace_degree 1 -mesh_transform shear -transform_data 3.0,4.0

    test:
      suffix: shear_7
      args: -dm_refine 1 -dm_coord_petscspace_degree 2 -mesh_transform shear -transform_data 3.0,4.0

  testset:
    args: -dm_coord_space 0 -dm_plex_simplex 0 -dm_plex_box_faces 1,1 -dm_plex_box_lower 1.,-1. -dm_plex_box_upper 3.,1. \
          -dm_coord_petscspace_degree 1 -mesh_transform flare -volume 8.

    test:
      suffix: flare_0
      args:

    test:
      suffix: flare_1
      args: -dm_refine 2

  testset:
    # Area: 3/4 \pi = 2.3562
    args: -dm_plex_simplex 0 -dm_plex_box_faces 1,1 -mesh_transform annulus -volume 2.35619449019235 -dm_coord_space 0

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
      args: -dm_refine 2 -dm_coord_petscspace_degree 2 -petscfe_default_quadrature_order 2 -tol .00012

    test:
      suffix: annulus_5
      requires: double
      args: -dm_refine 2 -dm_coord_petscspace_degree 3 -petscfe_default_quadrature_order 3 -tol 1.2e-7

  testset:
    # Volume: 4/3 \pi (8 - 1)/2 = 14/3 \pi = 14.66076571675238
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_plex_box_faces 1,1,1 -dm_plex_box_lower -1.,-1.,-1. -dm_plex_box_upper 1.,1.,1. -mesh_transform shell -volume 14.66076571675238 -dm_coord_space 0

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
    args: -coord_space 0 -dm_plex_filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/quads-q2.msh -dm_plex_gmsh_project -volume 1.0 -tol 1e-6

  test:
    # Volume: 1.0
    suffix: gmsh_q3
    requires: double
    nsize: {{1 2}}
    args: -coord_space 0 -dm_plex_filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/quads-q3.msh -dm_plex_gmsh_project -volume 1.0 -tol 1e-6

TEST*/
