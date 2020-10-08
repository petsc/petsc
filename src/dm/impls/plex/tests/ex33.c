static char help[] = "Tests for high order geometry\n\n";

#include <petscdmplex.h>
#include <petscds.h>

typedef enum {TRANSFORM_NONE, TRANSFORM_SHEAR, TRANSFORM_ANNULUS, TRANSFORM_SHELL} Transform;
const char * const TransformTypes[] = {"none", "shear", "annulus", "shell", "Mesh Transform", "TRANSFORM_", NULL};

typedef struct {
  char        filename[PETSC_MAX_PATH_LEN]; /* Import mesh from file */
  Transform   meshTransform;      /* Transform for initial box mesh */
  PetscReal   *transformDataReal; /* Parameters for mesh transform */
  PetscScalar *transformData;     /* Parameters for mesh transform */
  PetscReal   volume;             /* Analytical volume of the mesh */
  PetscReal   tol;                /* Tolerance for volume check */
} AppCtx;

PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscInt       n = 0, i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->filename[0]       = '\0';
  options->meshTransform     = TRANSFORM_NONE;
  options->transformDataReal = NULL;
  options->transformData     = NULL;
  options->volume            = -1.0;
  options->tol               = PETSC_SMALL;

  ierr = PetscOptionsBegin(comm, "", "Meshing Interpolation Test Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsString("-filename", "The mesh file", "ex33.c", options->filename, options->filename, sizeof(options->filename), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-mesh_transform", "Method to transform initial box mesh <none, shear, annulus, shell>", "ex33.c", TransformTypes, (PetscEnum) options->meshTransform, (PetscEnum *) &options->meshTransform, NULL);CHKERRQ(ierr);
  switch (options->meshTransform) {
    case TRANSFORM_NONE: break;
    case TRANSFORM_SHEAR:
      n = 2;
      ierr = PetscMalloc1(n, &options->transformDataReal);CHKERRQ(ierr);
      for (i = 0; i < n; ++i) options->transformDataReal[i] = 1.0;
      ierr = PetscOptionsRealArray("-transform_data", "Parameters for mesh transforms", "ex33.c", options->transformDataReal, &n, NULL);CHKERRQ(ierr);
      break;
    case TRANSFORM_ANNULUS:
      n = 2;
      ierr = PetscMalloc1(n, &options->transformData);CHKERRQ(ierr);
      options->transformData[0] = 1.0;
      options->transformData[1] = 2.0;
      ierr = PetscOptionsScalarArray("-transform_data", "Parameters for mesh transforms", "ex33.c", options->transformData, &n, NULL);CHKERRQ(ierr);
      break;
    case TRANSFORM_SHELL:
      n = 2;
      ierr = PetscMalloc1(n, &options->transformData);CHKERRQ(ierr);
      options->transformData[0] = 1.0;
      options->transformData[1] = 2.0;
      ierr = PetscOptionsScalarArray("-transform_data", "Parameters for mesh transforms", "ex33.c", options->transformData, &n, NULL);CHKERRQ(ierr);
      break;
    default: SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "Unknown mesh transform %D", options->meshTransform);
  }
  ierr = PetscOptionsReal("-volume", "The analytical volume of the mesh", "ex33.c", options->volume, &options->volume, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tol", "The tolerance for the volume check", "ex33.c", options->tol, &options->tol, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetCoordinateDim(dm, &dE);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(cdm, 0, &cStart, NULL);CHKERRQ(ierr);
  ierr = DMPlexGetCellType(dm, cStart, &ct);CHKERRQ(ierr);
  simplex = DMPolytopeTypeGetNumVertices(ct) == DMPolytopeTypeGetDim(ct)+1 ? PETSC_TRUE : PETSC_FALSE;
  ierr = PetscFECreateDefault(PETSC_COMM_SELF, dim, dE, simplex, "geom_", -1, &fe);CHKERRQ(ierr);
  ierr = DMProjectCoordinates(dm, fe);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *ctx, DM *dm)
{
  const char    *filename = ctx->filename;
  DM             cdm;
  PetscDS        cds;
  size_t         len;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
  ierr = PetscStrlen(filename, &len);CHKERRQ(ierr);
  if (len) {ierr = DMPlexCreateFromFile(comm, filename, PETSC_TRUE, dm);CHKERRQ(ierr);}
  else     {ierr = DMPlexCreateBoxMesh(comm, 2, PETSC_FALSE, NULL, NULL, NULL, NULL, PETSC_TRUE, dm);CHKERRQ(ierr);}
  ierr = DMLocalizeCoordinates(*dm);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *dm, "Mesh");CHKERRQ(ierr);
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  if (!len) {ierr = DMCreateCoordinateDisc(*dm);CHKERRQ(ierr);}
  switch (ctx->meshTransform) {
    case TRANSFORM_NONE:
      ierr = DMPlexRemapGeometry(*dm, 0.0, identity);CHKERRQ(ierr);
      break;
    case TRANSFORM_SHEAR:
      ierr = DMPlexShearGeometry(*dm, DM_X, ctx->transformDataReal);CHKERRQ(ierr);
      break;
    case TRANSFORM_ANNULUS:
      ierr = DMGetCoordinateDM(*dm, &cdm);CHKERRQ(ierr);
      ierr = DMGetDS(cdm, &cds);CHKERRQ(ierr);
      ierr = PetscDSSetConstants(cds, 2, ctx->transformData);CHKERRQ(ierr);
      ierr = DMPlexRemapGeometry(*dm, 0.0, f0_annulus);CHKERRQ(ierr);
      break;
    case TRANSFORM_SHELL:
      ierr = DMGetCoordinateDM(*dm, &cdm);CHKERRQ(ierr);
      ierr = DMGetDS(cdm, &cds);CHKERRQ(ierr);
      ierr = PetscDSSetConstants(cds, 2, ctx->transformData);CHKERRQ(ierr);
      ierr = DMPlexRemapGeometry(*dm, 0.0, f0_shell);CHKERRQ(ierr);
      break;
    default: SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "Unknown mesh transform %D", ctx->meshTransform);
  }
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, NULL);CHKERRQ(ierr);
  ierr = DMPlexGetCellType(dm, cStart, &ct);CHKERRQ(ierr);
  simplex = DMPolytopeTypeGetNumVertices(ct) == DMPolytopeTypeGetDim(ct)+1 ? PETSC_TRUE : PETSC_FALSE;
  ierr = PetscFECreateDefault(PETSC_COMM_SELF, dim, 1, simplex, NULL, PETSC_DETERMINE, &fe);CHKERRQ(ierr);
  ierr = PetscFESetName(fe, "scalar");CHKERRQ(ierr);
  ierr = DMAddField(dm, NULL, (PetscObject) fe);
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  ierr = DMCreateDS(dm);CHKERRQ(ierr);
  ierr = DMGetDS(dm, &ds);CHKERRQ(ierr);
  ierr = PetscDSSetObjective(ds, 0, volume);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CheckVolume(DM dm, AppCtx *ctx)
{
  Vec            u;
  PetscScalar    result;
  PetscReal      vol, tol = ctx->tol;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetGlobalVector(dm, &u);CHKERRQ(ierr);
  ierr = DMPlexComputeIntegralFEM(dm, u, &result, ctx);CHKERRQ(ierr);
  vol  = PetscRealPart(result);
  ierr = DMRestoreGlobalVector(dm, &u);CHKERRQ(ierr);
  ierr = PetscPrintf(PetscObjectComm((PetscObject) dm), "Volume: %g\n", (double) vol);CHKERRQ(ierr);
  if (ctx->volume > 0.0 && PetscAbsReal(ctx->volume - vol) > tol) {
    SETERRQ4(PetscObjectComm((PetscObject) dm), PETSC_ERR_PLIB, "Calculated volume %g != %g actual volume (error %g > %g tol)", (double) vol, (double) ctx->volume, (double) PetscAbsReal(ctx->volume - vol), (double) tol);
  }
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;
  AppCtx         user;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &dm);CHKERRQ(ierr);
  ierr = CreateDiscretization(dm, &user);CHKERRQ(ierr);
  ierr = CheckVolume(dm, &user);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFree(user.transformDataReal);CHKERRQ(ierr);
  ierr = PetscFree(user.transformData);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: square_0
    args: -dm_plex_box_faces 1,1 -dm_plex_box_lower -1.,-1. -dm_plex_box_upper 1.,1. -geom_petscspace_degree 1 -volume 4.

  test:
    suffix: square_1
    args: -dm_plex_box_faces 1,1 -dm_plex_box_lower -1.,-1. -dm_plex_box_upper 1.,1. -geom_petscspace_degree 2 -volume 4.

  test:
    suffix: square_2
    args: -dm_plex_box_faces 1,1 -dm_plex_box_lower -1.,-1. -dm_plex_box_upper 1.,1. -dm_refine 1 -geom_petscspace_degree 1 -volume 4.

  test:
    suffix: square_3
    args: -dm_plex_box_faces 1,1 -dm_plex_box_lower -1.,-1. -dm_plex_box_upper 1.,1. -dm_refine 1 -geom_petscspace_degree 2 -volume 4.

  test:
    suffix: cube_0
    args: -dm_plex_box_dim 3 -dm_plex_box_faces 1,1,1 -dm_plex_box_lower -1.,-1.,-1. -dm_plex_box_upper 1.,1.,1. -geom_petscspace_degree 1 -volume 8.

  test:
    suffix: cube_1
    args: -dm_plex_box_dim 3 -dm_plex_box_faces 1,1,1 -dm_plex_box_lower -1.,-1.,-1. -dm_plex_box_upper 1.,1.,1. -geom_petscspace_degree 2 -volume 8.

  test:
    suffix: cube_2
    args: -dm_plex_box_dim 3 -dm_plex_box_faces 1,1,1 -dm_plex_box_lower -1.,-1.,-1. -dm_plex_box_upper 1.,1.,1. -dm_refine 1 -geom_petscspace_degree 1 -volume 8.

  test:
    suffix: cube_3
    args: -dm_plex_box_dim 3 -dm_plex_box_faces 1,1,1 -dm_plex_box_lower -1.,-1.,-1. -dm_plex_box_upper 1.,1.,1. -dm_refine 1 -geom_petscspace_degree 2 -volume 8.

  test:
    suffix: shear_0
    args: -dm_plex_box_faces 1,1 -dm_plex_box_lower -1.,-1. -dm_plex_box_upper 1.,1. -geom_petscspace_degree 1 -mesh_transform shear -transform_data 3.0 -volume 4.

  test:
    suffix: shear_1
    args: -dm_plex_box_faces 1,1 -dm_plex_box_lower -1.,-1. -dm_plex_box_upper 1.,1. -geom_petscspace_degree 2 -mesh_transform shear -transform_data 3.0 -volume 4.

  test:
    suffix: shear_2
    args: -dm_plex_box_faces 1,1 -dm_plex_box_lower -1.,-1. -dm_plex_box_upper 1.,1. -dm_refine 1 -geom_petscspace_degree 1 -mesh_transform shear -transform_data 3.0 -volume 4.

  test:
    suffix: shear_3
    args: -dm_plex_box_faces 1,1 -dm_plex_box_lower -1.,-1. -dm_plex_box_upper 1.,1. -dm_refine 1 -geom_petscspace_degree 2 -mesh_transform shear -transform_data 3.0 -volume 4.

  test:
    suffix: shear_4
    args: -dm_plex_box_dim 3 -dm_plex_box_faces 1,1,1 -dm_plex_box_lower -1.,-1.,-1. -dm_plex_box_upper 1.,1.,1. -geom_petscspace_degree 1 -mesh_transform shear -transform_data 3.0 -volume 8.

  test:
    suffix: shear_5
    args: -dm_plex_box_dim 3 -dm_plex_box_faces 1,1,1 -dm_plex_box_lower -1.,-1.,-1. -dm_plex_box_upper 1.,1.,1. -geom_petscspace_degree 2 -mesh_transform shear -transform_data 3.0 -volume 8.

  test:
    suffix: shear_6
    args: -dm_plex_box_dim 3 -dm_plex_box_faces 1,1,1 -dm_plex_box_lower -1.,-1,-1. -dm_plex_box_upper 1.,1.,1. -dm_refine 1 -geom_petscspace_degree 1 -mesh_transform shear -transform_data 3.0,4.0 -volume 8.

  test:
    suffix: shear_7
    args: -dm_plex_box_dim 3 -dm_plex_box_faces 1,1,1 -dm_plex_box_lower -1.,-1.,-1. -dm_plex_box_upper 1.,1.,1. -dm_refine 1 -geom_petscspace_degree 2 -mesh_transform shear -transform_data 3.0,4.0 -volume 8.

  test:
    # Area: (a+b)/2 h = 3/\sqrt{2} (sqrt{2} - 1/\sqrt{2}) = 3/2
    suffix: annulus_0
    requires: double
    args: -dm_plex_box_faces 1,1 -geom_petscspace_degree 1 -mesh_transform annulus -volume 1.5

  test:
    # Area: 3/4 \pi = 2.3562
    suffix: annulus_1
    requires: double
    args: -dm_plex_box_faces 1,1 -dm_refine 3 -geom_petscspace_degree 1 -mesh_transform annulus -volume 2.35619449019235 -tol .016

  test:
    # Area: 3/4 \pi = 2.3562
    suffix: annulus_2
    requires: double
    args: -dm_plex_box_faces 1,1 -dm_refine 3 -geom_petscspace_degree 2 -mesh_transform annulus -volume 2.35619449019235 -tol .0038

  test:
    # Area: 3/4 \pi = 2.3562
    suffix: annulus_3
    requires: double
    args: -dm_plex_box_faces 1,1 -dm_refine 3 -geom_petscspace_degree 3 -mesh_transform annulus -volume 2.35619449019235 -tol 2.2e-6

  test:
    # Area: 3/4 \pi = 2.3562
    suffix: annulus_4
    requires: double
    args: -dm_plex_box_faces 1,1 -dm_refine 2 -geom_petscspace_degree 2 -petscfe_default_quadrature_order 2 -mesh_transform annulus -volume 2.35619449019235 -tol .00012

  test:
    # Area: 3/4 \pi = 2.3562
    suffix: annulus_5
    requires: double
    args: -dm_plex_box_faces 1,1 -dm_refine 2 -geom_petscspace_degree 3 -petscfe_default_quadrature_order 3 -mesh_transform annulus -volume 2.35619449019235 -tol 1.2e-7

  test:
    suffix: shell_0
    requires: double
    args: -dm_plex_box_dim 3 -dm_plex_box_faces 1,1,1 -dm_plex_box_lower -1.,-1.,-1. -dm_plex_box_upper 1.,1.,1. -dm_refine 1 -geom_petscspace_degree 1 -petscfe_default_quadrature_order 1 -mesh_transform shell -volume 5.633164922 -tol 1.0e-7

  test:
    # Volume: 4/3 \pi (8 - 1)/2 = 14/3 \pi = 14.66076571675238
    suffix: shell_1
    requires: double
    args: -dm_plex_box_dim 3 -dm_plex_box_faces 1,1,1 -dm_plex_box_lower -1.,-1.,-1. -dm_plex_box_upper 1.,1.,1. -dm_refine 2 -geom_petscspace_degree 1 -petscfe_default_quadrature_order 1 -mesh_transform shell -volume 14.66076571675238 -tol 3.1

  test:
    # Volume: 4/3 \pi (8 - 1)/2 = 14/3 \pi = 14.66076571675238
    suffix: shell_2
    requires: double
    args: -dm_plex_box_dim 3 -dm_plex_box_faces 1,1,1 -dm_plex_box_lower -1.,-1.,-1. -dm_plex_box_upper 1.,1.,1. -dm_refine 2 -geom_petscspace_degree 2 -petscfe_default_quadrature_order 2 -mesh_transform shell -volume 14.66076571675238 -tol .1

  test:
    # Volume: 4/3 \pi (8 - 1)/2 = 14/3 \pi = 14.66076571675238
    suffix: shell_3
    requires: double
    args: -dm_plex_box_dim 3 -dm_plex_box_faces 1,1,1 -dm_plex_box_lower -1.,-1.,-1. -dm_plex_box_upper 1.,1.,1. -dm_refine 2 -geom_petscspace_degree 3 -petscfe_default_quadrature_order 3 -mesh_transform shell -volume 14.66076571675238 -tol .02

  test:
    # Volume: 4/3 \pi (8 - 1)/2 = 14/3 \pi = 14.66076571675238
    suffix: shell_4
    requires: double
    args: -dm_plex_box_dim 3 -dm_plex_box_faces 1,1,1 -dm_plex_box_lower -1.,-1.,-1. -dm_plex_box_upper 1.,1.,1. -dm_refine 2 -geom_petscspace_degree 4 -petscfe_default_quadrature_order 4 -mesh_transform shell -volume 14.66076571675238 -tol .006

  test:
    # Volume: 1.0
    suffix: gmsh_q2
    requires: double
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/quads-q2.msh -dm_plex_gmsh_project -volume 1.0 -tol 1e-6

  test:
    # Volume: 1.0
    suffix: gmsh_q3
    requires: double
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/quads-q3.msh -dm_plex_gmsh_project -volume 1.0 -tol 1e-6

TEST*/
