static char help[] = "Test for function and field projection\n\n";

#include <petscdmplex.h>
#include <petscds.h>

typedef struct {
  PetscBool multifield;  /* Different numbers of input and output fields */
  PetscBool subdomain;   /* Try with a volumetric submesh */
  PetscBool submesh;     /* Try with a boundary submesh */
  PetscBool auxfield;    /* Try with auxiliary fields */
} AppCtx;

/* (x + y)*dim + d */
static PetscErrorCode linear(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscInt c;
  for (c = 0; c < Nc; ++c) u[c] = (x[0] + x[1])*Nc + c;
  return 0;
}

/* {x, y, z} */
static PetscErrorCode linear2(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscInt c;
  for (c = 0; c < Nc; ++c) u[c] = x[c];
  return 0;
}

/* {u_x, u_y, u_z} */
static void linear_vector(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                          const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                          PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f[])
{
  PetscInt d;
  for (d = 0; d < uOff[1]-uOff[0]; ++d) f[d] = u[d+uOff[0]];
}

/* p */
static void linear_scalar(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                          const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                          PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f[])
{
  f[0] = u[uOff[1]];
}

/* {div u, p^2} */
static void divergence_sq(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                          const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                          PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f[])
{
  PetscInt d;
  f[0] = 0.0;
  for (d = 0; d < dim; ++d) f[0] += u_x[uOff_x[0]+d*dim+d];
  f[1] = PetscSqr(u[uOff[1]]);
}

static PetscErrorCode ProcessOptions(AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->multifield  = PETSC_FALSE;
  options->subdomain   = PETSC_FALSE;
  options->submesh     = PETSC_FALSE;
  options->auxfield    = PETSC_FALSE;

  ierr = PetscOptionsBegin(PETSC_COMM_SELF, "", "Meshing Problem Options", "DMPLEX");CHKERRQ(ierr);
  CHKERRQ(PetscOptionsBool("-multifield", "Flag for trying different numbers of input/output fields", "ex23.c", options->multifield, &options->multifield, NULL));
  CHKERRQ(PetscOptionsBool("-subdomain", "Flag for trying volumetric submesh", "ex23.c", options->subdomain, &options->subdomain, NULL));
  CHKERRQ(PetscOptionsBool("-submesh", "Flag for trying boundary submesh", "ex23.c", options->submesh, &options->submesh, NULL));
  CHKERRQ(PetscOptionsBool("-auxfield", "Flag for trying auxiliary fields", "ex23.c", options->auxfield, &options->auxfield, NULL));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscFunctionBegin;
  CHKERRQ(DMCreate(comm, dm));
  CHKERRQ(DMSetType(*dm, DMPLEX));
  CHKERRQ(DMSetFromOptions(*dm));
  CHKERRQ(DMViewFromOptions(*dm, NULL, "-orig_dm_view"));
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupDiscretization(DM dm, PetscInt dim, PetscBool simplex, AppCtx *user)
{
  PetscFE        fe;
  MPI_Comm       comm;

  PetscFunctionBeginUser;
  CHKERRQ(PetscObjectGetComm((PetscObject) dm, &comm));
  CHKERRQ(PetscFECreateDefault(comm, dim, dim, simplex, "velocity_", -1, &fe));
  CHKERRQ(PetscObjectSetName((PetscObject) fe, "velocity"));
  CHKERRQ(DMSetField(dm, 0, NULL, (PetscObject) fe));
  CHKERRQ(PetscFEDestroy(&fe));
  CHKERRQ(PetscFECreateDefault(comm, dim, 1, simplex, "pressure_", -1, &fe));
  CHKERRQ(PetscObjectSetName((PetscObject) fe, "pressure"));
  CHKERRQ(DMSetField(dm, 1, NULL, (PetscObject) fe));
  CHKERRQ(PetscFEDestroy(&fe));
  CHKERRQ(DMCreateDS(dm));
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupOutputDiscretization(DM dm, PetscInt dim, PetscBool simplex, AppCtx *user)
{
  PetscFE        fe;
  MPI_Comm       comm;

  PetscFunctionBeginUser;
  CHKERRQ(PetscObjectGetComm((PetscObject) dm, &comm));
  CHKERRQ(PetscFECreateDefault(comm, dim, dim, simplex, "output_", -1, &fe));
  CHKERRQ(PetscObjectSetName((PetscObject) fe, "output"));
  CHKERRQ(DMSetField(dm, 0, NULL, (PetscObject) fe));
  CHKERRQ(PetscFEDestroy(&fe));
  CHKERRQ(DMCreateDS(dm));
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateSubdomainMesh(DM dm, DMLabel *domLabel, DM *subdm, AppCtx *user)
{
  DMLabel        label;
  PetscBool      simplex;
  PetscInt       dim, cStart, cEnd, c;

  PetscFunctionBeginUser;
  CHKERRQ(DMPlexIsSimplex(dm, &simplex));
  CHKERRQ(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  CHKERRQ(DMLabelCreate(PETSC_COMM_SELF, "subdomain", &label));
  for (c = cStart + (cEnd-cStart)/2; c < cEnd; ++c) CHKERRQ(DMLabelSetValue(label, c, 1));
  CHKERRQ(DMPlexFilter(dm, label, 1, subdm));
  CHKERRQ(DMGetDimension(*subdm, &dim));
  CHKERRQ(SetupDiscretization(*subdm, dim, simplex, user));
  CHKERRQ(PetscObjectSetName((PetscObject) *subdm, "subdomain"));
  CHKERRQ(DMViewFromOptions(*subdm, NULL, "-sub_dm_view"));
  if (domLabel) *domLabel = label;
  else          CHKERRQ(DMLabelDestroy(&label));
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateBoundaryMesh(DM dm, DMLabel *bdLabel, DM *subdm, AppCtx *user)
{
  DMLabel        label;
  PetscBool      simplex;
  PetscInt       dim;

  PetscFunctionBeginUser;
  CHKERRQ(DMPlexIsSimplex(dm, &simplex));
  CHKERRQ(DMLabelCreate(PETSC_COMM_SELF, "sub", &label));
  CHKERRQ(DMPlexMarkBoundaryFaces(dm, 1, label));
  CHKERRQ(DMPlexLabelComplete(dm, label));
  CHKERRQ(DMPlexCreateSubmesh(dm, label, 1, PETSC_TRUE, subdm));
  CHKERRQ(DMGetDimension(*subdm, &dim));
  CHKERRQ(SetupDiscretization(*subdm, dim, simplex, user));
  CHKERRQ(PetscObjectSetName((PetscObject) *subdm, "boundary"));
  CHKERRQ(DMViewFromOptions(*subdm, NULL, "-sub_dm_view"));
  if (bdLabel) *bdLabel = label;
  else         CHKERRQ(DMLabelDestroy(&label));
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateAuxiliaryVec(DM dm, DM *auxdm, Vec *la, AppCtx *user)
{
  PetscErrorCode (**afuncs)(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *);
  PetscBool         simplex;
  PetscInt          dim, Nf, f;

  PetscFunctionBeginUser;
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMPlexIsSimplex(dm, &simplex));
  CHKERRQ(DMGetNumFields(dm, &Nf));
  CHKERRQ(PetscMalloc1(Nf, &afuncs));
  for (f = 0; f < Nf; ++f) afuncs[f]  = linear;
  CHKERRQ(DMClone(dm, auxdm));
  CHKERRQ(SetupDiscretization(*auxdm, dim, simplex, user));
  CHKERRQ(DMCreateLocalVector(*auxdm, la));
  CHKERRQ(DMProjectFunctionLocal(dm, 0.0, afuncs, NULL, INSERT_VALUES, *la));
  CHKERRQ(VecViewFromOptions(*la, NULL, "-local_aux_view"));
  CHKERRQ(PetscFree(afuncs));
  PetscFunctionReturn(0);
}

static PetscErrorCode TestFunctionProjection(DM dm, DM dmAux, DMLabel label, Vec la, const char name[], AppCtx *user)
{
  PetscErrorCode (**funcs)(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *);
  Vec               x, lx;
  PetscInt          Nf, f;
  PetscInt          val[1] = {1};
  char              lname[PETSC_MAX_PATH_LEN];

  PetscFunctionBeginUser;
  if (dmAux) CHKERRQ(DMSetAuxiliaryVec(dm, NULL, 0, 0, la));
  CHKERRQ(DMGetNumFields(dm, &Nf));
  CHKERRQ(PetscMalloc1(Nf, &funcs));
  for (f = 0; f < Nf; ++f) funcs[f] = linear;
  CHKERRQ(DMGetGlobalVector(dm, &x));
  CHKERRQ(PetscStrcpy(lname, "Function "));
  CHKERRQ(PetscStrcat(lname, name));
  CHKERRQ(PetscObjectSetName((PetscObject) x, lname));
  if (!label) CHKERRQ(DMProjectFunction(dm, 0.0, funcs, NULL, INSERT_VALUES, x));
  else        CHKERRQ(DMProjectFunctionLabel(dm, 0.0, label, 1, val, 0, NULL, funcs, NULL, INSERT_VALUES, x));
  CHKERRQ(VecViewFromOptions(x, NULL, "-func_view"));
  CHKERRQ(DMRestoreGlobalVector(dm, &x));
  CHKERRQ(DMGetLocalVector(dm, &lx));
  CHKERRQ(PetscStrcpy(lname, "Local Function "));
  CHKERRQ(PetscStrcat(lname, name));
  CHKERRQ(PetscObjectSetName((PetscObject) lx, lname));
  if (!label) CHKERRQ(DMProjectFunctionLocal(dm, 0.0, funcs, NULL, INSERT_VALUES, lx));
  else        CHKERRQ(DMProjectFunctionLabelLocal(dm, 0.0, label, 1, val, 0, NULL, funcs, NULL, INSERT_VALUES, lx));
  CHKERRQ(VecViewFromOptions(lx, NULL, "-local_func_view"));
  CHKERRQ(DMRestoreLocalVector(dm, &lx));
  CHKERRQ(PetscFree(funcs));
  if (dmAux) CHKERRQ(DMSetAuxiliaryVec(dm, NULL, 0, 0, NULL));
  PetscFunctionReturn(0);
}

static PetscErrorCode TestFieldProjection(DM dm, DM dmAux, DMLabel label, Vec la, const char name[], AppCtx *user)
{
  PetscErrorCode (**afuncs)(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *);
  void           (**funcs)(PetscInt, PetscInt, PetscInt,
                           const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                           const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                           PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]);
  Vec               lx, lu;
  PetscInt          Nf, f;
  PetscInt          val[1] = {1};
  char              lname[PETSC_MAX_PATH_LEN];

  PetscFunctionBeginUser;
  if (dmAux) CHKERRQ(DMSetAuxiliaryVec(dm, NULL, 0, 0, la));
  CHKERRQ(DMGetNumFields(dm, &Nf));
  CHKERRQ(PetscMalloc2(Nf, &funcs, Nf, &afuncs));
  for (f = 0; f < Nf; ++f) afuncs[f]  = linear;
  funcs[0] = linear_vector;
  funcs[1] = linear_scalar;
  CHKERRQ(DMGetLocalVector(dm, &lu));
  CHKERRQ(PetscStrcpy(lname, "Local Field Input "));
  CHKERRQ(PetscStrcat(lname, name));
  CHKERRQ(PetscObjectSetName((PetscObject) lu, lname));
  if (!label) CHKERRQ(DMProjectFunctionLocal(dm, 0.0, afuncs, NULL, INSERT_VALUES, lu));
  else        CHKERRQ(DMProjectFunctionLabelLocal(dm, 0.0, label, 1, val, 0, NULL, afuncs, NULL, INSERT_VALUES, lu));
  CHKERRQ(VecViewFromOptions(lu, NULL, "-local_input_view"));
  CHKERRQ(DMGetLocalVector(dm, &lx));
  CHKERRQ(PetscStrcpy(lname, "Local Field "));
  CHKERRQ(PetscStrcat(lname, name));
  CHKERRQ(PetscObjectSetName((PetscObject) lx, lname));
  if (!label) CHKERRQ(DMProjectFieldLocal(dm, 0.0, lu, funcs, INSERT_VALUES, lx));
  else        CHKERRQ(DMProjectFieldLabelLocal(dm, 0.0, label, 1, val, 0, NULL, lu, funcs, INSERT_VALUES, lx));
  CHKERRQ(VecViewFromOptions(lx, NULL, "-local_field_view"));
  CHKERRQ(DMRestoreLocalVector(dm, &lx));
  CHKERRQ(DMRestoreLocalVector(dm, &lu));
  CHKERRQ(PetscFree2(funcs, afuncs));
  if (dmAux) CHKERRQ(DMSetAuxiliaryVec(dm, NULL, 0, 0, NULL));
  PetscFunctionReturn(0);
}

static PetscErrorCode TestFieldProjectionMultiple(DM dm, DM dmIn, DM dmAux, DMLabel label, Vec la, const char name[], AppCtx *user)
{
  PetscErrorCode (**afuncs)(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *);
  void           (**funcs)(PetscInt, PetscInt, PetscInt,
                           const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                           const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                           PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]);
  Vec               lx, lu;
  PetscInt          Nf, NfIn;
  PetscInt          val[1] = {1};
  char              lname[PETSC_MAX_PATH_LEN];

  PetscFunctionBeginUser;
  if (dmAux) CHKERRQ(DMSetAuxiliaryVec(dm, NULL, 0, 0, la));
  CHKERRQ(DMGetNumFields(dm, &Nf));
  CHKERRQ(DMGetNumFields(dmIn, &NfIn));
  CHKERRQ(PetscMalloc2(Nf, &funcs, NfIn, &afuncs));
  funcs[0]  = divergence_sq;
  afuncs[0] = linear2;
  afuncs[1] = linear;
  CHKERRQ(DMGetLocalVector(dmIn, &lu));
  CHKERRQ(PetscStrcpy(lname, "Local MultiField Input "));
  CHKERRQ(PetscStrcat(lname, name));
  CHKERRQ(PetscObjectSetName((PetscObject) lu, lname));
  if (!label) CHKERRQ(DMProjectFunctionLocal(dmIn, 0.0, afuncs, NULL, INSERT_VALUES, lu));
  else        CHKERRQ(DMProjectFunctionLabelLocal(dmIn, 0.0, label, 1, val, 0, NULL, afuncs, NULL, INSERT_VALUES, lu));
  CHKERRQ(VecViewFromOptions(lu, NULL, "-local_input_view"));
  CHKERRQ(DMGetLocalVector(dm, &lx));
  CHKERRQ(PetscStrcpy(lname, "Local MultiField "));
  CHKERRQ(PetscStrcat(lname, name));
  CHKERRQ(PetscObjectSetName((PetscObject) lx, lname));
  if (!label) CHKERRQ(DMProjectFieldLocal(dm, 0.0, lu, funcs, INSERT_VALUES, lx));
  else        CHKERRQ(DMProjectFieldLabelLocal(dm, 0.0, label, 1, val, 0, NULL, lu, funcs, INSERT_VALUES, lx));
  CHKERRQ(VecViewFromOptions(lx, NULL, "-local_field_view"));
  CHKERRQ(DMRestoreLocalVector(dm, &lx));
  CHKERRQ(DMRestoreLocalVector(dmIn, &lu));
  CHKERRQ(PetscFree2(funcs, afuncs));
  if (dmAux) CHKERRQ(DMSetAuxiliaryVec(dm, NULL, 0, 0, NULL));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm, subdm, auxdm;
  Vec            la;
  PetscInt       dim;
  PetscBool      simplex;
  AppCtx         user;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  CHKERRQ(ProcessOptions(&user));
  CHKERRQ(CreateMesh(PETSC_COMM_WORLD, &user, &dm));
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMPlexIsSimplex(dm, &simplex));
  CHKERRQ(SetupDiscretization(dm, dim, simplex, &user));
  /* Volumetric Mesh Projection */
  if (!user.multifield) {
    CHKERRQ(TestFunctionProjection(dm, NULL, NULL, NULL, "Volumetric Primary", &user));
    CHKERRQ(TestFieldProjection(dm, NULL, NULL, NULL, "Volumetric Primary", &user));
  } else {
    DM dmOut;

    CHKERRQ(DMClone(dm, &dmOut));
    CHKERRQ(SetupOutputDiscretization(dmOut, dim, simplex, &user));
    CHKERRQ(TestFieldProjectionMultiple(dmOut, dm, NULL, NULL, NULL, "Volumetric Primary", &user));
    CHKERRQ(DMDestroy(&dmOut));
  }
  if (user.auxfield) {
    /* Volumetric Mesh Projection with Volumetric Data */
    CHKERRQ(CreateAuxiliaryVec(dm, &auxdm, &la, &user));
    CHKERRQ(TestFunctionProjection(dm, auxdm, NULL, la, "Volumetric Primary and Volumetric Auxiliary", &user));
    CHKERRQ(TestFieldProjection(dm, auxdm, NULL, la, "Volumetric Primary and Volumetric Auxiliary", &user));
    CHKERRQ(VecDestroy(&la));
    /* Update of Volumetric Auxiliary Data with primary Volumetric Data */
    CHKERRQ(DMGetLocalVector(dm, &la));
    CHKERRQ(VecSet(la, 1.0));
    CHKERRQ(TestFieldProjection(auxdm, dm, NULL, la, "Volumetric Auxiliary Update with Volumetric Primary", &user));
    CHKERRQ(DMRestoreLocalVector(dm, &la));
    CHKERRQ(DMDestroy(&auxdm));
  }
  if (user.subdomain) {
    DMLabel domLabel;

    /* Subdomain Mesh Projection */
    CHKERRQ(CreateSubdomainMesh(dm, &domLabel, &subdm, &user));
    CHKERRQ(TestFunctionProjection(subdm, NULL, NULL, NULL, "Subdomain Primary", &user));
    CHKERRQ(TestFieldProjection(subdm, NULL, NULL, NULL, "Subdomain Primary", &user));
    if (user.auxfield) {
      /* Subdomain Mesh Projection with Subdomain Data */
      CHKERRQ(CreateAuxiliaryVec(subdm, &auxdm, &la, &user));
      CHKERRQ(TestFunctionProjection(subdm, auxdm, NULL, la, "Subdomain Primary and Subdomain Auxiliary", &user));
      CHKERRQ(TestFieldProjection(subdm, auxdm, NULL, la, "Subdomain Primary and Subdomain Auxiliary", &user));
      CHKERRQ(VecDestroy(&la));
      CHKERRQ(DMDestroy(&auxdm));
      /* Subdomain Mesh Projection with Volumetric Data */
      CHKERRQ(CreateAuxiliaryVec(dm, &auxdm, &la, &user));
      CHKERRQ(TestFunctionProjection(subdm, auxdm, NULL, la, "Subdomain Primary and Volumetric Auxiliary", &user));
      CHKERRQ(TestFieldProjection(subdm, auxdm, NULL, la, "Subdomain Primary and Volumetric Auxiliary", &user));
      CHKERRQ(VecDestroy(&la));
      CHKERRQ(DMDestroy(&auxdm));
      /* Volumetric Mesh Projection with Subdomain Data */
      CHKERRQ(CreateAuxiliaryVec(subdm, &auxdm, &la, &user));
      CHKERRQ(TestFunctionProjection(subdm, auxdm, domLabel, la, "Volumetric Primary and Subdomain Auxiliary", &user));
      CHKERRQ(TestFieldProjection(subdm, auxdm, domLabel, la, "Volumetric Primary and Subdomain Auxiliary", &user));
      CHKERRQ(VecDestroy(&la));
      CHKERRQ(DMDestroy(&auxdm));
    }
    CHKERRQ(DMDestroy(&subdm));
    CHKERRQ(DMLabelDestroy(&domLabel));
  }
  if (user.submesh) {
    DMLabel bdLabel;

    /* Boundary Mesh Projection */
    CHKERRQ(CreateBoundaryMesh(dm, &bdLabel, &subdm, &user));
    CHKERRQ(TestFunctionProjection(subdm, NULL, NULL, NULL, "Boundary Primary", &user));
    CHKERRQ(TestFieldProjection(subdm, NULL, NULL, NULL, "Boundary Primary", &user));
    if (user.auxfield) {
      /* Boundary Mesh Projection with Boundary Data */
      CHKERRQ(CreateAuxiliaryVec(subdm, &auxdm, &la, &user));
      CHKERRQ(TestFunctionProjection(subdm, auxdm, NULL, la, "Boundary Primary and Boundary Auxiliary", &user));
      CHKERRQ(TestFieldProjection(subdm, auxdm, NULL, la, "Boundary Primary and Boundary Auxiliary", &user));
      CHKERRQ(VecDestroy(&la));
      CHKERRQ(DMDestroy(&auxdm));
      /* Volumetric Mesh Projection with Boundary Data */
      CHKERRQ(CreateAuxiliaryVec(subdm, &auxdm, &la, &user));
      CHKERRQ(TestFunctionProjection(dm, auxdm, bdLabel, la, "Volumetric Primary and Boundary Auxiliary", &user));
      CHKERRQ(TestFieldProjection(dm, auxdm, bdLabel, la, "Volumetric Primary and Boundary Auxiliary", &user));
      CHKERRQ(VecDestroy(&la));
      CHKERRQ(DMDestroy(&auxdm));
    }
    CHKERRQ(DMLabelDestroy(&bdLabel));
    CHKERRQ(DMDestroy(&subdm));
  }
  CHKERRQ(DMDestroy(&dm));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: 0
    requires: triangle
    args: -dm_plex_box_faces 1,1 -func_view -local_func_view -local_input_view -local_field_view
  test:
    suffix: mf_0
    requires: triangle
    args: -dm_plex_box_faces 1,1 -velocity_petscspace_degree 1 -velocity_petscfe_default_quadrature_order 2 \
         -pressure_petscspace_degree 2 -pressure_petscfe_default_quadrature_order 2 \
         -multifield -output_petscspace_degree 1 -output_petscfe_default_quadrature_order 2 \
         -local_input_view -local_field_view
  test:
    suffix: 1
    requires: triangle
    args: -dm_plex_box_faces 1,1 -velocity_petscspace_degree 1 -velocity_petscfe_default_quadrature_order 2 -pressure_petscspace_degree 2 -pressure_petscfe_default_quadrature_order 2 -func_view -local_func_view -local_input_view -local_field_view -submesh -auxfield
  test:
    suffix: 2
    requires: triangle
    args: -dm_plex_box_faces 1,1 -velocity_petscspace_degree 1 -velocity_petscfe_default_quadrature_order 2 -pressure_petscspace_degree 2 -pressure_petscfe_default_quadrature_order 2 -func_view -local_func_view -local_input_view -local_field_view -subdomain -auxfield

TEST*/

/*
  Post-processing wants to project a function of the fields into some FE space
  - This is DMProjectField()
  - What about changing the number of components of the output, like displacement to stress? Aux vars

  Update of state variables
  - This is DMProjectField(), but solution must be the aux var
*/
