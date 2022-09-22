static char help[] = "Test for function and field projection\n\n";

#include <petscdmplex.h>
#include <petscds.h>

typedef struct {
  PetscBool multifield; /* Different numbers of input and output fields */
  PetscBool subdomain;  /* Try with a volumetric submesh */
  PetscBool submesh;    /* Try with a boundary submesh */
  PetscBool auxfield;   /* Try with auxiliary fields */
} AppCtx;

/* (x + y)*dim + d */
static PetscErrorCode linear(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscInt c;
  for (c = 0; c < Nc; ++c) u[c] = (x[0] + x[1]) * Nc + c;
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
static void linear_vector(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f[])
{
  PetscInt d;
  for (d = 0; d < uOff[1] - uOff[0]; ++d) f[d] = u[d + uOff[0]];
}

/* p */
static void linear_scalar(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f[])
{
  f[0] = u[uOff[1]];
}

/* {div u, p^2} */
static void divergence_sq(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f[])
{
  PetscInt d;
  f[0] = 0.0;
  for (d = 0; d < dim; ++d) f[0] += u_x[uOff_x[0] + d * dim + d];
  f[1] = PetscSqr(u[uOff[1]]);
}

static PetscErrorCode ProcessOptions(AppCtx *options)
{
  PetscFunctionBegin;
  options->multifield = PETSC_FALSE;
  options->subdomain  = PETSC_FALSE;
  options->submesh    = PETSC_FALSE;
  options->auxfield   = PETSC_FALSE;

  PetscOptionsBegin(PETSC_COMM_SELF, "", "Meshing Problem Options", "DMPLEX");
  PetscCall(PetscOptionsBool("-multifield", "Flag for trying different numbers of input/output fields", "ex23.c", options->multifield, &options->multifield, NULL));
  PetscCall(PetscOptionsBool("-subdomain", "Flag for trying volumetric submesh", "ex23.c", options->subdomain, &options->subdomain, NULL));
  PetscCall(PetscOptionsBool("-submesh", "Flag for trying boundary submesh", "ex23.c", options->submesh, &options->submesh, NULL));
  PetscCall(PetscOptionsBool("-auxfield", "Flag for trying auxiliary fields", "ex23.c", options->auxfield, &options->auxfield, NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscFunctionBegin;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-orig_dm_view"));
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupDiscretization(DM dm, PetscInt dim, PetscBool simplex, AppCtx *user)
{
  PetscFE  fe;
  MPI_Comm comm;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCall(PetscFECreateDefault(comm, dim, dim, simplex, "velocity_", -1, &fe));
  PetscCall(PetscObjectSetName((PetscObject)fe, "velocity"));
  PetscCall(DMSetField(dm, 0, NULL, (PetscObject)fe));
  PetscCall(PetscFEDestroy(&fe));
  PetscCall(PetscFECreateDefault(comm, dim, 1, simplex, "pressure_", -1, &fe));
  PetscCall(PetscObjectSetName((PetscObject)fe, "pressure"));
  PetscCall(DMSetField(dm, 1, NULL, (PetscObject)fe));
  PetscCall(PetscFEDestroy(&fe));
  PetscCall(DMCreateDS(dm));
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupOutputDiscretization(DM dm, PetscInt dim, PetscBool simplex, AppCtx *user)
{
  PetscFE  fe;
  MPI_Comm comm;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCall(PetscFECreateDefault(comm, dim, dim, simplex, "output_", -1, &fe));
  PetscCall(PetscObjectSetName((PetscObject)fe, "output"));
  PetscCall(DMSetField(dm, 0, NULL, (PetscObject)fe));
  PetscCall(PetscFEDestroy(&fe));
  PetscCall(DMCreateDS(dm));
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateSubdomainMesh(DM dm, DMLabel *domLabel, DM *subdm, AppCtx *user)
{
  DMLabel   label;
  PetscBool simplex;
  PetscInt  dim, cStart, cEnd, c;

  PetscFunctionBeginUser;
  PetscCall(DMPlexIsSimplex(dm, &simplex));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(DMLabelCreate(PETSC_COMM_SELF, "subdomain", &label));
  for (c = cStart + (cEnd - cStart) / 2; c < cEnd; ++c) PetscCall(DMLabelSetValue(label, c, 1));
  PetscCall(DMPlexFilter(dm, label, 1, subdm));
  PetscCall(DMGetDimension(*subdm, &dim));
  PetscCall(SetupDiscretization(*subdm, dim, simplex, user));
  PetscCall(PetscObjectSetName((PetscObject)*subdm, "subdomain"));
  PetscCall(DMViewFromOptions(*subdm, NULL, "-sub_dm_view"));
  if (domLabel) *domLabel = label;
  else PetscCall(DMLabelDestroy(&label));
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateBoundaryMesh(DM dm, DMLabel *bdLabel, DM *subdm, AppCtx *user)
{
  DMLabel   label;
  PetscBool simplex;
  PetscInt  dim;

  PetscFunctionBeginUser;
  PetscCall(DMPlexIsSimplex(dm, &simplex));
  PetscCall(DMLabelCreate(PETSC_COMM_SELF, "sub", &label));
  PetscCall(DMPlexMarkBoundaryFaces(dm, 1, label));
  PetscCall(DMPlexLabelComplete(dm, label));
  PetscCall(DMPlexCreateSubmesh(dm, label, 1, PETSC_TRUE, subdm));
  PetscCall(DMGetDimension(*subdm, &dim));
  PetscCall(SetupDiscretization(*subdm, dim, simplex, user));
  PetscCall(PetscObjectSetName((PetscObject)*subdm, "boundary"));
  PetscCall(DMViewFromOptions(*subdm, NULL, "-sub_dm_view"));
  if (bdLabel) *bdLabel = label;
  else PetscCall(DMLabelDestroy(&label));
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateAuxiliaryVec(DM dm, DM *auxdm, Vec *la, AppCtx *user)
{
  PetscErrorCode (**afuncs)(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *);
  PetscBool simplex;
  PetscInt  dim, Nf, f;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexIsSimplex(dm, &simplex));
  PetscCall(DMGetNumFields(dm, &Nf));
  PetscCall(PetscMalloc1(Nf, &afuncs));
  for (f = 0; f < Nf; ++f) afuncs[f] = linear;
  PetscCall(DMClone(dm, auxdm));
  PetscCall(SetupDiscretization(*auxdm, dim, simplex, user));
  PetscCall(DMCreateLocalVector(*auxdm, la));
  PetscCall(DMProjectFunctionLocal(dm, 0.0, afuncs, NULL, INSERT_VALUES, *la));
  PetscCall(VecViewFromOptions(*la, NULL, "-local_aux_view"));
  PetscCall(PetscFree(afuncs));
  PetscFunctionReturn(0);
}

static PetscErrorCode TestFunctionProjection(DM dm, DM dmAux, DMLabel label, Vec la, const char name[], AppCtx *user)
{
  PetscErrorCode (**funcs)(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *);
  Vec      x, lx;
  PetscInt Nf, f;
  PetscInt val[1] = {1};
  char     lname[PETSC_MAX_PATH_LEN];

  PetscFunctionBeginUser;
  if (dmAux) PetscCall(DMSetAuxiliaryVec(dm, NULL, 0, 0, la));
  PetscCall(DMGetNumFields(dm, &Nf));
  PetscCall(PetscMalloc1(Nf, &funcs));
  for (f = 0; f < Nf; ++f) funcs[f] = linear;
  PetscCall(DMGetGlobalVector(dm, &x));
  PetscCall(PetscStrcpy(lname, "Function "));
  PetscCall(PetscStrcat(lname, name));
  PetscCall(PetscObjectSetName((PetscObject)x, lname));
  if (!label) PetscCall(DMProjectFunction(dm, 0.0, funcs, NULL, INSERT_VALUES, x));
  else PetscCall(DMProjectFunctionLabel(dm, 0.0, label, 1, val, 0, NULL, funcs, NULL, INSERT_VALUES, x));
  PetscCall(VecViewFromOptions(x, NULL, "-func_view"));
  PetscCall(DMRestoreGlobalVector(dm, &x));
  PetscCall(DMGetLocalVector(dm, &lx));
  PetscCall(PetscStrcpy(lname, "Local Function "));
  PetscCall(PetscStrcat(lname, name));
  PetscCall(PetscObjectSetName((PetscObject)lx, lname));
  if (!label) PetscCall(DMProjectFunctionLocal(dm, 0.0, funcs, NULL, INSERT_VALUES, lx));
  else PetscCall(DMProjectFunctionLabelLocal(dm, 0.0, label, 1, val, 0, NULL, funcs, NULL, INSERT_VALUES, lx));
  PetscCall(VecViewFromOptions(lx, NULL, "-local_func_view"));
  PetscCall(DMRestoreLocalVector(dm, &lx));
  PetscCall(PetscFree(funcs));
  if (dmAux) PetscCall(DMSetAuxiliaryVec(dm, NULL, 0, 0, NULL));
  PetscFunctionReturn(0);
}

static PetscErrorCode TestFieldProjection(DM dm, DM dmAux, DMLabel label, Vec la, const char name[], AppCtx *user)
{
  PetscErrorCode (**afuncs)(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *);
  void (**funcs)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]);
  Vec      lx, lu;
  PetscInt Nf, f;
  PetscInt val[1] = {1};
  char     lname[PETSC_MAX_PATH_LEN];

  PetscFunctionBeginUser;
  if (dmAux) PetscCall(DMSetAuxiliaryVec(dm, NULL, 0, 0, la));
  PetscCall(DMGetNumFields(dm, &Nf));
  PetscCall(PetscMalloc2(Nf, &funcs, Nf, &afuncs));
  for (f = 0; f < Nf; ++f) afuncs[f] = linear;
  funcs[0] = linear_vector;
  funcs[1] = linear_scalar;
  PetscCall(DMGetLocalVector(dm, &lu));
  PetscCall(PetscStrcpy(lname, "Local Field Input "));
  PetscCall(PetscStrcat(lname, name));
  PetscCall(PetscObjectSetName((PetscObject)lu, lname));
  if (!label) PetscCall(DMProjectFunctionLocal(dm, 0.0, afuncs, NULL, INSERT_VALUES, lu));
  else PetscCall(DMProjectFunctionLabelLocal(dm, 0.0, label, 1, val, 0, NULL, afuncs, NULL, INSERT_VALUES, lu));
  PetscCall(VecViewFromOptions(lu, NULL, "-local_input_view"));
  PetscCall(DMGetLocalVector(dm, &lx));
  PetscCall(PetscStrcpy(lname, "Local Field "));
  PetscCall(PetscStrcat(lname, name));
  PetscCall(PetscObjectSetName((PetscObject)lx, lname));
  if (!label) PetscCall(DMProjectFieldLocal(dm, 0.0, lu, funcs, INSERT_VALUES, lx));
  else PetscCall(DMProjectFieldLabelLocal(dm, 0.0, label, 1, val, 0, NULL, lu, funcs, INSERT_VALUES, lx));
  PetscCall(VecViewFromOptions(lx, NULL, "-local_field_view"));
  PetscCall(DMRestoreLocalVector(dm, &lx));
  PetscCall(DMRestoreLocalVector(dm, &lu));
  PetscCall(PetscFree2(funcs, afuncs));
  if (dmAux) PetscCall(DMSetAuxiliaryVec(dm, NULL, 0, 0, NULL));
  PetscFunctionReturn(0);
}

static PetscErrorCode TestFieldProjectionMultiple(DM dm, DM dmIn, DM dmAux, DMLabel label, Vec la, const char name[], AppCtx *user)
{
  PetscErrorCode (**afuncs)(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *);
  void (**funcs)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]);
  Vec      lx, lu;
  PetscInt Nf, NfIn;
  PetscInt val[1] = {1};
  char     lname[PETSC_MAX_PATH_LEN];

  PetscFunctionBeginUser;
  if (dmAux) PetscCall(DMSetAuxiliaryVec(dm, NULL, 0, 0, la));
  PetscCall(DMGetNumFields(dm, &Nf));
  PetscCall(DMGetNumFields(dmIn, &NfIn));
  PetscCall(PetscMalloc2(Nf, &funcs, NfIn, &afuncs));
  funcs[0]  = divergence_sq;
  afuncs[0] = linear2;
  afuncs[1] = linear;
  PetscCall(DMGetLocalVector(dmIn, &lu));
  PetscCall(PetscStrcpy(lname, "Local MultiField Input "));
  PetscCall(PetscStrcat(lname, name));
  PetscCall(PetscObjectSetName((PetscObject)lu, lname));
  if (!label) PetscCall(DMProjectFunctionLocal(dmIn, 0.0, afuncs, NULL, INSERT_VALUES, lu));
  else PetscCall(DMProjectFunctionLabelLocal(dmIn, 0.0, label, 1, val, 0, NULL, afuncs, NULL, INSERT_VALUES, lu));
  PetscCall(VecViewFromOptions(lu, NULL, "-local_input_view"));
  PetscCall(DMGetLocalVector(dm, &lx));
  PetscCall(PetscStrcpy(lname, "Local MultiField "));
  PetscCall(PetscStrcat(lname, name));
  PetscCall(PetscObjectSetName((PetscObject)lx, lname));
  if (!label) PetscCall(DMProjectFieldLocal(dm, 0.0, lu, funcs, INSERT_VALUES, lx));
  else PetscCall(DMProjectFieldLabelLocal(dm, 0.0, label, 1, val, 0, NULL, lu, funcs, INSERT_VALUES, lx));
  PetscCall(VecViewFromOptions(lx, NULL, "-local_field_view"));
  PetscCall(DMRestoreLocalVector(dm, &lx));
  PetscCall(DMRestoreLocalVector(dmIn, &lu));
  PetscCall(PetscFree2(funcs, afuncs));
  if (dmAux) PetscCall(DMSetAuxiliaryVec(dm, NULL, 0, 0, NULL));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM        dm, subdm, auxdm;
  Vec       la;
  PetscInt  dim;
  PetscBool simplex;
  AppCtx    user;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(ProcessOptions(&user));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &user, &dm));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexIsSimplex(dm, &simplex));
  PetscCall(SetupDiscretization(dm, dim, simplex, &user));
  /* Volumetric Mesh Projection */
  if (!user.multifield) {
    PetscCall(TestFunctionProjection(dm, NULL, NULL, NULL, "Volumetric Primary", &user));
    PetscCall(TestFieldProjection(dm, NULL, NULL, NULL, "Volumetric Primary", &user));
  } else {
    DM dmOut;

    PetscCall(DMClone(dm, &dmOut));
    PetscCall(SetupOutputDiscretization(dmOut, dim, simplex, &user));
    PetscCall(TestFieldProjectionMultiple(dmOut, dm, NULL, NULL, NULL, "Volumetric Primary", &user));
    PetscCall(DMDestroy(&dmOut));
  }
  if (user.auxfield) {
    /* Volumetric Mesh Projection with Volumetric Data */
    PetscCall(CreateAuxiliaryVec(dm, &auxdm, &la, &user));
    PetscCall(TestFunctionProjection(dm, auxdm, NULL, la, "Volumetric Primary and Volumetric Auxiliary", &user));
    PetscCall(TestFieldProjection(dm, auxdm, NULL, la, "Volumetric Primary and Volumetric Auxiliary", &user));
    PetscCall(VecDestroy(&la));
    /* Update of Volumetric Auxiliary Data with primary Volumetric Data */
    PetscCall(DMGetLocalVector(dm, &la));
    PetscCall(VecSet(la, 1.0));
    PetscCall(TestFieldProjection(auxdm, dm, NULL, la, "Volumetric Auxiliary Update with Volumetric Primary", &user));
    PetscCall(DMRestoreLocalVector(dm, &la));
    PetscCall(DMDestroy(&auxdm));
  }
  if (user.subdomain) {
    DMLabel domLabel;

    /* Subdomain Mesh Projection */
    PetscCall(CreateSubdomainMesh(dm, &domLabel, &subdm, &user));
    PetscCall(TestFunctionProjection(subdm, NULL, NULL, NULL, "Subdomain Primary", &user));
    PetscCall(TestFieldProjection(subdm, NULL, NULL, NULL, "Subdomain Primary", &user));
    if (user.auxfield) {
      /* Subdomain Mesh Projection with Subdomain Data */
      PetscCall(CreateAuxiliaryVec(subdm, &auxdm, &la, &user));
      PetscCall(TestFunctionProjection(subdm, auxdm, NULL, la, "Subdomain Primary and Subdomain Auxiliary", &user));
      PetscCall(TestFieldProjection(subdm, auxdm, NULL, la, "Subdomain Primary and Subdomain Auxiliary", &user));
      PetscCall(VecDestroy(&la));
      PetscCall(DMDestroy(&auxdm));
      /* Subdomain Mesh Projection with Volumetric Data */
      PetscCall(CreateAuxiliaryVec(dm, &auxdm, &la, &user));
      PetscCall(TestFunctionProjection(subdm, auxdm, NULL, la, "Subdomain Primary and Volumetric Auxiliary", &user));
      PetscCall(TestFieldProjection(subdm, auxdm, NULL, la, "Subdomain Primary and Volumetric Auxiliary", &user));
      PetscCall(VecDestroy(&la));
      PetscCall(DMDestroy(&auxdm));
      /* Volumetric Mesh Projection with Subdomain Data */
      PetscCall(CreateAuxiliaryVec(subdm, &auxdm, &la, &user));
      PetscCall(TestFunctionProjection(subdm, auxdm, domLabel, la, "Volumetric Primary and Subdomain Auxiliary", &user));
      PetscCall(TestFieldProjection(subdm, auxdm, domLabel, la, "Volumetric Primary and Subdomain Auxiliary", &user));
      PetscCall(VecDestroy(&la));
      PetscCall(DMDestroy(&auxdm));
    }
    PetscCall(DMDestroy(&subdm));
    PetscCall(DMLabelDestroy(&domLabel));
  }
  if (user.submesh) {
    DMLabel bdLabel;

    /* Boundary Mesh Projection */
    PetscCall(CreateBoundaryMesh(dm, &bdLabel, &subdm, &user));
    PetscCall(TestFunctionProjection(subdm, NULL, NULL, NULL, "Boundary Primary", &user));
    PetscCall(TestFieldProjection(subdm, NULL, NULL, NULL, "Boundary Primary", &user));
    if (user.auxfield) {
      /* Boundary Mesh Projection with Boundary Data */
      PetscCall(CreateAuxiliaryVec(subdm, &auxdm, &la, &user));
      PetscCall(TestFunctionProjection(subdm, auxdm, NULL, la, "Boundary Primary and Boundary Auxiliary", &user));
      PetscCall(TestFieldProjection(subdm, auxdm, NULL, la, "Boundary Primary and Boundary Auxiliary", &user));
      PetscCall(VecDestroy(&la));
      PetscCall(DMDestroy(&auxdm));
      /* Volumetric Mesh Projection with Boundary Data */
      PetscCall(CreateAuxiliaryVec(subdm, &auxdm, &la, &user));
      PetscCall(TestFunctionProjection(dm, auxdm, bdLabel, la, "Volumetric Primary and Boundary Auxiliary", &user));
      PetscCall(TestFieldProjection(dm, auxdm, bdLabel, la, "Volumetric Primary and Boundary Auxiliary", &user));
      PetscCall(VecDestroy(&la));
      PetscCall(DMDestroy(&auxdm));
    }
    PetscCall(DMLabelDestroy(&bdLabel));
    PetscCall(DMDestroy(&subdm));
  }
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
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
