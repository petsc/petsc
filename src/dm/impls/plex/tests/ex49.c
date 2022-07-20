static char help[] = "Tests dof numberings for external integrators such as LibCEED.\n\n";

#include <petscdmplex.h>
#include <petscds.h>

typedef struct {
  PetscInt dummy;
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscFunctionBeginUser;
  options->dummy = 1;
  PetscOptionsBegin(comm, "", "Dof Ordering Options", "DMPLEX");
  //PetscCall(PetscOptionsInt("-num_refine", "Refine cycles", "ex46.c", options->Nr, &options->Nr, NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscFunctionBeginUser;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMSetApplicationContext(*dm, user));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupDiscretization(DM dm, AppCtx *user)
{
  DM       cdm = dm;
  PetscFE  fe;
  PetscInt dim;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(PetscFECreateDefault(PETSC_COMM_SELF, dim, 1, PETSC_FALSE, NULL, -1, &fe));
  PetscCall(PetscObjectSetName((PetscObject) fe, "scalar"));
  PetscCall(DMSetField(dm, 0, NULL, (PetscObject) fe));
  PetscCall(DMSetField(dm, 1, NULL, (PetscObject) fe));
  PetscCall(PetscFEDestroy(&fe));
  PetscCall(DMCreateDS(dm));
  while (cdm) {
    PetscCall(DMCopyDisc(dm,cdm));
    PetscCall(DMGetCoarseDM(cdm, &cdm));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode CheckOffsets(DM dm, const char *domain_name, PetscInt label_value, PetscInt height)
{
  const char *height_name[] = {"cells", "faces"};
  DMLabel   domain_label = NULL;
  DM        cdm;
  IS        offIS;
  PetscInt *offsets, Ncell, Ncl, Nc, n;
  PetscInt  Nf, f;

  PetscFunctionBeginUser;
  if (domain_name) PetscCall(DMGetLabel(dm, domain_name, &domain_label));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "## %s: '%s' {%" PetscInt_FMT "}\n", height_name[height], domain_name?domain_name:"default", label_value));
  // Offsets for cell closures
  PetscCall(DMGetNumFields(dm, &Nf));
  for (f = 0; f < Nf; ++f) {
    char name[PETSC_MAX_PATH_LEN];

    PetscCall(DMPlexGetLocalOffsets(dm, domain_label, label_value, height, f, &Ncell, &Ncl, &Nc, &n, &offsets));
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF, Ncell*Ncl, offsets, PETSC_OWN_POINTER, &offIS));
    PetscCall(PetscSNPrintf(name, PETSC_MAX_PATH_LEN, "Field %" PetscInt_FMT " Offsets", f));
    PetscCall(PetscObjectSetName((PetscObject) offIS, name));
    PetscCall(ISViewFromOptions(offIS,  NULL, "-offsets_view"));
    PetscCall(ISDestroy(&offIS));
  }
  // Offsets for coordinates
  {
    Vec                X;
    PetscSection       s;
    const PetscScalar *x;
    const char        *cname;
    PetscInt           cdim;
    PetscBool          isDG = PETSC_FALSE;

    PetscCall(DMGetCellCoordinateDM(dm, &cdm));
    if (!cdm) {
      PetscCall(DMGetCoordinateDM(dm, &cdm)); cname = "Coordinates";
      PetscCall(DMGetCoordinatesLocal(dm, &X));
    } else {
      isDG  = PETSC_TRUE;
      cname = "DG Coordinates";
      PetscCall(DMGetCellCoordinatesLocal(dm, &X));
    }
    if (isDG && height) PetscFunctionReturn(0);
    if (domain_name) PetscCall(DMGetLabel(cdm, domain_name, &domain_label));
    PetscCall(DMPlexGetLocalOffsets(cdm, domain_label, label_value, height, 0, &Ncell, &Ncl, &Nc, &n, &offsets));
    PetscCall(DMGetCoordinateDim(dm, &cdim));
    PetscCheck(Nc == cdim, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Geometric dimension %" PetscInt_FMT " should be %" PetscInt_FMT, Nc, cdim);
    PetscCall(DMGetLocalSection(cdm, &s));
    PetscCall(VecGetArrayRead(X, &x));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%s by element\n", cname));
    for (PetscInt c = 0; c < Ncell; ++c) {
      for (PetscInt v = 0; v < Ncl; ++v) {
        PetscInt off = offsets[c*Ncl+v], dgdof;
        const PetscScalar *vx = &x[off];

        if (isDG) {
          PetscCall(PetscSectionGetDof(s, c, &dgdof));
          PetscCheck(Ncl*Nc == dgdof, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Offset size %" PetscInt_FMT " should be %" PetscInt_FMT, Ncl*Nc, dgdof);
        }
        switch (cdim) {
        case 2:
          PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[%" PetscInt_FMT "] %" PetscInt_FMT " <-- %2" PetscInt_FMT " (% 4.2f, % 4.2f)\n", c, v, off, (double) PetscRealPart(vx[0]), (double) PetscRealPart(vx[1])));
          break;
        case 3:
          PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[%" PetscInt_FMT "] %" PetscInt_FMT " <-- %2" PetscInt_FMT " (% 4.2f, % 4.2f, % 4.2f)\n", c, v, off, (double) PetscRealPart(vx[0]), (double) PetscRealPart(vx[1]), (double) PetscRealPart(vx[2])));
          break;
        }
      }
    }
    PetscCall(VecRestoreArrayRead(X, &x));
    PetscCall(PetscFree(offsets));
  }
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM     dm;
  AppCtx user;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &user, &dm));
  PetscCall(SetupDiscretization(dm, &user));
  PetscCall(CheckOffsets(dm, NULL, 0, 0));
  PetscCall(CheckOffsets(dm, "Face Sets", 1, 1));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0
    requires: triangle
    args: -dm_refine 1 -petscspace_degree 1 -dm_view -offsets_view

  test:
    suffix: 1
    args: -dm_plex_simplex 0 -dm_plex_box_bd periodic,none -dm_plex_box_faces 3,3 -dm_sparse_localize 0 -petscspace_degree 1 \
          -dm_view -offsets_view

  test:
    suffix: cg_2d
    args: -dm_plex_simplex 0 -dm_plex_box_bd none,none -dm_plex_box_faces 3,3 -petscspace_degree 1 \
          -dm_view -offsets_view
TEST*/
