static char help[] = "Spectral element access patterns with Plex\n\n";

#include <petscdmplex.h>

typedef struct {
  PetscInt  Nf; /* Number of fields */
  PetscInt *Nc; /* Number of components per field */
  PetscInt *k;  /* Spectral order per field */
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscInt  len;
  PetscBool flg;

  PetscFunctionBeginUser;
  options->Nf = 0;
  options->Nc = NULL;
  options->k  = NULL;

  PetscOptionsBegin(comm, "", "SEM Problem Options", "DMPLEX");
  PetscCall(PetscOptionsBoundedInt("-num_fields", "The number of fields", "ex6.c", options->Nf, &options->Nf, NULL, 0));
  if (options->Nf) {
    len = options->Nf;
    PetscCall(PetscMalloc1(len, &options->Nc));
    PetscCall(PetscOptionsIntArray("-num_components", "The number of components per field", "ex6.c", options->Nc, &len, &flg));
    PetscCheck(!flg || !(len != options->Nf), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Length of components array is %" PetscInt_FMT " should be %" PetscInt_FMT, len, options->Nf);
    len = options->Nf;
    PetscCall(PetscMalloc1(len, &options->k));
    PetscCall(PetscOptionsIntArray("-order", "The spectral order per field", "ex6.c", options->k, &len, &flg));
    PetscCheck(!flg || !(len != options->Nf), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Length of order array is %" PetscInt_FMT " should be %" PetscInt_FMT, len, options->Nf);
  }
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LoadData2D(DM dm, PetscInt Ni, PetscInt Nj, PetscInt clSize, Vec u, AppCtx *user)
{
  PetscInt     i, j, f, c;
  PetscScalar *closure;

  PetscFunctionBeginUser;
  PetscCall(PetscMalloc1(clSize, &closure));
  for (j = 0; j < Nj; ++j) {
    for (i = 0; i < Ni; ++i) {
      PetscInt ki, kj, o = 0;
      PetscCall(PetscArrayzero(closure, clSize));

      for (f = 0; f < user->Nf; ++f) {
        PetscInt ioff = i * user->k[f], joff = j * user->k[f];

        for (kj = 0; kj <= user->k[f]; ++kj) {
          for (ki = 0; ki <= user->k[f]; ++ki) {
            for (c = 0; c < user->Nc[f]; ++c) closure[o++] = ((kj + joff) * (Ni * user->k[f] + 1) + ki + ioff) * user->Nc[f] + c;
          }
        }
      }
      PetscCall(DMPlexVecSetClosure(dm, NULL, u, j * Ni + i, closure, INSERT_VALUES));
    }
  }
  PetscCall(PetscFree(closure));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LoadData3D(DM dm, PetscInt Ni, PetscInt Nj, PetscInt Nk, PetscInt clSize, Vec u, AppCtx *user)
{
  PetscInt     i, j, k, f, c;
  PetscScalar *closure;

  PetscFunctionBeginUser;
  PetscCall(PetscMalloc1(clSize, &closure));
  for (k = 0; k < Nk; ++k) {
    for (j = 0; j < Nj; ++j) {
      for (i = 0; i < Ni; ++i) {
        PetscInt ki, kj, kk, o = 0;
        PetscCall(PetscArrayzero(closure, clSize));

        for (f = 0; f < user->Nf; ++f) {
          PetscInt ioff = i * user->k[f], joff = j * user->k[f], koff = k * user->k[f];

          for (kk = 0; kk <= user->k[f]; ++kk) {
            for (kj = 0; kj <= user->k[f]; ++kj) {
              for (ki = 0; ki <= user->k[f]; ++ki) {
                for (c = 0; c < user->Nc[f]; ++c) closure[o++] = (((kk + koff) * (Nj * user->k[f] + 1) + kj + joff) * (Ni * user->k[f] + 1) + ki + ioff) * user->Nc[f] + c;
              }
            }
          }
        }
        PetscCall(DMPlexVecSetClosure(dm, NULL, u, (k * Nj + j) * Ni + i, closure, INSERT_VALUES));
      }
    }
  }
  PetscCall(PetscFree(closure));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckPoint(DM dm, Vec u, PetscInt point, AppCtx *user)
{
  PetscSection       s;
  PetscScalar       *a;
  const PetscScalar *array;
  PetscInt           dof, d;

  PetscFunctionBeginUser;
  PetscCall(DMGetLocalSection(dm, &s));
  PetscCall(VecGetArrayRead(u, &array));
  PetscCall(DMPlexPointLocalRead(dm, point, array, &a));
  PetscCall(PetscSectionGetDof(s, point, &dof));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "Point %" PetscInt_FMT ": ", point));
  for (d = 0; d < dof; ++d) {
    if (d > 0) PetscCall(PetscPrintf(PETSC_COMM_SELF, ", "));
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "%2.0f", (double)PetscRealPart(a[d])));
  }
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "\n"));
  PetscCall(VecRestoreArrayRead(u, &array));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ReadData2D(DM dm, Vec u, AppCtx *user)
{
  PetscInt cStart, cEnd, cell;

  PetscFunctionBeginUser;
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  for (cell = cStart; cell < cEnd; ++cell) {
    PetscScalar *closure = NULL;
    PetscInt     closureSize, ki, kj, f, c, foff = 0;

    PetscCall(DMPlexVecGetClosure(dm, NULL, u, cell, &closureSize, &closure));
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "Cell %" PetscInt_FMT "\n", cell));
    for (f = 0; f < user->Nf; ++f) {
      PetscCall(PetscPrintf(PETSC_COMM_SELF, "  Field %" PetscInt_FMT "\n", f));
      for (kj = user->k[f]; kj >= 0; --kj) {
        for (ki = 0; ki <= user->k[f]; ++ki) {
          if (ki > 0) PetscCall(PetscPrintf(PETSC_COMM_SELF, "  "));
          for (c = 0; c < user->Nc[f]; ++c) {
            if (c > 0) PetscCall(PetscPrintf(PETSC_COMM_SELF, ","));
            PetscCall(PetscPrintf(PETSC_COMM_SELF, "%2.0f", (double)PetscRealPart(closure[(kj * (user->k[f] + 1) + ki) * user->Nc[f] + c + foff])));
          }
        }
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "\n"));
      }
      PetscCall(PetscPrintf(PETSC_COMM_SELF, "\n\n"));
      foff += PetscSqr(user->k[f] + 1);
    }
    PetscCall(DMPlexVecRestoreClosure(dm, NULL, u, cell, &closureSize, &closure));
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "\n\n"));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ReadData3D(DM dm, Vec u, AppCtx *user)
{
  PetscInt cStart, cEnd, cell;

  PetscFunctionBeginUser;
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  for (cell = cStart; cell < cEnd; ++cell) {
    PetscScalar *closure = NULL;
    PetscInt     closureSize, ki, kj, kk, f, c, foff = 0;

    PetscCall(DMPlexVecGetClosure(dm, NULL, u, cell, &closureSize, &closure));
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "Cell %" PetscInt_FMT "\n", cell));
    for (f = 0; f < user->Nf; ++f) {
      PetscCall(PetscPrintf(PETSC_COMM_SELF, "  Field %" PetscInt_FMT "\n", f));
      for (kk = user->k[f]; kk >= 0; --kk) {
        for (kj = user->k[f]; kj >= 0; --kj) {
          for (ki = 0; ki <= user->k[f]; ++ki) {
            if (ki > 0) PetscCall(PetscPrintf(PETSC_COMM_SELF, "  "));
            for (c = 0; c < user->Nc[f]; ++c) {
              if (c > 0) PetscCall(PetscPrintf(PETSC_COMM_SELF, ","));
              PetscCall(PetscPrintf(PETSC_COMM_SELF, "%2.0f", (double)PetscRealPart(closure[((kk * (user->k[f] + 1) + kj) * (user->k[f] + 1) + ki) * user->Nc[f] + c + foff])));
            }
          }
          PetscCall(PetscPrintf(PETSC_COMM_SELF, "\n"));
        }
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "\n"));
      }
      PetscCall(PetscPrintf(PETSC_COMM_SELF, "\n\n"));
      foff += PetscSqr(user->k[f] + 1);
    }
    PetscCall(DMPlexVecRestoreClosure(dm, NULL, u, cell, &closureSize, &closure));
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "\n\n"));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetSymmetries(DM dm, PetscSection s, AppCtx *user)
{
  PetscInt dim, f, o, i, j, k, c, d;
  DMLabel  depthLabel;

  PetscFunctionBegin;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetLabel(dm, "depth", &depthLabel));
  for (f = 0; f < user->Nf; f++) {
    PetscSectionSym sym;

    if (user->k[f] < 3) continue; /* No symmetries needed for order < 3, because no cell, facet, edge or vertex has more than one node */
    PetscCall(PetscSectionSymCreateLabel(PetscObjectComm((PetscObject)s), depthLabel, &sym));

    for (d = 0; d <= dim; d++) {
      if (d == 1) {
        PetscInt   numDof  = user->k[f] - 1;
        PetscInt   numComp = user->Nc[f];
        PetscInt   minOrnt = -1;
        PetscInt   maxOrnt = 1;
        PetscInt **perms;

        PetscCall(PetscCalloc1(maxOrnt - minOrnt, &perms));
        for (o = minOrnt; o < maxOrnt; o++) {
          PetscInt *perm;

          if (!o) { /* identity */
            perms[o - minOrnt] = NULL;
          } else {
            PetscCall(PetscMalloc1(numDof * numComp, &perm));
            for (i = numDof - 1, k = 0; i >= 0; i--) {
              for (j = 0; j < numComp; j++, k++) perm[k] = i * numComp + j;
            }
            perms[o - minOrnt] = perm;
          }
        }
        PetscCall(PetscSectionSymLabelSetStratum(sym, d, numDof * numComp, minOrnt, maxOrnt, PETSC_OWN_POINTER, (const PetscInt **)perms, NULL));
      } else if (d == 2) {
        PetscInt   perEdge = user->k[f] - 1;
        PetscInt   numDof  = perEdge * perEdge;
        PetscInt   numComp = user->Nc[f];
        PetscInt   minOrnt = -4;
        PetscInt   maxOrnt = 4;
        PetscInt **perms;

        PetscCall(PetscCalloc1(maxOrnt - minOrnt, &perms));
        for (o = minOrnt; o < maxOrnt; o++) {
          PetscInt *perm;

          if (!o) continue; /* identity */
          PetscCall(PetscMalloc1(numDof * numComp, &perm));
          /* We want to perm[k] to list which *localArray* position the *sectionArray* position k should go to for the given orientation*/
          switch (o) {
          case 0:
            break; /* identity */
          case -2: /* flip along (-1,-1)--( 1, 1), which swaps edges 0 and 3 and edges 1 and 2.  This swaps the i and j variables */
            for (i = 0, k = 0; i < perEdge; i++) {
              for (j = 0; j < perEdge; j++, k++) {
                for (c = 0; c < numComp; c++) perm[k * numComp + c] = (perEdge * j + i) * numComp + c;
              }
            }
            break;
          case -1: /* flip along (-1, 0)--( 1, 0), which swaps edges 0 and 2.  This reverses the i variable */
            for (i = 0, k = 0; i < perEdge; i++) {
              for (j = 0; j < perEdge; j++, k++) {
                for (c = 0; c < numComp; c++) perm[k * numComp + c] = (perEdge * (perEdge - 1 - i) + j) * numComp + c;
              }
            }
            break;
          case -4: /* flip along ( 1,-1)--(-1, 1), which swaps edges 0 and 1 and edges 2 and 3.  This swaps the i and j variables and reverse both */
            for (i = 0, k = 0; i < perEdge; i++) {
              for (j = 0; j < perEdge; j++, k++) {
                for (c = 0; c < numComp; c++) perm[k * numComp + c] = (perEdge * (perEdge - 1 - j) + (perEdge - 1 - i)) * numComp + c;
              }
            }
            break;
          case -3: /* flip along ( 0,-1)--( 0, 1), which swaps edges 3 and 1.  This reverses the j variable */
            for (i = 0, k = 0; i < perEdge; i++) {
              for (j = 0; j < perEdge; j++, k++) {
                for (c = 0; c < numComp; c++) perm[k * numComp + c] = (perEdge * i + (perEdge - 1 - j)) * numComp + c;
              }
            }
            break;
          case 1: /* rotate section edge 1 to local edge 0.  This swaps the i and j variables and then reverses the j variable */
            for (i = 0, k = 0; i < perEdge; i++) {
              for (j = 0; j < perEdge; j++, k++) {
                for (c = 0; c < numComp; c++) perm[k * numComp + c] = (perEdge * (perEdge - 1 - j) + i) * numComp + c;
              }
            }
            break;
          case 2: /* rotate section edge 2 to local edge 0.  This reverse both i and j variables */
            for (i = 0, k = 0; i < perEdge; i++) {
              for (j = 0; j < perEdge; j++, k++) {
                for (c = 0; c < numComp; c++) perm[k * numComp + c] = (perEdge * (perEdge - 1 - i) + (perEdge - 1 - j)) * numComp + c;
              }
            }
            break;
          case 3: /* rotate section edge 3 to local edge 0.  This swaps the i and j variables and then reverses the i variable */
            for (i = 0, k = 0; i < perEdge; i++) {
              for (j = 0; j < perEdge; j++, k++) {
                for (c = 0; c < numComp; c++) perm[k * numComp + c] = (perEdge * j + (perEdge - 1 - i)) * numComp + c;
              }
            }
            break;
          default:
            break;
          }
          perms[o - minOrnt] = perm;
        }
        PetscCall(PetscSectionSymLabelSetStratum(sym, d, numDof * numComp, minOrnt, maxOrnt, PETSC_OWN_POINTER, (const PetscInt **)perms, NULL));
      }
    }
    PetscCall(PetscSectionSetFieldSym(s, f, sym));
    PetscCall(PetscSectionSymDestroy(&sym));
  }
  PetscCall(PetscSectionViewFromOptions(s, NULL, "-section_with_sym_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM           dm;
  PetscSection s;
  Vec          u;
  AppCtx       user;
  PetscInt     dim, size = 0, f;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  PetscCall(DMCreate(PETSC_COMM_WORLD, &dm));
  PetscCall(DMSetType(dm, DMPLEX));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));
  PetscCall(DMGetDimension(dm, &dim));
  /* Create a section for SEM order k */
  {
    PetscInt *numDof, d;

    PetscCall(PetscMalloc1(user.Nf * (dim + 1), &numDof));
    for (f = 0; f < user.Nf; ++f) {
      for (d = 0; d <= dim; ++d) numDof[f * (dim + 1) + d] = PetscPowInt(user.k[f] - 1, d) * user.Nc[f];
      size += PetscPowInt(user.k[f] + 1, d) * user.Nc[f];
    }
    PetscCall(DMSetNumFields(dm, user.Nf));
    PetscCall(DMPlexCreateSection(dm, NULL, user.Nc, numDof, 0, NULL, NULL, NULL, NULL, &s));
    PetscCall(SetSymmetries(dm, s, &user));
    PetscCall(PetscFree(numDof));
  }
  PetscCall(DMSetLocalSection(dm, s));
  /* Create spectral ordering and load in data */
  PetscCall(DMPlexSetClosurePermutationTensor(dm, PETSC_DETERMINE, NULL));
  PetscCall(DMGetLocalVector(dm, &u));
  switch (dim) {
  case 2:
    PetscCall(LoadData2D(dm, 2, 2, size, u, &user));
    break;
  case 3:
    PetscCall(LoadData3D(dm, 2, 2, 2, size, u, &user));
    break;
  }
  /* Remove ordering and check some values */
  PetscCall(PetscSectionSetClosurePermutation(s, (PetscObject)dm, dim, NULL));
  switch (dim) {
  case 2:
    PetscCall(CheckPoint(dm, u, 0, &user));
    PetscCall(CheckPoint(dm, u, 13, &user));
    PetscCall(CheckPoint(dm, u, 15, &user));
    PetscCall(CheckPoint(dm, u, 19, &user));
    break;
  case 3:
    PetscCall(CheckPoint(dm, u, 0, &user));
    PetscCall(CheckPoint(dm, u, 13, &user));
    PetscCall(CheckPoint(dm, u, 15, &user));
    PetscCall(CheckPoint(dm, u, 19, &user));
    break;
  }
  /* Recreate spectral ordering and read out data */
  PetscCall(DMPlexSetClosurePermutationTensor(dm, PETSC_DETERMINE, s));
  switch (dim) {
  case 2:
    PetscCall(ReadData2D(dm, u, &user));
    break;
  case 3:
    PetscCall(ReadData3D(dm, u, &user));
    break;
  }
  PetscCall(DMRestoreLocalVector(dm, &u));
  PetscCall(PetscSectionDestroy(&s));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFree(user.Nc));
  PetscCall(PetscFree(user.k));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  # Spectral ordering 2D 0-5
  testset:
    args: -dm_plex_simplex 0 -dm_plex_box_faces 2,2

    test:
      suffix: 0
      args: -num_fields 1 -num_components 1 -order 2
    test:
      suffix: 1
      args: -num_fields 1 -num_components 1 -order 3
    test:
      suffix: 2
      args: -num_fields 1 -num_components 1 -order 5
    test:
      suffix: 3
      args: -num_fields 1 -num_components 2 -order 2
    test:
      suffix: 4
      args: -num_fields 2 -num_components 1,1 -order 2,2
    test:
      suffix: 5
      args: -num_fields 2 -num_components 1,2 -order 2,3

  # Spectral ordering 3D 6-11
  testset:
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_plex_box_faces 2,2,2

    test:
      suffix: 6
      args: -num_fields 1 -num_components 1 -order 2
    test:
      suffix: 7
      args: -num_fields 1 -num_components 1 -order 3
    test:
      suffix: 8
      args: -num_fields 1 -num_components 1 -order 5
    test:
      suffix: 9
      args: -num_fields 1 -num_components 2 -order 2
    test:
      suffix: 10
      args: -num_fields 2 -num_components 1,1 -order 2,2
    test:
      suffix: 11
      args: -num_fields 2 -num_components 1,2 -order 2,3

TEST*/
