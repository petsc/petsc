static char help[] = "Spectral element access patterns with Plex\n\n";

#include <petscdmplex.h>

typedef struct {
  PetscInt  Nf;  /* Number of fields */
  PetscInt *Nc;  /* Number of components per field */
  PetscInt *k;   /* Spectral order per field */
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscInt       len;
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->Nf = 0;
  options->Nc = NULL;
  options->k  = NULL;

  ierr = PetscOptionsBegin(comm, "", "SEM Problem Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsBoundedInt("-num_fields", "The number of fields", "ex6.c", options->Nf, &options->Nf, NULL,0);CHKERRQ(ierr);
  if (options->Nf) {
    len  = options->Nf;
    ierr = PetscMalloc1(len, &options->Nc);CHKERRQ(ierr);
    ierr = PetscOptionsIntArray("-num_components", "The number of components per field", "ex6.c", options->Nc, &len, &flg);CHKERRQ(ierr);
    if (flg && (len != options->Nf)) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Length of components array is %d should be %d", len, options->Nf);
    len  = options->Nf;
    ierr = PetscMalloc1(len, &options->k);CHKERRQ(ierr);
    ierr = PetscOptionsIntArray("-order", "The spectral order per field", "ex6.c", options->k, &len, &flg);CHKERRQ(ierr);
    if (flg && (len != options->Nf)) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Length of order array is %d should be %d", len, options->Nf);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode LoadData2D(DM dm, PetscInt Ni, PetscInt Nj, PetscInt clSize, Vec u, AppCtx *user)
{
  PetscInt       i, j, f, c;
  PetscErrorCode ierr;
  PetscScalar *closure;

  PetscFunctionBeginUser;
  ierr = PetscMalloc1(clSize,&closure);CHKERRQ(ierr);
  for (j = 0; j < Nj; ++j) {
    for (i = 0; i < Ni; ++i) {
      PetscInt    ki, kj, o = 0;
      ierr = PetscArrayzero(closure,clSize);CHKERRQ(ierr);

      for (f = 0; f < user->Nf; ++f) {
        PetscInt ioff = i*user->k[f], joff = j*user->k[f];

        for (kj = 0; kj <= user->k[f]; ++kj) {
          for (ki = 0; ki <= user->k[f]; ++ki) {
            for (c = 0; c < user->Nc[f]; ++c) {
              closure[o++] = ((kj + joff)*(Ni*user->k[f]+1) + ki + ioff)*user->Nc[f]+c;
            }
          }
        }
      }
      ierr = DMPlexVecSetClosure(dm, NULL, u, j*Ni+i, closure, INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(closure);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode LoadData3D(DM dm, PetscInt Ni, PetscInt Nj, PetscInt Nk, PetscInt clSize, Vec u, AppCtx *user)
{
  PetscInt       i, j, k, f, c;
  PetscErrorCode ierr;
  PetscScalar *closure;

  PetscFunctionBeginUser;
  ierr = PetscMalloc1(clSize,&closure);CHKERRQ(ierr);
  for (k = 0; k < Nk; ++k) {
    for (j = 0; j < Nj; ++j) {
      for (i = 0; i < Ni; ++i) {
        PetscInt    ki, kj, kk, o = 0;
        ierr = PetscArrayzero(closure,clSize);CHKERRQ(ierr);

        for (f = 0; f < user->Nf; ++f) {
          PetscInt ioff = i*user->k[f], joff = j*user->k[f], koff = k*user->k[f];

          for (kk = 0; kk <= user->k[f]; ++kk) {
            for (kj = 0; kj <= user->k[f]; ++kj) {
              for (ki = 0; ki <= user->k[f]; ++ki) {
                for (c = 0; c < user->Nc[f]; ++c) {
                  closure[o++] = (((kk + koff)*(Nj*user->k[f]+1) + kj + joff)*(Ni*user->k[f]+1) + ki + ioff)*user->Nc[f]+c;
                }
              }
            }
          }
        }
        ierr = DMPlexVecSetClosure(dm, NULL, u, (k*Nj+j)*Ni+i, closure, INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }
  ierr = PetscFree(closure);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CheckPoint(DM dm, Vec u, PetscInt point, AppCtx *user)
{
  PetscSection       s;
  PetscScalar        *a;
  const PetscScalar  *array;
  PetscInt           dof, d;
  PetscErrorCode     ierr;

  PetscFunctionBeginUser;
  ierr = DMGetLocalSection(dm, &s);CHKERRQ(ierr);
  ierr = VecGetArrayRead(u, &array);CHKERRQ(ierr);
  ierr = DMPlexPointLocalRead(dm, point, array, &a);CHKERRQ(ierr);
  ierr = PetscSectionGetDof(s, point, &dof);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF, "Point %D: ", point);CHKERRQ(ierr);
  for (d = 0; d < dof; ++d) {
    if (d > 0) {ierr = PetscPrintf(PETSC_COMM_SELF, ", ");CHKERRQ(ierr);}
    ierr = PetscPrintf(PETSC_COMM_SELF, "%2.0f", (double) PetscRealPart(a[d]));CHKERRQ(ierr);
  }
  ierr = PetscPrintf(PETSC_COMM_SELF, "\n");CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(u, &array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode ReadData2D(DM dm, Vec u, AppCtx *user)
{
  PetscInt       cStart, cEnd, cell;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  for (cell = cStart; cell < cEnd; ++cell) {
    PetscScalar *closure = NULL;
    PetscInt     closureSize, ki, kj, f, c, foff = 0;

    ierr = DMPlexVecGetClosure(dm, NULL, u, cell, &closureSize, &closure);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF, "Cell %D\n", cell);CHKERRQ(ierr);
    for (f = 0; f < user->Nf; ++f) {
      ierr = PetscPrintf(PETSC_COMM_SELF, "  Field %D\n", f);CHKERRQ(ierr);
      for (kj = user->k[f]; kj >= 0; --kj) {
        for (ki = 0; ki <= user->k[f]; ++ki) {
          if (ki > 0) {ierr = PetscPrintf(PETSC_COMM_SELF, "  ");CHKERRQ(ierr);}
          for (c = 0; c < user->Nc[f]; ++c) {
            if (c > 0) {ierr = PetscPrintf(PETSC_COMM_SELF, ",");CHKERRQ(ierr);}
            ierr = PetscPrintf(PETSC_COMM_SELF, "%2.0f", (double) PetscRealPart(closure[(kj*(user->k[f]+1) + ki)*user->Nc[f]+c + foff]));CHKERRQ(ierr);
          }
        }
        ierr = PetscPrintf(PETSC_COMM_SELF, "\n");CHKERRQ(ierr);
      }
      ierr = PetscPrintf(PETSC_COMM_SELF, "\n\n");CHKERRQ(ierr);
      foff += PetscSqr(user->k[f]+1);
    }
    ierr = DMPlexVecRestoreClosure(dm, NULL, u, cell, &closureSize, &closure);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF, "\n\n");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ReadData3D(DM dm, Vec u, AppCtx *user)
{
  PetscInt       cStart, cEnd, cell;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  for (cell = cStart; cell < cEnd; ++cell) {
    PetscScalar *closure = NULL;
    PetscInt     closureSize, ki, kj, kk, f, c, foff = 0;

    ierr = DMPlexVecGetClosure(dm, NULL, u, cell, &closureSize, &closure);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF, "Cell %D\n", cell);CHKERRQ(ierr);
    for (f = 0; f < user->Nf; ++f) {
      ierr = PetscPrintf(PETSC_COMM_SELF, "  Field %D\n", f);CHKERRQ(ierr);
      for (kk = user->k[f]; kk >= 0; --kk) {
        for (kj = user->k[f]; kj >= 0; --kj) {
          for (ki = 0; ki <= user->k[f]; ++ki) {
            if (ki > 0) {ierr = PetscPrintf(PETSC_COMM_SELF, "  ");CHKERRQ(ierr);}
            for (c = 0; c < user->Nc[f]; ++c) {
              if (c > 0) {ierr = PetscPrintf(PETSC_COMM_SELF, ",");CHKERRQ(ierr);}
              ierr = PetscPrintf(PETSC_COMM_SELF, "%2.0f", (double) PetscRealPart(closure[((kk*(user->k[f]+1) + kj)*(user->k[f]+1) + ki)*user->Nc[f]+c + foff]));CHKERRQ(ierr);
            }
          }
          ierr = PetscPrintf(PETSC_COMM_SELF, "\n");CHKERRQ(ierr);
        }
        ierr = PetscPrintf(PETSC_COMM_SELF, "\n");CHKERRQ(ierr);
      }
      ierr = PetscPrintf(PETSC_COMM_SELF, "\n\n");CHKERRQ(ierr);
      foff += PetscSqr(user->k[f]+1);
    }
    ierr = DMPlexVecRestoreClosure(dm, NULL, u, cell, &closureSize, &closure);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF, "\n\n");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SetSymmetries(DM dm, PetscSection s, AppCtx *user)
{
  PetscInt       dim, f, o, i, j, k, c, d;
  DMLabel        depthLabel;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetLabel(dm,"depth",&depthLabel);CHKERRQ(ierr);
  for (f = 0; f < user->Nf; f++) {
    PetscSectionSym sym;

    if (user->k[f] < 3) continue; /* No symmetries needed for order < 3, because no cell, facet, edge or vertex has more than one node */
    ierr = PetscSectionSymCreateLabel(PetscObjectComm((PetscObject)s),depthLabel,&sym);CHKERRQ(ierr);

    for (d = 0; d <= dim; d++) {
      if (d == 1) {
        PetscInt        numDof  = user->k[f] - 1;
        PetscInt        numComp = user->Nc[f];
        PetscInt        minOrnt = -1;
        PetscInt        maxOrnt = 1;
        PetscInt        **perms;

        ierr = PetscCalloc1(maxOrnt - minOrnt,&perms);CHKERRQ(ierr);
        for (o = minOrnt; o < maxOrnt; o++) {
          PetscInt *perm;

          if (!o) { /* identity */
            perms[o - minOrnt] = NULL;
          } else {
            ierr = PetscMalloc1(numDof * numComp, &perm);CHKERRQ(ierr);
            for (i = numDof - 1, k = 0; i >= 0; i--) {
              for (j = 0; j < numComp; j++, k++) perm[k] = i * numComp + j;
            }
            perms[o - minOrnt] = perm;
          }
        }
        ierr = PetscSectionSymLabelSetStratum(sym,d,numDof*numComp,minOrnt,maxOrnt,PETSC_OWN_POINTER,(const PetscInt **) perms,NULL);CHKERRQ(ierr);
      } else if (d == 2) {
        PetscInt        perEdge = user->k[f] - 1;
        PetscInt        numDof  = perEdge * perEdge;
        PetscInt        numComp = user->Nc[f];
        PetscInt        minOrnt = -4;
        PetscInt        maxOrnt = 4;
        PetscInt        **perms;

        ierr = PetscCalloc1(maxOrnt-minOrnt,&perms);CHKERRQ(ierr);
        for (o = minOrnt; o < maxOrnt; o++) {
          PetscInt *perm;

          if (!o) continue; /* identity */
          ierr = PetscMalloc1(numDof * numComp, &perm);CHKERRQ(ierr);
          /* We want to perm[k] to list which *localArray* position the *sectionArray* position k should go to for the given orientation*/
          switch (o) {
          case 0:
            break; /* identity */
          case -2: /* flip along (-1,-1)--( 1, 1), which swaps edges 0 and 3 and edges 1 and 2.  This swaps the i and j variables */
            for (i = 0, k = 0; i < perEdge; i++) {
              for (j = 0; j < perEdge; j++, k++) {
                for (c = 0; c < numComp; c++) {
                  perm[k * numComp + c] = (perEdge * j + i) * numComp + c;
                }
              }
            }
            break;
          case -1: /* flip along (-1, 0)--( 1, 0), which swaps edges 0 and 2.  This reverses the i variable */
            for (i = 0, k = 0; i < perEdge; i++) {
              for (j = 0; j < perEdge; j++, k++) {
                for (c = 0; c < numComp; c++) {
                  perm[k * numComp + c] = (perEdge * (perEdge - 1 - i) + j) * numComp + c;
                }
              }
            }
            break;
          case -4: /* flip along ( 1,-1)--(-1, 1), which swaps edges 0 and 1 and edges 2 and 3.  This swaps the i and j variables and reverse both */
            for (i = 0, k = 0; i < perEdge; i++) {
              for (j = 0; j < perEdge; j++, k++) {
                for (c = 0; c < numComp; c++) {
                  perm[k * numComp + c] = (perEdge * (perEdge - 1 - j) + (perEdge - 1 - i)) * numComp + c;
                }
              }
            }
            break;
          case -3: /* flip along ( 0,-1)--( 0, 1), which swaps edges 3 and 1.  This reverses the j variable */
            for (i = 0, k = 0; i < perEdge; i++) {
              for (j = 0; j < perEdge; j++, k++) {
                for (c = 0; c < numComp; c++) {
                  perm[k * numComp + c] = (perEdge * i + (perEdge - 1 - j)) * numComp + c;
                }
              }
            }
            break;
          case  1: /* rotate section edge 1 to local edge 0.  This swaps the i and j variables and then reverses the j variable */
            for (i = 0, k = 0; i < perEdge; i++) {
              for (j = 0; j < perEdge; j++, k++) {
                for (c = 0; c < numComp; c++) {
                  perm[k * numComp + c] = (perEdge * (perEdge - 1 - j) + i) * numComp + c;
                }
              }
            }
            break;
          case  2: /* rotate section edge 2 to local edge 0.  This reverse both i and j variables */
            for (i = 0, k = 0; i < perEdge; i++) {
              for (j = 0; j < perEdge; j++, k++) {
                for (c = 0; c < numComp; c++) {
                  perm[k * numComp + c] = (perEdge * (perEdge - 1 - i) + (perEdge - 1 - j)) * numComp + c;
                }
              }
            }
            break;
          case  3: /* rotate section edge 3 to local edge 0.  This swaps the i and j variables and then reverses the i variable */
            for (i = 0, k = 0; i < perEdge; i++) {
              for (j = 0; j < perEdge; j++, k++) {
                for (c = 0; c < numComp; c++) {
                  perm[k * numComp + c] = (perEdge * j + (perEdge - 1 - i)) * numComp + c;
                }
              }
            }
            break;
          default:
            break;
          }
          perms[o - minOrnt] = perm;
        }
        ierr = PetscSectionSymLabelSetStratum(sym,d,numDof*numComp,minOrnt,maxOrnt,PETSC_OWN_POINTER,(const PetscInt **) perms,NULL);CHKERRQ(ierr);
      }
    }
    ierr = PetscSectionSetFieldSym(s,f,sym);CHKERRQ(ierr);
    ierr = PetscSectionSymDestroy(&sym);CHKERRQ(ierr);
  }
  ierr = PetscSectionViewFromOptions(s,NULL,"-section_with_sym_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;
  PetscSection   s;
  Vec            u;
  AppCtx         user;
  PetscInt       dim, size = 0, f;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = DMCreate(PETSC_COMM_WORLD, &dm);CHKERRQ(ierr);
  ierr = DMSetType(dm, DMPLEX);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  /* Create a section for SEM order k */
  {
    PetscInt *numDof, d;

    ierr = PetscMalloc1(user.Nf*(dim+1), &numDof);CHKERRQ(ierr);
    for (f = 0; f < user.Nf; ++f) {
      for (d = 0; d <= dim; ++d) numDof[f*(dim+1)+d] = PetscPowInt(user.k[f]-1, d)*user.Nc[f];
      size += PetscPowInt(user.k[f]+1, d)*user.Nc[f];
    }
    ierr = DMSetNumFields(dm, user.Nf);CHKERRQ(ierr);
    ierr = DMPlexCreateSection(dm, NULL, user.Nc, numDof, 0, NULL, NULL, NULL, NULL, &s);CHKERRQ(ierr);
    ierr = SetSymmetries(dm, s, &user);CHKERRQ(ierr);
    ierr = PetscFree(numDof);CHKERRQ(ierr);
  }
  ierr = DMSetLocalSection(dm, s);CHKERRQ(ierr);
  /* Create spectral ordering and load in data */
  ierr = DMPlexSetClosurePermutationTensor(dm, PETSC_DETERMINE, NULL);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &u);CHKERRQ(ierr);
  switch (dim) {
  case 2: ierr = LoadData2D(dm, 2, 2, size, u, &user);CHKERRQ(ierr);break;
  case 3: ierr = LoadData3D(dm, 2, 2, 2, size, u, &user);CHKERRQ(ierr);break;
  }
  /* Remove ordering and check some values */
  ierr = PetscSectionSetClosurePermutation(s, (PetscObject) dm, dim, NULL);CHKERRQ(ierr);
  switch (dim) {
  case 2:
    ierr = CheckPoint(dm, u,  0, &user);CHKERRQ(ierr);
    ierr = CheckPoint(dm, u, 13, &user);CHKERRQ(ierr);
    ierr = CheckPoint(dm, u, 15, &user);CHKERRQ(ierr);
    ierr = CheckPoint(dm, u, 19, &user);CHKERRQ(ierr);
    break;
  case 3:
    ierr = CheckPoint(dm, u,  0, &user);CHKERRQ(ierr);
    ierr = CheckPoint(dm, u, 13, &user);CHKERRQ(ierr);
    ierr = CheckPoint(dm, u, 15, &user);CHKERRQ(ierr);
    ierr = CheckPoint(dm, u, 19, &user);CHKERRQ(ierr);
    break;
  }
  /* Recreate spectral ordering and read out data */
  ierr = DMPlexSetClosurePermutationTensor(dm, PETSC_DETERMINE, s);CHKERRQ(ierr);
  switch (dim) {
  case 2: ierr = ReadData2D(dm, u, &user);CHKERRQ(ierr);break;
  case 3: ierr = ReadData3D(dm, u, &user);CHKERRQ(ierr);break;
  }
  ierr = DMRestoreLocalVector(dm, &u);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&s);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFree(user.Nc);CHKERRQ(ierr);
  ierr = PetscFree(user.k);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
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
