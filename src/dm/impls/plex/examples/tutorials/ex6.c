static char help[] = "Spectral element access patterns with Plex\n\n";

#include <petscdmplex.h>

typedef struct {
  PetscInt  dim; /* Topological problem dimension */
  PetscInt  Nf;  /* Number of fields */
  PetscInt *Nc;  /* Number of components per field */
  PetscInt *k;   /* Spectral order per field */
} AppCtx;

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscInt       len;
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->dim = 2;
  options->Nf  = 0;
  options->Nc  = NULL;
  options->k   = NULL;

  ierr = PetscOptionsBegin(comm, "", "SEM Problem Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "Problem dimension", "ex6.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-num_fields", "The number of fields", "ex6.c", options->Nf, &options->Nf, NULL);CHKERRQ(ierr);
  if (options->Nf) {
    len  = options->Nf;
    ierr = PetscMalloc1(len, &options->Nc);CHKERRQ(ierr);
    ierr = PetscOptionsIntArray("-num_components", "The number of components per field", "ex6.c", options->Nc, &len, &flg);CHKERRQ(ierr);
    if (flg && (len != options->Nf)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Length of components array is %d should be %d", len, options->Nf);
    len  = options->Nf;
    ierr = PetscMalloc1(len, &options->k);CHKERRQ(ierr);
    ierr = PetscOptionsIntArray("-order", "The spectral order per field", "ex6.c", options->k, &len, &flg);CHKERRQ(ierr);
    if (flg && (len != options->Nf)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Length of order array is %d should be %d", len, options->Nf);
  }
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
};

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  DM             dm;
  PetscSection   s;
  Vec            u;
  AppCtx         user;
  PetscInt       cells[3] = {2, 2, 2};
  PetscInt       size = 0, dsize, cStart, cEnd, cell, c, f, i, j;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = DMPlexCreateHexBoxMesh(PETSC_COMM_WORLD, 2, cells, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, &dm);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  /* Create a section for SEM order k */
  {
    PetscInt *numDof, d;

    ierr = PetscMalloc1(user.Nf*(user.dim+1), &numDof);CHKERRQ(ierr);
    for (f = 0; f < user.Nf; ++f) {
      for (d = 0; d <= user.dim; ++d) numDof[f*(user.dim+1)+d] = PetscPowInt(user.k[f]-1, d)*user.Nc[f];
      size += PetscSqr(user.k[f]+1)*user.Nc[f];
    }
    dsize = PetscSqrtReal(size);
    ierr = DMPlexCreateSection(dm, user.dim, user.Nf, user.Nc, numDof, 0, NULL, NULL, NULL, NULL, &s);CHKERRQ(ierr);
    ierr = PetscFree(numDof);CHKERRQ(ierr);
  }
  ierr = DMSetDefaultSection(dm, s);CHKERRQ(ierr);
  /* Create spectral ordering and load in data */
  ierr = DMPlexCreateSpectralClosurePermutation(dm, NULL);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &u);CHKERRQ(ierr);
  for (j = 0; j < 2; ++j) {
    for (i = 0; i < 2; ++i) {
      PetscScalar closure[size];
      PetscInt    ki, kj, o = 0;

      for (f = 0; f < user.Nf; ++f) {
        PetscInt ioff = i*user.k[f], joff = j*user.k[f];

        for (kj = 0; kj <= user.k[f]; ++kj) {
          for (ki = 0; ki <= user.k[f]; ++ki) {
            for (c = 0; c < user.Nc[f]; ++c) {
              closure[o++] = ((kj + joff)*(2*user.k[f]+1) + ki + ioff)*user.Nc[f]+c;
            }
          }
        }
      }
      ierr = DMPlexVecSetClosure(dm, NULL, u, j*2+i, closure, INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  /* Remove ordering and check some values */
  {
    const PetscScalar *array, *a;
    PetscInt           dof, point, d;

    ierr = PetscSectionSetClosurePermutation(s, (PetscObject) dm, NULL);CHKERRQ(ierr);
    ierr = VecGetArrayRead(u, &array);CHKERRQ(ierr);
    point = 0;
    ierr = DMPlexPointLocalRead(dm, point, array, &a);CHKERRQ(ierr);
    ierr = PetscSectionGetDof(s, point, &dof);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF, "Point %D: ", point);CHKERRQ(ierr);
    for (d = 0; d < dof; ++d) {
      if (d > 0) {ierr = PetscPrintf(PETSC_COMM_SELF, ", ");CHKERRQ(ierr);}
      ierr = PetscPrintf(PETSC_COMM_SELF, "%2.0f", a[d]);CHKERRQ(ierr);
    }
    ierr = PetscPrintf(PETSC_COMM_SELF, "\n");CHKERRQ(ierr);
    point = 13;
    ierr = DMPlexPointLocalRead(dm, point, array, &a);CHKERRQ(ierr);
    ierr = PetscSectionGetDof(s, point, &dof);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF, "Point %D: ", point);CHKERRQ(ierr);
    for (d = 0; d < dof; ++d) {
      if (d > 0) {ierr = PetscPrintf(PETSC_COMM_SELF, ", ");CHKERRQ(ierr);}
      ierr = PetscPrintf(PETSC_COMM_SELF, "%2.0f", a[d]);CHKERRQ(ierr);
    }
    ierr = PetscPrintf(PETSC_COMM_SELF, "\n");CHKERRQ(ierr);
    point = 15;
    ierr = DMPlexPointLocalRead(dm, point, array, &a);CHKERRQ(ierr);
    ierr = PetscSectionGetDof(s, point, &dof);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF, "Point %D: ", point);CHKERRQ(ierr);
    for (d = 0; d < dof; ++d) {
      if (d > 0) {ierr = PetscPrintf(PETSC_COMM_SELF, ", ");CHKERRQ(ierr);}
      ierr = PetscPrintf(PETSC_COMM_SELF, "%2.0f", a[d]);CHKERRQ(ierr);
    }
    ierr = PetscPrintf(PETSC_COMM_SELF, "\n");CHKERRQ(ierr);
    point = 19;
    ierr = DMPlexPointLocalRead(dm, point, array, &a);CHKERRQ(ierr);
    ierr = PetscSectionGetDof(s, point, &dof);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF, "Point %D: ", point);CHKERRQ(ierr);
    for (d = 0; d < dof; ++d) {
      if (d > 0) {ierr = PetscPrintf(PETSC_COMM_SELF, ", ");CHKERRQ(ierr);}
      ierr = PetscPrintf(PETSC_COMM_SELF, "%2.0f", a[d]);CHKERRQ(ierr);
    }
    ierr = PetscPrintf(PETSC_COMM_SELF, "\n");CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(u, &array);CHKERRQ(ierr);
  }
  /* Recreate spectral ordering and read out data */
  ierr = DMPlexCreateSpectralClosurePermutation(dm, s);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  for (cell = cStart; cell < cEnd; ++cell) {
    PetscScalar *closure = NULL;
    PetscInt     closureSize, ki, kj, foff = 0, o = 0;

    ierr = DMPlexVecGetClosure(dm, NULL, u, cell, &closureSize, &closure);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF, "Cell %D\n", cell);CHKERRQ(ierr);
    for (f = 0; f < user.Nf; ++f) {
      ierr = PetscPrintf(PETSC_COMM_SELF, "  Field %D\n", f);CHKERRQ(ierr);
      for (kj = user.k[f]; kj >= 0; --kj) {
        for (ki = 0; ki <= user.k[f]; ++ki) {
          if (ki > 0) {ierr = PetscPrintf(PETSC_COMM_SELF, "  ");CHKERRQ(ierr);}
          for (c = 0; c < user.Nc[f]; ++c) {
            if (c > 0) ierr = PetscPrintf(PETSC_COMM_SELF, ",");CHKERRQ(ierr);
            ierr = PetscPrintf(PETSC_COMM_SELF, "%2.0f", closure[(kj*(user.k[f]+1) + ki)*user.Nc[f]+c + foff]);CHKERRQ(ierr);
          }
        }
        ierr = PetscPrintf(PETSC_COMM_SELF, "\n");CHKERRQ(ierr);
      }
      ierr = PetscPrintf(PETSC_COMM_SELF, "\n\n");CHKERRQ(ierr);
      foff += PetscSqr(user.k[f]+1);
    }
    ierr = DMPlexVecRestoreClosure(dm, NULL, u, cell, &closureSize, &closure);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF, "\n\n");CHKERRQ(ierr);
  }

  ierr = DMRestoreLocalVector(dm, &u);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&s);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFree(user.Nc);CHKERRQ(ierr);
  ierr = PetscFree(user.k);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
