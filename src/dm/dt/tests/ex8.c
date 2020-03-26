const char help[] = "Tests PetscDTBaryToIndex() and PetscDTIndexToBary()";

#include <petsc/private/petscimpl.h>
#include <petsc/private/dtimpl.h>
#include <petsc/private/petscfeimpl.h>

int main(int argc, char **argv)
{
  PetscInt       d, n, maxdim = 4;
  PetscInt       *btupprev, *btup;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  ierr = PetscMalloc2(maxdim + 1, &btup, maxdim + 1, &btupprev);CHKERRQ(ierr);
  for (d = 0; d <= maxdim; d++) {
    for (n = 0; n <= d + 2; n++) {
      PetscInt j, k, Nk, kchk;

      ierr = PetscDTBinomialInt(d + n, d, &Nk);CHKERRQ(ierr);
      for (k = 0; k < Nk; k++) {
        PetscInt sum;

        ierr = PetscDTIndexToBary(d + 1, n, k, btup);CHKERRQ(ierr);
        for (j = 0, sum = 0; j < d + 1; j++) {
          if (btup[j] < 0) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "PetscDTIndexToBary, d = %d, n = %d, k = %d negative entry\n", d, n, k);
          sum += btup[j];
        }
        if (sum != n) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "PetscDTIndexToBary, d = %d, n = %d, k = %d incorrect sum\n", d, n, k);
        ierr = PetscDTBaryToIndex(d + 1, n, btup, &kchk);CHKERRQ(ierr);
        if (kchk != k) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "PetscDTBaryToIndex, d = %d, n = %d, k = %d mismatch\n", d, n, k);
        if (k) {
          j = d;
          while (j >= 0 && btup[j] == btupprev[j]) j--;
          if (j < 0) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "PetscDTIndexToBary, d = %d, n = %d, k = %d equal to previous\n", d, n, k);
          if (btup[j] < btupprev[j]) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "PetscDTIndexToBary, d = %d, n = %d, k = %d less to previous\n", d, n, k);
        } else {
          ierr = PetscArraycpy(btupprev, btup, d + 1);CHKERRQ(ierr);
        }
      }
    }
  }
  ierr = PetscFree2(btup, btupprev);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:

TEST*/
