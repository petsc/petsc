const char help[] = "Tests PetscDTBaryToIndex(), PetscDTIndexToBary(), PetscDTIndexToGradedOrder() and PetscDTGradedOrderToIndex()";

#include <petsc/private/petscimpl.h>
#include <petsc/private/dtimpl.h>
#include <petsc/private/petscfeimpl.h>

int main(int argc, char **argv)
{
  PetscInt       d, n, maxdim = 4;
  PetscInt       *btupprev, *btup;
  PetscInt       *gtup;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  ierr = PetscMalloc3(maxdim + 1, &btup, maxdim + 1, &btupprev, maxdim, &gtup);CHKERRQ(ierr);
  for (d = 0; d <= maxdim; d++) {
    for (n = 0; n <= d + 2; n++) {
      PetscInt j, k, Nk, kchk;

      ierr = PetscDTBinomialInt(d + n, d, &Nk);CHKERRQ(ierr);
      for (k = 0; k < Nk; k++) {
        PetscInt sum;

        ierr = PetscDTIndexToBary(d + 1, n, k, btup);CHKERRQ(ierr);
        for (j = 0, sum = 0; j < d + 1; j++) {
          if (btup[j] < 0) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "PetscDTIndexToBary, d = %D, n = %D, k = %D negative entry", d, n, k);
          sum += btup[j];
        }
        if (sum != n) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "PetscDTIndexToBary, d = %D, n = %D, k = %D incorrect sum", d, n, k);
        ierr = PetscDTBaryToIndex(d + 1, n, btup, &kchk);CHKERRQ(ierr);
        if (kchk != k) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "PetscDTBaryToIndex, d = %D, n = %D, k = %D mismatch", d, n, k);
        if (k) {
          j = d;
          while (j >= 0 && btup[j] == btupprev[j]) j--;
          if (j < 0) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "PetscDTIndexToBary, d = %D, n = %D, k = %D equal to previous", d, n, k);
          if (btup[j] < btupprev[j]) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "PetscDTIndexToBary, d = %D, n = %D, k = %D less to previous", d, n, k);
        } else {
          ierr = PetscArraycpy(btupprev, btup, d + 1);CHKERRQ(ierr);
        }
        ierr = PetscDTIndexToGradedOrder(d, Nk - 1 - k, gtup);CHKERRQ(ierr);
        ierr = PetscDTGradedOrderToIndex(d, gtup, &kchk);CHKERRQ(ierr);
        if (kchk != Nk - 1 - k) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "PetscDTGradedOrderToIndex, d = %D, n = %D, k = %D mismatch", d, n, Nk - 1 - k);
        for (j = 0; j < d; j++) {
          if (gtup[j] != btup[d - 1 - j]) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "PetscDTIndexToGradedOrder, d = %D, n = %D, k = %D incorrect", d, n, Nk - 1 - k);
        }
      }
    }
  }
  ierr = PetscFree3(btup, btupprev, gtup);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:

TEST*/
