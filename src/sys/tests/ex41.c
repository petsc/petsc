static char help[] = "Test PETSc integer hash set.\n\n";

#include <petsc/private/hashseti.h>
#include <petscsys.h>

#define PetscTestCheck(expr) PetscCheck(expr, PETSC_COMM_SELF, PETSC_ERR_LIB, "Assertion: `%s' failed.", PetscStringize(expr))

int main(int argc, char **argv)
{
  PetscHSetI ht = NULL, hd;
  PetscInt   n, off, array[4], na, nb, i, *marray, size;
  PetscBool  has, flag;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  PetscCall(PetscHSetICreate(&ht));
  PetscTestCheck(ht != NULL);
  PetscCall(PetscHSetIGetSize(ht, &n));
  PetscTestCheck(n == 0);

  PetscCall(PetscHSetIResize(ht, 0));
  PetscCall(PetscHSetIGetSize(ht, &n));
  PetscTestCheck(n == 0);

  PetscCall(PetscHSetIHas(ht, 42, &has));
  PetscTestCheck(has == PETSC_FALSE);

  PetscCall(PetscHSetIAdd(ht, 42));
  PetscCall(PetscHSetIGetSize(ht, &n));
  PetscTestCheck(n == 1);
  PetscCall(PetscHSetIHas(ht, 42, &has));
  PetscTestCheck(has == PETSC_TRUE);

  PetscCall(PetscHSetIDel(ht, 42));
  PetscCall(PetscHSetIGetSize(ht, &n));
  PetscTestCheck(n == 0);
  PetscCall(PetscHSetIHas(ht, 42, &has));
  PetscTestCheck(has == PETSC_FALSE);
  PetscCall(PetscHSetIDel(ht, 42));
  PetscCall(PetscHSetIDel(ht, 24));

  PetscCall(PetscHSetIQueryAdd(ht, 123, &flag));
  PetscTestCheck(flag == PETSC_TRUE);
  PetscCall(PetscHSetIQueryAdd(ht, 123, &flag));
  PetscTestCheck(flag == PETSC_FALSE);
  PetscCall(PetscHSetIQueryDel(ht, 123, &flag));
  PetscTestCheck(flag == PETSC_TRUE);
  PetscCall(PetscHSetIQueryDel(ht, 123, &flag));
  PetscTestCheck(flag == PETSC_FALSE);

  PetscCall(PetscHSetIResize(ht, 13));
  PetscCall(PetscHSetIGetSize(ht, &n));
  PetscTestCheck(n == 0);

  PetscCall(PetscHSetIClear(ht));
  PetscCall(PetscHSetIGetSize(ht, &n));
  PetscTestCheck(n == 0);

  PetscCall(PetscHSetIAdd(ht, 42));
  PetscCall(PetscHSetIAdd(ht, 13));
  PetscCall(PetscHSetIGetSize(ht, &n));
  PetscTestCheck(n == 2);

  off = 0;
  PetscCall(PetscHSetIGetElems(ht, &off, array));
  PetscCall(PetscSortInt(off, array));
  PetscTestCheck(off == 2);
  PetscTestCheck(array[0] == 13);
  PetscTestCheck(array[1] == 42);
  PetscCall(PetscHSetIGetElems(ht, &off, array));
  PetscCall(PetscSortInt(2, array + 2));
  PetscTestCheck(off == 4);
  PetscTestCheck(array[0] == 13);
  PetscTestCheck(array[1] == 42);
  PetscTestCheck(array[0] == 13);
  PetscTestCheck(array[1] == 42);

  off = 0;
  PetscCall(PetscHSetIDuplicate(ht, &hd));
  PetscCall(PetscHSetIGetElems(hd, &off, array));
  PetscCall(PetscSortInt(off, array));
  PetscTestCheck(off == 2);
  PetscTestCheck(array[0] == 13);
  PetscTestCheck(array[1] == 42);
  PetscCall(PetscHSetIDestroy(&hd));

  PetscCall(PetscHSetIAdd(ht, 0));
  PetscCall(PetscHSetIGetSize(ht, &n));
  PetscTestCheck(n != 0);
  PetscCall(PetscHSetIReset(ht));
  PetscCall(PetscHSetIGetSize(ht, &n));
  PetscTestCheck(n == 0);
  PetscCall(PetscHSetIReset(ht));
  PetscCall(PetscHSetIGetSize(ht, &n));
  PetscTestCheck(n == 0);
  PetscCall(PetscHSetIAdd(ht, 0));
  PetscCall(PetscHSetIGetSize(ht, &n));
  PetscTestCheck(n != 0);

  PetscCall(PetscHSetIDestroy(&ht));
  PetscTestCheck(ht == NULL);

  PetscCall(PetscHSetICreate(&ht));
  PetscCall(PetscHSetIReset(ht));
  PetscCall(PetscHSetIGetSize(ht, &n));
  PetscTestCheck(n == 0);
  PetscCall(PetscHSetIDestroy(&ht));

  PetscCall(PetscHSetICreate(&ht));
  PetscCall(PetscHSetICreate(&hd));
  n = 10;
  PetscCall(PetscHSetIResize(ht, n));
  PetscCall(PetscHSetIResize(hd, n));
  PetscCall(PetscHSetIGetCapacity(ht, &na));
  PetscCall(PetscHSetIGetCapacity(hd, &nb));
  PetscTestCheck(na >= n);
  PetscTestCheck(nb >= n);
  for (i = 0; i < n; i++) {
    PetscCall(PetscHSetIAdd(ht, i + 1));
    PetscCall(PetscHSetIAdd(hd, i + 1 + n));
  }
  PetscCall(PetscHSetIGetCapacity(ht, &nb));
  PetscTestCheck(nb >= na);
  /* Merge ht and hd, and the result is in ht */
  PetscCall(PetscHSetIUpdate(ht, hd));
  PetscCall(PetscHSetIDestroy(&hd));
  PetscCall(PetscHSetIGetSize(ht, &size));
  PetscTestCheck(size == (2 * n));
  PetscCall(PetscMalloc1(n * 2, &marray));
  off = 0;
  PetscCall(PetscHSetIGetElems(ht, &off, marray));
  PetscCall(PetscHSetIDestroy(&ht));
  PetscTestCheck(off == (2 * n));
  PetscCall(PetscSortInt(off, marray));
  for (i = 0; i < n; i++) {
    PetscTestCheck(marray[i] == (i + 1));
    PetscTestCheck(marray[n + i] == (i + 1 + n));
  }
  PetscCall(PetscFree(marray));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
