static char help[] = "Test PETSc integer hash set.\n\n";

#include <petsc/private/hashseti.h>
#include <petscsys.h>

#define PetscAssert(expr) do {            \
if (PetscUnlikely(!(expr)))               \
  SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB, \
           "Assertion: `%s' failed.",     \
           PetscStringize(expr));         \
} while(0)


int main(int argc,char **argv)
{
  PetscHSetI     ht = NULL, hd;
  PetscInt       n, off, array[4];
  PetscBool      has, flag;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;

  ierr = PetscHSetICreate(&ht);CHKERRQ(ierr);
  PetscAssert(ht != NULL);
  ierr = PetscHSetIGetSize(ht,&n);CHKERRQ(ierr);
  PetscAssert(n == 0);

  ierr = PetscHSetIResize(ht,0);CHKERRQ(ierr);
  ierr = PetscHSetIGetSize(ht,&n);CHKERRQ(ierr);
  PetscAssert(n == 0);

  ierr = PetscHSetIHas(ht,42,&has);CHKERRQ(ierr);
  PetscAssert(has == PETSC_FALSE);

  ierr = PetscHSetIAdd(ht,42);CHKERRQ(ierr);
  ierr = PetscHSetIGetSize(ht,&n);CHKERRQ(ierr);
  PetscAssert(n == 1);
  ierr = PetscHSetIHas(ht,42,&has);CHKERRQ(ierr);
  PetscAssert(has == PETSC_TRUE);

  ierr = PetscHSetIDel(ht,42);CHKERRQ(ierr);
  ierr = PetscHSetIGetSize(ht,&n);CHKERRQ(ierr);
  PetscAssert(n == 0);
  ierr = PetscHSetIHas(ht,42,&has);CHKERRQ(ierr);
  PetscAssert(has == PETSC_FALSE);
  ierr = PetscHSetIDel(ht,42);CHKERRQ(ierr);
  ierr = PetscHSetIDel(ht,24);CHKERRQ(ierr);

  ierr = PetscHSetIQueryAdd(ht,123,&flag);CHKERRQ(ierr);
  PetscAssert(flag == PETSC_TRUE);
  ierr = PetscHSetIQueryAdd(ht,123,&flag);CHKERRQ(ierr);
  PetscAssert(flag == PETSC_FALSE);
  ierr = PetscHSetIQueryDel(ht,123,&flag);CHKERRQ(ierr);
  PetscAssert(flag == PETSC_TRUE);
  ierr = PetscHSetIQueryDel(ht,123,&flag);CHKERRQ(ierr);
  PetscAssert(flag == PETSC_FALSE);

  ierr = PetscHSetIResize(ht,13);CHKERRQ(ierr);
  ierr = PetscHSetIGetSize(ht,&n);CHKERRQ(ierr);
  PetscAssert(n == 0);

  ierr = PetscHSetIClear(ht);CHKERRQ(ierr);
  ierr = PetscHSetIGetSize(ht,&n);CHKERRQ(ierr);
  PetscAssert(n == 0);

  ierr = PetscHSetIAdd(ht,42);CHKERRQ(ierr);
  ierr = PetscHSetIAdd(ht,13);CHKERRQ(ierr);
  ierr = PetscHSetIGetSize(ht,&n);CHKERRQ(ierr);
  PetscAssert(n == 2);

  off = 0;
  ierr = PetscHSetIGetElems(ht,&off,array);CHKERRQ(ierr);
  ierr = PetscSortInt(off,array);CHKERRQ(ierr);
  PetscAssert(off == 2);
  PetscAssert(array[0] == 13);
  PetscAssert(array[1] == 42);
  ierr = PetscHSetIGetElems(ht,&off,array);CHKERRQ(ierr);
  ierr = PetscSortInt(2,array+2);CHKERRQ(ierr);
  PetscAssert(off == 4);
  PetscAssert(array[0] == 13);
  PetscAssert(array[1] == 42);
  PetscAssert(array[0] == 13);
  PetscAssert(array[1] == 42);

  off = 0;
  ierr = PetscHSetIDuplicate(ht,&hd);CHKERRQ(ierr);
  ierr = PetscHSetIGetElems(hd,&off,array);CHKERRQ(ierr);
  ierr = PetscSortInt(off,array);CHKERRQ(ierr);
  PetscAssert(off == 2);
  PetscAssert(array[0] == 13);
  PetscAssert(array[1] == 42);
  ierr = PetscHSetIDestroy(&hd);CHKERRQ(ierr);

  ierr = PetscHSetIAdd(ht,0);CHKERRQ(ierr);
  ierr = PetscHSetIGetSize(ht,&n);CHKERRQ(ierr);
  PetscAssert(n != 0);
  ierr = PetscHSetIReset(ht);CHKERRQ(ierr);
  ierr = PetscHSetIGetSize(ht,&n);CHKERRQ(ierr);
  PetscAssert(n == 0);
  ierr = PetscHSetIReset(ht);CHKERRQ(ierr);
  ierr = PetscHSetIGetSize(ht,&n);CHKERRQ(ierr);
  PetscAssert(n == 0);
  ierr = PetscHSetIAdd(ht,0);CHKERRQ(ierr);
  ierr = PetscHSetIGetSize(ht,&n);CHKERRQ(ierr);
  PetscAssert(n != 0);

  ierr = PetscHSetIDestroy(&ht);CHKERRQ(ierr);
  PetscAssert(ht == NULL);

  ierr = PetscHSetICreate(&ht);CHKERRQ(ierr);
  ierr = PetscHSetIReset(ht);CHKERRQ(ierr);
  ierr = PetscHSetIGetSize(ht,&n);CHKERRQ(ierr);
  PetscAssert(n == 0);
  ierr = PetscHSetIDestroy(&ht);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}


/*TEST

   test:

TEST*/
