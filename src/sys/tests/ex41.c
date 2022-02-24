static char help[] = "Test PETSc integer hash set.\n\n";

#include <petsc/private/hashseti.h>
#include <petscsys.h>

#define PetscTestCheck(expr) PetscCheck(expr,PETSC_COMM_SELF,PETSC_ERR_LIB,"Assertion: `%s' failed.",PetscStringize(expr))

int main(int argc,char **argv)
{
  PetscHSetI     ht = NULL, hd;
  PetscInt       n, off, array[4],na,nb,i,*marray,size;
  PetscBool      has, flag;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;

  CHKERRQ(PetscHSetICreate(&ht));
  PetscTestCheck(ht != NULL);
  CHKERRQ(PetscHSetIGetSize(ht,&n));
  PetscTestCheck(n == 0);

  CHKERRQ(PetscHSetIResize(ht,0));
  CHKERRQ(PetscHSetIGetSize(ht,&n));
  PetscTestCheck(n == 0);

  CHKERRQ(PetscHSetIHas(ht,42,&has));
  PetscTestCheck(has == PETSC_FALSE);

  CHKERRQ(PetscHSetIAdd(ht,42));
  CHKERRQ(PetscHSetIGetSize(ht,&n));
  PetscTestCheck(n == 1);
  CHKERRQ(PetscHSetIHas(ht,42,&has));
  PetscTestCheck(has == PETSC_TRUE);

  CHKERRQ(PetscHSetIDel(ht,42));
  CHKERRQ(PetscHSetIGetSize(ht,&n));
  PetscTestCheck(n == 0);
  CHKERRQ(PetscHSetIHas(ht,42,&has));
  PetscTestCheck(has == PETSC_FALSE);
  CHKERRQ(PetscHSetIDel(ht,42));
  CHKERRQ(PetscHSetIDel(ht,24));

  CHKERRQ(PetscHSetIQueryAdd(ht,123,&flag));
  PetscTestCheck(flag == PETSC_TRUE);
  CHKERRQ(PetscHSetIQueryAdd(ht,123,&flag));
  PetscTestCheck(flag == PETSC_FALSE);
  CHKERRQ(PetscHSetIQueryDel(ht,123,&flag));
  PetscTestCheck(flag == PETSC_TRUE);
  CHKERRQ(PetscHSetIQueryDel(ht,123,&flag));
  PetscTestCheck(flag == PETSC_FALSE);

  CHKERRQ(PetscHSetIResize(ht,13));
  CHKERRQ(PetscHSetIGetSize(ht,&n));
  PetscTestCheck(n == 0);

  CHKERRQ(PetscHSetIClear(ht));
  CHKERRQ(PetscHSetIGetSize(ht,&n));
  PetscTestCheck(n == 0);

  CHKERRQ(PetscHSetIAdd(ht,42));
  CHKERRQ(PetscHSetIAdd(ht,13));
  CHKERRQ(PetscHSetIGetSize(ht,&n));
  PetscTestCheck(n == 2);

  off = 0;
  CHKERRQ(PetscHSetIGetElems(ht,&off,array));
  CHKERRQ(PetscSortInt(off,array));
  PetscTestCheck(off == 2);
  PetscTestCheck(array[0] == 13);
  PetscTestCheck(array[1] == 42);
  CHKERRQ(PetscHSetIGetElems(ht,&off,array));
  CHKERRQ(PetscSortInt(2,array+2));
  PetscTestCheck(off == 4);
  PetscTestCheck(array[0] == 13);
  PetscTestCheck(array[1] == 42);
  PetscTestCheck(array[0] == 13);
  PetscTestCheck(array[1] == 42);

  off = 0;
  CHKERRQ(PetscHSetIDuplicate(ht,&hd));
  CHKERRQ(PetscHSetIGetElems(hd,&off,array));
  CHKERRQ(PetscSortInt(off,array));
  PetscTestCheck(off == 2);
  PetscTestCheck(array[0] == 13);
  PetscTestCheck(array[1] == 42);
  CHKERRQ(PetscHSetIDestroy(&hd));

  CHKERRQ(PetscHSetIAdd(ht,0));
  CHKERRQ(PetscHSetIGetSize(ht,&n));
  PetscTestCheck(n != 0);
  CHKERRQ(PetscHSetIReset(ht));
  CHKERRQ(PetscHSetIGetSize(ht,&n));
  PetscTestCheck(n == 0);
  CHKERRQ(PetscHSetIReset(ht));
  CHKERRQ(PetscHSetIGetSize(ht,&n));
  PetscTestCheck(n == 0);
  CHKERRQ(PetscHSetIAdd(ht,0));
  CHKERRQ(PetscHSetIGetSize(ht,&n));
  PetscTestCheck(n != 0);

  CHKERRQ(PetscHSetIDestroy(&ht));
  PetscTestCheck(ht == NULL);

  CHKERRQ(PetscHSetICreate(&ht));
  CHKERRQ(PetscHSetIReset(ht));
  CHKERRQ(PetscHSetIGetSize(ht,&n));
  PetscTestCheck(n == 0);
  CHKERRQ(PetscHSetIDestroy(&ht));

  CHKERRQ(PetscHSetICreate(&ht));
  CHKERRQ(PetscHSetICreate(&hd));
  n = 10;
  CHKERRQ(PetscHSetIResize(ht,n));
  CHKERRQ(PetscHSetIResize(hd,n));
  CHKERRQ(PetscHSetIGetCapacity(ht,&na));
  CHKERRQ(PetscHSetIGetCapacity(hd,&nb));
  PetscTestCheck(na>=n);
  PetscTestCheck(nb>=n);
  for (i=0; i<n; i++) {
    CHKERRQ(PetscHSetIAdd(ht,i+1));
    CHKERRQ(PetscHSetIAdd(hd,i+1+n));
  }
  CHKERRQ(PetscHSetIGetCapacity(ht,&nb));
  PetscTestCheck(nb>=na);
  /* Merge ht and hd, and the result is in ht */
  CHKERRQ(PetscHSetIUpdate(ht,hd));
  CHKERRQ(PetscHSetIDestroy(&hd));
  CHKERRQ(PetscHSetIGetSize(ht,&size));
  PetscTestCheck(size==(2*n));
  CHKERRQ(PetscMalloc1(n*2,&marray));
  off = 0;
  CHKERRQ(PetscHSetIGetElems(ht,&off,marray));
  CHKERRQ(PetscHSetIDestroy(&ht));
  PetscTestCheck(off==(2*n));
  CHKERRQ(PetscSortInt(off,marray));
  for (i=0; i<n; i++) {
    PetscTestCheck(marray[i]==(i+1));
    PetscTestCheck(marray[n+i]==(i+1+n));
  }
  CHKERRQ(PetscFree(marray));

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

TEST*/
