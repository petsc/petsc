static char help[] = "Test PETSc integer hash map.\n\n";

#include <petsc/private/hashmapi.h>
#include <petsc/private/hashmapiv.h>
#include <petscsys.h>

/* Unused, keep it for testing purposes */
PETSC_HASH_MAP(HMapIP, PetscInt, void*, PetscHashInt, PetscHashEqual, NULL)

/* Unused, keep it for testing purposes */
typedef struct { double x; double y; double z; } Point;
static Point origin = {0.0, 0.0, 0.0};
PETSC_HASH_MAP(HMapIS, PetscInt, Point, PetscHashInt, PetscHashEqual, origin)

#define PetscAssert(expr) do {            \
if (PetscUnlikely(!(expr)))               \
  SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB, \
           "Assertion: `%s' failed.",     \
           PetscStringize(expr));         \
} while (0)

int main(int argc,char **argv)
{
  PetscHMapI     ht = NULL, hd;
  PetscHMapIV    htv;
  PetscInt       n, v, koff, keys[4], voff, vals[4],na,nb,i,size,*karray,off;
  PetscScalar    *varray,*vwork;
  PetscBool      has, flag;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;

  ierr = PetscHMapICreate(&ht);CHKERRQ(ierr);
  PetscAssert(ht != NULL);
  ierr = PetscHMapIGetSize(ht,&n);CHKERRQ(ierr);
  PetscAssert(n == 0);

  ierr = PetscHMapIResize(ht,0);CHKERRQ(ierr);
  ierr = PetscHMapIGetSize(ht,&n);CHKERRQ(ierr);
  PetscAssert(n == 0);

  ierr = PetscHMapIHas(ht,123,&has);CHKERRQ(ierr);
  PetscAssert(has == PETSC_FALSE);
  ierr = PetscHMapIGet(ht,123,&v);CHKERRQ(ierr);
  PetscAssert(v == -1);

  ierr = PetscHMapISet(ht,123,42);CHKERRQ(ierr);
  ierr = PetscHMapIGetSize(ht,&n);CHKERRQ(ierr);
  PetscAssert(n == 1);
  ierr = PetscHMapIHas(ht,123,&has);CHKERRQ(ierr);
  PetscAssert(has == PETSC_TRUE);
  ierr = PetscHMapIGet(ht,123,&v);CHKERRQ(ierr);
  PetscAssert(v == 42);

  ierr = PetscHMapIDel(ht,123);CHKERRQ(ierr);
  ierr = PetscHMapIGetSize(ht,&n);CHKERRQ(ierr);
  PetscAssert(n == 0);
  ierr = PetscHMapIHas(ht,123,&has);CHKERRQ(ierr);
  PetscAssert(has == PETSC_FALSE);
  ierr = PetscHMapIGet(ht,123,&v);CHKERRQ(ierr);
  PetscAssert(v == -1);

  ierr = PetscHMapIQuerySet(ht,123,1,&flag);CHKERRQ(ierr);
  PetscAssert(flag == PETSC_TRUE);
  ierr = PetscHMapIQuerySet(ht,123,1,&flag);CHKERRQ(ierr);
  PetscAssert(flag == PETSC_FALSE);
  ierr = PetscHMapIQueryDel(ht,123,&flag);CHKERRQ(ierr);
  PetscAssert(flag == PETSC_TRUE);
  ierr = PetscHMapIQueryDel(ht,123,&flag);CHKERRQ(ierr);
  PetscAssert(flag == PETSC_FALSE);

  ierr = PetscHMapIResize(ht,13);CHKERRQ(ierr);
  ierr = PetscHMapIGetSize(ht,&n);CHKERRQ(ierr);
  PetscAssert(n == 0);

  ierr = PetscHMapIClear(ht);CHKERRQ(ierr);
  ierr = PetscHMapIGetSize(ht,&n);CHKERRQ(ierr);
  PetscAssert(n == 0);

  ierr = PetscHMapISet(ht,321,24);CHKERRQ(ierr);
  ierr = PetscHMapISet(ht,123,42);CHKERRQ(ierr);
  ierr = PetscHMapIGetSize(ht,&n);CHKERRQ(ierr);
  PetscAssert(n == 2);

  koff = 0; keys[0] = keys[1] = 0;CHKERRQ(ierr);
  ierr = PetscHMapIGetKeys(ht,&koff,keys);CHKERRQ(ierr);
  ierr = PetscSortInt(koff,keys);CHKERRQ(ierr);
  PetscAssert(koff == 2);
  PetscAssert(keys[0] == 123);
  PetscAssert(keys[1] == 321);

  voff = 0; vals[0] = vals[1] = 0;
  ierr = PetscHMapIGetVals(ht,&voff,vals);CHKERRQ(ierr);
  ierr = PetscSortInt(voff,vals);CHKERRQ(ierr);
  PetscAssert(voff == 2);
  PetscAssert(vals[0] == 24);
  PetscAssert(vals[1] == 42);

  koff = 0; keys[0] = keys[1] = 0;
  voff = 0; vals[0] = vals[1] = 0;
  ierr = PetscHMapIDuplicate(ht,&hd);CHKERRQ(ierr);
  ierr = PetscHMapIGetKeys(ht,&koff,keys);CHKERRQ(ierr);
  ierr = PetscHMapIGetVals(ht,&voff,vals);CHKERRQ(ierr);
  ierr = PetscSortInt(koff,keys);CHKERRQ(ierr);
  ierr = PetscSortInt(voff,vals);CHKERRQ(ierr);
  PetscAssert(koff == 2);
  PetscAssert(voff == 2);
  PetscAssert(keys[0] == 123);
  PetscAssert(keys[1] == 321);
  PetscAssert(vals[0] == 24);
  PetscAssert(vals[1] == 42);
  ierr = PetscHMapIDestroy(&hd);CHKERRQ(ierr);

  ierr = PetscHMapISet(ht,0,0);CHKERRQ(ierr);
  ierr = PetscHMapIGetSize(ht,&n);CHKERRQ(ierr);
  PetscAssert(n != 0);
  ierr = PetscHMapIReset(ht);CHKERRQ(ierr);
  ierr = PetscHMapIGetSize(ht,&n);CHKERRQ(ierr);
  PetscAssert(n == 0);
  ierr = PetscHMapIReset(ht);CHKERRQ(ierr);
  ierr = PetscHMapIGetSize(ht,&n);CHKERRQ(ierr);
  PetscAssert(n == 0);
  ierr = PetscHMapISet(ht,0,0);CHKERRQ(ierr);
  ierr = PetscHMapIGetSize(ht,&n);CHKERRQ(ierr);
  PetscAssert(n != 0);

  ierr = PetscHMapIDestroy(&ht);CHKERRQ(ierr);
  PetscAssert(ht == NULL);

  ierr = PetscHMapICreate(&ht);CHKERRQ(ierr);
  ierr = PetscHMapIReset(ht);CHKERRQ(ierr);
  ierr = PetscHMapIGetSize(ht,&n);CHKERRQ(ierr);
  PetscAssert(n == 0);
  ierr = PetscHMapIDestroy(&ht);CHKERRQ(ierr);

  ierr = PetscHMapIVCreate(&htv);CHKERRQ(ierr);
  n = 10;
  ierr = PetscHMapIVResize(htv,n);CHKERRQ(ierr);
  ierr = PetscHMapIVGetCapacity(htv,&na);CHKERRQ(ierr);
  PetscAssert(na>=n);
  for (i=0; i<n; i++) {
    ierr = PetscHMapIVSet(htv,i+100,10.);CHKERRQ(ierr);
  }
  ierr = PetscHMapIVGetCapacity(htv,&nb);CHKERRQ(ierr);
  PetscAssert(nb>=na);
  for (i=0; i<(2*n); i++) {
    ierr = PetscHMapIVAddValue(htv,i+100,5.);CHKERRQ(ierr);
  }
  ierr = PetscHMapIVGetSize(htv,&size);CHKERRQ(ierr);
  PetscAssert(size==(2*n));
  ierr = PetscMalloc3(size,&karray,size,&varray,size,&vwork);CHKERRQ(ierr);
  off = 0;
  ierr = PetscHMapIVGetPairs(htv,&off,karray,varray);CHKERRQ(ierr);
  PetscAssert(off==(2*n));
  ierr = PetscSortIntWithDataArray(off,karray,varray,sizeof(PetscScalar),vwork);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    PetscAssert(karray[i]==(i+100));
    PetscAssert(karray[n+i]==(n+i+100));
    PetscAssert(varray[i]==15.);
    PetscAssert(varray[n+i]==5.);
  }
  ierr = PetscFree3(karray,varray,vwork);CHKERRQ(ierr);
  ierr = PetscHMapIVDestroy(&htv);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

TEST*/
