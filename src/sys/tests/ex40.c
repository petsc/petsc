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

#define PetscTestCheck(expr) PetscCheck(expr,PETSC_COMM_SELF,PETSC_ERR_LIB, "Assertion: `%s' failed.", PetscStringize(expr))

int main(int argc,char **argv)
{
  PetscHMapI     ht = NULL, hd;
  PetscHMapIV    htv;
  PetscInt       n, v, koff, keys[4], voff, vals[4],na,nb,i,size,*karray,off;
  PetscScalar    *varray,*vwork;
  PetscBool      has, flag;

  CHKERRQ(PetscInitialize(&argc,&argv,NULL,help));

  CHKERRQ(PetscHMapICreate(&ht));
  PetscTestCheck(ht != NULL);
  CHKERRQ(PetscHMapIGetSize(ht,&n));
  PetscTestCheck(n == 0);

  CHKERRQ(PetscHMapIResize(ht,0));
  CHKERRQ(PetscHMapIGetSize(ht,&n));
  PetscTestCheck(n == 0);

  CHKERRQ(PetscHMapIHas(ht,123,&has));
  PetscTestCheck(has == PETSC_FALSE);
  CHKERRQ(PetscHMapIGet(ht,123,&v));
  PetscTestCheck(v == -1);

  CHKERRQ(PetscHMapISet(ht,123,42));
  CHKERRQ(PetscHMapIGetSize(ht,&n));
  PetscTestCheck(n == 1);
  CHKERRQ(PetscHMapIHas(ht,123,&has));
  PetscTestCheck(has == PETSC_TRUE);
  CHKERRQ(PetscHMapIGet(ht,123,&v));
  PetscTestCheck(v == 42);

  CHKERRQ(PetscHMapIDel(ht,123));
  CHKERRQ(PetscHMapIGetSize(ht,&n));
  PetscTestCheck(n == 0);
  CHKERRQ(PetscHMapIHas(ht,123,&has));
  PetscTestCheck(has == PETSC_FALSE);
  CHKERRQ(PetscHMapIGet(ht,123,&v));
  PetscTestCheck(v == -1);

  CHKERRQ(PetscHMapIQuerySet(ht,123,1,&flag));
  PetscTestCheck(flag == PETSC_TRUE);
  CHKERRQ(PetscHMapIQuerySet(ht,123,1,&flag));
  PetscTestCheck(flag == PETSC_FALSE);
  CHKERRQ(PetscHMapIQueryDel(ht,123,&flag));
  PetscTestCheck(flag == PETSC_TRUE);
  CHKERRQ(PetscHMapIQueryDel(ht,123,&flag));
  PetscTestCheck(flag == PETSC_FALSE);

  CHKERRQ(PetscHMapIResize(ht,13));
  CHKERRQ(PetscHMapIGetSize(ht,&n));
  PetscTestCheck(n == 0);

  CHKERRQ(PetscHMapIClear(ht));
  CHKERRQ(PetscHMapIGetSize(ht,&n));
  PetscTestCheck(n == 0);

  CHKERRQ(PetscHMapISet(ht,321,24));
  CHKERRQ(PetscHMapISet(ht,123,42));
  CHKERRQ(PetscHMapIGetSize(ht,&n));
  PetscTestCheck(n == 2);

  koff = 0; keys[0] = keys[1] = 0;
  CHKERRQ(PetscHMapIGetKeys(ht,&koff,keys));
  CHKERRQ(PetscSortInt(koff,keys));
  PetscTestCheck(koff == 2);
  PetscTestCheck(keys[0] == 123);
  PetscTestCheck(keys[1] == 321);

  voff = 0; vals[0] = vals[1] = 0;
  CHKERRQ(PetscHMapIGetVals(ht,&voff,vals));
  CHKERRQ(PetscSortInt(voff,vals));
  PetscTestCheck(voff == 2);
  PetscTestCheck(vals[0] == 24);
  PetscTestCheck(vals[1] == 42);

  koff = 0; keys[0] = keys[1] = 0;
  voff = 0; vals[0] = vals[1] = 0;
  CHKERRQ(PetscHMapIDuplicate(ht,&hd));
  CHKERRQ(PetscHMapIGetKeys(ht,&koff,keys));
  CHKERRQ(PetscHMapIGetVals(ht,&voff,vals));
  CHKERRQ(PetscSortInt(koff,keys));
  CHKERRQ(PetscSortInt(voff,vals));
  PetscTestCheck(koff == 2);
  PetscTestCheck(voff == 2);
  PetscTestCheck(keys[0] == 123);
  PetscTestCheck(keys[1] == 321);
  PetscTestCheck(vals[0] == 24);
  PetscTestCheck(vals[1] == 42);
  CHKERRQ(PetscHMapIDestroy(&hd));

  CHKERRQ(PetscHMapISet(ht,0,0));
  CHKERRQ(PetscHMapIGetSize(ht,&n));
  PetscTestCheck(n != 0);
  CHKERRQ(PetscHMapIReset(ht));
  CHKERRQ(PetscHMapIGetSize(ht,&n));
  PetscTestCheck(n == 0);
  CHKERRQ(PetscHMapIReset(ht));
  CHKERRQ(PetscHMapIGetSize(ht,&n));
  PetscTestCheck(n == 0);
  CHKERRQ(PetscHMapISet(ht,0,0));
  CHKERRQ(PetscHMapIGetSize(ht,&n));
  PetscTestCheck(n != 0);

  CHKERRQ(PetscHMapIDestroy(&ht));
  PetscTestCheck(ht == NULL);

  CHKERRQ(PetscHMapICreate(&ht));
  CHKERRQ(PetscHMapIReset(ht));
  CHKERRQ(PetscHMapIGetSize(ht,&n));
  PetscTestCheck(n == 0);
  CHKERRQ(PetscHMapIDestroy(&ht));

  CHKERRQ(PetscHMapIVCreate(&htv));
  n = 10;
  CHKERRQ(PetscHMapIVResize(htv,n));
  CHKERRQ(PetscHMapIVGetCapacity(htv,&na));
  PetscTestCheck(na>=n);
  for (i=0; i<n; i++) CHKERRQ(PetscHMapIVSet(htv,i+100,10.));

  CHKERRQ(PetscHMapIVGetCapacity(htv,&nb));
  PetscTestCheck(nb>=na);
  for (i=0; i<(2*n); i++) CHKERRQ(PetscHMapIVAddValue(htv,i+100,5.));

  CHKERRQ(PetscHMapIVGetSize(htv,&size));
  PetscTestCheck(size==(2*n));
  CHKERRQ(PetscMalloc3(size,&karray,size,&varray,size,&vwork));
  off = 0;
  CHKERRQ(PetscHMapIVGetPairs(htv,&off,karray,varray));
  PetscTestCheck(off==(2*n));
  CHKERRQ(PetscSortIntWithDataArray(off,karray,varray,sizeof(PetscScalar),vwork));
  for (i=0; i<n; i++) {
    PetscTestCheck(karray[i]==(i+100));
    PetscTestCheck(karray[n+i]==(n+i+100));
    PetscTestCheck(varray[i]==15.);
    PetscTestCheck(varray[n+i]==5.);
  }
  CHKERRQ(PetscFree3(karray,varray,vwork));
  CHKERRQ(PetscHMapIVDestroy(&htv));

  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
