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

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,NULL,help));

  PetscCall(PetscHMapICreate(&ht));
  PetscTestCheck(ht != NULL);
  PetscCall(PetscHMapIGetSize(ht,&n));
  PetscTestCheck(n == 0);

  PetscCall(PetscHMapIResize(ht,0));
  PetscCall(PetscHMapIGetSize(ht,&n));
  PetscTestCheck(n == 0);

  PetscCall(PetscHMapIHas(ht,123,&has));
  PetscTestCheck(has == PETSC_FALSE);
  PetscCall(PetscHMapIGet(ht,123,&v));
  PetscTestCheck(v == -1);

  PetscCall(PetscHMapISet(ht,123,42));
  PetscCall(PetscHMapIGetSize(ht,&n));
  PetscTestCheck(n == 1);
  PetscCall(PetscHMapIHas(ht,123,&has));
  PetscTestCheck(has == PETSC_TRUE);
  PetscCall(PetscHMapIGet(ht,123,&v));
  PetscTestCheck(v == 42);

  PetscCall(PetscHMapIDel(ht,123));
  PetscCall(PetscHMapIGetSize(ht,&n));
  PetscTestCheck(n == 0);
  PetscCall(PetscHMapIHas(ht,123,&has));
  PetscTestCheck(has == PETSC_FALSE);
  PetscCall(PetscHMapIGet(ht,123,&v));
  PetscTestCheck(v == -1);

  PetscCall(PetscHMapIQuerySet(ht,123,1,&flag));
  PetscTestCheck(flag == PETSC_TRUE);
  PetscCall(PetscHMapIQuerySet(ht,123,1,&flag));
  PetscTestCheck(flag == PETSC_FALSE);
  PetscCall(PetscHMapIQueryDel(ht,123,&flag));
  PetscTestCheck(flag == PETSC_TRUE);
  PetscCall(PetscHMapIQueryDel(ht,123,&flag));
  PetscTestCheck(flag == PETSC_FALSE);

  PetscCall(PetscHMapIResize(ht,13));
  PetscCall(PetscHMapIGetSize(ht,&n));
  PetscTestCheck(n == 0);

  PetscCall(PetscHMapIClear(ht));
  PetscCall(PetscHMapIGetSize(ht,&n));
  PetscTestCheck(n == 0);

  PetscCall(PetscHMapISet(ht,321,24));
  PetscCall(PetscHMapISet(ht,123,42));
  PetscCall(PetscHMapIGetSize(ht,&n));
  PetscTestCheck(n == 2);

  koff = 0; keys[0] = keys[1] = 0;
  PetscCall(PetscHMapIGetKeys(ht,&koff,keys));
  PetscCall(PetscSortInt(koff,keys));
  PetscTestCheck(koff == 2);
  PetscTestCheck(keys[0] == 123);
  PetscTestCheck(keys[1] == 321);

  voff = 0; vals[0] = vals[1] = 0;
  PetscCall(PetscHMapIGetVals(ht,&voff,vals));
  PetscCall(PetscSortInt(voff,vals));
  PetscTestCheck(voff == 2);
  PetscTestCheck(vals[0] == 24);
  PetscTestCheck(vals[1] == 42);

  koff = 0; keys[0] = keys[1] = 0;
  voff = 0; vals[0] = vals[1] = 0;
  PetscCall(PetscHMapIDuplicate(ht,&hd));
  PetscCall(PetscHMapIGetKeys(ht,&koff,keys));
  PetscCall(PetscHMapIGetVals(ht,&voff,vals));
  PetscCall(PetscSortInt(koff,keys));
  PetscCall(PetscSortInt(voff,vals));
  PetscTestCheck(koff == 2);
  PetscTestCheck(voff == 2);
  PetscTestCheck(keys[0] == 123);
  PetscTestCheck(keys[1] == 321);
  PetscTestCheck(vals[0] == 24);
  PetscTestCheck(vals[1] == 42);
  PetscCall(PetscHMapIDestroy(&hd));

  PetscCall(PetscHMapISet(ht,0,0));
  PetscCall(PetscHMapIGetSize(ht,&n));
  PetscTestCheck(n != 0);
  PetscCall(PetscHMapIReset(ht));
  PetscCall(PetscHMapIGetSize(ht,&n));
  PetscTestCheck(n == 0);
  PetscCall(PetscHMapIReset(ht));
  PetscCall(PetscHMapIGetSize(ht,&n));
  PetscTestCheck(n == 0);
  PetscCall(PetscHMapISet(ht,0,0));
  PetscCall(PetscHMapIGetSize(ht,&n));
  PetscTestCheck(n != 0);

  PetscCall(PetscHMapIDestroy(&ht));
  PetscTestCheck(ht == NULL);

  PetscCall(PetscHMapICreate(&ht));
  PetscCall(PetscHMapIReset(ht));
  PetscCall(PetscHMapIGetSize(ht,&n));
  PetscTestCheck(n == 0);
  PetscCall(PetscHMapIDestroy(&ht));

  PetscCall(PetscHMapIVCreate(&htv));
  n = 10;
  PetscCall(PetscHMapIVResize(htv,n));
  PetscCall(PetscHMapIVGetCapacity(htv,&na));
  PetscTestCheck(na>=n);
  for (i=0; i<n; i++) PetscCall(PetscHMapIVSet(htv,i+100,10.));

  PetscCall(PetscHMapIVGetCapacity(htv,&nb));
  PetscTestCheck(nb>=na);
  for (i=0; i<(2*n); i++) PetscCall(PetscHMapIVAddValue(htv,i+100,5.));

  PetscCall(PetscHMapIVGetSize(htv,&size));
  PetscTestCheck(size==(2*n));
  PetscCall(PetscMalloc3(size,&karray,size,&varray,size,&vwork));
  off = 0;
  PetscCall(PetscHMapIVGetPairs(htv,&off,karray,varray));
  PetscTestCheck(off==(2*n));
  PetscCall(PetscSortIntWithDataArray(off,karray,varray,sizeof(PetscScalar),vwork));
  for (i=0; i<n; i++) {
    PetscTestCheck(karray[i]==(i+100));
    PetscTestCheck(karray[n+i]==(n+i+100));
    PetscTestCheck(varray[i]==15.);
    PetscTestCheck(varray[n+i]==5.);
  }
  PetscCall(PetscFree3(karray,varray,vwork));
  PetscCall(PetscHMapIVDestroy(&htv));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
