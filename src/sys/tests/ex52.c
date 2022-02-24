static char help[] = "A benchmark for testing PetscSortInt(), PetscSortIntSemiOrdered(), PetscSortIntWithArrayPair(), PetscIntSortSemiOrderedWithArray(), and PetscSortIntWithArray()\n\
  The array is filled with random numbers, but one can control average duplicates for each unique integer with the -d option.\n\
  Usage:\n\
   mpirun -n 1 ./ex52 -n <length of the array to sort>, default=100 \n\
                      -r <repeat times for each sort>, default=10 \n\
                      -d <average duplicates for each unique integer>, default=1, i.e., no duplicates \n\n";

#include <petscsys.h>
#include <petsctime.h>
#include <petscviewer.h>
#include <petscvec.h>
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       i,l,n=100,r=10,d=1,vsize=1;
  PetscInt       *X,*X1,*XR,*XSO,*W,*Y,*Z,*XP,*X1P;
  PetscReal      val,norm1,nreal;
  PetscRandom    rdm,rdm2;
  PetscLogDouble time, time1, time2;
  PetscMPIInt    size;
  PetscViewer    vwr;
  Vec            x;
  unsigned long  seedr, seedo;
  PetscBool      order=PETSC_FALSE;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheckFalse(size != 1,PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"This is a uniprocessor example only!");

  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-r",&r,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-d",&d,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-vsize",&vsize,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-order",NULL,&order));
  CHKERRQ(PetscOptionsGetViewer(PETSC_COMM_WORLD,NULL,NULL,"-array_view",&vwr,NULL,NULL));
  PetscCheckFalse(n<1 || r<1 || d<1 || d>n,PETSC_COMM_WORLD,PETSC_ERR_SUP,"Wrong input n=%" PetscInt_FMT ",r=%" PetscInt_FMT ",d=%" PetscInt_FMT ". They must be >=1 and n>=d",n,r,d);

  CHKERRQ(PetscCalloc6(n,&X,n,&X1,n,&XR,n,&XSO,n,&Y,n,&Z));
  CHKERRQ(PetscRandomCreate(PETSC_COMM_SELF,&rdm));
  CHKERRQ(PetscRandomSetFromOptions(rdm));
  CHKERRQ(PetscRandomGetSeed(rdm, &seedr));

  for (i=0; i<n; ++i) {
    CHKERRQ(PetscRandomGetValueReal(rdm,&val));
    XR[i] = val*PETSC_MAX_INT;
    if (d > 1) XR[i] = XR[i] % (n/d);
    XSO[i] = i;
    if (d > 1) XSO[i] = XSO[i] % (n/d);
  }

  nreal = (PetscReal) n;
  CHKERRQ(PetscRandomCreate(PETSC_COMM_SELF,&rdm2));
  CHKERRQ(PetscRandomGetSeed(rdm, &seedo));
  CHKERRQ(PetscRandomSetInterval(rdm2,0,nreal));
  for (i = 0; i < n/10; ++i) {
    PetscInt swapi, t;
    CHKERRQ(PetscRandomGetValueReal(rdm2,&val));
    swapi = (PetscInt) val;
    t = XSO[swapi-1];
    XSO[swapi-1] = XSO[swapi];
    XSO[swapi] = t;
  }
  CHKERRQ(PetscRandomDestroy(&rdm2));

  if (vwr) CHKERRQ(PetscIntView(n, order ? XSO : XR, vwr));
  CHKERRQ(PetscViewerDestroy(&vwr));
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&x));
  CHKERRQ(VecSetSizes(x,PETSC_DECIDE,vsize));
  CHKERRQ(VecSetFromOptions(x));
  CHKERRQ(VecSetRandom(x,rdm));
  time = 0.0;
  time1 = 0.0;
  for (l=0; l<r; l++) { /* r loops */
    CHKERRQ(PetscArraycpy(X,order ? XSO : XR,n));
    CHKERRQ(PetscArraycpy(X1,order ? XSO : XR,n));

    CHKERRQ(VecNorm(x,NORM_1,&norm1));
    CHKERRQ(PetscTimeSubtract(&time1));
    CHKERRQ(PetscIntSortSemiOrdered(n,X1));
    CHKERRQ(PetscTimeAdd(&time1));

    CHKERRQ(VecNorm(x,NORM_1,&norm1));
    CHKERRQ(PetscTimeSubtract(&time));
    CHKERRQ(PetscSortInt(n,X));
    CHKERRQ(PetscTimeAdd(&time));

    for (i=0; i<n-1; i++) {PetscCheckFalse(X[i] > X[i+1],PETSC_COMM_SELF,PETSC_ERR_PLIB,"PetscSortInt() produced wrong results!");}
    for (i=0; i<n; i++) {PetscCheckFalse(X[i] != X1[i],PETSC_COMM_SELF,PETSC_ERR_PLIB,"PetscIntSortSemiOrdered() rep %" PetscInt_FMT " X1[%" PetscInt_FMT "]:%" PetscInt_FMT " does not match PetscSortInt() X[%" PetscInt_FMT "]:%" PetscInt_FMT "! randomSeed %lu, orderedSeed %lu",l,i,X1[i],i,X[i],seedr,seedo);}
    for (i=0; i<n-1; i++) {PetscCheckFalse(X1[i] > X1[i+1],PETSC_COMM_SELF,PETSC_ERR_PLIB,"PetscIntSortSemiOrdered() produced wrong results! randomSeed %lu orderedSeed %lu",seedr,seedo);}
    CHKERRQ(PetscArrayzero(X,n));
    CHKERRQ(PetscArrayzero(X1,n));
  }
  CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"PetscSortInt()              with %" PetscInt_FMT " integers, %" PetscInt_FMT " duplicate(s) per unique value took %g seconds\n",n,d,time/r));
  CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"PetscIntSortSemiOrdered()   with %" PetscInt_FMT " integers, %" PetscInt_FMT " duplicate(s) per unique value took %g seconds\n",n,d,time1/r));
  CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Speedup of PetscIntSortSemiOrdered() was %g (0:1 = slower, >1 means faster)\n",time/time1));

  for (i=0; i<n; i++) { /* Init X[] */
    CHKERRQ(PetscRandomGetValueReal(rdm,&val));
    X[i] = val*PETSC_MAX_INT;
    if (d > 1) X[i] = X[i] % (n/d);
  }
  CHKERRQ(PetscCalloc3(n,&XP,n,&X1P,n,&W));

  time = 0.0;
  time1 = 0.0;
  time2 = 0.0;
  for (l=0; l<r; l++) { /* r loops */
    CHKERRQ(PetscArraycpy(X1, order ? XSO : XR,n));
    CHKERRQ(PetscArraycpy(X1P,order ? XSO : XR,n));
    CHKERRQ(PetscArraycpy(X,  order ? XSO : XR,n));
    CHKERRQ(PetscArraycpy(XP, order ? XSO : XR,n));
    CHKERRQ(PetscArraycpy(W, order ? XSO : XR,n));

    CHKERRQ(VecNorm(x,NORM_1,&norm1));
    CHKERRQ(PetscTimeSubtract(&time1));
    CHKERRQ(PetscIntSortSemiOrderedWithArray(n,X1,X1P));
    CHKERRQ(PetscTimeAdd(&time1));

    CHKERRQ(VecNorm(x,NORM_1,&norm1));
    CHKERRQ(PetscTimeSubtract(&time2));
    CHKERRQ(PetscSortIntWithArray(n,X,XP));
    CHKERRQ(PetscTimeAdd(&time2));

    CHKERRQ(PetscTimeSubtract(&time));
    CHKERRQ(PetscSortIntWithArrayPair(n,W,Y,Z));
    CHKERRQ(PetscTimeAdd(&time));

    for (i=0; i<n-1; i++) {if (Y[i] > Y[i+1]) {PetscIntView(n,Y,0);SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"PetscSortIntWithArray() produced wrong results!");}}
    for (i=0; i<n-1; i++) {PetscCheckFalse(W[i] > W[i+1],PETSC_COMM_SELF,PETSC_ERR_PLIB,"PetscSortIntWithArrayPair() produced wrong results!");}
    for (i=0; i<n; i++) {PetscCheckFalse(X1P[i] != X[i],PETSC_COMM_SELF,PETSC_ERR_PLIB,"PetscIntSortSemiOrdered() rep %" PetscInt_FMT " X1[%" PetscInt_FMT "]:%" PetscInt_FMT " does not match PetscSortIntWithArray() X[%" PetscInt_FMT "]:%" PetscInt_FMT "! randomSeed %lu, orderedSeed %lu",l,i,X1[i],i,X[i],seedr,seedo);}
    for (i=0; i<n-1; i++) {PetscCheckFalse(X1[i] > X1[i+1],PETSC_COMM_SELF,PETSC_ERR_PLIB,"PetscIntSortSemiOrdered() produced wrong results! randomSeed %lu orderedSeed %lu",seedr,seedo);}
    CHKERRQ(PetscArrayzero(X1,n));
    CHKERRQ(PetscArrayzero(X1P,n));
    CHKERRQ(PetscArrayzero(X,n));
    CHKERRQ(PetscArrayzero(XP,n));
    CHKERRQ(PetscArrayzero(W,n));
  }
  CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"PetscSortIntWithArrayPair()        with %" PetscInt_FMT " integers, %" PetscInt_FMT " duplicate(s) per unique value took %g seconds\n",n,d,time/r));
  CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"PetscSortIntWithArray()            with %" PetscInt_FMT " integers, %" PetscInt_FMT " duplicate(s) per unique value took %g seconds\n",n,d,time2/r));
  CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"PetscIntSortSemiOrderedWithArray() with %" PetscInt_FMT " integers, %" PetscInt_FMT " duplicate(s) per unique value took %g seconds\n",n,d,time1/r));
  CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Speedup of PetscIntSortSemiOrderedWithArray() was %g (0:1 = slower, >1 means faster)\n",time2/time1));
  CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"SUCCEEDED\n"));

  CHKERRQ(VecDestroy(&x));
  CHKERRQ(PetscRandomDestroy(&rdm));
  CHKERRQ(PetscFree3(XP,X1P,W));
  CHKERRQ(PetscFree6(X,X1,XR,XSO,Y,Z));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   testset:
     filter: grep -vE "per unique value took|Speedup of "

     test:
       suffix: small
       args: -n 9 -r 1

     test:
       suffix: large
       args: -n 1000 -r 10 -d 1
       # Do not need to output timing results for test

TEST*/
