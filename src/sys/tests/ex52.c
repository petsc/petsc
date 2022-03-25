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

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheckFalse(size != 1,PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"This is a uniprocessor example only!");

  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-r",&r,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-d",&d,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-vsize",&vsize,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-order",NULL,&order));
  PetscCall(PetscOptionsGetViewer(PETSC_COMM_WORLD,NULL,NULL,"-array_view",&vwr,NULL,NULL));
  PetscCheckFalse(n<1 || r<1 || d<1 || d>n,PETSC_COMM_WORLD,PETSC_ERR_SUP,"Wrong input n=%" PetscInt_FMT ",r=%" PetscInt_FMT ",d=%" PetscInt_FMT ". They must be >=1 and n>=d",n,r,d);

  PetscCall(PetscCalloc6(n,&X,n,&X1,n,&XR,n,&XSO,n,&Y,n,&Z));
  PetscCall(PetscRandomCreate(PETSC_COMM_SELF,&rdm));
  PetscCall(PetscRandomSetFromOptions(rdm));
  PetscCall(PetscRandomGetSeed(rdm, &seedr));

  for (i=0; i<n; ++i) {
    PetscCall(PetscRandomGetValueReal(rdm,&val));
    XR[i] = val*PETSC_MAX_INT;
    if (d > 1) XR[i] = XR[i] % (n/d);
    XSO[i] = i;
    if (d > 1) XSO[i] = XSO[i] % (n/d);
  }

  nreal = (PetscReal) n;
  PetscCall(PetscRandomCreate(PETSC_COMM_SELF,&rdm2));
  PetscCall(PetscRandomGetSeed(rdm, &seedo));
  PetscCall(PetscRandomSetInterval(rdm2,0,nreal));
  for (i = 0; i < n/10; ++i) {
    PetscInt swapi, t;
    PetscCall(PetscRandomGetValueReal(rdm2,&val));
    swapi = (PetscInt) val;
    t = XSO[swapi-1];
    XSO[swapi-1] = XSO[swapi];
    XSO[swapi] = t;
  }
  PetscCall(PetscRandomDestroy(&rdm2));

  if (vwr) PetscCall(PetscIntView(n, order ? XSO : XR, vwr));
  PetscCall(PetscViewerDestroy(&vwr));
  PetscCall(VecCreate(PETSC_COMM_WORLD,&x));
  PetscCall(VecSetSizes(x,PETSC_DECIDE,vsize));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecSetRandom(x,rdm));
  time = 0.0;
  time1 = 0.0;
  for (l=0; l<r; l++) { /* r loops */
    PetscCall(PetscArraycpy(X,order ? XSO : XR,n));
    PetscCall(PetscArraycpy(X1,order ? XSO : XR,n));

    PetscCall(VecNorm(x,NORM_1,&norm1));
    PetscCall(PetscTimeSubtract(&time1));
    PetscCall(PetscIntSortSemiOrdered(n,X1));
    PetscCall(PetscTimeAdd(&time1));

    PetscCall(VecNorm(x,NORM_1,&norm1));
    PetscCall(PetscTimeSubtract(&time));
    PetscCall(PetscSortInt(n,X));
    PetscCall(PetscTimeAdd(&time));

    for (i=0; i<n-1; i++) {PetscCheckFalse(X[i] > X[i+1],PETSC_COMM_SELF,PETSC_ERR_PLIB,"PetscSortInt() produced wrong results!");}
    for (i=0; i<n; i++) {PetscCheckFalse(X[i] != X1[i],PETSC_COMM_SELF,PETSC_ERR_PLIB,"PetscIntSortSemiOrdered() rep %" PetscInt_FMT " X1[%" PetscInt_FMT "]:%" PetscInt_FMT " does not match PetscSortInt() X[%" PetscInt_FMT "]:%" PetscInt_FMT "! randomSeed %lu, orderedSeed %lu",l,i,X1[i],i,X[i],seedr,seedo);}
    for (i=0; i<n-1; i++) {PetscCheckFalse(X1[i] > X1[i+1],PETSC_COMM_SELF,PETSC_ERR_PLIB,"PetscIntSortSemiOrdered() produced wrong results! randomSeed %lu orderedSeed %lu",seedr,seedo);}
    PetscCall(PetscArrayzero(X,n));
    PetscCall(PetscArrayzero(X1,n));
  }
  PetscCall(PetscPrintf(PETSC_COMM_SELF,"PetscSortInt()              with %" PetscInt_FMT " integers, %" PetscInt_FMT " duplicate(s) per unique value took %g seconds\n",n,d,time/r));
  PetscCall(PetscPrintf(PETSC_COMM_SELF,"PetscIntSortSemiOrdered()   with %" PetscInt_FMT " integers, %" PetscInt_FMT " duplicate(s) per unique value took %g seconds\n",n,d,time1/r));
  PetscCall(PetscPrintf(PETSC_COMM_SELF,"Speedup of PetscIntSortSemiOrdered() was %g (0:1 = slower, >1 means faster)\n",time/time1));

  for (i=0; i<n; i++) { /* Init X[] */
    PetscCall(PetscRandomGetValueReal(rdm,&val));
    X[i] = val*PETSC_MAX_INT;
    if (d > 1) X[i] = X[i] % (n/d);
  }
  PetscCall(PetscCalloc3(n,&XP,n,&X1P,n,&W));

  time = 0.0;
  time1 = 0.0;
  time2 = 0.0;
  for (l=0; l<r; l++) { /* r loops */
    PetscCall(PetscArraycpy(X1, order ? XSO : XR,n));
    PetscCall(PetscArraycpy(X1P,order ? XSO : XR,n));
    PetscCall(PetscArraycpy(X,  order ? XSO : XR,n));
    PetscCall(PetscArraycpy(XP, order ? XSO : XR,n));
    PetscCall(PetscArraycpy(W, order ? XSO : XR,n));

    PetscCall(VecNorm(x,NORM_1,&norm1));
    PetscCall(PetscTimeSubtract(&time1));
    PetscCall(PetscIntSortSemiOrderedWithArray(n,X1,X1P));
    PetscCall(PetscTimeAdd(&time1));

    PetscCall(VecNorm(x,NORM_1,&norm1));
    PetscCall(PetscTimeSubtract(&time2));
    PetscCall(PetscSortIntWithArray(n,X,XP));
    PetscCall(PetscTimeAdd(&time2));

    PetscCall(PetscTimeSubtract(&time));
    PetscCall(PetscSortIntWithArrayPair(n,W,Y,Z));
    PetscCall(PetscTimeAdd(&time));

    for (i=0; i<n-1; i++) {if (Y[i] > Y[i+1]) {PetscIntView(n,Y,0);SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"PetscSortIntWithArray() produced wrong results!");}}
    for (i=0; i<n-1; i++) {PetscCheckFalse(W[i] > W[i+1],PETSC_COMM_SELF,PETSC_ERR_PLIB,"PetscSortIntWithArrayPair() produced wrong results!");}
    for (i=0; i<n; i++) {PetscCheckFalse(X1P[i] != X[i],PETSC_COMM_SELF,PETSC_ERR_PLIB,"PetscIntSortSemiOrdered() rep %" PetscInt_FMT " X1[%" PetscInt_FMT "]:%" PetscInt_FMT " does not match PetscSortIntWithArray() X[%" PetscInt_FMT "]:%" PetscInt_FMT "! randomSeed %lu, orderedSeed %lu",l,i,X1[i],i,X[i],seedr,seedo);}
    for (i=0; i<n-1; i++) {PetscCheckFalse(X1[i] > X1[i+1],PETSC_COMM_SELF,PETSC_ERR_PLIB,"PetscIntSortSemiOrdered() produced wrong results! randomSeed %lu orderedSeed %lu",seedr,seedo);}
    PetscCall(PetscArrayzero(X1,n));
    PetscCall(PetscArrayzero(X1P,n));
    PetscCall(PetscArrayzero(X,n));
    PetscCall(PetscArrayzero(XP,n));
    PetscCall(PetscArrayzero(W,n));
  }
  PetscCall(PetscPrintf(PETSC_COMM_SELF,"PetscSortIntWithArrayPair()        with %" PetscInt_FMT " integers, %" PetscInt_FMT " duplicate(s) per unique value took %g seconds\n",n,d,time/r));
  PetscCall(PetscPrintf(PETSC_COMM_SELF,"PetscSortIntWithArray()            with %" PetscInt_FMT " integers, %" PetscInt_FMT " duplicate(s) per unique value took %g seconds\n",n,d,time2/r));
  PetscCall(PetscPrintf(PETSC_COMM_SELF,"PetscIntSortSemiOrderedWithArray() with %" PetscInt_FMT " integers, %" PetscInt_FMT " duplicate(s) per unique value took %g seconds\n",n,d,time1/r));
  PetscCall(PetscPrintf(PETSC_COMM_SELF,"Speedup of PetscIntSortSemiOrderedWithArray() was %g (0:1 = slower, >1 means faster)\n",time2/time1));
  PetscCall(PetscPrintf(PETSC_COMM_SELF,"SUCCEEDED\n"));

  PetscCall(VecDestroy(&x));
  PetscCall(PetscRandomDestroy(&rdm));
  PetscCall(PetscFree3(XP,X1P,W));
  PetscCall(PetscFree6(X,X1,XR,XSO,Y,Z));
  PetscCall(PetscFinalize());
  return 0;
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
