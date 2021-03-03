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
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  if (size != 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"This is a uniprocessor example only!");

  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-r",&r,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-d",&d,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-vsize",&vsize,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-order",NULL,&order);CHKERRQ(ierr);
  ierr = PetscOptionsGetViewer(PETSC_COMM_WORLD,NULL,NULL,"-array_view",&vwr,NULL,NULL);CHKERRQ(ierr);
  if (n<1 || r<1 || d<1 || d>n) SETERRQ3(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Wrong input n=%D,r=%D,d=%D. They must be >=1 and n>=d\n",n,r,d);

  ierr = PetscCalloc6(n,&X,n,&X1,n,&XR,n,&XSO,n,&Y,n,&Z);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_SELF,&rdm);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rdm);CHKERRQ(ierr);
  ierr = PetscRandomGetSeed(rdm, &seedr);CHKERRQ(ierr);

  for (i=0; i<n; ++i) {
    ierr = PetscRandomGetValueReal(rdm,&val);CHKERRQ(ierr);
    XR[i] = val*PETSC_MAX_INT;
    if (d > 1) XR[i] = XR[i] % (n/d);
    XSO[i] = i;
    if (d > 1) XSO[i] = XSO[i] % (n/d);
  }

  nreal = (PetscReal) n;
  ierr = PetscRandomCreate(PETSC_COMM_SELF,&rdm2);CHKERRQ(ierr);
  ierr = PetscRandomGetSeed(rdm, &seedo);CHKERRQ(ierr);
  ierr = PetscRandomSetInterval(rdm2,0,nreal);CHKERRQ(ierr);
  for (i = 0; i < n/10; ++i) {
    PetscInt swapi, t;
    ierr = PetscRandomGetValueReal(rdm2,&val);CHKERRQ(ierr);
    swapi = (PetscInt) val;
    t = XSO[swapi-1];
    XSO[swapi-1] = XSO[swapi];
    XSO[swapi] = t;
  }
  ierr = PetscRandomDestroy(&rdm2);CHKERRQ(ierr);

  if (vwr) {ierr = PetscIntView(n, order ? XSO : XR, vwr);CHKERRQ(ierr);}
  ierr = PetscViewerDestroy(&vwr);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,vsize);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecSetRandom(x,rdm);CHKERRQ(ierr);
  time = 0.0;
  time1 = 0.0;
  for (l=0; l<r; l++) { /* r loops */
    ierr = PetscArraycpy(X,order ? XSO : XR,n);CHKERRQ(ierr);
    ierr = PetscArraycpy(X1,order ? XSO : XR,n);CHKERRQ(ierr);

    ierr = VecNorm(x,NORM_1,&norm1);CHKERRQ(ierr);
    ierr = PetscTimeSubtract(&time1);CHKERRQ(ierr);
    ierr = PetscIntSortSemiOrdered(n,X1);CHKERRQ(ierr);
    ierr = PetscTimeAdd(&time1);CHKERRQ(ierr);

    ierr = VecNorm(x,NORM_1,&norm1);CHKERRQ(ierr);
    ierr = PetscTimeSubtract(&time);CHKERRQ(ierr);
    ierr = PetscSortInt(n,X);CHKERRQ(ierr);
    ierr = PetscTimeAdd(&time);CHKERRQ(ierr);

    for (i=0; i<n-1; i++) {if (X[i] > X[i+1]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"PetscSortInt() produced wrong results!");}
    for (i=0; i<n; i++) {if (X[i] != X1[i]) SETERRQ7(PETSC_COMM_SELF,PETSC_ERR_PLIB,"PetscIntSortSemiOrdered() rep %D X1[%D]:%D does not match PetscSortInt() X[%D]:%D! randomSeed %lu, orderedSeed %lu",l,i,X1[i],i,X[i],seedr,seedo);}
    for (i=0; i<n-1; i++) {if (X1[i] > X1[i+1]) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"PetscIntSortSemiOrdered() produced wrong results! randomSeed %lu orderedSeed %lu",seedr,seedo);}
    ierr = PetscArrayzero(X,n);CHKERRQ(ierr);
    ierr = PetscArrayzero(X1,n);CHKERRQ(ierr);
  }
  ierr = PetscPrintf(PETSC_COMM_SELF,"PetscSortInt()              with %D integers, %D duplicate(s) per unique value took %g seconds\n",n,d,time/r);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"PetscIntSortSemiOrdered()   with %D integers, %D duplicate(s) per unique value took %g seconds\n",n,d,time1/r);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"Speedup of PetscIntSortSemiOrdered() was %g (0:1 = slower, >1 means faster)\n",time/time1);CHKERRQ(ierr);


  for (i=0; i<n; i++) { /* Init X[] */
    ierr = PetscRandomGetValueReal(rdm,&val);CHKERRQ(ierr);
    X[i] = val*PETSC_MAX_INT;
    if (d > 1) X[i] = X[i] % (n/d);
  }
  ierr = PetscCalloc3(n,&XP,n,&X1P,n,&W);CHKERRQ(ierr);

  time = 0.0;
  time1 = 0.0;
  time2 = 0.0;
  for (l=0; l<r; l++) { /* r loops */
    ierr = PetscArraycpy(X1, order ? XSO : XR,n);CHKERRQ(ierr);
    ierr = PetscArraycpy(X1P,order ? XSO : XR,n);CHKERRQ(ierr);
    ierr = PetscArraycpy(X,  order ? XSO : XR,n);CHKERRQ(ierr);
    ierr = PetscArraycpy(XP, order ? XSO : XR,n);CHKERRQ(ierr);
    ierr = PetscArraycpy(W, order ? XSO : XR,n);CHKERRQ(ierr);

    ierr = VecNorm(x,NORM_1,&norm1);CHKERRQ(ierr);
    ierr = PetscTimeSubtract(&time1);CHKERRQ(ierr);
    ierr = PetscIntSortSemiOrderedWithArray(n,X1,X1P);CHKERRQ(ierr);
    ierr = PetscTimeAdd(&time1);CHKERRQ(ierr);

    ierr = VecNorm(x,NORM_1,&norm1);CHKERRQ(ierr);
    ierr = PetscTimeSubtract(&time2);CHKERRQ(ierr);
    ierr = PetscSortIntWithArray(n,X,XP);CHKERRQ(ierr);
    ierr = PetscTimeAdd(&time2);CHKERRQ(ierr);

    ierr = PetscTimeSubtract(&time);CHKERRQ(ierr);
    ierr = PetscSortIntWithArrayPair(n,W,Y,Z);CHKERRQ(ierr);
    ierr = PetscTimeAdd(&time);CHKERRQ(ierr);

    for (i=0; i<n-1; i++) {if (Y[i] > Y[i+1]) {PetscIntView(n,Y,0);SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"PetscSortIntWithArray() produced wrong results!");}}
    for (i=0; i<n-1; i++) {if (W[i] > W[i+1]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"PetscSortIntWithArrayPair() produced wrong results!");}
    for (i=0; i<n; i++) {if (X1P[i] != X[i]) SETERRQ7(PETSC_COMM_SELF,PETSC_ERR_PLIB,"PetscIntSortSemiOrdered() rep %D X1[%D]:%D does not match PetscSortIntWithArray() X[%D]:%D! randomSeed %lu, orderedSeed %lu",l,i,X1[i],i,X[i],seedr,seedo);}
    for (i=0; i<n-1; i++) {if (X1[i] > X1[i+1]) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"PetscIntSortSemiOrdered() produced wrong results! randomSeed %lu orderedSeed %lu",seedr,seedo);}
    ierr = PetscArrayzero(X1,n);CHKERRQ(ierr);
    ierr = PetscArrayzero(X1P,n);CHKERRQ(ierr);
    ierr = PetscArrayzero(X,n);CHKERRQ(ierr);
    ierr = PetscArrayzero(XP,n);CHKERRQ(ierr);
    ierr = PetscArrayzero(W,n);CHKERRQ(ierr);
  }
  ierr = PetscPrintf(PETSC_COMM_SELF,"PetscSortIntWithArrayPair()        with %D integers, %D duplicate(s) per unique value took %g seconds\n",n,d,time/r);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"PetscSortIntWithArray()            with %D integers, %D duplicate(s) per unique value took %g seconds\n",n,d,time2/r);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"PetscIntSortSemiOrderedWithArray() with %D integers, %D duplicate(s) per unique value took %g seconds\n",n,d,time1/r);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"Speedup of PetscIntSortSemiOrderedWithArray() was %g (0:1 = slower, >1 means faster)\n",time2/time1);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"SUCCEEDED\n");CHKERRQ(ierr);

  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rdm);CHKERRQ(ierr);
  ierr = PetscFree3(XP,X1P,W);CHKERRQ(ierr);
  ierr = PetscFree6(X,X1,XR,XSO,Y,Z);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      args: -n 1000 -r 10 -d 1
      # Do not need to output timing results for test
      filter: grep -vE "per unique value took|Speedup of "
TEST*/
