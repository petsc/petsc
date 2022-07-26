
#include <petscvec.h>

static char help[] = "Tests vecScatter Sequential to Sequential for (CUDA) vectors\n\
  -m # : the size of the vectors\n                                      \
  -n # : the numer of indices (with n<=m)\n                             \
  -toFirst # : the starting index of the output vector for strided scatters\n \
  -toStep # : the step size into the output vector for strided scatters\n \
  -fromFirst # : the starting index of the input vector for strided scatters\n\
  -fromStep # : the step size into the input vector for strided scatters\n\n";

int main(int argc, char * argv[]) {

  Vec            X,Y;
  PetscInt       m,n,i,n1,n2;
  PetscInt       toFirst,toStep,fromFirst,fromStep;
  PetscInt       *idx,*idy;
  PetscBool      flg;
  IS             toISStrided,fromISStrided,toISGeneral,fromISGeneral;
  VecScatter     vscatSStoSS,vscatSStoSG,vscatSGtoSS,vscatSGtoSG;
  ScatterMode    mode;
  InsertMode     addv;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-m",&m,&flg));
  if (!flg) m = 100;

  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,&flg));
  if (!flg) n = 30;

  PetscCall(PetscOptionsGetInt(NULL,NULL,"-toFirst",&toFirst,&flg));
  if (!flg) toFirst = 3;

  PetscCall(PetscOptionsGetInt(NULL,NULL,"-toStep",&toStep,&flg));
  if (!flg) toStep = 3;

  PetscCall(PetscOptionsGetInt(NULL,NULL,"-fromFirst",&fromFirst,&flg));
  if (!flg) fromFirst = 2;

  PetscCall(PetscOptionsGetInt(NULL,NULL,"-fromStep",&fromStep,&flg));
  if (!flg) fromStep = 2;

  if (n>m) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"The vector sizes are %" PetscInt_FMT ". The number of elements being scattered is %" PetscInt_FMT "\n",m,n));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Adjust the parameters such that m>=n\n"));
  } else if (toFirst+(n-1)*toStep >=m) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"The vector sizes are %" PetscInt_FMT ". The number of elements being scattered is %" PetscInt_FMT "\n",m,n));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"For the Strided Scatter, toFirst=%" PetscInt_FMT " and toStep=%" PetscInt_FMT ".\n",toFirst,toStep));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"This produces an index (toFirst+(n-1)*toStep)>=m\n"));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Adjust the parameterrs accordingly with -m, -n, -toFirst, or -toStep\n"));
  } else if (fromFirst+(n-1)*fromStep>=m) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"The vector sizes are %" PetscInt_FMT ". The number of elements being scattered is %" PetscInt_FMT "\n",m,n));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"For the Strided Scatter, fromFirst=%" PetscInt_FMT " and fromStep=%" PetscInt_FMT ".\n",fromFirst,toStep));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"This produces an index (fromFirst+(n-1)*fromStep)>=m\n"));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Adjust the parameterrs accordingly with -m, -n, -fromFirst, or -fromStep\n"));
  } else {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"m=%" PetscInt_FMT "\tn=%" PetscInt_FMT "\tfromFirst=%" PetscInt_FMT "\tfromStep=%" PetscInt_FMT "\ttoFirst=%" PetscInt_FMT "\ttoStep=%" PetscInt_FMT "\n",m,n,fromFirst,fromStep,toFirst,toStep));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"fromFirst+(n-1)*fromStep=%" PetscInt_FMT "\ttoFirst+(n-1)*toStep=%" PetscInt_FMT "\n",fromFirst+(n-1)*fromStep,toFirst+(n-1)*toStep));

    /* Build the vectors */
    PetscCall(VecCreate(PETSC_COMM_WORLD,&Y));
    PetscCall(VecSetSizes(Y,m,PETSC_DECIDE));
    PetscCall(VecCreate(PETSC_COMM_WORLD,&X));
    PetscCall(VecSetSizes(X,m,PETSC_DECIDE));

    PetscCall(VecSetFromOptions(Y));
    PetscCall(VecSetFromOptions(X));
    PetscCall(VecSet(X,2.0));
    PetscCall(VecSet(Y,1.0));

    /* Build the strided index sets */
    PetscCall(ISCreate(PETSC_COMM_WORLD,&toISStrided));
    PetscCall(ISCreate(PETSC_COMM_WORLD,&fromISStrided));
    PetscCall(ISSetType(toISStrided, ISSTRIDE));
    PetscCall(ISSetType(fromISStrided, ISSTRIDE));
    PetscCall(ISStrideSetStride(fromISStrided,n,fromFirst,fromStep));
    PetscCall(ISStrideSetStride(toISStrided,n,toFirst,toStep));

    /* Build the general index sets */
    PetscCall(PetscMalloc1(n,&idx));
    PetscCall(PetscMalloc1(n,&idy));
    for (i=0; i<n; i++) {
      idx[i] = i % m;
      idy[i] = (i+m) % m;
    }
    n1 = n;
    n2 = n;
    PetscCall(PetscSortRemoveDupsInt(&n1,idx));
    PetscCall(PetscSortRemoveDupsInt(&n2,idy));

    PetscCall(ISCreateGeneral(PETSC_COMM_WORLD,n1,idx,PETSC_COPY_VALUES,&toISGeneral));
    PetscCall(ISCreateGeneral(PETSC_COMM_WORLD,n2,idy,PETSC_COPY_VALUES,&fromISGeneral));

    /* set the mode and the insert/add parameter */
    mode = SCATTER_FORWARD;
    addv = ADD_VALUES;

    /* VecScatter : Seq Strided to Seq Strided */
    PetscCall(VecScatterCreate(X,fromISStrided,Y,toISStrided,&vscatSStoSS));
    PetscCall(VecScatterBegin(vscatSStoSS,X,Y,addv,mode));
    PetscCall(VecScatterEnd(vscatSStoSS,X,Y,addv,mode));
    PetscCall(VecScatterDestroy(&vscatSStoSS));

    /* VecScatter : Seq General to Seq Strided */
    PetscCall(VecScatterCreate(Y,fromISGeneral,X,toISStrided,&vscatSGtoSS));
    PetscCall(VecScatterBegin(vscatSGtoSS,Y,X,addv,mode));
    PetscCall(VecScatterEnd(vscatSGtoSS,Y,X,addv,mode));
    PetscCall(VecScatterDestroy(&vscatSGtoSS));

    /* VecScatter : Seq General to Seq General */
    PetscCall(VecScatterCreate(X,fromISGeneral,Y,toISGeneral,&vscatSGtoSG));
    PetscCall(VecScatterBegin(vscatSGtoSG,X,Y,addv,mode));
    PetscCall(VecScatterEnd(vscatSGtoSG,X,Y,addv,mode));
    PetscCall(VecScatterDestroy(&vscatSGtoSG));

    /* VecScatter : Seq Strided to Seq General */
    PetscCall(VecScatterCreate(Y,fromISStrided,X,toISGeneral,&vscatSStoSG));
    PetscCall(VecScatterBegin(vscatSStoSG,Y,X,addv,mode));
    PetscCall(VecScatterEnd(vscatSStoSG,Y,X,addv,mode));
    PetscCall(VecScatterDestroy(&vscatSStoSG));

    /* view the results */
    PetscCall(VecView(Y,PETSC_VIEWER_STDOUT_WORLD));

    /* Cleanup */
    PetscCall(VecDestroy(&X));
    PetscCall(VecDestroy(&Y));
    PetscCall(ISDestroy(&toISStrided));
    PetscCall(ISDestroy(&fromISStrided));
    PetscCall(ISDestroy(&toISGeneral));
    PetscCall(ISDestroy(&fromISGeneral));
    PetscCall(PetscFree(idx));
    PetscCall(PetscFree(idy));
  }
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      suffix: cuda
      args: -vec_type cuda
      requires: cuda

TEST*/
