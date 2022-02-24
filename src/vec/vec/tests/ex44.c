
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
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,&flg));
  if (!flg) m = 100;

  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,&flg));
  if (!flg) n = 30;

  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-toFirst",&toFirst,&flg));
  if (!flg) toFirst = 3;

  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-toStep",&toStep,&flg));
  if (!flg) toStep = 3;

  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-fromFirst",&fromFirst,&flg));
  if (!flg) fromFirst = 2;

  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-fromStep",&fromStep,&flg));
  if (!flg) fromStep = 2;

  if (n>m) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"The vector sizes are %" PetscInt_FMT ". The number of elements being scattered is %" PetscInt_FMT "\n",m,n));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Adjust the parameters such that m>=n\n"));
  } else if (toFirst+(n-1)*toStep >=m) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"The vector sizes are %" PetscInt_FMT ". The number of elements being scattered is %" PetscInt_FMT "\n",m,n));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"For the Strided Scatter, toFirst=%" PetscInt_FMT " and toStep=%" PetscInt_FMT ".\n",toFirst,toStep));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"This produces an index (toFirst+(n-1)*toStep)>=m\n"));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Adjust the parameterrs accordingly with -m, -n, -toFirst, or -toStep\n"));
  } else if (fromFirst+(n-1)*fromStep>=m) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"The vector sizes are %" PetscInt_FMT ". The number of elements being scattered is %" PetscInt_FMT "\n",m,n));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"For the Strided Scatter, fromFirst=%" PetscInt_FMT " and fromStep=%" PetscInt_FMT ".\n",fromFirst,toStep));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"This produces an index (fromFirst+(n-1)*fromStep)>=m\n"));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Adjust the parameterrs accordingly with -m, -n, -fromFirst, or -fromStep\n"));
  } else {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"m=%" PetscInt_FMT "\tn=%" PetscInt_FMT "\tfromFirst=%" PetscInt_FMT "\tfromStep=%" PetscInt_FMT "\ttoFirst=%" PetscInt_FMT "\ttoStep=%" PetscInt_FMT "\n",m,n,fromFirst,fromStep,toFirst,toStep));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"fromFirst+(n-1)*fromStep=%" PetscInt_FMT "\ttoFirst+(n-1)*toStep=%" PetscInt_FMT "\n",fromFirst+(n-1)*fromStep,toFirst+(n-1)*toStep));

    /* Build the vectors */
    CHKERRQ(VecCreate(PETSC_COMM_WORLD,&Y));
    CHKERRQ(VecSetSizes(Y,m,PETSC_DECIDE));
    CHKERRQ(VecCreate(PETSC_COMM_WORLD,&X));
    CHKERRQ(VecSetSizes(X,m,PETSC_DECIDE));

    CHKERRQ(VecSetFromOptions(Y));
    CHKERRQ(VecSetFromOptions(X));
    CHKERRQ(VecSet(X,2.0));
    CHKERRQ(VecSet(Y,1.0));

    /* Build the strided index sets */
    CHKERRQ(ISCreate(PETSC_COMM_WORLD,&toISStrided));
    CHKERRQ(ISCreate(PETSC_COMM_WORLD,&fromISStrided));
    CHKERRQ(ISSetType(toISStrided, ISSTRIDE));
    CHKERRQ(ISSetType(fromISStrided, ISSTRIDE));
    CHKERRQ(ISStrideSetStride(fromISStrided,n,fromFirst,fromStep));
    CHKERRQ(ISStrideSetStride(toISStrided,n,toFirst,toStep));

    /* Build the general index sets */
    CHKERRQ(PetscMalloc1(n,&idx));
    CHKERRQ(PetscMalloc1(n,&idy));
    for (i=0; i<n; i++) {
      idx[i] = i % m;
      idy[i] = (i+m) % m;
    }
    n1 = n;
    n2 = n;
    CHKERRQ(PetscSortRemoveDupsInt(&n1,idx));
    CHKERRQ(PetscSortRemoveDupsInt(&n2,idy));

    CHKERRQ(ISCreateGeneral(PETSC_COMM_WORLD,n1,idx,PETSC_COPY_VALUES,&toISGeneral));
    CHKERRQ(ISCreateGeneral(PETSC_COMM_WORLD,n2,idy,PETSC_COPY_VALUES,&fromISGeneral));

    /* set the mode and the insert/add parameter */
    mode = SCATTER_FORWARD;
    addv = ADD_VALUES;

    /* VecScatter : Seq Strided to Seq Strided */
    CHKERRQ(VecScatterCreate(X,fromISStrided,Y,toISStrided,&vscatSStoSS));
    CHKERRQ(VecScatterBegin(vscatSStoSS,X,Y,addv,mode));
    CHKERRQ(VecScatterEnd(vscatSStoSS,X,Y,addv,mode));
    CHKERRQ(VecScatterDestroy(&vscatSStoSS));

    /* VecScatter : Seq General to Seq Strided */
    CHKERRQ(VecScatterCreate(Y,fromISGeneral,X,toISStrided,&vscatSGtoSS));
    CHKERRQ(VecScatterBegin(vscatSGtoSS,Y,X,addv,mode));
    CHKERRQ(VecScatterEnd(vscatSGtoSS,Y,X,addv,mode));
    CHKERRQ(VecScatterDestroy(&vscatSGtoSS));

    /* VecScatter : Seq General to Seq General */
    CHKERRQ(VecScatterCreate(X,fromISGeneral,Y,toISGeneral,&vscatSGtoSG));
    CHKERRQ(VecScatterBegin(vscatSGtoSG,X,Y,addv,mode));
    CHKERRQ(VecScatterEnd(vscatSGtoSG,X,Y,addv,mode));
    CHKERRQ(VecScatterDestroy(&vscatSGtoSG));

    /* VecScatter : Seq Strided to Seq General */
    CHKERRQ(VecScatterCreate(Y,fromISStrided,X,toISGeneral,&vscatSStoSG));
    CHKERRQ(VecScatterBegin(vscatSStoSG,Y,X,addv,mode));
    CHKERRQ(VecScatterEnd(vscatSStoSG,Y,X,addv,mode));
    CHKERRQ(VecScatterDestroy(&vscatSStoSG));

    /* view the results */
    CHKERRQ(VecView(Y,PETSC_VIEWER_STDOUT_WORLD));

    /* Cleanup */
    CHKERRQ(VecDestroy(&X));
    CHKERRQ(VecDestroy(&Y));
    CHKERRQ(ISDestroy(&toISStrided));
    CHKERRQ(ISDestroy(&fromISStrided));
    CHKERRQ(ISDestroy(&toISGeneral));
    CHKERRQ(ISDestroy(&fromISGeneral));
    CHKERRQ(PetscFree(idx));
    CHKERRQ(PetscFree(idy));
  }
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      suffix: cuda
      args: -vec_type cuda
      requires: cuda

TEST*/
