
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
  ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,&flg);CHKERRQ(ierr);
  if (!flg) m = 100;

  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,&flg);CHKERRQ(ierr);
  if (!flg) n = 30;

  ierr = PetscOptionsGetInt(NULL,NULL,"-toFirst",&toFirst,&flg);CHKERRQ(ierr);
  if (!flg) toFirst = 3;

  ierr = PetscOptionsGetInt(NULL,NULL,"-toStep",&toStep,&flg);CHKERRQ(ierr);
  if (!flg) toStep = 3;

  ierr = PetscOptionsGetInt(NULL,NULL,"-fromFirst",&fromFirst,&flg);CHKERRQ(ierr);
  if (!flg) fromFirst = 2;

  ierr = PetscOptionsGetInt(NULL,NULL,"-fromStep",&fromStep,&flg);CHKERRQ(ierr);
  if (!flg) fromStep = 2;

  if (n>m) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"The vector sizes are %" PetscInt_FMT ". The number of elements being scattered is %" PetscInt_FMT "\n",m,n);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Adjust the parameters such that m>=n\n");CHKERRQ(ierr);
  } else if (toFirst+(n-1)*toStep >=m) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"The vector sizes are %" PetscInt_FMT ". The number of elements being scattered is %" PetscInt_FMT "\n",m,n);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"For the Strided Scatter, toFirst=%" PetscInt_FMT " and toStep=%" PetscInt_FMT ".\n",toFirst,toStep);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"This produces an index (toFirst+(n-1)*toStep)>=m\n");CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Adjust the parameterrs accordingly with -m, -n, -toFirst, or -toStep\n");CHKERRQ(ierr);
  } else if (fromFirst+(n-1)*fromStep>=m) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"The vector sizes are %" PetscInt_FMT ". The number of elements being scattered is %" PetscInt_FMT "\n",m,n);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"For the Strided Scatter, fromFirst=%" PetscInt_FMT " and fromStep=%" PetscInt_FMT ".\n",fromFirst,toStep);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"This produces an index (fromFirst+(n-1)*fromStep)>=m\n");CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Adjust the parameterrs accordingly with -m, -n, -fromFirst, or -fromStep\n");CHKERRQ(ierr);
  } else {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"m=%" PetscInt_FMT "\tn=%" PetscInt_FMT "\tfromFirst=%" PetscInt_FMT "\tfromStep=%" PetscInt_FMT "\ttoFirst=%" PetscInt_FMT "\ttoStep=%" PetscInt_FMT "\n",m,n,fromFirst,fromStep,toFirst,toStep);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"fromFirst+(n-1)*fromStep=%" PetscInt_FMT "\ttoFirst+(n-1)*toStep=%" PetscInt_FMT "\n",fromFirst+(n-1)*fromStep,toFirst+(n-1)*toStep);CHKERRQ(ierr);

    /* Build the vectors */
    ierr = VecCreate(PETSC_COMM_WORLD,&Y);CHKERRQ(ierr);
    ierr = VecSetSizes(Y,m,PETSC_DECIDE);CHKERRQ(ierr);
    ierr = VecCreate(PETSC_COMM_WORLD,&X);CHKERRQ(ierr);
    ierr = VecSetSizes(X,m,PETSC_DECIDE);CHKERRQ(ierr);

    ierr = VecSetFromOptions(Y);CHKERRQ(ierr);
    ierr = VecSetFromOptions(X);CHKERRQ(ierr);
    ierr = VecSet(X,2.0);CHKERRQ(ierr);
    ierr = VecSet(Y,1.0);CHKERRQ(ierr);

    /* Build the strided index sets */
    ierr = ISCreate(PETSC_COMM_WORLD,&toISStrided);CHKERRQ(ierr);
    ierr = ISCreate(PETSC_COMM_WORLD,&fromISStrided);CHKERRQ(ierr);
    ierr = ISSetType(toISStrided, ISSTRIDE);CHKERRQ(ierr);
    ierr = ISSetType(fromISStrided, ISSTRIDE);CHKERRQ(ierr);
    ierr = ISStrideSetStride(fromISStrided,n,fromFirst,fromStep);CHKERRQ(ierr);
    ierr = ISStrideSetStride(toISStrided,n,toFirst,toStep);CHKERRQ(ierr);

    /* Build the general index sets */
    ierr = PetscMalloc1(n,&idx);CHKERRQ(ierr);
    ierr = PetscMalloc1(n,&idy);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      idx[i] = i % m;
      idy[i] = (i+m) % m;
    }
    n1 = n;
    n2 = n;
    ierr = PetscSortRemoveDupsInt(&n1,idx);CHKERRQ(ierr);
    ierr = PetscSortRemoveDupsInt(&n2,idy);CHKERRQ(ierr);

    ierr = ISCreateGeneral(PETSC_COMM_WORLD,n1,idx,PETSC_COPY_VALUES,&toISGeneral);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PETSC_COMM_WORLD,n2,idy,PETSC_COPY_VALUES,&fromISGeneral);CHKERRQ(ierr);

    /* set the mode and the insert/add parameter */
    mode = SCATTER_FORWARD;
    addv = ADD_VALUES;

    /* VecScatter : Seq Strided to Seq Strided */
    ierr = VecScatterCreate(X,fromISStrided,Y,toISStrided,&vscatSStoSS);CHKERRQ(ierr);
    ierr = VecScatterBegin(vscatSStoSS,X,Y,addv,mode);CHKERRQ(ierr);
    ierr = VecScatterEnd(vscatSStoSS,X,Y,addv,mode);CHKERRQ(ierr);
    ierr = VecScatterDestroy(&vscatSStoSS);CHKERRQ(ierr);

    /* VecScatter : Seq General to Seq Strided */
    ierr = VecScatterCreate(Y,fromISGeneral,X,toISStrided,&vscatSGtoSS);CHKERRQ(ierr);
    ierr = VecScatterBegin(vscatSGtoSS,Y,X,addv,mode);CHKERRQ(ierr);
    ierr = VecScatterEnd(vscatSGtoSS,Y,X,addv,mode);CHKERRQ(ierr);
    ierr = VecScatterDestroy(&vscatSGtoSS);CHKERRQ(ierr);

    /* VecScatter : Seq General to Seq General */
    ierr = VecScatterCreate(X,fromISGeneral,Y,toISGeneral,&vscatSGtoSG);CHKERRQ(ierr);
    ierr = VecScatterBegin(vscatSGtoSG,X,Y,addv,mode);CHKERRQ(ierr);
    ierr = VecScatterEnd(vscatSGtoSG,X,Y,addv,mode);CHKERRQ(ierr);
    ierr = VecScatterDestroy(&vscatSGtoSG);CHKERRQ(ierr);

    /* VecScatter : Seq Strided to Seq General */
    ierr = VecScatterCreate(Y,fromISStrided,X,toISGeneral,&vscatSStoSG);CHKERRQ(ierr);
    ierr = VecScatterBegin(vscatSStoSG,Y,X,addv,mode);CHKERRQ(ierr);
    ierr = VecScatterEnd(vscatSStoSG,Y,X,addv,mode);CHKERRQ(ierr);
    ierr = VecScatterDestroy(&vscatSStoSG);CHKERRQ(ierr);

    /* view the results */
    ierr = VecView(Y,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

    /* Cleanup */
    ierr = VecDestroy(&X);CHKERRQ(ierr);
    ierr = VecDestroy(&Y);CHKERRQ(ierr);
    ierr = ISDestroy(&toISStrided);CHKERRQ(ierr);
    ierr = ISDestroy(&fromISStrided);CHKERRQ(ierr);
    ierr = ISDestroy(&toISGeneral);CHKERRQ(ierr);
    ierr = ISDestroy(&fromISGeneral);CHKERRQ(ierr);
    ierr = PetscFree(idx);CHKERRQ(ierr);
    ierr = PetscFree(idy);CHKERRQ(ierr);
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
