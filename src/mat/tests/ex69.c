static char help[] = "Tests MatCreateDenseCUDA(), MatDenseCUDAPlaceArray(), MatDenseCUDAReplaceArray(), MatDenseCUDAResetArray()\n";

#include <petscmat.h>

static PetscErrorCode MatMult_S(Mat S,Vec x,Vec y)
{
  PetscErrorCode ierr;
  Mat            A;

  PetscFunctionBeginUser;
  ierr = MatShellGetContext(S,&A);CHKERRQ(ierr);
  ierr = MatMult(A,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscBool test_cusparse_transgen = PETSC_FALSE;

static PetscErrorCode MatMultTranspose_S(Mat S,Vec x,Vec y)
{
  PetscErrorCode ierr;
  Mat            A;

  PetscFunctionBeginUser;
  ierr = MatShellGetContext(S,&A);CHKERRQ(ierr);
  ierr = MatMultTranspose(A,x,y);CHKERRQ(ierr);

  /* alternate transgen true and false to test code logic */
  ierr = MatSetOption(A,MAT_FORM_EXPLICIT_TRANSPOSE,test_cusparse_transgen);CHKERRQ(ierr);
  test_cusparse_transgen = (PetscBool)!test_cusparse_transgen;
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  Mat            A,B,C,S;
  Vec            t,v;
  PetscScalar    *vv,*aa;
  PetscInt       n=30,k=6,l=0,i,Istart,Iend,nloc,bs,test=1;
  PetscBool      flg,reset,use_shell = PETSC_FALSE;
  PetscErrorCode ierr;
  VecType        vtype;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-k",&k,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-l",&l,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-test",&test,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-use_shell",&use_shell,NULL);CHKERRQ(ierr);
  PetscCheckFalse(k < 0,PETSC_COMM_WORLD,PETSC_ERR_USER,"k %" PetscInt_FMT " must be positive",k);
  PetscCheckFalse(l < 0,PETSC_COMM_WORLD,PETSC_ERR_USER,"l %" PetscInt_FMT " must be positive",l);
  PetscCheckFalse(l > k,PETSC_COMM_WORLD,PETSC_ERR_USER,"l %" PetscInt_FMT " must be smaller or equal than k %" PetscInt_FMT,l,k);

  /* sparse matrix */
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
  ierr = MatSetType(A,MATAIJCUSPARSE);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(A,"A_");CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);

  /* test special case for SeqAIJCUSPARSE to generate explicit transpose (not default) */
  ierr = MatSetOption(A,MAT_FORM_EXPLICIT_TRANSPOSE,PETSC_TRUE);CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);
  for (i=Istart;i<Iend;i++) {
    if (i>0) { ierr = MatSetValue(A,i,i-1,-1.0,INSERT_VALUES);CHKERRQ(ierr); }
    if (i<n-1) { ierr = MatSetValue(A,i,i+1,-1.0,INSERT_VALUES);CHKERRQ(ierr); }
    ierr = MatSetValue(A,i,i,2.0,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* template vector */
  ierr = MatCreateVecs(A,NULL,&t);CHKERRQ(ierr);
  ierr = VecGetType(t,&vtype);CHKERRQ(ierr);

  /* long vector, contains the stacked columns of an nxk dense matrix */
  ierr = VecGetLocalSize(t,&nloc);CHKERRQ(ierr);
  ierr = VecGetBlockSize(t,&bs);CHKERRQ(ierr);
  ierr = VecCreate(PetscObjectComm((PetscObject)t),&v);CHKERRQ(ierr);
  ierr = VecSetType(v,vtype);CHKERRQ(ierr);
  ierr = VecSetSizes(v,k*nloc,k*n);CHKERRQ(ierr);
  ierr = VecSetBlockSize(v,bs);CHKERRQ(ierr);
  ierr = VecSetRandom(v,NULL);CHKERRQ(ierr);

  /* dense matrix that contains the columns of v */
  ierr = VecCUDAGetArray(v,&vv);CHKERRQ(ierr);

  /* test few cases for MatDenseCUDA handling pointers */
  switch (test) {
  case 1:
    ierr = MatCreateDenseCUDA(PetscObjectComm((PetscObject)v),nloc,PETSC_DECIDE,n,k-l,vv,&B);CHKERRQ(ierr); /* pass a pointer to avoid allocation of storage */
    ierr = MatDenseCUDAReplaceArray(B,NULL);CHKERRQ(ierr);  /* replace with a null pointer, the value after BVRestoreMat */
    ierr = MatDenseCUDAPlaceArray(B,vv+l*nloc);CHKERRQ(ierr);  /* set the actual pointer */
    reset = PETSC_TRUE;
    break;
  case 2:
    ierr = MatCreateDenseCUDA(PetscObjectComm((PetscObject)v),nloc,PETSC_DECIDE,n,k-l,NULL,&B);CHKERRQ(ierr);
    ierr = MatDenseCUDAPlaceArray(B,vv+l*nloc);CHKERRQ(ierr);  /* set the actual pointer */
    reset = PETSC_TRUE;
    break;
  default:
    ierr = MatCreateDenseCUDA(PetscObjectComm((PetscObject)v),nloc,PETSC_DECIDE,n,k-l,vv+l*nloc,&B);CHKERRQ(ierr);
    reset = PETSC_FALSE;
    break;
  }
  ierr = VecCUDARestoreArray(v,&vv);CHKERRQ(ierr);

  /* Test MatMatMult */
  if (use_shell) {
    /* we could have called the general convertor below, but we explicit set the operations
       ourselves to test MatProductSymbolic_X_Dense, MatProductNumeric_X_Dense code */
    /* ierr = MatConvert(A,MATSHELL,MAT_INITIAL_MATRIX,&S);CHKERRQ(ierr); */
    ierr = MatCreateShell(PetscObjectComm((PetscObject)v),nloc,nloc,n,n,A,&S);CHKERRQ(ierr);
    ierr = MatShellSetOperation(S,MATOP_MULT,(void(*)(void))MatMult_S);CHKERRQ(ierr);
    ierr = MatShellSetOperation(S,MATOP_MULT_TRANSPOSE,(void(*)(void))MatMultTranspose_S);CHKERRQ(ierr);
    ierr = MatShellSetVecType(S,vtype);CHKERRQ(ierr);
  } else {
    ierr = PetscObjectReference((PetscObject)A);CHKERRQ(ierr);
    S    = A;
  }

  ierr = MatCreateDenseCUDA(PetscObjectComm((PetscObject)v),nloc,PETSC_DECIDE,n,k-l,NULL,&C);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* test MatMatMult */
  ierr = MatProductCreateWithMat(S,B,NULL,C);CHKERRQ(ierr);
  ierr = MatProductSetType(C,MATPRODUCT_AB);CHKERRQ(ierr);
  ierr = MatProductSetFromOptions(C);CHKERRQ(ierr);
  ierr = MatProductSymbolic(C);CHKERRQ(ierr);
  ierr = MatProductNumeric(C);CHKERRQ(ierr);
  ierr = MatMatMultEqual(S,B,C,10,&flg);CHKERRQ(ierr);
  if (!flg) { ierr = PetscPrintf(PETSC_COMM_WORLD,"Error MatMatMult\n");CHKERRQ(ierr); }

  /* test MatTransposeMatMult */
  ierr = MatProductCreateWithMat(S,B,NULL,C);CHKERRQ(ierr);
  ierr = MatProductSetType(C,MATPRODUCT_AtB);CHKERRQ(ierr);
  ierr = MatProductSetFromOptions(C);CHKERRQ(ierr);
  ierr = MatProductSymbolic(C);CHKERRQ(ierr);
  ierr = MatProductNumeric(C);CHKERRQ(ierr);
  ierr = MatTransposeMatMultEqual(S,B,C,10,&flg);CHKERRQ(ierr);
  if (!flg) { ierr = PetscPrintf(PETSC_COMM_WORLD,"Error MatTransposeMatMult\n");CHKERRQ(ierr); }

  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = MatDestroy(&S);CHKERRQ(ierr);

  /* finished using B */
  ierr = MatDenseCUDAGetArray(B,&aa);CHKERRQ(ierr);
  PetscCheckFalse(vv != aa-l*nloc,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Wrong array");
  ierr = MatDenseCUDARestoreArray(B,&aa);CHKERRQ(ierr);
  if (reset) {
    ierr = MatDenseCUDAResetArray(B);CHKERRQ(ierr);
  }
  ierr = VecCUDARestoreArray(v,&vv);CHKERRQ(ierr);

  if (test == 1) {
    ierr = MatDenseCUDAGetArray(B,&aa);CHKERRQ(ierr);
    PetscCheckFalse(aa,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Expected a null pointer");
    ierr = MatDenseCUDARestoreArray(B,&aa);CHKERRQ(ierr);
  }

  /* free work space */
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&t);CHKERRQ(ierr);
  ierr = VecDestroy(&v);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  build:
    requires: cuda

  test:
    requires: cuda
    suffix: 1
    nsize: {{1 2}}
    args: -A_mat_type {{aij aijcusparse}} -test {{0 1 2}} -k 6 -l {{0 5}} -use_shell {{0 1}}

TEST*/
