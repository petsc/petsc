static char help[] = "Tests MatCreateDenseCUDA(), MatDenseCUDAPlaceArray(), MatDenseCUDAReplaceArray(), MatDenseCUDAResetArray()\n";

#include <petscmat.h>

static PetscErrorCode MatMult_S(Mat S,Vec x,Vec y)
{
  Mat A;

  PetscFunctionBeginUser;
  CHKERRQ(MatShellGetContext(S,&A));
  CHKERRQ(MatMult(A,x,y));
  PetscFunctionReturn(0);
}

static PetscBool test_cusparse_transgen = PETSC_FALSE;

static PetscErrorCode MatMultTranspose_S(Mat S,Vec x,Vec y)
{
  Mat A;

  PetscFunctionBeginUser;
  CHKERRQ(MatShellGetContext(S,&A));
  CHKERRQ(MatMultTranspose(A,x,y));

  /* alternate transgen true and false to test code logic */
  CHKERRQ(MatSetOption(A,MAT_FORM_EXPLICIT_TRANSPOSE,test_cusparse_transgen));
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
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-k",&k,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-l",&l,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-test",&test,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-use_shell",&use_shell,NULL));
  PetscCheckFalse(k < 0,PETSC_COMM_WORLD,PETSC_ERR_USER,"k %" PetscInt_FMT " must be positive",k);
  PetscCheckFalse(l < 0,PETSC_COMM_WORLD,PETSC_ERR_USER,"l %" PetscInt_FMT " must be positive",l);
  PetscCheckFalse(l > k,PETSC_COMM_WORLD,PETSC_ERR_USER,"l %" PetscInt_FMT " must be smaller or equal than k %" PetscInt_FMT,l,k);

  /* sparse matrix */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetType(A,MATAIJCUSPARSE));
  CHKERRQ(MatSetOptionsPrefix(A,"A_"));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));

  /* test special case for SeqAIJCUSPARSE to generate explicit transpose (not default) */
  CHKERRQ(MatSetOption(A,MAT_FORM_EXPLICIT_TRANSPOSE,PETSC_TRUE));

  CHKERRQ(MatGetOwnershipRange(A,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    if (i>0) CHKERRQ(MatSetValue(A,i,i-1,-1.0,INSERT_VALUES));
    if (i<n-1) CHKERRQ(MatSetValue(A,i,i+1,-1.0,INSERT_VALUES));
    CHKERRQ(MatSetValue(A,i,i,2.0,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* template vector */
  CHKERRQ(MatCreateVecs(A,NULL,&t));
  CHKERRQ(VecGetType(t,&vtype));

  /* long vector, contains the stacked columns of an nxk dense matrix */
  CHKERRQ(VecGetLocalSize(t,&nloc));
  CHKERRQ(VecGetBlockSize(t,&bs));
  CHKERRQ(VecCreate(PetscObjectComm((PetscObject)t),&v));
  CHKERRQ(VecSetType(v,vtype));
  CHKERRQ(VecSetSizes(v,k*nloc,k*n));
  CHKERRQ(VecSetBlockSize(v,bs));
  CHKERRQ(VecSetRandom(v,NULL));

  /* dense matrix that contains the columns of v */
  CHKERRQ(VecCUDAGetArray(v,&vv));

  /* test few cases for MatDenseCUDA handling pointers */
  switch (test) {
  case 1:
    CHKERRQ(MatCreateDenseCUDA(PetscObjectComm((PetscObject)v),nloc,PETSC_DECIDE,n,k-l,vv,&B)); /* pass a pointer to avoid allocation of storage */
    CHKERRQ(MatDenseCUDAReplaceArray(B,NULL));  /* replace with a null pointer, the value after BVRestoreMat */
    CHKERRQ(MatDenseCUDAPlaceArray(B,vv+l*nloc));  /* set the actual pointer */
    reset = PETSC_TRUE;
    break;
  case 2:
    CHKERRQ(MatCreateDenseCUDA(PetscObjectComm((PetscObject)v),nloc,PETSC_DECIDE,n,k-l,NULL,&B));
    CHKERRQ(MatDenseCUDAPlaceArray(B,vv+l*nloc));  /* set the actual pointer */
    reset = PETSC_TRUE;
    break;
  default:
    CHKERRQ(MatCreateDenseCUDA(PetscObjectComm((PetscObject)v),nloc,PETSC_DECIDE,n,k-l,vv+l*nloc,&B));
    reset = PETSC_FALSE;
    break;
  }
  CHKERRQ(VecCUDARestoreArray(v,&vv));

  /* Test MatMatMult */
  if (use_shell) {
    /* we could have called the general convertor below, but we explicit set the operations
       ourselves to test MatProductSymbolic_X_Dense, MatProductNumeric_X_Dense code */
    /* CHKERRQ(MatConvert(A,MATSHELL,MAT_INITIAL_MATRIX,&S)); */
    CHKERRQ(MatCreateShell(PetscObjectComm((PetscObject)v),nloc,nloc,n,n,A,&S));
    CHKERRQ(MatShellSetOperation(S,MATOP_MULT,(void(*)(void))MatMult_S));
    CHKERRQ(MatShellSetOperation(S,MATOP_MULT_TRANSPOSE,(void(*)(void))MatMultTranspose_S));
    CHKERRQ(MatShellSetVecType(S,vtype));
  } else {
    CHKERRQ(PetscObjectReference((PetscObject)A));
    S    = A;
  }

  CHKERRQ(MatCreateDenseCUDA(PetscObjectComm((PetscObject)v),nloc,PETSC_DECIDE,n,k-l,NULL,&C));
  CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));

  /* test MatMatMult */
  CHKERRQ(MatProductCreateWithMat(S,B,NULL,C));
  CHKERRQ(MatProductSetType(C,MATPRODUCT_AB));
  CHKERRQ(MatProductSetFromOptions(C));
  CHKERRQ(MatProductSymbolic(C));
  CHKERRQ(MatProductNumeric(C));
  CHKERRQ(MatMatMultEqual(S,B,C,10,&flg));
  if (!flg) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Error MatMatMult\n"));

  /* test MatTransposeMatMult */
  CHKERRQ(MatProductCreateWithMat(S,B,NULL,C));
  CHKERRQ(MatProductSetType(C,MATPRODUCT_AtB));
  CHKERRQ(MatProductSetFromOptions(C));
  CHKERRQ(MatProductSymbolic(C));
  CHKERRQ(MatProductNumeric(C));
  CHKERRQ(MatTransposeMatMultEqual(S,B,C,10,&flg));
  if (!flg) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Error MatTransposeMatMult\n"));

  CHKERRQ(MatDestroy(&C));
  CHKERRQ(MatDestroy(&S));

  /* finished using B */
  CHKERRQ(MatDenseCUDAGetArray(B,&aa));
  PetscCheckFalse(vv != aa-l*nloc,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Wrong array");
  CHKERRQ(MatDenseCUDARestoreArray(B,&aa));
  if (reset) {
    CHKERRQ(MatDenseCUDAResetArray(B));
  }
  CHKERRQ(VecCUDARestoreArray(v,&vv));

  if (test == 1) {
    CHKERRQ(MatDenseCUDAGetArray(B,&aa));
    PetscCheck(!aa,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Expected a null pointer");
    CHKERRQ(MatDenseCUDARestoreArray(B,&aa));
  }

  /* free work space */
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(VecDestroy(&t));
  CHKERRQ(VecDestroy(&v));
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
