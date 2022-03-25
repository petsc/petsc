static char help[] = "Tests MatCreateDenseCUDA(), MatDenseCUDAPlaceArray(), MatDenseCUDAReplaceArray(), MatDenseCUDAResetArray()\n";

#include <petscmat.h>

static PetscErrorCode MatMult_S(Mat S,Vec x,Vec y)
{
  Mat A;

  PetscFunctionBeginUser;
  PetscCall(MatShellGetContext(S,&A));
  PetscCall(MatMult(A,x,y));
  PetscFunctionReturn(0);
}

static PetscBool test_cusparse_transgen = PETSC_FALSE;

static PetscErrorCode MatMultTranspose_S(Mat S,Vec x,Vec y)
{
  Mat A;

  PetscFunctionBeginUser;
  PetscCall(MatShellGetContext(S,&A));
  PetscCall(MatMultTranspose(A,x,y));

  /* alternate transgen true and false to test code logic */
  PetscCall(MatSetOption(A,MAT_FORM_EXPLICIT_TRANSPOSE,test_cusparse_transgen));
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
  VecType        vtype;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-k",&k,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-l",&l,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-test",&test,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-use_shell",&use_shell,NULL));
  PetscCheckFalse(k < 0,PETSC_COMM_WORLD,PETSC_ERR_USER,"k %" PetscInt_FMT " must be positive",k);
  PetscCheckFalse(l < 0,PETSC_COMM_WORLD,PETSC_ERR_USER,"l %" PetscInt_FMT " must be positive",l);
  PetscCheckFalse(l > k,PETSC_COMM_WORLD,PETSC_ERR_USER,"l %" PetscInt_FMT " must be smaller or equal than k %" PetscInt_FMT,l,k);

  /* sparse matrix */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n));
  PetscCall(MatSetType(A,MATAIJCUSPARSE));
  PetscCall(MatSetOptionsPrefix(A,"A_"));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));

  /* test special case for SeqAIJCUSPARSE to generate explicit transpose (not default) */
  PetscCall(MatSetOption(A,MAT_FORM_EXPLICIT_TRANSPOSE,PETSC_TRUE));

  PetscCall(MatGetOwnershipRange(A,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    if (i>0) PetscCall(MatSetValue(A,i,i-1,-1.0,INSERT_VALUES));
    if (i<n-1) PetscCall(MatSetValue(A,i,i+1,-1.0,INSERT_VALUES));
    PetscCall(MatSetValue(A,i,i,2.0,INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* template vector */
  PetscCall(MatCreateVecs(A,NULL,&t));
  PetscCall(VecGetType(t,&vtype));

  /* long vector, contains the stacked columns of an nxk dense matrix */
  PetscCall(VecGetLocalSize(t,&nloc));
  PetscCall(VecGetBlockSize(t,&bs));
  PetscCall(VecCreate(PetscObjectComm((PetscObject)t),&v));
  PetscCall(VecSetType(v,vtype));
  PetscCall(VecSetSizes(v,k*nloc,k*n));
  PetscCall(VecSetBlockSize(v,bs));
  PetscCall(VecSetRandom(v,NULL));

  /* dense matrix that contains the columns of v */
  PetscCall(VecCUDAGetArray(v,&vv));

  /* test few cases for MatDenseCUDA handling pointers */
  switch (test) {
  case 1:
    PetscCall(MatCreateDenseCUDA(PetscObjectComm((PetscObject)v),nloc,PETSC_DECIDE,n,k-l,vv,&B)); /* pass a pointer to avoid allocation of storage */
    PetscCall(MatDenseCUDAReplaceArray(B,NULL));  /* replace with a null pointer, the value after BVRestoreMat */
    PetscCall(MatDenseCUDAPlaceArray(B,vv+l*nloc));  /* set the actual pointer */
    reset = PETSC_TRUE;
    break;
  case 2:
    PetscCall(MatCreateDenseCUDA(PetscObjectComm((PetscObject)v),nloc,PETSC_DECIDE,n,k-l,NULL,&B));
    PetscCall(MatDenseCUDAPlaceArray(B,vv+l*nloc));  /* set the actual pointer */
    reset = PETSC_TRUE;
    break;
  default:
    PetscCall(MatCreateDenseCUDA(PetscObjectComm((PetscObject)v),nloc,PETSC_DECIDE,n,k-l,vv+l*nloc,&B));
    reset = PETSC_FALSE;
    break;
  }
  PetscCall(VecCUDARestoreArray(v,&vv));

  /* Test MatMatMult */
  if (use_shell) {
    /* we could have called the general convertor below, but we explicit set the operations
       ourselves to test MatProductSymbolic_X_Dense, MatProductNumeric_X_Dense code */
    /* PetscCall(MatConvert(A,MATSHELL,MAT_INITIAL_MATRIX,&S)); */
    PetscCall(MatCreateShell(PetscObjectComm((PetscObject)v),nloc,nloc,n,n,A,&S));
    PetscCall(MatShellSetOperation(S,MATOP_MULT,(void(*)(void))MatMult_S));
    PetscCall(MatShellSetOperation(S,MATOP_MULT_TRANSPOSE,(void(*)(void))MatMultTranspose_S));
    PetscCall(MatShellSetVecType(S,vtype));
  } else {
    PetscCall(PetscObjectReference((PetscObject)A));
    S    = A;
  }

  PetscCall(MatCreateDenseCUDA(PetscObjectComm((PetscObject)v),nloc,PETSC_DECIDE,n,k-l,NULL,&C));
  PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));

  /* test MatMatMult */
  PetscCall(MatProductCreateWithMat(S,B,NULL,C));
  PetscCall(MatProductSetType(C,MATPRODUCT_AB));
  PetscCall(MatProductSetFromOptions(C));
  PetscCall(MatProductSymbolic(C));
  PetscCall(MatProductNumeric(C));
  PetscCall(MatMatMultEqual(S,B,C,10,&flg));
  if (!flg) PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Error MatMatMult\n"));

  /* test MatTransposeMatMult */
  PetscCall(MatProductCreateWithMat(S,B,NULL,C));
  PetscCall(MatProductSetType(C,MATPRODUCT_AtB));
  PetscCall(MatProductSetFromOptions(C));
  PetscCall(MatProductSymbolic(C));
  PetscCall(MatProductNumeric(C));
  PetscCall(MatTransposeMatMultEqual(S,B,C,10,&flg));
  if (!flg) PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Error MatTransposeMatMult\n"));

  PetscCall(MatDestroy(&C));
  PetscCall(MatDestroy(&S));

  /* finished using B */
  PetscCall(MatDenseCUDAGetArray(B,&aa));
  PetscCheckFalse(vv != aa-l*nloc,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Wrong array");
  PetscCall(MatDenseCUDARestoreArray(B,&aa));
  if (reset) {
    PetscCall(MatDenseCUDAResetArray(B));
  }
  PetscCall(VecCUDARestoreArray(v,&vv));

  if (test == 1) {
    PetscCall(MatDenseCUDAGetArray(B,&aa));
    PetscCheck(!aa,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Expected a null pointer");
    PetscCall(MatDenseCUDARestoreArray(B,&aa));
  }

  /* free work space */
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&t));
  PetscCall(VecDestroy(&v));
  PetscCall(PetscFinalize());
  return 0;
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
