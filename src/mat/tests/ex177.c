
static char help[] = "Tests various routines in MatKAIJ format.\n";

#include <petscmat.h>
#define IMAX 15

int main(int argc,char **args)
{
  Mat            A,B,TA;
  PetscScalar    *S,*T;
  PetscViewer    fd;
  char           file[PETSC_MAX_PATH_LEN];
  PetscInt       m,n,M,N,p=1,q=1,i,j;
  PetscMPIInt    rank,size;
  PetscBool      flg;

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  /* Load AIJ matrix A */
  PetscCall(PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_USER,"Must indicate binary file with the -f option");
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd));
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatLoad(A,fd));
  PetscCall(PetscViewerDestroy(&fd));

  /* Get dof, then create S and T */
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-p",&p,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-q",&q,NULL));
  PetscCall(PetscMalloc2(p*q,&S,p*q,&T));
  for (i=0; i<p*q; i++) S[i] = 0;

  for (i=0; i<p; i++) {
    for (j=0; j<q; j++) {
      /* Set some random non-zero values */
      S[i+p*j] = ((PetscReal) ((i+1)*(j+1))) / ((PetscReal) (p+q));
      T[i+p*j] = ((PetscReal) ((p-i)+j)) / ((PetscReal) (p*q));
    }
  }

  /* Test KAIJ when both S & T are not NULL */

  /* Create KAIJ matrix TA */
  PetscCall(MatCreateKAIJ(A,p,q,S,T,&TA));
  PetscCall(MatGetLocalSize(A,&m,&n));
  PetscCall(MatGetSize(A,&M,&N));

  PetscCall(MatConvert(TA,MATAIJ,MAT_INITIAL_MATRIX,&B));

  /* Test MatKAIJGetScaledIdentity() */
  PetscCall(MatKAIJGetScaledIdentity(TA,&flg));
  PetscCheck(!flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Error in Test 1: MatKAIJGetScaledIdentity()");
  /* Test MatMult() */
  PetscCall(MatMultEqual(TA,B,10,&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_CONV_FAILED,"Error in Test 1: MatMult() for KAIJ matrix");
  /* Test MatMultAdd() */
  PetscCall(MatMultAddEqual(TA,B,10,&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_CONV_FAILED,"Error in Test 1: MatMultAdd() for KAIJ matrix");

  PetscCall(MatDestroy(&TA));
  PetscCall(MatDestroy(&B));

  /* Test KAIJ when S is NULL */

  /* Create KAIJ matrix TA */
  PetscCall(MatCreateKAIJ(A,p,q,NULL,T,&TA));
  PetscCall(MatGetLocalSize(A,&m,&n));
  PetscCall(MatGetSize(A,&M,&N));

  PetscCall(MatConvert(TA,MATAIJ,MAT_INITIAL_MATRIX,&B));

  /* Test MatKAIJGetScaledIdentity() */
  PetscCall(MatKAIJGetScaledIdentity(TA,&flg));
  PetscCheck(!flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Error in Test 2: MatKAIJGetScaledIdentity()");
  /* Test MatMult() */
  PetscCall(MatMultEqual(TA,B,10,&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_CONV_FAILED,"Error in Test 2: MatMult() for KAIJ matrix");
  /* Test MatMultAdd() */
  PetscCall(MatMultAddEqual(TA,B,10,&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_CONV_FAILED,"Error in Test 2: MatMultAdd() for KAIJ matrix");

  PetscCall(MatDestroy(&TA));
  PetscCall(MatDestroy(&B));

  /* Test KAIJ when T is NULL */

  /* Create KAIJ matrix TA */
  PetscCall(MatCreateKAIJ(A,p,q,S,NULL,&TA));
  PetscCall(MatGetLocalSize(A,&m,&n));
  PetscCall(MatGetSize(A,&M,&N));

  PetscCall(MatConvert(TA,MATAIJ,MAT_INITIAL_MATRIX,&B));

  /* Test MatKAIJGetScaledIdentity() */
  PetscCall(MatKAIJGetScaledIdentity(TA,&flg));
  PetscCheck(!flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Error in Test 3: MatKAIJGetScaledIdentity()");
  /* Test MatMult() */
  PetscCall(MatMultEqual(TA,B,10,&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_CONV_FAILED,"Error in Test 3: MatMult() for KAIJ matrix");
  /* Test MatMultAdd() */
  PetscCall(MatMultAddEqual(TA,B,10,&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_CONV_FAILED,"Error in Test 3: MatMultAdd() for KAIJ matrix");

  PetscCall(MatDestroy(&TA));
  PetscCall(MatDestroy(&B));

  /* Test KAIJ when T is is an identity matrix */

  if (p == q) {
    for (i=0; i<p; i++) {
      for (j=0; j<q; j++) {
        if (i==j) T[i+j*p] = 1.0;
        else      T[i+j*p] = 0.0;
      }
    }

    /* Create KAIJ matrix TA */
    PetscCall(MatCreateKAIJ(A,p,q,S,T,&TA));
    PetscCall(MatGetLocalSize(A,&m,&n));
    PetscCall(MatGetSize(A,&M,&N));

    PetscCall(MatConvert(TA,MATAIJ,MAT_INITIAL_MATRIX,&B));

    /* Test MatKAIJGetScaledIdentity() */
    PetscCall(MatKAIJGetScaledIdentity(TA,&flg));
    PetscCheck(!flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Error in Test 4: MatKAIJGetScaledIdentity()");
    /* Test MatMult() */
    PetscCall(MatMultEqual(TA,B,10,&flg));
    PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_CONV_FAILED,"Error in Test 4: MatMult() for KAIJ matrix");
    /* Test MatMultAdd() */
    PetscCall(MatMultAddEqual(TA,B,10,&flg));
    PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_CONV_FAILED,"Error in Test 4: MatMultAdd() for KAIJ matrix");

    PetscCall(MatDestroy(&TA));
    PetscCall(MatDestroy(&B));

    PetscCall(MatCreateKAIJ(A,p,q,NULL,T,&TA));
    /* Test MatKAIJGetScaledIdentity() */
    PetscCall(MatKAIJGetScaledIdentity(TA,&flg));
    PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Error in Test 5: MatKAIJGetScaledIdentity()");
    PetscCall(MatDestroy(&TA));

    for (i=0; i<p; i++) {
      for (j=0; j<q; j++) {
        if (i==j) S[i+j*p] = T[i+j*p] = 2.0;
        else      S[i+j*p] = T[i+j*p] = 0.0;
      }
    }
    PetscCall(MatCreateKAIJ(A,p,q,S,T,&TA));
    /* Test MatKAIJGetScaledIdentity() */
    PetscCall(MatKAIJGetScaledIdentity(TA,&flg));
    PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Error in Test 6: MatKAIJGetScaledIdentity()");
    PetscCall(MatDestroy(&TA));
  }

  /* Done with all tests */

  PetscCall(PetscFree2(S,T));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  build:
    requires: !complex

  test:
    requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
    output_file: output/ex176.out
    nsize: {{1 2 3 4}}
    args: -f ${DATAFILESPATH}/matrices/small -p {{2 3 7}} -q {{3 7}} -viewer_binary_skip_info

TEST*/
