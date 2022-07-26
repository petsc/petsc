static char help[] = "Test ViennaCL Matrix Conversions";

#include <petscmat.h>

int main(int argc,char **args)
{
  PetscMPIInt rank,size;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));

  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  /* Construct a sequential ViennaCL matrix and vector */
  if (rank == 0) {
    Mat A_vcl;
    Vec v_vcl,r_vcl;
    PetscInt n = 17, m = 31,nz = 5,i,cnt,j;
    const PetscScalar val = 1.0;

    PetscCall(MatCreateSeqAIJViennaCL(PETSC_COMM_SELF,m,n,nz,NULL,&A_vcl));

    /* Add nz arbitrary entries per row in arbitrary columns */
    for (i=0;i<m;++i) {
      for (cnt = 0; cnt<nz; ++cnt) {
        j = (19 * cnt + (7*i + 3)) % n;
        PetscCall(MatSetValue(A_vcl,i,j,(PetscScalar)(0.3 * i + j),INSERT_VALUES));
      }
    }
    PetscCall(MatAssemblyBegin(A_vcl,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A_vcl,MAT_FINAL_ASSEMBLY));

    PetscCall(VecCreateSeqViennaCL(PETSC_COMM_SELF,n,&v_vcl));
    PetscCall(VecCreateSeqViennaCL(PETSC_COMM_SELF,m,&r_vcl));
    PetscCall(VecSet(v_vcl,val));

    PetscCall(MatMult(A_vcl,v_vcl,r_vcl));

    PetscCall(VecDestroy(&v_vcl));
    PetscCall(VecDestroy(&r_vcl));
    PetscCall(MatDestroy(&A_vcl));
  }

  /* Create a sequential AIJ matrix on rank 0 convert it to a new ViennaCL matrix, and apply it to a ViennaCL vector */
  if (rank == 0) {
    Mat               A,A_vcl;
    Vec               v,r,v_vcl,r_vcl,d_vcl;
    PetscInt          n=17,m=31,nz=5,i,cnt,j;
    const PetscScalar val = 1.0;
    PetscReal         dnorm;
    const PetscReal   tol = 1e-5;

    PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF,m,n,nz,NULL,&A));

    /* Add nz arbitrary entries per row in arbitrary columns */
    for (i=0;i<m;++i) {
      for (cnt = 0; cnt<nz; ++cnt) {
        j = (19 * cnt + (7*i + 3)) % n;
        PetscCall(MatSetValue(A,i,j,(PetscScalar) (0.3 * i + j),INSERT_VALUES));
      }
    }
    PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

    PetscCall(PetscObjectSetName((PetscObject)A,"Sequential CPU Matrix"));

    PetscCall(VecCreateSeq(PETSC_COMM_SELF,n,&v));
    PetscCall(VecCreateSeq(PETSC_COMM_SELF,m,&r));
    PetscCall(PetscObjectSetName((PetscObject)r,"CPU result vector"));
    PetscCall(VecSet(v,val));
    PetscCall(MatMult(A,v,r));

    PetscCall(MatConvert(A,MATSEQAIJVIENNACL,MAT_INITIAL_MATRIX,&A_vcl));
    PetscCall(PetscObjectSetName((PetscObject)A_vcl,"New ViennaCL Matrix"));

    PetscCall(VecCreateSeqViennaCL(PETSC_COMM_SELF,n,&v_vcl));
    PetscCall(VecCreateSeqViennaCL(PETSC_COMM_SELF,m,&r_vcl));
    PetscCall(PetscObjectSetName((PetscObject)r_vcl,"ViennaCL result vector"));
    PetscCall(VecSet(v_vcl,val));
    PetscCall(MatMult(A_vcl,v_vcl,r_vcl));

    PetscCall(VecDuplicate(r_vcl,&d_vcl));
    PetscCall(VecCopy(r_vcl,d_vcl));
    PetscCall(VecAXPY(d_vcl,-1.0,r_vcl));
    PetscCall(VecNorm(d_vcl,NORM_INFINITY,&dnorm));
    PetscCheck(dnorm <= tol,PETSC_COMM_WORLD,PETSC_ERR_ARG_INCOMP,"Sequential CPU and MPI ViennaCL vector results incompatible with inf norm greater than tolerance of %g",tol);

    PetscCall(VecDestroy(&v));
    PetscCall(VecDestroy(&r));
    PetscCall(VecDestroy(&v_vcl));
    PetscCall(VecDestroy(&r_vcl));
    PetscCall(VecDestroy(&d_vcl));
    PetscCall(MatDestroy(&A));
    PetscCall(MatDestroy(&A_vcl));
  }

  /* Create a sequential AIJ matrix on rank 0 convert it inplace to a new ViennaCL matrix, and apply it to a ViennaCL vector */
  if (rank == 0) {
    Mat               A;
    Vec               v,r,v_vcl,r_vcl,d_vcl;
    PetscInt          n=17,m=31,nz=5,i,cnt,j;
    const PetscScalar val = 1.0;
    PetscReal         dnorm;
    const PetscReal   tol = 1e-5;

    PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF,m,n,nz,NULL,&A));

    /* Add nz arbitrary entries per row in arbitrary columns */
    for (i=0;i<m;++i) {
      for (cnt = 0; cnt<nz; ++cnt) {
        j = (19 * cnt + (7*i + 3)) % n;
        PetscCall(MatSetValue(A,i,j,(PetscScalar)(0.3 * i + j),INSERT_VALUES));
      }
    }
    PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

    PetscCall(PetscObjectSetName((PetscObject)A,"Sequential CPU Matrix"));

    PetscCall(VecCreateSeq(PETSC_COMM_SELF,n,&v));
    PetscCall(VecCreateSeq(PETSC_COMM_SELF,m,&r));
    PetscCall(PetscObjectSetName((PetscObject)r,"CPU result vector"));
    PetscCall(VecSet(v,val));
    PetscCall(MatMult(A,v,r));

    PetscCall(MatConvert(A,MATSEQAIJVIENNACL,MAT_INPLACE_MATRIX,&A));
    PetscCall(PetscObjectSetName((PetscObject)A,"Converted ViennaCL Matrix"));

    PetscCall(VecCreateSeqViennaCL(PETSC_COMM_SELF,n,&v_vcl));
    PetscCall(VecCreateSeqViennaCL(PETSC_COMM_SELF,m,&r_vcl));
    PetscCall(PetscObjectSetName((PetscObject)r_vcl,"ViennaCL result vector"));
    PetscCall(VecSet(v_vcl,val));
    PetscCall(MatMult(A,v_vcl,r_vcl));

    PetscCall(VecDuplicate(r_vcl,&d_vcl));
    PetscCall(VecCopy(r_vcl,d_vcl));
    PetscCall(VecAXPY(d_vcl,-1.0,r_vcl));
    PetscCall(VecNorm(d_vcl,NORM_INFINITY,&dnorm));
    PetscCheck(dnorm <= tol,PETSC_COMM_WORLD,PETSC_ERR_ARG_INCOMP,"MPI CPU and MPI ViennaCL Vector results incompatible with inf norm greater than tolerance of %g",tol);

    PetscCall(VecDestroy(&v));
    PetscCall(VecDestroy(&r));
    PetscCall(VecDestroy(&v_vcl));
    PetscCall(VecDestroy(&r_vcl));
    PetscCall(VecDestroy(&d_vcl));
    PetscCall(MatDestroy(&A));
  }

  /* Create a parallel AIJ matrix, convert it to a new ViennaCL matrix, and apply it to a ViennaCL vector */
  if (size > 1) {
    Mat               A,A_vcl;
    Vec               v,r,v_vcl,r_vcl,d_vcl;
    PetscInt          N=17,M=31,nz=5,i,cnt,j,rlow,rhigh;
    const PetscScalar val = 1.0;
    PetscReal         dnorm;
    const PetscReal   tol=1e-5;

    PetscCall(MatCreateAIJ(PETSC_COMM_WORLD,PETSC_DETERMINE,PETSC_DETERMINE,M,N,nz,NULL,nz,NULL,&A));

    /* Add nz arbitrary entries per row in arbitrary columns */
    PetscCall(MatGetOwnershipRange(A,&rlow,&rhigh));
    for (i=rlow;i<rhigh;++i) {
      for (cnt = 0; cnt<nz; ++cnt) {
        j = (19 * cnt + (7*i + 3)) % N;
        PetscCall(MatSetValue(A,i,j,(PetscScalar)(0.3 * i + j),INSERT_VALUES));
      }
    }
    PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

    PetscCall(PetscObjectSetName((PetscObject)A,"MPI CPU Matrix"));

    PetscCall(MatCreateVecs(A,&v,&r));
    PetscCall(PetscObjectSetName((PetscObject)r,"MPI CPU result vector"));
    PetscCall(VecSet(v,val));
    PetscCall(MatMult(A,v,r));

    PetscCall(MatConvert(A,MATMPIAIJVIENNACL,MAT_INITIAL_MATRIX,&A_vcl));
    PetscCall(PetscObjectSetName((PetscObject)A_vcl,"New MPI ViennaCL Matrix"));

    PetscCall(MatCreateVecs(A_vcl,&v_vcl,&r_vcl));
    PetscCall(PetscObjectSetName((PetscObject)r_vcl,"ViennaCL result vector"));
    PetscCall(VecSet(v_vcl,val));
    PetscCall(MatMult(A_vcl,v_vcl,r_vcl));

    PetscCall(VecDuplicate(r_vcl,&d_vcl));
    PetscCall(VecCopy(r_vcl,d_vcl));
    PetscCall(VecAXPY(d_vcl,-1.0,r_vcl));
    PetscCall(VecNorm(d_vcl,NORM_INFINITY,&dnorm));
    PetscCheck(dnorm <= tol,PETSC_COMM_WORLD,PETSC_ERR_ARG_INCOMP,"MPI CPU and MPI ViennaCL Vector results incompatible with inf norm greater than tolerance of %g",tol);

    PetscCall(VecDestroy(&v));
    PetscCall(VecDestroy(&r));
    PetscCall(VecDestroy(&v_vcl));
    PetscCall(VecDestroy(&r_vcl));
    PetscCall(VecDestroy(&d_vcl));
    PetscCall(MatDestroy(&A));
    PetscCall(MatDestroy(&A_vcl));
  }

  /* Create a parallel AIJ matrix, convert it in-place to a ViennaCL matrix, and apply it to a ViennaCL vector */
  if (size > 1) {
    Mat               A;
    Vec               v,r,v_vcl,r_vcl,d_vcl;
    PetscInt          N=17,M=31,nz=5,i,cnt,j,rlow,rhigh;
    const PetscScalar val = 1.0;
    PetscReal         dnorm;
    const PetscReal   tol=1e-5;

    PetscCall(MatCreateAIJ(PETSC_COMM_WORLD,PETSC_DETERMINE,PETSC_DETERMINE,M,N,nz,NULL,nz,NULL,&A));

    /* Add nz arbitrary entries per row in arbitrary columns */
    PetscCall(MatGetOwnershipRange(A,&rlow,&rhigh));
    for (i=rlow;i<rhigh;++i) {
      for (cnt = 0; cnt<nz; ++cnt) {
        j = (19 * cnt + (7*i + 3)) % N;
        PetscCall(MatSetValue(A,i,j,(PetscScalar)(0.3 * i + j),INSERT_VALUES));
      }
    }
    PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

    PetscCall(PetscObjectSetName((PetscObject)A,"MPI CPU Matrix"));

    PetscCall(MatCreateVecs(A,&v,&r));
    PetscCall(PetscObjectSetName((PetscObject)r,"MPI CPU result vector"));
    PetscCall(VecSet(v,val));
    PetscCall(MatMult(A,v,r));

    PetscCall(MatConvert(A,MATMPIAIJVIENNACL,MAT_INPLACE_MATRIX,&A));
    PetscCall(PetscObjectSetName((PetscObject)A,"Converted MPI ViennaCL Matrix"));

    PetscCall(MatCreateVecs(A,&v_vcl,&r_vcl));
    PetscCall(PetscObjectSetName((PetscObject)r_vcl,"ViennaCL result vector"));
    PetscCall(VecSet(v_vcl,val));
    PetscCall(MatMult(A,v_vcl,r_vcl));

    PetscCall(VecDuplicate(r_vcl,&d_vcl));
    PetscCall(VecCopy(r_vcl,d_vcl));
    PetscCall(VecAXPY(d_vcl,-1.0,r_vcl));
    PetscCall(VecNorm(d_vcl,NORM_INFINITY,&dnorm));
    PetscCheck(dnorm <= tol,PETSC_COMM_WORLD,PETSC_ERR_ARG_INCOMP,"MPI CPU and MPI ViennaCL Vector results incompatible with inf norm greater than tolerance of %g",tol);

    PetscCall(VecDestroy(&v));
    PetscCall(VecDestroy(&r));
    PetscCall(VecDestroy(&v_vcl));
    PetscCall(VecDestroy(&r_vcl));
    PetscCall(VecDestroy(&d_vcl));
    PetscCall(MatDestroy(&A));
  }

  PetscCall(PetscFinalize());
  return 0;

}

/*TEST

   build:
      requires: viennacl

   test:
      output_file: output/ex204.out

   test:
      suffix: 2
      nsize: 2
      output_file: output/ex204.out

TEST*/
