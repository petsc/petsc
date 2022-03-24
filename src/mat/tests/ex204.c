static char help[] = "Test ViennaCL Matrix Conversions";

#include <petscmat.h>

int main(int argc,char **args)
{
  PetscMPIInt rank,size;

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));

  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  /* Construct a sequential ViennaCL matrix and vector */
  if (rank == 0) {
    Mat A_vcl;
    Vec v_vcl,r_vcl;
    PetscInt n = 17, m = 31,nz = 5,i,cnt,j;
    const PetscScalar val = 1.0;

    CHKERRQ(MatCreateSeqAIJViennaCL(PETSC_COMM_SELF,m,n,nz,NULL,&A_vcl));

    /* Add nz arbitrary entries per row in arbitrary columns */
    for (i=0;i<m;++i) {
      for (cnt = 0; cnt<nz; ++cnt) {
        j = (19 * cnt + (7*i + 3)) % n;
        CHKERRQ(MatSetValue(A_vcl,i,j,(PetscScalar)(0.3 * i + j),INSERT_VALUES));
      }
    }
    CHKERRQ(MatAssemblyBegin(A_vcl,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(A_vcl,MAT_FINAL_ASSEMBLY));

    CHKERRQ(VecCreateSeqViennaCL(PETSC_COMM_SELF,n,&v_vcl));
    CHKERRQ(VecCreateSeqViennaCL(PETSC_COMM_SELF,m,&r_vcl));
    CHKERRQ(VecSet(v_vcl,val));

    CHKERRQ(MatMult(A_vcl,v_vcl,r_vcl));

    CHKERRQ(VecDestroy(&v_vcl));
    CHKERRQ(VecDestroy(&r_vcl));
    CHKERRQ(MatDestroy(&A_vcl));
  }

  /* Create a sequential AIJ matrix on rank 0 convert it to a new ViennaCL matrix, and apply it to a ViennaCL vector */
  if (rank == 0) {
    Mat               A,A_vcl;
    Vec               v,r,v_vcl,r_vcl,d_vcl;
    PetscInt          n=17,m=31,nz=5,i,cnt,j;
    const PetscScalar val = 1.0;
    PetscReal         dnorm;
    const PetscReal   tol = 1e-5;

    CHKERRQ(MatCreateSeqAIJ(PETSC_COMM_SELF,m,n,nz,NULL,&A));

    /* Add nz arbitrary entries per row in arbitrary columns */
    for (i=0;i<m;++i) {
      for (cnt = 0; cnt<nz; ++cnt) {
        j = (19 * cnt + (7*i + 3)) % n;
        CHKERRQ(MatSetValue(A,i,j,(PetscScalar) (0.3 * i + j),INSERT_VALUES));
      }
    }
    CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

    CHKERRQ(PetscObjectSetName((PetscObject)A,"Sequential CPU Matrix"));

    CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,n,&v));
    CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,m,&r));
    CHKERRQ(PetscObjectSetName((PetscObject)r,"CPU result vector"));
    CHKERRQ(VecSet(v,val));
    CHKERRQ(MatMult(A,v,r));

    CHKERRQ(MatConvert(A,MATSEQAIJVIENNACL,MAT_INITIAL_MATRIX,&A_vcl));
    CHKERRQ(PetscObjectSetName((PetscObject)A_vcl,"New ViennaCL Matrix"));

    CHKERRQ(VecCreateSeqViennaCL(PETSC_COMM_SELF,n,&v_vcl));
    CHKERRQ(VecCreateSeqViennaCL(PETSC_COMM_SELF,m,&r_vcl));
    CHKERRQ(PetscObjectSetName((PetscObject)r_vcl,"ViennaCL result vector"));
    CHKERRQ(VecSet(v_vcl,val));
    CHKERRQ(MatMult(A_vcl,v_vcl,r_vcl));

    CHKERRQ(VecDuplicate(r_vcl,&d_vcl));
    CHKERRQ(VecCopy(r_vcl,d_vcl));
    CHKERRQ(VecAXPY(d_vcl,-1.0,r_vcl));
    CHKERRQ(VecNorm(d_vcl,NORM_INFINITY,&dnorm));
    PetscCheckFalse(dnorm > tol,PETSC_COMM_WORLD,PETSC_ERR_ARG_INCOMP,"Sequential CPU and MPI ViennaCL vector results incompatible with inf norm greater than tolerance of %g",tol);

    CHKERRQ(VecDestroy(&v));
    CHKERRQ(VecDestroy(&r));
    CHKERRQ(VecDestroy(&v_vcl));
    CHKERRQ(VecDestroy(&r_vcl));
    CHKERRQ(VecDestroy(&d_vcl));
    CHKERRQ(MatDestroy(&A));
    CHKERRQ(MatDestroy(&A_vcl));
  }

  /* Create a sequential AIJ matrix on rank 0 convert it inplace to a new ViennaCL matrix, and apply it to a ViennaCL vector */
  if (rank == 0) {
    Mat               A;
    Vec               v,r,v_vcl,r_vcl,d_vcl;
    PetscInt          n=17,m=31,nz=5,i,cnt,j;
    const PetscScalar val = 1.0;
    PetscReal         dnorm;
    const PetscReal   tol = 1e-5;

    CHKERRQ(MatCreateSeqAIJ(PETSC_COMM_SELF,m,n,nz,NULL,&A));

    /* Add nz arbitrary entries per row in arbitrary columns */
    for (i=0;i<m;++i) {
      for (cnt = 0; cnt<nz; ++cnt) {
        j = (19 * cnt + (7*i + 3)) % n;
        CHKERRQ(MatSetValue(A,i,j,(PetscScalar)(0.3 * i + j),INSERT_VALUES));
      }
    }
    CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

    CHKERRQ(PetscObjectSetName((PetscObject)A,"Sequential CPU Matrix"));

    CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,n,&v));
    CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,m,&r));
    CHKERRQ(PetscObjectSetName((PetscObject)r,"CPU result vector"));
    CHKERRQ(VecSet(v,val));
    CHKERRQ(MatMult(A,v,r));

    CHKERRQ(MatConvert(A,MATSEQAIJVIENNACL,MAT_INPLACE_MATRIX,&A));
    CHKERRQ(PetscObjectSetName((PetscObject)A,"Converted ViennaCL Matrix"));

    CHKERRQ(VecCreateSeqViennaCL(PETSC_COMM_SELF,n,&v_vcl));
    CHKERRQ(VecCreateSeqViennaCL(PETSC_COMM_SELF,m,&r_vcl));
    CHKERRQ(PetscObjectSetName((PetscObject)r_vcl,"ViennaCL result vector"));
    CHKERRQ(VecSet(v_vcl,val));
    CHKERRQ(MatMult(A,v_vcl,r_vcl));

    CHKERRQ(VecDuplicate(r_vcl,&d_vcl));
    CHKERRQ(VecCopy(r_vcl,d_vcl));
    CHKERRQ(VecAXPY(d_vcl,-1.0,r_vcl));
    CHKERRQ(VecNorm(d_vcl,NORM_INFINITY,&dnorm));
    PetscCheckFalse(dnorm > tol,PETSC_COMM_WORLD,PETSC_ERR_ARG_INCOMP,"MPI CPU and MPI ViennaCL Vector results incompatible with inf norm greater than tolerance of %g",tol);

    CHKERRQ(VecDestroy(&v));
    CHKERRQ(VecDestroy(&r));
    CHKERRQ(VecDestroy(&v_vcl));
    CHKERRQ(VecDestroy(&r_vcl));
    CHKERRQ(VecDestroy(&d_vcl));
    CHKERRQ(MatDestroy(&A));
  }

  /* Create a parallel AIJ matrix, convert it to a new ViennaCL matrix, and apply it to a ViennaCL vector */
  if (size > 1) {
    Mat               A,A_vcl;
    Vec               v,r,v_vcl,r_vcl,d_vcl;
    PetscInt          N=17,M=31,nz=5,i,cnt,j,rlow,rhigh;
    const PetscScalar val = 1.0;
    PetscReal         dnorm;
    const PetscReal   tol=1e-5;

    CHKERRQ(MatCreateAIJ(PETSC_COMM_WORLD,PETSC_DETERMINE,PETSC_DETERMINE,M,N,nz,NULL,nz,NULL,&A));

    /* Add nz arbitrary entries per row in arbitrary columns */
    CHKERRQ(MatGetOwnershipRange(A,&rlow,&rhigh));
    for (i=rlow;i<rhigh;++i) {
      for (cnt = 0; cnt<nz; ++cnt) {
        j = (19 * cnt + (7*i + 3)) % N;
        CHKERRQ(MatSetValue(A,i,j,(PetscScalar)(0.3 * i + j),INSERT_VALUES));
      }
    }
    CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

    CHKERRQ(PetscObjectSetName((PetscObject)A,"MPI CPU Matrix"));

    CHKERRQ(MatCreateVecs(A,&v,&r));
    CHKERRQ(PetscObjectSetName((PetscObject)r,"MPI CPU result vector"));
    CHKERRQ(VecSet(v,val));
    CHKERRQ(MatMult(A,v,r));

    CHKERRQ(MatConvert(A,MATMPIAIJVIENNACL,MAT_INITIAL_MATRIX,&A_vcl));
    CHKERRQ(PetscObjectSetName((PetscObject)A_vcl,"New MPI ViennaCL Matrix"));

    CHKERRQ(MatCreateVecs(A_vcl,&v_vcl,&r_vcl));
    CHKERRQ(PetscObjectSetName((PetscObject)r_vcl,"ViennaCL result vector"));
    CHKERRQ(VecSet(v_vcl,val));
    CHKERRQ(MatMult(A_vcl,v_vcl,r_vcl));

    CHKERRQ(VecDuplicate(r_vcl,&d_vcl));
    CHKERRQ(VecCopy(r_vcl,d_vcl));
    CHKERRQ(VecAXPY(d_vcl,-1.0,r_vcl));
    CHKERRQ(VecNorm(d_vcl,NORM_INFINITY,&dnorm));
    PetscCheckFalse(dnorm > tol,PETSC_COMM_WORLD,PETSC_ERR_ARG_INCOMP,"MPI CPU and MPI ViennaCL Vector results incompatible with inf norm greater than tolerance of %g",tol);

    CHKERRQ(VecDestroy(&v));
    CHKERRQ(VecDestroy(&r));
    CHKERRQ(VecDestroy(&v_vcl));
    CHKERRQ(VecDestroy(&r_vcl));
    CHKERRQ(VecDestroy(&d_vcl));
    CHKERRQ(MatDestroy(&A));
    CHKERRQ(MatDestroy(&A_vcl));
  }

  /* Create a parallel AIJ matrix, convert it in-place to a ViennaCL matrix, and apply it to a ViennaCL vector */
  if (size > 1) {
    Mat               A;
    Vec               v,r,v_vcl,r_vcl,d_vcl;
    PetscInt          N=17,M=31,nz=5,i,cnt,j,rlow,rhigh;
    const PetscScalar val = 1.0;
    PetscReal         dnorm;
    const PetscReal   tol=1e-5;

    CHKERRQ(MatCreateAIJ(PETSC_COMM_WORLD,PETSC_DETERMINE,PETSC_DETERMINE,M,N,nz,NULL,nz,NULL,&A));

    /* Add nz arbitrary entries per row in arbitrary columns */
    CHKERRQ(MatGetOwnershipRange(A,&rlow,&rhigh));
    for (i=rlow;i<rhigh;++i) {
      for (cnt = 0; cnt<nz; ++cnt) {
        j = (19 * cnt + (7*i + 3)) % N;
        CHKERRQ(MatSetValue(A,i,j,(PetscScalar)(0.3 * i + j),INSERT_VALUES));
      }
    }
    CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

    CHKERRQ(PetscObjectSetName((PetscObject)A,"MPI CPU Matrix"));

    CHKERRQ(MatCreateVecs(A,&v,&r));
    CHKERRQ(PetscObjectSetName((PetscObject)r,"MPI CPU result vector"));
    CHKERRQ(VecSet(v,val));
    CHKERRQ(MatMult(A,v,r));

    CHKERRQ(MatConvert(A,MATMPIAIJVIENNACL,MAT_INPLACE_MATRIX,&A));
    CHKERRQ(PetscObjectSetName((PetscObject)A,"Converted MPI ViennaCL Matrix"));

    CHKERRQ(MatCreateVecs(A,&v_vcl,&r_vcl));
    CHKERRQ(PetscObjectSetName((PetscObject)r_vcl,"ViennaCL result vector"));
    CHKERRQ(VecSet(v_vcl,val));
    CHKERRQ(MatMult(A,v_vcl,r_vcl));

    CHKERRQ(VecDuplicate(r_vcl,&d_vcl));
    CHKERRQ(VecCopy(r_vcl,d_vcl));
    CHKERRQ(VecAXPY(d_vcl,-1.0,r_vcl));
    CHKERRQ(VecNorm(d_vcl,NORM_INFINITY,&dnorm));
    PetscCheckFalse(dnorm > tol,PETSC_COMM_WORLD,PETSC_ERR_ARG_INCOMP,"MPI CPU and MPI ViennaCL Vector results incompatible with inf norm greater than tolerance of %g",tol);

    CHKERRQ(VecDestroy(&v));
    CHKERRQ(VecDestroy(&r));
    CHKERRQ(VecDestroy(&v_vcl));
    CHKERRQ(VecDestroy(&r_vcl));
    CHKERRQ(VecDestroy(&d_vcl));
    CHKERRQ(MatDestroy(&A));
  }

  CHKERRQ(PetscFinalize());
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
