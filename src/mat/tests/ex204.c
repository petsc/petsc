static char help[] = "Test ViennaCL Matrix Conversions";

#include <petscmat.h>

int main(int argc,char **args)
{
  PetscErrorCode ierr;
  PetscMPIInt rank,size;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;

  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);

  /* Construct a sequential ViennaCL matrix and vector */
  if (rank == 0) {
    Mat A_vcl;
    Vec v_vcl,r_vcl;
    PetscInt n = 17, m = 31,nz = 5,i,cnt,j;
    const PetscScalar val = 1.0;

    ierr = MatCreateSeqAIJViennaCL(PETSC_COMM_SELF,m,n,nz,NULL,&A_vcl);CHKERRQ(ierr);

    /* Add nz arbitrary entries per row in arbitrary columns */
    for (i=0;i<m;++i) {
      for (cnt = 0; cnt<nz; ++cnt) {
        j = (19 * cnt + (7*i + 3)) % n;
        ierr = MatSetValue(A_vcl,i,j,(PetscScalar)(0.3 * i + j),INSERT_VALUES);CHKERRQ(ierr);
      }
    }
    ierr = MatAssemblyBegin(A_vcl,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A_vcl,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    ierr = VecCreateSeqViennaCL(PETSC_COMM_SELF,n,&v_vcl);CHKERRQ(ierr);
    ierr = VecCreateSeqViennaCL(PETSC_COMM_SELF,m,&r_vcl);CHKERRQ(ierr);
    ierr = VecSet(v_vcl,val);CHKERRQ(ierr);

    ierr = MatMult(A_vcl,v_vcl,r_vcl);CHKERRQ(ierr);

    ierr = VecDestroy(&v_vcl);CHKERRQ(ierr);
    ierr = VecDestroy(&r_vcl);CHKERRQ(ierr);
    ierr = MatDestroy(&A_vcl);CHKERRQ(ierr);
  }

  /* Create a sequential AIJ matrix on rank 0 convert it to a new ViennaCL matrix, and apply it to a ViennaCL vector */
  if (rank == 0) {
    Mat               A,A_vcl;
    Vec               v,r,v_vcl,r_vcl,d_vcl;
    PetscInt          n=17,m=31,nz=5,i,cnt,j;
    const PetscScalar val = 1.0;
    PetscReal         dnorm;
    const PetscReal   tol = 1e-5;

    ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,m,n,nz,NULL,&A);CHKERRQ(ierr);

    /* Add nz arbitrary entries per row in arbitrary columns */
    for (i=0;i<m;++i) {
      for (cnt = 0; cnt<nz; ++cnt) {
        j = (19 * cnt + (7*i + 3)) % n;
        ierr = MatSetValue(A,i,j,(PetscScalar) (0.3 * i + j),INSERT_VALUES);CHKERRQ(ierr);
      }
    }
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    ierr = PetscObjectSetName((PetscObject)A,"Sequential CPU Matrix");CHKERRQ(ierr);

    ierr = VecCreateSeq(PETSC_COMM_SELF,n,&v);CHKERRQ(ierr);
    ierr = VecCreateSeq(PETSC_COMM_SELF,m,&r);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)r,"CPU result vector");CHKERRQ(ierr);
    ierr = VecSet(v,val);CHKERRQ(ierr);
    ierr = MatMult(A,v,r);CHKERRQ(ierr);

    ierr = MatConvert(A,MATSEQAIJVIENNACL,MAT_INITIAL_MATRIX,&A_vcl);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)A_vcl,"New ViennaCL Matrix");CHKERRQ(ierr);

    ierr = VecCreateSeqViennaCL(PETSC_COMM_SELF,n,&v_vcl);CHKERRQ(ierr);
    ierr = VecCreateSeqViennaCL(PETSC_COMM_SELF,m,&r_vcl);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)r_vcl,"ViennaCL result vector");CHKERRQ(ierr);
    ierr = VecSet(v_vcl,val);CHKERRQ(ierr);
    ierr = MatMult(A_vcl,v_vcl,r_vcl);CHKERRQ(ierr);

    ierr = VecDuplicate(r_vcl,&d_vcl);CHKERRQ(ierr);
    ierr = VecCopy(r_vcl,d_vcl);CHKERRQ(ierr);
    ierr = VecAXPY(d_vcl,-1.0,r_vcl);CHKERRQ(ierr);
    ierr = VecNorm(d_vcl,NORM_INFINITY,&dnorm);CHKERRQ(ierr);
    PetscAssertFalse(dnorm > tol,PETSC_COMM_WORLD,PETSC_ERR_ARG_INCOMP,"Sequential CPU and MPI ViennaCL vector results incompatible with inf norm greater than tolerance of %g",tol);

    ierr = VecDestroy(&v);CHKERRQ(ierr);
    ierr = VecDestroy(&r);CHKERRQ(ierr);
    ierr = VecDestroy(&v_vcl);CHKERRQ(ierr);
    ierr = VecDestroy(&r_vcl);CHKERRQ(ierr);
    ierr = VecDestroy(&d_vcl);CHKERRQ(ierr);
    ierr = MatDestroy(&A);CHKERRQ(ierr);
    ierr = MatDestroy(&A_vcl);CHKERRQ(ierr);
  }

  /* Create a sequential AIJ matrix on rank 0 convert it inplace to a new ViennaCL matrix, and apply it to a ViennaCL vector */
  if (rank == 0) {
    Mat               A;
    Vec               v,r,v_vcl,r_vcl,d_vcl;
    PetscInt          n=17,m=31,nz=5,i,cnt,j;
    const PetscScalar val = 1.0;
    PetscReal         dnorm;
    const PetscReal   tol = 1e-5;

    ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,m,n,nz,NULL,&A);CHKERRQ(ierr);

    /* Add nz arbitrary entries per row in arbitrary columns */
    for (i=0;i<m;++i) {
      for (cnt = 0; cnt<nz; ++cnt) {
        j = (19 * cnt + (7*i + 3)) % n;
        ierr = MatSetValue(A,i,j,(PetscScalar)(0.3 * i + j),INSERT_VALUES);CHKERRQ(ierr);
      }
    }
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    ierr = PetscObjectSetName((PetscObject)A,"Sequential CPU Matrix");CHKERRQ(ierr);

    ierr = VecCreateSeq(PETSC_COMM_SELF,n,&v);CHKERRQ(ierr);
    ierr = VecCreateSeq(PETSC_COMM_SELF,m,&r);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)r,"CPU result vector");CHKERRQ(ierr);
    ierr = VecSet(v,val);CHKERRQ(ierr);
    ierr = MatMult(A,v,r);CHKERRQ(ierr);

    ierr = MatConvert(A,MATSEQAIJVIENNACL,MAT_INPLACE_MATRIX,&A);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)A,"Converted ViennaCL Matrix");CHKERRQ(ierr);

    ierr = VecCreateSeqViennaCL(PETSC_COMM_SELF,n,&v_vcl);CHKERRQ(ierr);
    ierr = VecCreateSeqViennaCL(PETSC_COMM_SELF,m,&r_vcl);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)r_vcl,"ViennaCL result vector");CHKERRQ(ierr);
    ierr = VecSet(v_vcl,val);CHKERRQ(ierr);
    ierr = MatMult(A,v_vcl,r_vcl);CHKERRQ(ierr);

    ierr = VecDuplicate(r_vcl,&d_vcl);CHKERRQ(ierr);
    ierr = VecCopy(r_vcl,d_vcl);CHKERRQ(ierr);
    ierr = VecAXPY(d_vcl,-1.0,r_vcl);CHKERRQ(ierr);
    ierr = VecNorm(d_vcl,NORM_INFINITY,&dnorm);CHKERRQ(ierr);
    PetscAssertFalse(dnorm > tol,PETSC_COMM_WORLD,PETSC_ERR_ARG_INCOMP,"MPI CPU and MPI ViennaCL Vector results incompatible with inf norm greater than tolerance of %g",tol);

    ierr = VecDestroy(&v);CHKERRQ(ierr);
    ierr = VecDestroy(&r);CHKERRQ(ierr);
    ierr = VecDestroy(&v_vcl);CHKERRQ(ierr);
    ierr = VecDestroy(&r_vcl);CHKERRQ(ierr);
    ierr = VecDestroy(&d_vcl);CHKERRQ(ierr);
    ierr = MatDestroy(&A);CHKERRQ(ierr);
  }

  /* Create a parallel AIJ matrix, convert it to a new ViennaCL matrix, and apply it to a ViennaCL vector */
  if (size > 1) {
    Mat               A,A_vcl;
    Vec               v,r,v_vcl,r_vcl,d_vcl;
    PetscInt          N=17,M=31,nz=5,i,cnt,j,rlow,rhigh;
    const PetscScalar val = 1.0;
    PetscReal         dnorm;
    const PetscReal   tol=1e-5;

    ierr = MatCreateAIJ(PETSC_COMM_WORLD,PETSC_DETERMINE,PETSC_DETERMINE,M,N,nz,NULL,nz,NULL,&A);CHKERRQ(ierr);

    /* Add nz arbitrary entries per row in arbitrary columns */
    ierr = MatGetOwnershipRange(A,&rlow,&rhigh);CHKERRQ(ierr);
    for (i=rlow;i<rhigh;++i) {
      for (cnt = 0; cnt<nz; ++cnt) {
        j = (19 * cnt + (7*i + 3)) % N;
        ierr = MatSetValue(A,i,j,(PetscScalar)(0.3 * i + j),INSERT_VALUES);CHKERRQ(ierr);
      }
    }
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    ierr = PetscObjectSetName((PetscObject)A,"MPI CPU Matrix");CHKERRQ(ierr);

    ierr = MatCreateVecs(A,&v,&r);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)r,"MPI CPU result vector");CHKERRQ(ierr);
    ierr = VecSet(v,val);CHKERRQ(ierr);
    ierr = MatMult(A,v,r);CHKERRQ(ierr);

    ierr = MatConvert(A,MATMPIAIJVIENNACL,MAT_INITIAL_MATRIX,&A_vcl);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)A_vcl,"New MPI ViennaCL Matrix");CHKERRQ(ierr);

    ierr = MatCreateVecs(A_vcl,&v_vcl,&r_vcl);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)r_vcl,"ViennaCL result vector");CHKERRQ(ierr);
    ierr = VecSet(v_vcl,val);CHKERRQ(ierr);
    ierr = MatMult(A_vcl,v_vcl,r_vcl);CHKERRQ(ierr);

    ierr = VecDuplicate(r_vcl,&d_vcl);CHKERRQ(ierr);
    ierr = VecCopy(r_vcl,d_vcl);CHKERRQ(ierr);
    ierr = VecAXPY(d_vcl,-1.0,r_vcl);CHKERRQ(ierr);
    ierr = VecNorm(d_vcl,NORM_INFINITY,&dnorm);CHKERRQ(ierr);
    PetscAssertFalse(dnorm > tol,PETSC_COMM_WORLD,PETSC_ERR_ARG_INCOMP,"MPI CPU and MPI ViennaCL Vector results incompatible with inf norm greater than tolerance of %g",tol);

    ierr = VecDestroy(&v);CHKERRQ(ierr);
    ierr = VecDestroy(&r);CHKERRQ(ierr);
    ierr = VecDestroy(&v_vcl);CHKERRQ(ierr);
    ierr = VecDestroy(&r_vcl);CHKERRQ(ierr);
    ierr = VecDestroy(&d_vcl);CHKERRQ(ierr);
    ierr = MatDestroy(&A);CHKERRQ(ierr);
    ierr = MatDestroy(&A_vcl);CHKERRQ(ierr);
  }

  /* Create a parallel AIJ matrix, convert it in-place to a ViennaCL matrix, and apply it to a ViennaCL vector */
  if (size > 1) {
    Mat               A;
    Vec               v,r,v_vcl,r_vcl,d_vcl;
    PetscInt          N=17,M=31,nz=5,i,cnt,j,rlow,rhigh;
    const PetscScalar val = 1.0;
    PetscReal         dnorm;
    const PetscReal   tol=1e-5;

    ierr = MatCreateAIJ(PETSC_COMM_WORLD,PETSC_DETERMINE,PETSC_DETERMINE,M,N,nz,NULL,nz,NULL,&A);CHKERRQ(ierr);

    /* Add nz arbitrary entries per row in arbitrary columns */
    ierr = MatGetOwnershipRange(A,&rlow,&rhigh);CHKERRQ(ierr);
    for (i=rlow;i<rhigh;++i) {
      for (cnt = 0; cnt<nz; ++cnt) {
        j = (19 * cnt + (7*i + 3)) % N;
        ierr = MatSetValue(A,i,j,(PetscScalar)(0.3 * i + j),INSERT_VALUES);CHKERRQ(ierr);
      }
    }
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    ierr = PetscObjectSetName((PetscObject)A,"MPI CPU Matrix");CHKERRQ(ierr);

    ierr = MatCreateVecs(A,&v,&r);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)r,"MPI CPU result vector");CHKERRQ(ierr);
    ierr = VecSet(v,val);CHKERRQ(ierr);
    ierr = MatMult(A,v,r);CHKERRQ(ierr);

    ierr = MatConvert(A,MATMPIAIJVIENNACL,MAT_INPLACE_MATRIX,&A);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)A,"Converted MPI ViennaCL Matrix");CHKERRQ(ierr);

    ierr = MatCreateVecs(A,&v_vcl,&r_vcl);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)r_vcl,"ViennaCL result vector");CHKERRQ(ierr);
    ierr = VecSet(v_vcl,val);CHKERRQ(ierr);
    ierr = MatMult(A,v_vcl,r_vcl);CHKERRQ(ierr);

    ierr = VecDuplicate(r_vcl,&d_vcl);CHKERRQ(ierr);
    ierr = VecCopy(r_vcl,d_vcl);CHKERRQ(ierr);
    ierr = VecAXPY(d_vcl,-1.0,r_vcl);CHKERRQ(ierr);
    ierr = VecNorm(d_vcl,NORM_INFINITY,&dnorm);CHKERRQ(ierr);
    PetscAssertFalse(dnorm > tol,PETSC_COMM_WORLD,PETSC_ERR_ARG_INCOMP,"MPI CPU and MPI ViennaCL Vector results incompatible with inf norm greater than tolerance of %g",tol);

    ierr = VecDestroy(&v);CHKERRQ(ierr);
    ierr = VecDestroy(&r);CHKERRQ(ierr);
    ierr = VecDestroy(&v_vcl);CHKERRQ(ierr);
    ierr = VecDestroy(&r_vcl);CHKERRQ(ierr);
    ierr = VecDestroy(&d_vcl);CHKERRQ(ierr);
    ierr = MatDestroy(&A);CHKERRQ(ierr);
  }

  ierr = PetscFinalize();
  return ierr;

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
