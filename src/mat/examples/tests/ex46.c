
static char help[] = "Tests MatHYPRE\n";

#include <petscmathypre.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat                A,B,C,D;
  hypre_ParCSRMatrix *parcsr;
  Vec                x,x2,y,y2;
  PetscReal          err;
  PetscInt           i,j,N = 6, M = 6;
  PetscErrorCode     ierr;
  PetscBool          flg;
  char               file[256];

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetString(NULL,NULL,"-f",file,256,&flg);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  if (!flg) { /* Create a matrix and test MatSetValues */
    PetscMPIInt NP;

    ierr = MPI_Comm_size(PETSC_COMM_WORLD,&NP);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL);CHKERRQ(ierr);
    if (N < 6) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Matrix has to have more than 6 columns");
    ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,M,N);CHKERRQ(ierr);
    ierr = MatSetType(A,MATAIJ);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(A,9,NULL);CHKERRQ(ierr);
    ierr = MatMPIAIJSetPreallocation(A,9,NULL,9,NULL);CHKERRQ(ierr);
    ierr = MatCreate(PETSC_COMM_WORLD,&B);CHKERRQ(ierr);
    ierr = MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,M,N);CHKERRQ(ierr);
    ierr = MatSetType(B,MATHYPRE);CHKERRQ(ierr);
    if (M == N) {
      ierr = MatHYPRESetPreallocation(B,9,NULL,9,NULL);CHKERRQ(ierr);
    } else {
      ierr = MatHYPRESetPreallocation(B,6,NULL,6,NULL);CHKERRQ(ierr);
    }
    if (M == N) {
      for (i=0; i<M; i++) {
        PetscInt    cols[] = {0,1,2,3,4,5};
        PetscScalar vals[] = {0,1./NP,2./NP,3./NP,4./NP,5./NP};
        for (j=i-2; j<i+1; j++) {
          if (j >= N) {
            ierr = MatSetValue(A,i,N-1,(1.*j*N+i)/(3.*N*NP),ADD_VALUES);CHKERRQ(ierr);
            ierr = MatSetValue(B,i,N-1,(1.*j*N+i)/(3.*N*NP),ADD_VALUES);CHKERRQ(ierr);
          } else if (i > j) {
            ierr = MatSetValue(A,i,j,(1.*j*N+i)/(2.*N*NP),ADD_VALUES);CHKERRQ(ierr);
            ierr = MatSetValue(B,i,j,(1.*j*N+i)/(2.*N*NP),ADD_VALUES);CHKERRQ(ierr);
          } else {
            ierr = MatSetValue(A,i,j,-1.-(1.*j*N+i)/(4.*N*NP),ADD_VALUES);CHKERRQ(ierr);
            ierr = MatSetValue(B,i,j,-1.-(1.*j*N+i)/(4.*N*NP),ADD_VALUES);CHKERRQ(ierr);
          }
        }
        ierr = MatSetValues(A,1,&i,6,cols,vals,ADD_VALUES);CHKERRQ(ierr);
        ierr = MatSetValues(B,1,&i,6,cols,vals,ADD_VALUES);CHKERRQ(ierr);
      }
    } else { /* HYPRE_IJMatrix does not support INSERT_VALUES with off-proc entries */
      PetscInt rows[2];
      ierr = MatGetOwnershipRange(A,&rows[0],&rows[1]);CHKERRQ(ierr);
      for (i=rows[0];i<rows[1];i++) {
        PetscInt    cols[] = {0,1,2,3,4,5};
        PetscScalar vals[] = {-1,1,-2,2,-3,3};

        ierr = MatSetValues(A,1,&i,6,cols,vals,INSERT_VALUES);CHKERRQ(ierr);
        ierr = MatSetValues(B,1,&i,6,cols,vals,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
    /* MAT_FLUSH_ASSEMBLY currently not supported */
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    /* MatAXPY_Basic further exercises MatSetValues_HYPRE */
    ierr = MatAXPY(B,-1.,A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatConvert(B,MATAIJ,MAT_INITIAL_MATRIX,&C);CHKERRQ(ierr);
    ierr = MatNorm(C,NORM_INFINITY,&err);CHKERRQ(ierr);
    if (err > PETSC_SMALL) SETERRQ1(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Error MatSetValues %g",err);
    ierr = MatDestroy(&B);CHKERRQ(ierr);
    ierr = MatDestroy(&C);CHKERRQ(ierr);
  } else {
    PetscViewer viewer;
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
    ierr = MatSetFromOptions(A);CHKERRQ(ierr);
    ierr = MatLoad(A,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    ierr = MatGetSize(A,&M,&N);CHKERRQ(ierr);
  }
  /* check conversion routines */
  ierr = MatConvert(A,MATHYPRE,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);
  ierr = MatConvert(A,MATHYPRE,MAT_REUSE_MATRIX,&B);CHKERRQ(ierr);
  ierr = MatConvert(B,MATIS,MAT_INITIAL_MATRIX,&D);CHKERRQ(ierr);
  ierr = MatConvert(B,MATIS,MAT_REUSE_MATRIX,&D);CHKERRQ(ierr);
  ierr = MatConvert(B,MATAIJ,MAT_INITIAL_MATRIX,&C);CHKERRQ(ierr);
  ierr = MatConvert(B,MATAIJ,MAT_REUSE_MATRIX,&C);CHKERRQ(ierr);
  ierr = MatAXPY(C,-1.,A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatNorm(C,NORM_INFINITY,&err);CHKERRQ(ierr);
  if (err > PETSC_SMALL) SETERRQ1(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Error Mat AIJ %g",err);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = MatISGetMPIXAIJ(D,MAT_INITIAL_MATRIX,&C);CHKERRQ(ierr);
  ierr = MatAXPY(C,-1.,A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatNorm(C,NORM_INFINITY,&err);CHKERRQ(ierr);
  if (err > PETSC_SMALL) SETERRQ1(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Error Mat IS %g",err);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = MatDestroy(&D);CHKERRQ(ierr);

  /* check MatCreateFromParCSR */
  ierr = MatHYPREGetParCSR(B,&parcsr);CHKERRQ(ierr);
  ierr = MatCreateFromParCSR(parcsr,MATAIJ,PETSC_COPY_VALUES,&D);CHKERRQ(ierr);
  ierr = MatDestroy(&D);CHKERRQ(ierr);
  ierr = MatCreateFromParCSR(parcsr,MATHYPRE,PETSC_USE_POINTER,&C);CHKERRQ(ierr);

  /* check matmult */
  ierr = MatCreateVecs(A,&x,&y);CHKERRQ(ierr);
  ierr = MatCreateVecs(A,&x2,&y2);CHKERRQ(ierr);
  ierr = VecSet(x,1.);CHKERRQ(ierr);
  ierr = MatMult(A,x,y);CHKERRQ(ierr);
  ierr = MatMult(B,x,y2);CHKERRQ(ierr);
  ierr = VecAXPY(y,-1.,y2);CHKERRQ(ierr);
  ierr = VecNorm(y,NORM_2,&err);CHKERRQ(ierr);
  if (err > PETSC_SMALL) SETERRQ1(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Error MatMult B %g",err);
  ierr = MatMult(C,x,y);CHKERRQ(ierr);
  ierr = VecAXPY(y,-1.,y2);CHKERRQ(ierr);
  ierr = VecNorm(y,NORM_2,&err);CHKERRQ(ierr);
  if (err > PETSC_SMALL) SETERRQ1(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Error MatMult C %g",err);
  ierr = VecSet(y,1.);CHKERRQ(ierr);
  ierr = MatMultTranspose(A,y,x);CHKERRQ(ierr);
  ierr = MatMultTranspose(B,y,x2);CHKERRQ(ierr);
  ierr = VecAXPY(x,-1.,x2);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,&err);CHKERRQ(ierr);
  if (err > PETSC_SMALL) SETERRQ1(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Error MatMultTranspose C %g",err);
  ierr = MatMultTranspose(C,y,x);CHKERRQ(ierr);
  ierr = VecAXPY(x,-1.,x2);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,&err);CHKERRQ(ierr);
  if (err > PETSC_SMALL) SETERRQ1(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Error MatMultTranspose C %g",err);

  /* check PtAP */
  if (M == N) {
    Mat       pP,hP;
    PetscReal norm;

    /* PETSc PtAP */
    ierr = MatPtAP(A,A,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&pP);CHKERRQ(ierr);
    ierr = MatNorm(pP,NORM_INFINITY,&norm);CHKERRQ(ierr);

    /* MatPtAP_HYPRE_HYPRE */
    ierr = MatPtAP(C,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&hP);CHKERRQ(ierr);
    ierr = MatConvert(hP,MATAIJ,MAT_INITIAL_MATRIX,&D);CHKERRQ(ierr);
    ierr = MatAXPY(D,-1.,pP,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatNorm(D,NORM_INFINITY,&err);CHKERRQ(ierr);
    if (err/norm > PETSC_SMALL) SETERRQ2(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Error MatPtAP %g %g",err,norm);
    ierr = MatDestroy(&hP);CHKERRQ(ierr);
    ierr = MatDestroy(&D);CHKERRQ(ierr);

    /* MatPtAP_AIJ_HYPRE */
    ierr = MatPtAP(A,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&hP);CHKERRQ(ierr);
    ierr = MatConvert(hP,MATAIJ,MAT_INITIAL_MATRIX,&D);CHKERRQ(ierr);
    ierr = MatAXPY(D,-1.,pP,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatNorm(D,NORM_INFINITY,&err);CHKERRQ(ierr);
    if (err/norm > PETSC_SMALL) SETERRQ2(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Error MatPtAP mixed %g %g",err,norm);
    ierr = MatDestroy(&D);CHKERRQ(ierr);
    ierr = MatDestroy(&hP);CHKERRQ(ierr);

    ierr = MatDestroy(&pP);CHKERRQ(ierr);
  }

  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&x2);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = VecDestroy(&y2);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}
