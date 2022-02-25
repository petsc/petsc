
static char help[] = "Tests MatHYPRE\n";

#include <petscmathypre.h>

int main(int argc,char **args)
{
  Mat                A,B,C,D;
  Mat                pAB,CD,CAB;
  hypre_ParCSRMatrix *parcsr;
  PetscReal          err;
  PetscInt           i,j,N = 6, M = 6;
  PetscErrorCode     ierr;
  PetscBool          flg,testptap = PETSC_TRUE,testmatmatmult = PETSC_TRUE;
  PetscReal          norm;
  char               file[256];

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),&flg);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  testptap = PETSC_FALSE;
  testmatmatmult = PETSC_FALSE;
  ierr = PetscOptionsInsertString(NULL,"-options_left 0");CHKERRQ(ierr);
#endif
  ierr = PetscOptionsGetBool(NULL,NULL,"-ptap",&testptap,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-matmatmult",&testmatmatmult,NULL);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  if (!flg) { /* Create a matrix and test MatSetValues */
    PetscMPIInt size;

    ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
    ierr = PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL);CHKERRQ(ierr);
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
        PetscScalar vals[] = {0,1./size,2./size,3./size,4./size,5./size};
        for (j=i-2; j<i+1; j++) {
          if (j >= N) {
            ierr = MatSetValue(A,i,N-1,(1.*j*N+i)/(3.*N*size),ADD_VALUES);CHKERRQ(ierr);
            ierr = MatSetValue(B,i,N-1,(1.*j*N+i)/(3.*N*size),ADD_VALUES);CHKERRQ(ierr);
          } else if (i > j) {
            ierr = MatSetValue(A,i,PetscMin(j,N-1),(1.*j*N+i)/(2.*N*size),ADD_VALUES);CHKERRQ(ierr);
            ierr = MatSetValue(B,i,PetscMin(j,N-1),(1.*j*N+i)/(2.*N*size),ADD_VALUES);CHKERRQ(ierr);
          } else {
            ierr = MatSetValue(A,i,PetscMin(j,N-1),-1.-(1.*j*N+i)/(4.*N*size),ADD_VALUES);CHKERRQ(ierr);
            ierr = MatSetValue(B,i,PetscMin(j,N-1),-1.-(1.*j*N+i)/(4.*N*size),ADD_VALUES);CHKERRQ(ierr);
          }
        }
        ierr = MatSetValues(A,1,&i,PetscMin(6,N),cols,vals,ADD_VALUES);CHKERRQ(ierr);
        ierr = MatSetValues(B,1,&i,PetscMin(6,N),cols,vals,ADD_VALUES);CHKERRQ(ierr);
      }
    } else {
      PetscInt  rows[2];
      PetscBool test_offproc = PETSC_FALSE;

      ierr = PetscOptionsGetBool(NULL,NULL,"-test_offproc",&test_offproc,NULL);CHKERRQ(ierr);
      if (test_offproc) {
        const PetscInt *ranges;
        PetscMPIInt    rank;

        ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
        ierr = MatGetOwnershipRanges(A,&ranges);CHKERRQ(ierr);
        rows[0] = ranges[(rank+1)%size];
        rows[1] = ranges[(rank+1)%size + 1];
      } else {
        ierr = MatGetOwnershipRange(A,&rows[0],&rows[1]);CHKERRQ(ierr);
      }
      for (i=rows[0];i<rows[1];i++) {
        PetscInt    cols[] = {0,1,2,3,4,5};
        PetscScalar vals[] = {-1,1,-2,2,-3,3};

        ierr = MatSetValues(A,1,&i,PetscMin(6,N),cols,vals,INSERT_VALUES);CHKERRQ(ierr);
        ierr = MatSetValues(B,1,&i,PetscMin(6,N),cols,vals,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
    /* MAT_FLUSH_ASSEMBLY currently not supported */
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

#if defined(PETSC_USE_COMPLEX)
    /* make the matrix imaginary */
    ierr = MatScale(A,PETSC_i);CHKERRQ(ierr);
    ierr = MatScale(B,PETSC_i);CHKERRQ(ierr);
#endif

    /* MatAXPY further exercises MatSetValues_HYPRE */
    ierr = MatAXPY(B,-1.,A,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatConvert(B,MATMPIAIJ,MAT_INITIAL_MATRIX,&C);CHKERRQ(ierr);
    ierr = MatNorm(C,NORM_INFINITY,&err);CHKERRQ(ierr);
    PetscCheckFalse(err > PETSC_SMALL,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Error MatSetValues %g",err);
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
  ierr = MatMultEqual(B,A,4,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Error Mat HYPRE");
  ierr = MatConvert(B,MATIS,MAT_INITIAL_MATRIX,&D);CHKERRQ(ierr);
  ierr = MatConvert(B,MATIS,MAT_REUSE_MATRIX,&D);CHKERRQ(ierr);
  ierr = MatMultEqual(D,A,4,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Error Mat IS");
  ierr = MatConvert(B,MATAIJ,MAT_INITIAL_MATRIX,&C);CHKERRQ(ierr);
  ierr = MatConvert(B,MATAIJ,MAT_REUSE_MATRIX,&C);CHKERRQ(ierr);
  ierr = MatMultEqual(C,A,4,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Error Mat AIJ");
  ierr = MatAXPY(C,-1.,A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatNorm(C,NORM_INFINITY,&err);CHKERRQ(ierr);
  PetscCheckFalse(err > PETSC_SMALL,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Error Mat AIJ %g",err);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = MatConvert(D,MATAIJ,MAT_INITIAL_MATRIX,&C);CHKERRQ(ierr);
  ierr = MatAXPY(C,-1.,A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatNorm(C,NORM_INFINITY,&err);CHKERRQ(ierr);
  PetscCheckFalse(err > PETSC_SMALL,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Error Mat IS %g",err);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = MatDestroy(&D);CHKERRQ(ierr);

  /* check MatCreateFromParCSR */
  ierr = MatHYPREGetParCSR(B,&parcsr);CHKERRQ(ierr);
  ierr = MatCreateFromParCSR(parcsr,MATAIJ,PETSC_COPY_VALUES,&D);CHKERRQ(ierr);
  ierr = MatDestroy(&D);CHKERRQ(ierr);
  ierr = MatCreateFromParCSR(parcsr,MATHYPRE,PETSC_USE_POINTER,&C);CHKERRQ(ierr);

  /* check MatMult operations */
  ierr = MatMultEqual(A,B,4,&flg);CHKERRQ(ierr);
  PetscCheckFalse(!flg,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Error MatMult B");
  ierr = MatMultEqual(A,C,4,&flg);CHKERRQ(ierr);
  PetscCheckFalse(!flg,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Error MatMult C");
  ierr = MatMultAddEqual(A,B,4,&flg);CHKERRQ(ierr);
  PetscCheckFalse(!flg,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Error MatMultAdd B");
  ierr = MatMultAddEqual(A,C,4,&flg);CHKERRQ(ierr);
  PetscCheckFalse(!flg,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Error MatMultAdd C");
  ierr = MatMultTransposeEqual(A,B,4,&flg);CHKERRQ(ierr);
  PetscCheckFalse(!flg,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Error MatMultTranspose B");
  ierr = MatMultTransposeEqual(A,C,4,&flg);CHKERRQ(ierr);
  PetscCheckFalse(!flg,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Error MatMultTranspose C");
  ierr = MatMultTransposeAddEqual(A,B,4,&flg);CHKERRQ(ierr);
  PetscCheckFalse(!flg,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Error MatMultTransposeAdd B");
  ierr = MatMultTransposeAddEqual(A,C,4,&flg);CHKERRQ(ierr);
  PetscCheckFalse(!flg,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Error MatMultTransposeAdd C");

  /* check PtAP */
  if (testptap && M == N) {
    Mat pP,hP;

    /* PETSc MatPtAP -> output is a MatAIJ
       It uses HYPRE functions when -matptap_via hypre is specified at command line */
    ierr = MatPtAP(A,A,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&pP);CHKERRQ(ierr);
    ierr = MatPtAP(A,A,MAT_REUSE_MATRIX,PETSC_DEFAULT,&pP);CHKERRQ(ierr);
    ierr = MatNorm(pP,NORM_INFINITY,&norm);CHKERRQ(ierr);
    ierr = MatPtAPMultEqual(A,A,pP,10,&flg);CHKERRQ(ierr);
    PetscCheckFalse(!flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Error in MatPtAP_MatAIJ");

    /* MatPtAP_HYPRE_HYPRE -> output is a MatHYPRE */
    ierr = MatPtAP(C,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&hP);CHKERRQ(ierr);
    ierr = MatPtAP(C,B,MAT_REUSE_MATRIX,PETSC_DEFAULT,&hP);CHKERRQ(ierr);
    ierr = MatPtAPMultEqual(C,B,hP,10,&flg);CHKERRQ(ierr);
    PetscCheckFalse(!flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Error in MatPtAP_HYPRE_HYPRE");

    /* Test MatAXPY_Basic() */
    ierr = MatAXPY(hP,-1.,pP,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatHasOperation(hP,MATOP_NORM,&flg);CHKERRQ(ierr);
    if (!flg) { /* TODO add MatNorm_HYPRE */
      ierr = MatConvert(hP,MATAIJ,MAT_INPLACE_MATRIX,&hP);CHKERRQ(ierr);
    }
    ierr = MatNorm(hP,NORM_INFINITY,&err);CHKERRQ(ierr);
    PetscCheckFalse(err/norm > PETSC_SMALL,PetscObjectComm((PetscObject)hP),PETSC_ERR_PLIB,"Error MatPtAP %g %g",err,norm);
    ierr = MatDestroy(&pP);CHKERRQ(ierr);
    ierr = MatDestroy(&hP);CHKERRQ(ierr);

    /* MatPtAP_AIJ_HYPRE -> output can be decided at runtime with -matptap_hypre_outtype */
    ierr = MatPtAP(A,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&hP);CHKERRQ(ierr);
    ierr = MatPtAP(A,B,MAT_REUSE_MATRIX,PETSC_DEFAULT,&hP);CHKERRQ(ierr);
    ierr = MatPtAPMultEqual(A,B,hP,10,&flg);CHKERRQ(ierr);
    PetscCheckFalse(!flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Error in MatPtAP_AIJ_HYPRE");
    ierr = MatDestroy(&hP);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);

  /* check MatMatMult */
  if (testmatmatmult) {
    ierr = MatTranspose(A,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);
    ierr = MatConvert(A,MATHYPRE,MAT_INITIAL_MATRIX,&C);CHKERRQ(ierr);
    ierr = MatConvert(B,MATHYPRE,MAT_INITIAL_MATRIX,&D);CHKERRQ(ierr);

    /* PETSc MatMatMult -> output is a MatAIJ
       It uses HYPRE functions when -matmatmult_via hypre is specified at command line */
    ierr = MatMatMult(A,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&pAB);CHKERRQ(ierr);
    ierr = MatMatMult(A,B,MAT_REUSE_MATRIX,PETSC_DEFAULT,&pAB);CHKERRQ(ierr);
    ierr = MatNorm(pAB,NORM_INFINITY,&norm);CHKERRQ(ierr);
    ierr = MatMatMultEqual(A,B,pAB,10,&flg);CHKERRQ(ierr);
    PetscCheckFalse(!flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Error in MatMatMult_AIJ_AIJ");

    /* MatMatMult_HYPRE_HYPRE -> output is a MatHYPRE */
    ierr = MatMatMult(C,D,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&CD);CHKERRQ(ierr);
    ierr = MatMatMult(C,D,MAT_REUSE_MATRIX,PETSC_DEFAULT,&CD);CHKERRQ(ierr);
    ierr = MatMatMultEqual(C,D,CD,10,&flg);CHKERRQ(ierr);
    PetscCheckFalse(!flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Error in MatMatMult_HYPRE_HYPRE");

    /* Test MatAXPY_Basic() */
    ierr = MatAXPY(CD,-1.,pAB,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);

    ierr = MatHasOperation(CD,MATOP_NORM,&flg);CHKERRQ(ierr);
    if (!flg) { /* TODO add MatNorm_HYPRE */
      ierr = MatConvert(CD,MATAIJ,MAT_INPLACE_MATRIX,&CD);CHKERRQ(ierr);
    }
    ierr = MatNorm(CD,NORM_INFINITY,&err);CHKERRQ(ierr);
    PetscCheckFalse(err/norm > PETSC_SMALL,PetscObjectComm((PetscObject)CD),PETSC_ERR_PLIB,"Error MatMatMult %g %g",err,norm);

    ierr = MatDestroy(&C);CHKERRQ(ierr);
    ierr = MatDestroy(&D);CHKERRQ(ierr);
    ierr = MatDestroy(&pAB);CHKERRQ(ierr);
    ierr = MatDestroy(&CD);CHKERRQ(ierr);

    /* When configured with HYPRE, MatMatMatMult is available for the triplet transpose(aij)-aij-aij */
    ierr = MatCreateTranspose(A,&C);CHKERRQ(ierr);
    ierr = MatMatMatMult(C,A,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&CAB);CHKERRQ(ierr);
    ierr = MatDestroy(&C);CHKERRQ(ierr);
    ierr = MatTranspose(A,MAT_INITIAL_MATRIX,&C);CHKERRQ(ierr);
    ierr = MatMatMult(C,A,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&D);CHKERRQ(ierr);
    ierr = MatDestroy(&C);CHKERRQ(ierr);
    ierr = MatMatMult(D,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&C);CHKERRQ(ierr);
    ierr = MatNorm(C,NORM_INFINITY,&norm);CHKERRQ(ierr);
    ierr = MatAXPY(C,-1.,CAB,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatNorm(C,NORM_INFINITY,&err);CHKERRQ(ierr);
    PetscCheckFalse(err/norm > PETSC_SMALL,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Error MatMatMatMult %g %g",err,norm);
    ierr = MatDestroy(&C);CHKERRQ(ierr);
    ierr = MatDestroy(&D);CHKERRQ(ierr);
    ierr = MatDestroy(&CAB);CHKERRQ(ierr);
    ierr = MatDestroy(&B);CHKERRQ(ierr);
  }

  /* Check MatView */
  ierr = MatViewFromOptions(A,NULL,"-view_A");CHKERRQ(ierr);
  ierr = MatConvert(A,MATHYPRE,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);
  ierr = MatViewFromOptions(B,NULL,"-view_B");CHKERRQ(ierr);

  /* Check MatDuplicate/MatCopy */
  for (j=0;j<3;j++) {
    MatDuplicateOption dop;

    dop = MAT_COPY_VALUES;
    if (j==1) dop = MAT_DO_NOT_COPY_VALUES;
    if (j==2) dop = MAT_SHARE_NONZERO_PATTERN;

    for (i=0;i<3;i++) {
      MatStructure str;

      ierr = PetscPrintf(PETSC_COMM_WORLD,"Dup/Copy tests: %" PetscInt_FMT " %" PetscInt_FMT "\n",j,i);CHKERRQ(ierr);

      str = DIFFERENT_NONZERO_PATTERN;
      if (i==1) str = SAME_NONZERO_PATTERN;
      if (i==2) str = SUBSET_NONZERO_PATTERN;

      ierr = MatDuplicate(A,dop,&C);CHKERRQ(ierr);
      ierr = MatDuplicate(B,dop,&D);CHKERRQ(ierr);
      if (dop != MAT_COPY_VALUES) {
        ierr = MatCopy(A,C,str);CHKERRQ(ierr);
        ierr = MatCopy(B,D,str);CHKERRQ(ierr);
      }
      /* AXPY with AIJ and HYPRE */
      ierr = MatAXPY(C,-1.0,D,str);CHKERRQ(ierr);
      ierr = MatNorm(C,NORM_INFINITY,&err);CHKERRQ(ierr);
      if (err > PETSC_SMALL) {
        ierr = MatViewFromOptions(A,NULL,"-view_duplicate_diff");CHKERRQ(ierr);
        ierr = MatViewFromOptions(B,NULL,"-view_duplicate_diff");CHKERRQ(ierr);
        ierr = MatViewFromOptions(C,NULL,"-view_duplicate_diff");CHKERRQ(ierr);
        SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Error test 1 MatDuplicate/MatCopy %g (%" PetscInt_FMT ",%" PetscInt_FMT ")",err,j,i);
      }
      /* AXPY with HYPRE and HYPRE */
      ierr = MatAXPY(D,-1.0,B,str);CHKERRQ(ierr);
      if (err > PETSC_SMALL) {
        ierr = MatViewFromOptions(A,NULL,"-view_duplicate_diff");CHKERRQ(ierr);
        ierr = MatViewFromOptions(B,NULL,"-view_duplicate_diff");CHKERRQ(ierr);
        ierr = MatViewFromOptions(D,NULL,"-view_duplicate_diff");CHKERRQ(ierr);
        SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Error test 2 MatDuplicate/MatCopy %g (%" PetscInt_FMT ",%" PetscInt_FMT ")",err,j,i);
      }
      /* Copy from HYPRE to AIJ */
      ierr = MatCopy(B,C,str);CHKERRQ(ierr);
      /* Copy from AIJ to HYPRE */
      ierr = MatCopy(A,D,str);CHKERRQ(ierr);
      /* AXPY with HYPRE and AIJ */
      ierr = MatAXPY(D,-1.0,C,str);CHKERRQ(ierr);
      ierr = MatHasOperation(D,MATOP_NORM,&flg);CHKERRQ(ierr);
      if (!flg) { /* TODO add MatNorm_HYPRE */
        ierr = MatConvert(D,MATAIJ,MAT_INPLACE_MATRIX,&D);CHKERRQ(ierr);
      }
      ierr = MatNorm(D,NORM_INFINITY,&err);CHKERRQ(ierr);
      if (err > PETSC_SMALL) {
        ierr = MatViewFromOptions(A,NULL,"-view_duplicate_diff");CHKERRQ(ierr);
        ierr = MatViewFromOptions(C,NULL,"-view_duplicate_diff");CHKERRQ(ierr);
        ierr = MatViewFromOptions(D,NULL,"-view_duplicate_diff");CHKERRQ(ierr);
        SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Error test 3 MatDuplicate/MatCopy %g (%" PetscInt_FMT ",%" PetscInt_FMT ")",err,j,i);
      }
      ierr = MatDestroy(&C);CHKERRQ(ierr);
      ierr = MatDestroy(&D);CHKERRQ(ierr);
    }
  }
  ierr = MatDestroy(&B);CHKERRQ(ierr);

  ierr = MatHasCongruentLayouts(A,&flg);CHKERRQ(ierr);
  if (flg) {
    Vec y,y2;

    ierr = MatConvert(A,MATHYPRE,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);
    ierr = MatCreateVecs(A,NULL,&y);CHKERRQ(ierr);
    ierr = MatCreateVecs(B,NULL,&y2);CHKERRQ(ierr);
    ierr = MatGetDiagonal(A,y);CHKERRQ(ierr);
    ierr = MatGetDiagonal(B,y2);CHKERRQ(ierr);
    ierr = VecAXPY(y2,-1.0,y);CHKERRQ(ierr);
    ierr = VecNorm(y2,NORM_INFINITY,&err);CHKERRQ(ierr);
    if (err > PETSC_SMALL) {
      ierr = VecViewFromOptions(y,NULL,"-view_diagonal_diff");CHKERRQ(ierr);
      ierr = VecViewFromOptions(y2,NULL,"-view_diagonal_diff");CHKERRQ(ierr);
      SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Error MatGetDiagonal %g",err);
    }
    ierr = MatDestroy(&B);CHKERRQ(ierr);
    ierr = VecDestroy(&y);CHKERRQ(ierr);
    ierr = VecDestroy(&y2);CHKERRQ(ierr);
  }

  ierr = MatDestroy(&A);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
      requires: hypre

   test:
      requires: !defined(PETSC_HAVE_HYPRE_DEVICE)
      suffix: 1
      args: -N 11 -M 11
      output_file: output/ex115_1.out

   test:
      requires: !defined(PETSC_HAVE_HYPRE_DEVICE)
      suffix: 2
      nsize: 3
      args: -N 13 -M 13 -matmatmult_via hypre
      output_file: output/ex115_1.out

   test:
      requires: !defined(PETSC_HAVE_HYPRE_DEVICE)
      suffix: 3
      nsize: 4
      args: -M 13 -N 7 -matmatmult_via hypre
      output_file: output/ex115_1.out

   test:
      requires: !defined(PETSC_HAVE_HYPRE_DEVICE)
      suffix: 4
      nsize: 2
      args: -M 12 -N 19
      output_file: output/ex115_1.out

   test:
      requires: !defined(PETSC_HAVE_HYPRE_DEVICE)
      suffix: 5
      nsize: 3
      args: -M 13 -N 13 -matptap_via hypre -matptap_hypre_outtype hypre
      output_file: output/ex115_1.out

   test:
      requires: !defined(PETSC_HAVE_HYPRE_DEVICE)
      suffix: 6
      nsize: 3
      args: -M 12 -N 19 -test_offproc
      output_file: output/ex115_1.out

   test:
      requires: !defined(PETSC_HAVE_HYPRE_DEVICE)
      suffix: 7
      nsize: 3
      args: -M 19 -N 12 -test_offproc -view_B ::ascii_info_detail
      output_file: output/ex115_7.out

   test:
      requires: !defined(PETSC_HAVE_HYPRE_DEVICE)
      suffix: 8
      nsize: 3
      args: -M 1 -N 12 -test_offproc
      output_file: output/ex115_1.out

   test:
      requires: !defined(PETSC_HAVE_HYPRE_DEVICE)
      suffix: 9
      nsize: 3
      args: -M 1 -N 2 -test_offproc
      output_file: output/ex115_1.out

TEST*/
