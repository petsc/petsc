
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
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),&flg));
#if defined(PETSC_USE_COMPLEX)
  testptap = PETSC_FALSE;
  testmatmatmult = PETSC_FALSE;
  CHKERRQ(PetscOptionsInsertString(NULL,"-options_left 0"));
#endif
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-ptap",&testptap,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-matmatmult",&testmatmatmult,NULL));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  if (!flg) { /* Create a matrix and test MatSetValues */
    PetscMPIInt size;

    CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
    CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL));
    CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL));
    CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,M,N));
    CHKERRQ(MatSetType(A,MATAIJ));
    CHKERRQ(MatSeqAIJSetPreallocation(A,9,NULL));
    CHKERRQ(MatMPIAIJSetPreallocation(A,9,NULL,9,NULL));
    CHKERRQ(MatCreate(PETSC_COMM_WORLD,&B));
    CHKERRQ(MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,M,N));
    CHKERRQ(MatSetType(B,MATHYPRE));
    if (M == N) {
      CHKERRQ(MatHYPRESetPreallocation(B,9,NULL,9,NULL));
    } else {
      CHKERRQ(MatHYPRESetPreallocation(B,6,NULL,6,NULL));
    }
    if (M == N) {
      for (i=0; i<M; i++) {
        PetscInt    cols[] = {0,1,2,3,4,5};
        PetscScalar vals[] = {0,1./size,2./size,3./size,4./size,5./size};
        for (j=i-2; j<i+1; j++) {
          if (j >= N) {
            CHKERRQ(MatSetValue(A,i,N-1,(1.*j*N+i)/(3.*N*size),ADD_VALUES));
            CHKERRQ(MatSetValue(B,i,N-1,(1.*j*N+i)/(3.*N*size),ADD_VALUES));
          } else if (i > j) {
            CHKERRQ(MatSetValue(A,i,PetscMin(j,N-1),(1.*j*N+i)/(2.*N*size),ADD_VALUES));
            CHKERRQ(MatSetValue(B,i,PetscMin(j,N-1),(1.*j*N+i)/(2.*N*size),ADD_VALUES));
          } else {
            CHKERRQ(MatSetValue(A,i,PetscMin(j,N-1),-1.-(1.*j*N+i)/(4.*N*size),ADD_VALUES));
            CHKERRQ(MatSetValue(B,i,PetscMin(j,N-1),-1.-(1.*j*N+i)/(4.*N*size),ADD_VALUES));
          }
        }
        CHKERRQ(MatSetValues(A,1,&i,PetscMin(6,N),cols,vals,ADD_VALUES));
        CHKERRQ(MatSetValues(B,1,&i,PetscMin(6,N),cols,vals,ADD_VALUES));
      }
    } else {
      PetscInt  rows[2];
      PetscBool test_offproc = PETSC_FALSE;

      CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-test_offproc",&test_offproc,NULL));
      if (test_offproc) {
        const PetscInt *ranges;
        PetscMPIInt    rank;

        CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
        CHKERRQ(MatGetOwnershipRanges(A,&ranges));
        rows[0] = ranges[(rank+1)%size];
        rows[1] = ranges[(rank+1)%size + 1];
      } else {
        CHKERRQ(MatGetOwnershipRange(A,&rows[0],&rows[1]));
      }
      for (i=rows[0];i<rows[1];i++) {
        PetscInt    cols[] = {0,1,2,3,4,5};
        PetscScalar vals[] = {-1,1,-2,2,-3,3};

        CHKERRQ(MatSetValues(A,1,&i,PetscMin(6,N),cols,vals,INSERT_VALUES));
        CHKERRQ(MatSetValues(B,1,&i,PetscMin(6,N),cols,vals,INSERT_VALUES));
      }
    }
    /* MAT_FLUSH_ASSEMBLY currently not supported */
    CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));

#if defined(PETSC_USE_COMPLEX)
    /* make the matrix imaginary */
    CHKERRQ(MatScale(A,PETSC_i));
    CHKERRQ(MatScale(B,PETSC_i));
#endif

    /* MatAXPY further exercises MatSetValues_HYPRE */
    CHKERRQ(MatAXPY(B,-1.,A,DIFFERENT_NONZERO_PATTERN));
    CHKERRQ(MatConvert(B,MATMPIAIJ,MAT_INITIAL_MATRIX,&C));
    CHKERRQ(MatNorm(C,NORM_INFINITY,&err));
    PetscCheck(err <= PETSC_SMALL,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Error MatSetValues %g",err);
    CHKERRQ(MatDestroy(&B));
    CHKERRQ(MatDestroy(&C));
  } else {
    PetscViewer viewer;

    CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&viewer));
    CHKERRQ(MatSetFromOptions(A));
    CHKERRQ(MatLoad(A,viewer));
    CHKERRQ(PetscViewerDestroy(&viewer));
    CHKERRQ(MatGetSize(A,&M,&N));
  }

  /* check conversion routines */
  CHKERRQ(MatConvert(A,MATHYPRE,MAT_INITIAL_MATRIX,&B));
  CHKERRQ(MatConvert(A,MATHYPRE,MAT_REUSE_MATRIX,&B));
  CHKERRQ(MatMultEqual(B,A,4,&flg));
  PetscCheck(flg,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Error Mat HYPRE");
  CHKERRQ(MatConvert(B,MATIS,MAT_INITIAL_MATRIX,&D));
  CHKERRQ(MatConvert(B,MATIS,MAT_REUSE_MATRIX,&D));
  CHKERRQ(MatMultEqual(D,A,4,&flg));
  PetscCheck(flg,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Error Mat IS");
  CHKERRQ(MatConvert(B,MATAIJ,MAT_INITIAL_MATRIX,&C));
  CHKERRQ(MatConvert(B,MATAIJ,MAT_REUSE_MATRIX,&C));
  CHKERRQ(MatMultEqual(C,A,4,&flg));
  PetscCheck(flg,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Error Mat AIJ");
  CHKERRQ(MatAXPY(C,-1.,A,SAME_NONZERO_PATTERN));
  CHKERRQ(MatNorm(C,NORM_INFINITY,&err));
  PetscCheck(err <= PETSC_SMALL,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Error Mat AIJ %g",err);
  CHKERRQ(MatDestroy(&C));
  CHKERRQ(MatConvert(D,MATAIJ,MAT_INITIAL_MATRIX,&C));
  CHKERRQ(MatAXPY(C,-1.,A,SAME_NONZERO_PATTERN));
  CHKERRQ(MatNorm(C,NORM_INFINITY,&err));
  PetscCheck(err <= PETSC_SMALL,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Error Mat IS %g",err);
  CHKERRQ(MatDestroy(&C));
  CHKERRQ(MatDestroy(&D));

  /* check MatCreateFromParCSR */
  CHKERRQ(MatHYPREGetParCSR(B,&parcsr));
  CHKERRQ(MatCreateFromParCSR(parcsr,MATAIJ,PETSC_COPY_VALUES,&D));
  CHKERRQ(MatDestroy(&D));
  CHKERRQ(MatCreateFromParCSR(parcsr,MATHYPRE,PETSC_USE_POINTER,&C));

  /* check MatMult operations */
  CHKERRQ(MatMultEqual(A,B,4,&flg));
  PetscCheck(flg,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Error MatMult B");
  CHKERRQ(MatMultEqual(A,C,4,&flg));
  PetscCheck(flg,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Error MatMult C");
  CHKERRQ(MatMultAddEqual(A,B,4,&flg));
  PetscCheck(flg,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Error MatMultAdd B");
  CHKERRQ(MatMultAddEqual(A,C,4,&flg));
  PetscCheck(flg,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Error MatMultAdd C");
  CHKERRQ(MatMultTransposeEqual(A,B,4,&flg));
  PetscCheck(flg,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Error MatMultTranspose B");
  CHKERRQ(MatMultTransposeEqual(A,C,4,&flg));
  PetscCheck(flg,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Error MatMultTranspose C");
  CHKERRQ(MatMultTransposeAddEqual(A,B,4,&flg));
  PetscCheck(flg,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Error MatMultTransposeAdd B");
  CHKERRQ(MatMultTransposeAddEqual(A,C,4,&flg));
  PetscCheck(flg,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Error MatMultTransposeAdd C");

  /* check PtAP */
  if (testptap && M == N) {
    Mat pP,hP;

    /* PETSc MatPtAP -> output is a MatAIJ
       It uses HYPRE functions when -matptap_via hypre is specified at command line */
    CHKERRQ(MatPtAP(A,A,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&pP));
    CHKERRQ(MatPtAP(A,A,MAT_REUSE_MATRIX,PETSC_DEFAULT,&pP));
    CHKERRQ(MatNorm(pP,NORM_INFINITY,&norm));
    CHKERRQ(MatPtAPMultEqual(A,A,pP,10,&flg));
    PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Error in MatPtAP_MatAIJ");

    /* MatPtAP_HYPRE_HYPRE -> output is a MatHYPRE */
    CHKERRQ(MatPtAP(C,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&hP));
    CHKERRQ(MatPtAP(C,B,MAT_REUSE_MATRIX,PETSC_DEFAULT,&hP));
    CHKERRQ(MatPtAPMultEqual(C,B,hP,10,&flg));
    PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Error in MatPtAP_HYPRE_HYPRE");

    /* Test MatAXPY_Basic() */
    CHKERRQ(MatAXPY(hP,-1.,pP,DIFFERENT_NONZERO_PATTERN));
    CHKERRQ(MatHasOperation(hP,MATOP_NORM,&flg));
    if (!flg) { /* TODO add MatNorm_HYPRE */
      CHKERRQ(MatConvert(hP,MATAIJ,MAT_INPLACE_MATRIX,&hP));
    }
    CHKERRQ(MatNorm(hP,NORM_INFINITY,&err));
    PetscCheckFalse(err/norm > PETSC_SMALL,PetscObjectComm((PetscObject)hP),PETSC_ERR_PLIB,"Error MatPtAP %g %g",err,norm);
    CHKERRQ(MatDestroy(&pP));
    CHKERRQ(MatDestroy(&hP));

    /* MatPtAP_AIJ_HYPRE -> output can be decided at runtime with -matptap_hypre_outtype */
    CHKERRQ(MatPtAP(A,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&hP));
    CHKERRQ(MatPtAP(A,B,MAT_REUSE_MATRIX,PETSC_DEFAULT,&hP));
    CHKERRQ(MatPtAPMultEqual(A,B,hP,10,&flg));
    PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Error in MatPtAP_AIJ_HYPRE");
    CHKERRQ(MatDestroy(&hP));
  }
  CHKERRQ(MatDestroy(&C));
  CHKERRQ(MatDestroy(&B));

  /* check MatMatMult */
  if (testmatmatmult) {
    CHKERRQ(MatTranspose(A,MAT_INITIAL_MATRIX,&B));
    CHKERRQ(MatConvert(A,MATHYPRE,MAT_INITIAL_MATRIX,&C));
    CHKERRQ(MatConvert(B,MATHYPRE,MAT_INITIAL_MATRIX,&D));

    /* PETSc MatMatMult -> output is a MatAIJ
       It uses HYPRE functions when -matmatmult_via hypre is specified at command line */
    CHKERRQ(MatMatMult(A,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&pAB));
    CHKERRQ(MatMatMult(A,B,MAT_REUSE_MATRIX,PETSC_DEFAULT,&pAB));
    CHKERRQ(MatNorm(pAB,NORM_INFINITY,&norm));
    CHKERRQ(MatMatMultEqual(A,B,pAB,10,&flg));
    PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Error in MatMatMult_AIJ_AIJ");

    /* MatMatMult_HYPRE_HYPRE -> output is a MatHYPRE */
    CHKERRQ(MatMatMult(C,D,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&CD));
    CHKERRQ(MatMatMult(C,D,MAT_REUSE_MATRIX,PETSC_DEFAULT,&CD));
    CHKERRQ(MatMatMultEqual(C,D,CD,10,&flg));
    PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Error in MatMatMult_HYPRE_HYPRE");

    /* Test MatAXPY_Basic() */
    CHKERRQ(MatAXPY(CD,-1.,pAB,DIFFERENT_NONZERO_PATTERN));

    CHKERRQ(MatHasOperation(CD,MATOP_NORM,&flg));
    if (!flg) { /* TODO add MatNorm_HYPRE */
      CHKERRQ(MatConvert(CD,MATAIJ,MAT_INPLACE_MATRIX,&CD));
    }
    CHKERRQ(MatNorm(CD,NORM_INFINITY,&err));
    PetscCheck((err/norm) <= PETSC_SMALL,PetscObjectComm((PetscObject)CD),PETSC_ERR_PLIB,"Error MatMatMult %g %g",err,norm);

    CHKERRQ(MatDestroy(&C));
    CHKERRQ(MatDestroy(&D));
    CHKERRQ(MatDestroy(&pAB));
    CHKERRQ(MatDestroy(&CD));

    /* When configured with HYPRE, MatMatMatMult is available for the triplet transpose(aij)-aij-aij */
    CHKERRQ(MatCreateTranspose(A,&C));
    CHKERRQ(MatMatMatMult(C,A,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&CAB));
    CHKERRQ(MatDestroy(&C));
    CHKERRQ(MatTranspose(A,MAT_INITIAL_MATRIX,&C));
    CHKERRQ(MatMatMult(C,A,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&D));
    CHKERRQ(MatDestroy(&C));
    CHKERRQ(MatMatMult(D,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&C));
    CHKERRQ(MatNorm(C,NORM_INFINITY,&norm));
    CHKERRQ(MatAXPY(C,-1.,CAB,DIFFERENT_NONZERO_PATTERN));
    CHKERRQ(MatNorm(C,NORM_INFINITY,&err));
    PetscCheck((err/norm) <= PETSC_SMALL,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Error MatMatMatMult %g %g",err,norm);
    CHKERRQ(MatDestroy(&C));
    CHKERRQ(MatDestroy(&D));
    CHKERRQ(MatDestroy(&CAB));
    CHKERRQ(MatDestroy(&B));
  }

  /* Check MatView */
  CHKERRQ(MatViewFromOptions(A,NULL,"-view_A"));
  CHKERRQ(MatConvert(A,MATHYPRE,MAT_INITIAL_MATRIX,&B));
  CHKERRQ(MatViewFromOptions(B,NULL,"-view_B"));

  /* Check MatDuplicate/MatCopy */
  for (j=0;j<3;j++) {
    MatDuplicateOption dop;

    dop = MAT_COPY_VALUES;
    if (j==1) dop = MAT_DO_NOT_COPY_VALUES;
    if (j==2) dop = MAT_SHARE_NONZERO_PATTERN;

    for (i=0;i<3;i++) {
      MatStructure str;

      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Dup/Copy tests: %" PetscInt_FMT " %" PetscInt_FMT "\n",j,i));

      str = DIFFERENT_NONZERO_PATTERN;
      if (i==1) str = SAME_NONZERO_PATTERN;
      if (i==2) str = SUBSET_NONZERO_PATTERN;

      CHKERRQ(MatDuplicate(A,dop,&C));
      CHKERRQ(MatDuplicate(B,dop,&D));
      if (dop != MAT_COPY_VALUES) {
        CHKERRQ(MatCopy(A,C,str));
        CHKERRQ(MatCopy(B,D,str));
      }
      /* AXPY with AIJ and HYPRE */
      CHKERRQ(MatAXPY(C,-1.0,D,str));
      CHKERRQ(MatNorm(C,NORM_INFINITY,&err));
      if (err > PETSC_SMALL) {
        CHKERRQ(MatViewFromOptions(A,NULL,"-view_duplicate_diff"));
        CHKERRQ(MatViewFromOptions(B,NULL,"-view_duplicate_diff"));
        CHKERRQ(MatViewFromOptions(C,NULL,"-view_duplicate_diff"));
        SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Error test 1 MatDuplicate/MatCopy %g (%" PetscInt_FMT ",%" PetscInt_FMT ")",err,j,i);
      }
      /* AXPY with HYPRE and HYPRE */
      CHKERRQ(MatAXPY(D,-1.0,B,str));
      if (err > PETSC_SMALL) {
        CHKERRQ(MatViewFromOptions(A,NULL,"-view_duplicate_diff"));
        CHKERRQ(MatViewFromOptions(B,NULL,"-view_duplicate_diff"));
        CHKERRQ(MatViewFromOptions(D,NULL,"-view_duplicate_diff"));
        SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Error test 2 MatDuplicate/MatCopy %g (%" PetscInt_FMT ",%" PetscInt_FMT ")",err,j,i);
      }
      /* Copy from HYPRE to AIJ */
      CHKERRQ(MatCopy(B,C,str));
      /* Copy from AIJ to HYPRE */
      CHKERRQ(MatCopy(A,D,str));
      /* AXPY with HYPRE and AIJ */
      CHKERRQ(MatAXPY(D,-1.0,C,str));
      CHKERRQ(MatHasOperation(D,MATOP_NORM,&flg));
      if (!flg) { /* TODO add MatNorm_HYPRE */
        CHKERRQ(MatConvert(D,MATAIJ,MAT_INPLACE_MATRIX,&D));
      }
      CHKERRQ(MatNorm(D,NORM_INFINITY,&err));
      if (err > PETSC_SMALL) {
        CHKERRQ(MatViewFromOptions(A,NULL,"-view_duplicate_diff"));
        CHKERRQ(MatViewFromOptions(C,NULL,"-view_duplicate_diff"));
        CHKERRQ(MatViewFromOptions(D,NULL,"-view_duplicate_diff"));
        SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Error test 3 MatDuplicate/MatCopy %g (%" PetscInt_FMT ",%" PetscInt_FMT ")",err,j,i);
      }
      CHKERRQ(MatDestroy(&C));
      CHKERRQ(MatDestroy(&D));
    }
  }
  CHKERRQ(MatDestroy(&B));

  CHKERRQ(MatHasCongruentLayouts(A,&flg));
  if (flg) {
    Vec y,y2;

    CHKERRQ(MatConvert(A,MATHYPRE,MAT_INITIAL_MATRIX,&B));
    CHKERRQ(MatCreateVecs(A,NULL,&y));
    CHKERRQ(MatCreateVecs(B,NULL,&y2));
    CHKERRQ(MatGetDiagonal(A,y));
    CHKERRQ(MatGetDiagonal(B,y2));
    CHKERRQ(VecAXPY(y2,-1.0,y));
    CHKERRQ(VecNorm(y2,NORM_INFINITY,&err));
    if (err > PETSC_SMALL) {
      CHKERRQ(VecViewFromOptions(y,NULL,"-view_diagonal_diff"));
      CHKERRQ(VecViewFromOptions(y2,NULL,"-view_diagonal_diff"));
      SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Error MatGetDiagonal %g",err);
    }
    CHKERRQ(MatDestroy(&B));
    CHKERRQ(VecDestroy(&y));
    CHKERRQ(VecDestroy(&y2));
  }

  CHKERRQ(MatDestroy(&A));

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
