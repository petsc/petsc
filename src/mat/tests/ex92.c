
static char help[] = "Tests MatIncreaseOverlap(), MatCreateSubMatrices() for parallel MatSBAIJ format.\n";
/* Example of usage:
      mpiexec -n 2 ./ex92 -nd 2 -ov 3 -mat_block_size 2 -view_id 0 -test_overlap -test_submat
*/
#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            A,Atrans,sA,*submatA,*submatsA;
  PetscErrorCode ierr;
  PetscMPIInt    size,rank;
  PetscInt       bs=1,mbs=10,ov=1,i,j,k,*rows,*cols,nd=2,*idx,rstart,rend,sz,M,N,Mbs;
  PetscScalar    *vals,rval,one=1.0;
  IS             *is1,*is2;
  PetscRandom    rand;
  PetscBool      flg,TestOverlap,TestSubMat,TestAllcols,test_sorted=PETSC_FALSE;
  PetscInt       vid = -1;
#if defined(PETSC_USE_LOG)
  PetscLogStage  stages[2];
#endif

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-mat_block_size",&bs,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-mat_mbs",&mbs,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-ov",&ov,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-nd",&nd,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-view_id",&vid,NULL));
  CHKERRQ(PetscOptionsHasName(NULL,NULL, "-test_overlap", &TestOverlap));
  CHKERRQ(PetscOptionsHasName(NULL,NULL, "-test_submat", &TestSubMat));
  CHKERRQ(PetscOptionsHasName(NULL,NULL, "-test_allcols", &TestAllcols));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-test_sorted",&test_sorted,NULL));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,mbs*bs,mbs*bs,PETSC_DECIDE,PETSC_DECIDE));
  CHKERRQ(MatSetType(A,MATBAIJ));
  CHKERRQ(MatSeqBAIJSetPreallocation(A,bs,PETSC_DEFAULT,NULL));
  CHKERRQ(MatMPIBAIJSetPreallocation(A,bs,PETSC_DEFAULT,NULL,PETSC_DEFAULT,NULL));

  CHKERRQ(PetscRandomCreate(PETSC_COMM_WORLD,&rand));
  CHKERRQ(PetscRandomSetFromOptions(rand));

  CHKERRQ(MatGetOwnershipRange(A,&rstart,&rend));
  CHKERRQ(MatGetSize(A,&M,&N));
  Mbs  = M/bs;

  CHKERRQ(PetscMalloc1(bs,&rows));
  CHKERRQ(PetscMalloc1(bs,&cols));
  CHKERRQ(PetscMalloc1(bs*bs,&vals));
  CHKERRQ(PetscMalloc1(M,&idx));

  /* Now set blocks of values */
  for (j=0; j<bs*bs; j++) vals[j] = 0.0;
  for (i=0; i<Mbs; i++) {
    cols[0] = i*bs; rows[0] = i*bs;
    for (j=1; j<bs; j++) {
      rows[j] = rows[j-1]+1;
      cols[j] = cols[j-1]+1;
    }
    CHKERRQ(MatSetValues(A,bs,rows,bs,cols,vals,ADD_VALUES));
  }
  /* second, add random blocks */
  for (i=0; i<20*bs; i++) {
    CHKERRQ(PetscRandomGetValue(rand,&rval));
    cols[0] = bs*(PetscInt)(PetscRealPart(rval)*Mbs);
    CHKERRQ(PetscRandomGetValue(rand,&rval));
    rows[0] = rstart + bs*(PetscInt)(PetscRealPart(rval)*mbs);
    for (j=1; j<bs; j++) {
      rows[j] = rows[j-1]+1;
      cols[j] = cols[j-1]+1;
    }

    for (j=0; j<bs*bs; j++) {
      CHKERRQ(PetscRandomGetValue(rand,&rval));
      vals[j] = rval;
    }
    CHKERRQ(MatSetValues(A,bs,rows,bs,cols,vals,ADD_VALUES));
  }

  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* make A a symmetric matrix: A <- A^T + A */
  CHKERRQ(MatTranspose(A,MAT_INITIAL_MATRIX, &Atrans));
  CHKERRQ(MatAXPY(A,one,Atrans,DIFFERENT_NONZERO_PATTERN));
  CHKERRQ(MatDestroy(&Atrans));
  CHKERRQ(MatTranspose(A,MAT_INITIAL_MATRIX, &Atrans));
  CHKERRQ(MatEqual(A, Atrans, &flg));
  if (flg) {
    CHKERRQ(MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE));
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"A+A^T is non-symmetric");
  CHKERRQ(MatDestroy(&Atrans));

  /* create a SeqSBAIJ matrix sA (= A) */
  CHKERRQ(MatConvert(A,MATSBAIJ,MAT_INITIAL_MATRIX,&sA));
  if (vid >= 0 && vid < size) {
    CHKERRQ(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"A:\n"));
    CHKERRQ(MatView(A,PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"sA:\n"));
    CHKERRQ(MatView(sA,PETSC_VIEWER_STDOUT_WORLD));
  }

  /* Test sA==A through MatMult() */
  CHKERRQ(MatMultEqual(A,sA,10,&flg));
  PetscCheckFalse(!flg,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Error in MatConvert(): A != sA");

  /* Test MatIncreaseOverlap() */
  CHKERRQ(PetscMalloc1(nd,&is1));
  CHKERRQ(PetscMalloc1(nd,&is2));

  for (i=0; i<nd; i++) {
    if (!TestAllcols) {
      CHKERRQ(PetscRandomGetValue(rand,&rval));
      sz   = (PetscInt)((0.5+0.2*PetscRealPart(rval))*mbs); /* 0.5*mbs < sz < 0.7*mbs */

      for (j=0; j<sz; j++) {
        CHKERRQ(PetscRandomGetValue(rand,&rval));
        idx[j*bs] = bs*(PetscInt)(PetscRealPart(rval)*Mbs);
        for (k=1; k<bs; k++) idx[j*bs+k] = idx[j*bs]+k;
      }
      CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,sz*bs,idx,PETSC_COPY_VALUES,is1+i));
      CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,sz*bs,idx,PETSC_COPY_VALUES,is2+i));
      if (rank == vid) {
        CHKERRQ(PetscPrintf(PETSC_COMM_SELF," [%d] IS sz[%" PetscInt_FMT "]: %" PetscInt_FMT "\n",rank,i,sz));
        CHKERRQ(ISView(is2[i],PETSC_VIEWER_STDOUT_SELF));
      }
    } else { /* Test all rows and columns */
      sz   = M;
      CHKERRQ(ISCreateStride(PETSC_COMM_SELF,sz,0,1,is1+i));
      CHKERRQ(ISCreateStride(PETSC_COMM_SELF,sz,0,1,is2+i));

      if (rank == vid) {
        PetscBool colflag;
        CHKERRQ(ISIdentity(is2[i],&colflag));
        CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"[%d] is2[%" PetscInt_FMT "], colflag %d\n",rank,i,colflag));
        CHKERRQ(ISView(is2[i],PETSC_VIEWER_STDOUT_SELF));
      }
    }
  }

  CHKERRQ(PetscLogStageRegister("MatOv_SBAIJ",&stages[0]));
  CHKERRQ(PetscLogStageRegister("MatOv_BAIJ",&stages[1]));

  /* Test MatIncreaseOverlap */
  if (TestOverlap) {
    CHKERRQ(PetscLogStagePush(stages[0]));
    CHKERRQ(MatIncreaseOverlap(sA,nd,is2,ov));
    CHKERRQ(PetscLogStagePop());

    CHKERRQ(PetscLogStagePush(stages[1]));
    CHKERRQ(MatIncreaseOverlap(A,nd,is1,ov));
    CHKERRQ(PetscLogStagePop());

    if (rank == vid) {
      CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"\n[%d] IS from BAIJ:\n",rank));
      CHKERRQ(ISView(is1[0],PETSC_VIEWER_STDOUT_SELF));
      CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"\n[%d] IS from SBAIJ:\n",rank));
      CHKERRQ(ISView(is2[0],PETSC_VIEWER_STDOUT_SELF));
    }

    for (i=0; i<nd; ++i) {
      CHKERRQ(ISEqual(is1[i],is2[i],&flg));
      if (!flg) {
        if (rank == 0) {
          CHKERRQ(ISSort(is1[i]));
          CHKERRQ(ISSort(is2[i]));
        }
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"i=%" PetscInt_FMT ", is1 != is2",i);
      }
    }
  }

  /* Test MatCreateSubmatrices */
  if (TestSubMat) {
    if (test_sorted) {
      for (i = 0; i < nd; ++i) {
        CHKERRQ(ISSort(is1[i]));
      }
    }
    CHKERRQ(MatCreateSubMatrices(A,nd,is1,is1,MAT_INITIAL_MATRIX,&submatA));
    CHKERRQ(MatCreateSubMatrices(sA,nd,is1,is1,MAT_INITIAL_MATRIX,&submatsA));

    CHKERRQ(MatMultEqual(A,sA,10,&flg));
    PetscCheckFalse(!flg,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"A != sA");

    /* Now test MatCreateSubmatrices with MAT_REUSE_MATRIX option */
    CHKERRQ(MatCreateSubMatrices(A,nd,is1,is1,MAT_REUSE_MATRIX,&submatA));
    CHKERRQ(MatCreateSubMatrices(sA,nd,is1,is1,MAT_REUSE_MATRIX,&submatsA));
    CHKERRQ(MatMultEqual(A,sA,10,&flg));
    PetscCheckFalse(!flg,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"MatCreateSubmatrices(): A != sA");

    CHKERRQ(MatDestroySubMatrices(nd,&submatA));
    CHKERRQ(MatDestroySubMatrices(nd,&submatsA));
  }

  /* Free allocated memory */
  for (i=0; i<nd; ++i) {
    CHKERRQ(ISDestroy(&is1[i]));
    CHKERRQ(ISDestroy(&is2[i]));
  }
  CHKERRQ(PetscFree(is1));
  CHKERRQ(PetscFree(is2));
  CHKERRQ(PetscFree(idx));
  CHKERRQ(PetscFree(rows));
  CHKERRQ(PetscFree(cols));
  CHKERRQ(PetscFree(vals));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&sA));
  CHKERRQ(PetscRandomDestroy(&rand));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      args: -ov {{1 3}} -mat_block_size {{2 8}} -test_overlap -test_submat
      output_file: output/ex92_1.out

   test:
      suffix: 2
      nsize: {{3 4}}
      args: -ov {{1 3}} -mat_block_size {{2 8}} -test_overlap -test_submat
      output_file: output/ex92_1.out

   test:
      suffix: 3
      nsize: {{3 4}}
      args: -ov {{1 3}} -mat_block_size {{2 8}} -test_overlap -test_allcols
      output_file: output/ex92_1.out

   test:
      suffix: 3_sorted
      nsize: {{3 4}}
      args: -ov {{1 3}} -mat_block_size {{2 8}} -test_overlap -test_allcols -test_sorted
      output_file: output/ex92_1.out

   test:
      suffix: 4
      nsize: {{3 4}}
      args: -ov {{1 3}} -mat_block_size {{2 8}} -test_submat -test_allcols
      output_file: output/ex92_1.out

TEST*/
