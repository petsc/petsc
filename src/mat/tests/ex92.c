
static char help[] = "Tests MatIncreaseOverlap(), MatCreateSubMatrices() for parallel MatSBAIJ format.\n";
/* Example of usage:
      mpiexec -n 2 ./ex92 -nd 2 -ov 3 -mat_block_size 2 -view_id 0 -test_overlap -test_submat
*/
#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            A,Atrans,sA,*submatA,*submatsA;
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

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  PetscCall(PetscOptionsGetInt(NULL,NULL,"-mat_block_size",&bs,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-mat_mbs",&mbs,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-ov",&ov,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-nd",&nd,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-view_id",&vid,NULL));
  PetscCall(PetscOptionsHasName(NULL,NULL, "-test_overlap", &TestOverlap));
  PetscCall(PetscOptionsHasName(NULL,NULL, "-test_submat", &TestSubMat));
  PetscCall(PetscOptionsHasName(NULL,NULL, "-test_allcols", &TestAllcols));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-test_sorted",&test_sorted,NULL));

  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,mbs*bs,mbs*bs,PETSC_DECIDE,PETSC_DECIDE));
  PetscCall(MatSetType(A,MATBAIJ));
  PetscCall(MatSeqBAIJSetPreallocation(A,bs,PETSC_DEFAULT,NULL));
  PetscCall(MatMPIBAIJSetPreallocation(A,bs,PETSC_DEFAULT,NULL,PETSC_DEFAULT,NULL));

  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD,&rand));
  PetscCall(PetscRandomSetFromOptions(rand));

  PetscCall(MatGetOwnershipRange(A,&rstart,&rend));
  PetscCall(MatGetSize(A,&M,&N));
  Mbs  = M/bs;

  PetscCall(PetscMalloc1(bs,&rows));
  PetscCall(PetscMalloc1(bs,&cols));
  PetscCall(PetscMalloc1(bs*bs,&vals));
  PetscCall(PetscMalloc1(M,&idx));

  /* Now set blocks of values */
  for (j=0; j<bs*bs; j++) vals[j] = 0.0;
  for (i=0; i<Mbs; i++) {
    cols[0] = i*bs; rows[0] = i*bs;
    for (j=1; j<bs; j++) {
      rows[j] = rows[j-1]+1;
      cols[j] = cols[j-1]+1;
    }
    PetscCall(MatSetValues(A,bs,rows,bs,cols,vals,ADD_VALUES));
  }
  /* second, add random blocks */
  for (i=0; i<20*bs; i++) {
    PetscCall(PetscRandomGetValue(rand,&rval));
    cols[0] = bs*(PetscInt)(PetscRealPart(rval)*Mbs);
    PetscCall(PetscRandomGetValue(rand,&rval));
    rows[0] = rstart + bs*(PetscInt)(PetscRealPart(rval)*mbs);
    for (j=1; j<bs; j++) {
      rows[j] = rows[j-1]+1;
      cols[j] = cols[j-1]+1;
    }

    for (j=0; j<bs*bs; j++) {
      PetscCall(PetscRandomGetValue(rand,&rval));
      vals[j] = rval;
    }
    PetscCall(MatSetValues(A,bs,rows,bs,cols,vals,ADD_VALUES));
  }

  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* make A a symmetric matrix: A <- A^T + A */
  PetscCall(MatTranspose(A,MAT_INITIAL_MATRIX, &Atrans));
  PetscCall(MatAXPY(A,one,Atrans,DIFFERENT_NONZERO_PATTERN));
  PetscCall(MatDestroy(&Atrans));
  PetscCall(MatTranspose(A,MAT_INITIAL_MATRIX, &Atrans));
  PetscCall(MatEqual(A, Atrans, &flg));
  if (flg) {
    PetscCall(MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE));
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"A+A^T is non-symmetric");
  PetscCall(MatDestroy(&Atrans));

  /* create a SeqSBAIJ matrix sA (= A) */
  PetscCall(MatConvert(A,MATSBAIJ,MAT_INITIAL_MATRIX,&sA));
  if (vid >= 0 && vid < size) {
    PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"A:\n"));
    PetscCall(MatView(A,PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"sA:\n"));
    PetscCall(MatView(sA,PETSC_VIEWER_STDOUT_WORLD));
  }

  /* Test sA==A through MatMult() */
  PetscCall(MatMultEqual(A,sA,10,&flg));
  PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Error in MatConvert(): A != sA");

  /* Test MatIncreaseOverlap() */
  PetscCall(PetscMalloc1(nd,&is1));
  PetscCall(PetscMalloc1(nd,&is2));

  for (i=0; i<nd; i++) {
    if (!TestAllcols) {
      PetscCall(PetscRandomGetValue(rand,&rval));
      sz   = (PetscInt)((0.5+0.2*PetscRealPart(rval))*mbs); /* 0.5*mbs < sz < 0.7*mbs */

      for (j=0; j<sz; j++) {
        PetscCall(PetscRandomGetValue(rand,&rval));
        idx[j*bs] = bs*(PetscInt)(PetscRealPart(rval)*Mbs);
        for (k=1; k<bs; k++) idx[j*bs+k] = idx[j*bs]+k;
      }
      PetscCall(ISCreateGeneral(PETSC_COMM_SELF,sz*bs,idx,PETSC_COPY_VALUES,is1+i));
      PetscCall(ISCreateGeneral(PETSC_COMM_SELF,sz*bs,idx,PETSC_COPY_VALUES,is2+i));
      if (rank == vid) {
        PetscCall(PetscPrintf(PETSC_COMM_SELF," [%d] IS sz[%" PetscInt_FMT "]: %" PetscInt_FMT "\n",rank,i,sz));
        PetscCall(ISView(is2[i],PETSC_VIEWER_STDOUT_SELF));
      }
    } else { /* Test all rows and columns */
      sz   = M;
      PetscCall(ISCreateStride(PETSC_COMM_SELF,sz,0,1,is1+i));
      PetscCall(ISCreateStride(PETSC_COMM_SELF,sz,0,1,is2+i));

      if (rank == vid) {
        PetscBool colflag;
        PetscCall(ISIdentity(is2[i],&colflag));
        PetscCall(PetscPrintf(PETSC_COMM_SELF,"[%d] is2[%" PetscInt_FMT "], colflag %d\n",rank,i,colflag));
        PetscCall(ISView(is2[i],PETSC_VIEWER_STDOUT_SELF));
      }
    }
  }

  PetscCall(PetscLogStageRegister("MatOv_SBAIJ",&stages[0]));
  PetscCall(PetscLogStageRegister("MatOv_BAIJ",&stages[1]));

  /* Test MatIncreaseOverlap */
  if (TestOverlap) {
    PetscCall(PetscLogStagePush(stages[0]));
    PetscCall(MatIncreaseOverlap(sA,nd,is2,ov));
    PetscCall(PetscLogStagePop());

    PetscCall(PetscLogStagePush(stages[1]));
    PetscCall(MatIncreaseOverlap(A,nd,is1,ov));
    PetscCall(PetscLogStagePop());

    if (rank == vid) {
      PetscCall(PetscPrintf(PETSC_COMM_SELF,"\n[%d] IS from BAIJ:\n",rank));
      PetscCall(ISView(is1[0],PETSC_VIEWER_STDOUT_SELF));
      PetscCall(PetscPrintf(PETSC_COMM_SELF,"\n[%d] IS from SBAIJ:\n",rank));
      PetscCall(ISView(is2[0],PETSC_VIEWER_STDOUT_SELF));
    }

    for (i=0; i<nd; ++i) {
      PetscCall(ISEqual(is1[i],is2[i],&flg));
      if (!flg) {
        if (rank == 0) {
          PetscCall(ISSort(is1[i]));
          PetscCall(ISSort(is2[i]));
        }
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"i=%" PetscInt_FMT ", is1 != is2",i);
      }
    }
  }

  /* Test MatCreateSubmatrices */
  if (TestSubMat) {
    if (test_sorted) {
      for (i = 0; i < nd; ++i) {
        PetscCall(ISSort(is1[i]));
      }
    }
    PetscCall(MatCreateSubMatrices(A,nd,is1,is1,MAT_INITIAL_MATRIX,&submatA));
    PetscCall(MatCreateSubMatrices(sA,nd,is1,is1,MAT_INITIAL_MATRIX,&submatsA));

    PetscCall(MatMultEqual(A,sA,10,&flg));
    PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"A != sA");

    /* Now test MatCreateSubmatrices with MAT_REUSE_MATRIX option */
    PetscCall(MatCreateSubMatrices(A,nd,is1,is1,MAT_REUSE_MATRIX,&submatA));
    PetscCall(MatCreateSubMatrices(sA,nd,is1,is1,MAT_REUSE_MATRIX,&submatsA));
    PetscCall(MatMultEqual(A,sA,10,&flg));
    PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"MatCreateSubmatrices(): A != sA");

    PetscCall(MatDestroySubMatrices(nd,&submatA));
    PetscCall(MatDestroySubMatrices(nd,&submatsA));
  }

  /* Free allocated memory */
  for (i=0; i<nd; ++i) {
    PetscCall(ISDestroy(&is1[i]));
    PetscCall(ISDestroy(&is2[i]));
  }
  PetscCall(PetscFree(is1));
  PetscCall(PetscFree(is2));
  PetscCall(PetscFree(idx));
  PetscCall(PetscFree(rows));
  PetscCall(PetscFree(cols));
  PetscCall(PetscFree(vals));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&sA));
  PetscCall(PetscRandomDestroy(&rand));
  PetscCall(PetscFinalize());
  return 0;
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
