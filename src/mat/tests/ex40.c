
static char help[] = "Tests the parallel case for MatIncreaseOverlap(). Input arguments are:\n\
  -f <input_file> : file to load.  For example see $PETSC_DIR/share/petsc/datafiles/matrices\n\
  -nd <size>      : > 0  number of domains per processor \n\
  -ov <overlap>   : >=0  amount of overlap between domains\n\n";

#include <petscmat.h>

PetscErrorCode ISAllGatherDisjoint(IS iis, IS** ois)
{
  IS             *is2,is;
  const PetscInt *idxs;
  PetscInt       i, ls,*sizes;
  PetscMPIInt    size;

  PetscFunctionBeginUser;
  CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)iis),&size));
  CHKERRQ(PetscMalloc1(size,&is2));
  CHKERRQ(PetscMalloc1(size,&sizes));
  CHKERRQ(ISGetLocalSize(iis,&ls));
  /* we don't have a public ISGetLayout */
  CHKERRMPI(MPI_Allgather(&ls,1,MPIU_INT,sizes,1,MPIU_INT,PetscObjectComm((PetscObject)iis)));
  CHKERRQ(ISAllGather(iis,&is));
  CHKERRQ(ISGetIndices(is,&idxs));
  for (i = 0, ls = 0; i < size; i++) {
    CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,sizes[i],idxs+ls,PETSC_COPY_VALUES,&is2[i]));
    ls += sizes[i];
  }
  CHKERRQ(ISRestoreIndices(is,&idxs));
  CHKERRQ(ISDestroy(&is));
  CHKERRQ(PetscFree(sizes));
  *ois = is2;
  PetscFunctionReturn(0);
}

int main(int argc,char **args)
{
  PetscInt       nd = 2,ov = 1,ndpar,i,start,m,n,end,lsize;
  PetscMPIInt    rank;
  PetscBool      flg, useND = PETSC_FALSE;
  Mat            A,B;
  char           file[PETSC_MAX_PATH_LEN];
  PetscViewer    fd;
  IS             *is1,*is2;
  PetscRandom    r;
  PetscScalar    rand;

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_USER,"Must use -f filename to indicate a file containing a PETSc binary matrix");
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-nd",&nd,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-ov",&ov,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-nested_dissection",&useND,NULL));

  /* Read matrix */
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetType(A,MATMPIAIJ));
  CHKERRQ(MatLoad(A,fd));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(PetscViewerDestroy(&fd));

  /* Read the matrix again as a sequential matrix */
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_SELF,file,FILE_MODE_READ,&fd));
  CHKERRQ(MatCreate(PETSC_COMM_SELF,&B));
  CHKERRQ(MatSetType(B,MATSEQAIJ));
  CHKERRQ(MatLoad(B,fd));
  CHKERRQ(MatSetFromOptions(B));
  CHKERRQ(PetscViewerDestroy(&fd));

  /* Create the IS corresponding to subdomains */
  if (useND) {
    MatPartitioning part;
    IS              ndmap;
    PetscMPIInt     size;

    ndpar = 1;
    CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
    nd   = (PetscInt)size;
    CHKERRQ(PetscMalloc1(ndpar,&is1));
    CHKERRQ(MatPartitioningCreate(PETSC_COMM_WORLD,&part));
    CHKERRQ(MatPartitioningSetAdjacency(part,A));
    CHKERRQ(MatPartitioningSetFromOptions(part));
    CHKERRQ(MatPartitioningApplyND(part,&ndmap));
    CHKERRQ(MatPartitioningDestroy(&part));
    CHKERRQ(ISBuildTwoSided(ndmap,NULL,&is1[0]));
    CHKERRQ(ISDestroy(&ndmap));
    CHKERRQ(ISAllGatherDisjoint(is1[0],&is2));
  } else {
    /* Create the random Index Sets */
    CHKERRQ(PetscMalloc1(nd,&is1));
    CHKERRQ(PetscMalloc1(nd,&is2));

    CHKERRQ(MatGetSize(A,&m,&n));
    CHKERRQ(PetscRandomCreate(PETSC_COMM_SELF,&r));
    CHKERRQ(PetscRandomSetFromOptions(r));
    for (i=0; i<nd; i++) {
      CHKERRQ(PetscRandomGetValue(r,&rand));
      start = (PetscInt)(rand*m);
      CHKERRQ(PetscRandomGetValue(r,&rand));
      end   = (PetscInt)(rand*m);
      lsize =  end - start;
      if (start > end) { start = end; lsize = -lsize;}
      CHKERRQ(ISCreateStride(PETSC_COMM_SELF,lsize,start,1,is1+i));
      CHKERRQ(ISCreateStride(PETSC_COMM_SELF,lsize,start,1,is2+i));
    }
    ndpar = nd;
    CHKERRQ(PetscRandomDestroy(&r));
  }
  CHKERRQ(MatIncreaseOverlap(A,ndpar,is1,ov));
  CHKERRQ(MatIncreaseOverlap(B,nd,is2,ov));
  if (useND) {
    IS *is;

    CHKERRQ(ISAllGatherDisjoint(is1[0],&is));
    CHKERRQ(ISDestroy(&is1[0]));
    CHKERRQ(PetscFree(is1));
    is1 = is;
  }
  /* Now see if the serial and parallel case have the same answers */
  for (i=0; i<nd; ++i) {
    CHKERRQ(ISEqual(is1[i],is2[i],&flg));
    if (!flg) {
      CHKERRQ(ISViewFromOptions(is1[i],NULL,"-err_view"));
      CHKERRQ(ISViewFromOptions(is2[i],NULL,"-err_view"));
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"proc:[%d], i=%" PetscInt_FMT ", flg =%d",rank,i,(int)flg);
    }
  }

  /* Free allocated memory */
  for (i=0; i<nd; ++i) {
    CHKERRQ(ISDestroy(&is1[i]));
    CHKERRQ(ISDestroy(&is2[i]));
  }
  CHKERRQ(PetscFree(is1));
  CHKERRQ(PetscFree(is2));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   build:
      requires: !complex

   testset:
      nsize: 5
      requires: datafilespath double !defined(PETSC_USE_64BIT_INDICES) !complex
      args: -f ${DATAFILESPATH}/matrices/arco1 -viewer_binary_skip_info -ov 2
      output_file: output/ex40_1.out
      test:
        suffix: 1
        args: -nd 7
      test:
        requires: parmetis
        suffix: 1_nd
        args: -nested_dissection -mat_partitioning_type parmetis

   testset:
      nsize: 3
      requires: double !defined(PETSC_USE_64BIT_INDICES) !complex
      args: -f ${wPETSC_DIR}/share/petsc/datafiles/matrices/ns-real-int32-float64 -mat_increase_overlap_scalable 1 -ov 2
      output_file: output/ex40_1.out
      test:
        suffix: 2
        args: -nd 7
      test:
        requires: parmetis
        suffix: 2_nd
        args: -nested_dissection -mat_partitioning_type parmetis

TEST*/
