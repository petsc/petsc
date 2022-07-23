
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
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)iis),&size));
  PetscCall(PetscMalloc1(size,&is2));
  PetscCall(PetscMalloc1(size,&sizes));
  PetscCall(ISGetLocalSize(iis,&ls));
  /* we don't have a public ISGetLayout */
  PetscCallMPI(MPI_Allgather(&ls,1,MPIU_INT,sizes,1,MPIU_INT,PetscObjectComm((PetscObject)iis)));
  PetscCall(ISAllGather(iis,&is));
  PetscCall(ISGetIndices(is,&idxs));
  for (i = 0, ls = 0; i < size; i++) {
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF,sizes[i],idxs+ls,PETSC_COPY_VALUES,&is2[i]));
    ls += sizes[i];
  }
  PetscCall(ISRestoreIndices(is,&idxs));
  PetscCall(ISDestroy(&is));
  PetscCall(PetscFree(sizes));
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

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCall(PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_USER,"Must use -f filename to indicate a file containing a PETSc binary matrix");
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-nd",&nd,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-ov",&ov,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-nested_dissection",&useND,NULL));

  /* Read matrix */
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd));
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetType(A,MATMPIAIJ));
  PetscCall(MatLoad(A,fd));
  PetscCall(MatSetFromOptions(A));
  PetscCall(PetscViewerDestroy(&fd));

  /* Read the matrix again as a sequential matrix */
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_SELF,file,FILE_MODE_READ,&fd));
  PetscCall(MatCreate(PETSC_COMM_SELF,&B));
  PetscCall(MatSetType(B,MATSEQAIJ));
  PetscCall(MatLoad(B,fd));
  PetscCall(MatSetFromOptions(B));
  PetscCall(PetscViewerDestroy(&fd));

  /* Create the IS corresponding to subdomains */
  if (useND) {
    MatPartitioning part;
    IS              ndmap;
    PetscMPIInt     size;

    ndpar = 1;
    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
    nd   = (PetscInt)size;
    PetscCall(PetscMalloc1(ndpar,&is1));
    PetscCall(MatPartitioningCreate(PETSC_COMM_WORLD,&part));
    PetscCall(MatPartitioningSetAdjacency(part,A));
    PetscCall(MatPartitioningSetFromOptions(part));
    PetscCall(MatPartitioningApplyND(part,&ndmap));
    PetscCall(MatPartitioningDestroy(&part));
    PetscCall(ISBuildTwoSided(ndmap,NULL,&is1[0]));
    PetscCall(ISDestroy(&ndmap));
    PetscCall(ISAllGatherDisjoint(is1[0],&is2));
  } else {
    /* Create the random Index Sets */
    PetscCall(PetscMalloc1(nd,&is1));
    PetscCall(PetscMalloc1(nd,&is2));

    PetscCall(MatGetSize(A,&m,&n));
    PetscCall(PetscRandomCreate(PETSC_COMM_SELF,&r));
    PetscCall(PetscRandomSetFromOptions(r));
    for (i=0; i<nd; i++) {
      PetscCall(PetscRandomGetValue(r,&rand));
      start = (PetscInt)(rand*m);
      PetscCall(PetscRandomGetValue(r,&rand));
      end   = (PetscInt)(rand*m);
      lsize =  end - start;
      if (start > end) { start = end; lsize = -lsize;}
      PetscCall(ISCreateStride(PETSC_COMM_SELF,lsize,start,1,is1+i));
      PetscCall(ISCreateStride(PETSC_COMM_SELF,lsize,start,1,is2+i));
    }
    ndpar = nd;
    PetscCall(PetscRandomDestroy(&r));
  }
  PetscCall(MatIncreaseOverlap(A,ndpar,is1,ov));
  PetscCall(MatIncreaseOverlap(B,nd,is2,ov));
  if (useND) {
    IS *is;

    PetscCall(ISAllGatherDisjoint(is1[0],&is));
    PetscCall(ISDestroy(&is1[0]));
    PetscCall(PetscFree(is1));
    is1 = is;
  }
  /* Now see if the serial and parallel case have the same answers */
  for (i=0; i<nd; ++i) {
    PetscCall(ISEqual(is1[i],is2[i],&flg));
    if (!flg) {
      PetscCall(ISViewFromOptions(is1[i],NULL,"-err_view"));
      PetscCall(ISViewFromOptions(is2[i],NULL,"-err_view"));
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"proc:[%d], i=%" PetscInt_FMT ", flg =%d",rank,i,(int)flg);
    }
  }

  /* Free allocated memory */
  for (i=0; i<nd; ++i) {
    PetscCall(ISDestroy(&is1[i]));
    PetscCall(ISDestroy(&is2[i]));
  }
  PetscCall(PetscFree(is1));
  PetscCall(PetscFree(is2));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(PetscFinalize());
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
