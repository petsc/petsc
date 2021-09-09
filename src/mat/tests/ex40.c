
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
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBeginUser;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)iis),&size);CHKERRMPI(ierr);
  ierr = PetscMalloc1(size,&is2);CHKERRQ(ierr);
  ierr = PetscMalloc1(size,&sizes);CHKERRQ(ierr);
  ierr = ISGetLocalSize(iis,&ls);CHKERRQ(ierr);
  /* we don't have a public ISGetLayout */
  ierr = MPI_Allgather(&ls,1,MPIU_INT,sizes,1,MPIU_INT,PetscObjectComm((PetscObject)iis));CHKERRMPI(ierr);
  ierr = ISAllGather(iis,&is);CHKERRQ(ierr);
  ierr = ISGetIndices(is,&idxs);CHKERRQ(ierr);
  for (i = 0, ls = 0; i < size; i++) {
    ierr = ISCreateGeneral(PETSC_COMM_SELF,sizes[i],idxs+ls,PETSC_COPY_VALUES,&is2[i]);CHKERRQ(ierr);
    ls += sizes[i];
  }
  ierr = ISRestoreIndices(is,&idxs);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);
  ierr = PetscFree(sizes);CHKERRQ(ierr);
  *ois = is2;
  PetscFunctionReturn(0);
}

int main(int argc,char **args)
{
  PetscErrorCode ierr;
  PetscInt       nd = 2,ov = 1,ndpar,i,start,m,n,end,lsize;
  PetscMPIInt    rank;
  PetscBool      flg, useND = PETSC_FALSE;
  Mat            A,B;
  char           file[PETSC_MAX_PATH_LEN];
  PetscViewer    fd;
  IS             *is1,*is2;
  PetscRandom    r;
  PetscScalar    rand;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Must use -f filename to indicate a file containing a PETSc binary matrix");
  ierr = PetscOptionsGetInt(NULL,NULL,"-nd",&nd,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-ov",&ov,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-nested_dissection",&useND,NULL);CHKERRQ(ierr);

  /* Read matrix */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetType(A,MATMPIAIJ);CHKERRQ(ierr);
  ierr = MatLoad(A,fd);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);

  /* Read the matrix again as a sequential matrix */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_SELF,file,FILE_MODE_READ,&fd);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_SELF,&B);CHKERRQ(ierr);
  ierr = MatSetType(B,MATSEQAIJ);CHKERRQ(ierr);
  ierr = MatLoad(B,fd);CHKERRQ(ierr);
  ierr = MatSetFromOptions(B);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);

  /* Create the IS corresponding to subdomains */
  if (useND) {
    MatPartitioning part;
    IS              ndmap;
    PetscMPIInt     size;

    ndpar = 1;
    ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
    nd   = (PetscInt)size;
    ierr = PetscMalloc1(ndpar,&is1);CHKERRQ(ierr);
    ierr = MatPartitioningCreate(PETSC_COMM_WORLD,&part);CHKERRQ(ierr);
    ierr = MatPartitioningSetAdjacency(part,A);CHKERRQ(ierr);
    ierr = MatPartitioningSetFromOptions(part);CHKERRQ(ierr);
    ierr = MatPartitioningApplyND(part,&ndmap);CHKERRQ(ierr);
    ierr = MatPartitioningDestroy(&part);CHKERRQ(ierr);
    ierr = ISBuildTwoSided(ndmap,NULL,&is1[0]);CHKERRQ(ierr);
    ierr = ISDestroy(&ndmap);CHKERRQ(ierr);
    ierr = ISAllGatherDisjoint(is1[0],&is2);CHKERRQ(ierr);
  } else {
    /* Create the random Index Sets */
    ierr = PetscMalloc1(nd,&is1);CHKERRQ(ierr);
    ierr = PetscMalloc1(nd,&is2);CHKERRQ(ierr);

    ierr = MatGetSize(A,&m,&n);CHKERRQ(ierr);
    ierr = PetscRandomCreate(PETSC_COMM_SELF,&r);CHKERRQ(ierr);
    ierr = PetscRandomSetFromOptions(r);CHKERRQ(ierr);
    for (i=0; i<nd; i++) {
      ierr  = PetscRandomGetValue(r,&rand);CHKERRQ(ierr);
      start = (PetscInt)(rand*m);
      ierr  = PetscRandomGetValue(r,&rand);CHKERRQ(ierr);
      end   = (PetscInt)(rand*m);
      lsize =  end - start;
      if (start > end) { start = end; lsize = -lsize;}
      ierr = ISCreateStride(PETSC_COMM_SELF,lsize,start,1,is1+i);CHKERRQ(ierr);
      ierr = ISCreateStride(PETSC_COMM_SELF,lsize,start,1,is2+i);CHKERRQ(ierr);
    }
    ndpar = nd;
    ierr = PetscRandomDestroy(&r);CHKERRQ(ierr);
  }
  ierr = MatIncreaseOverlap(A,ndpar,is1,ov);CHKERRQ(ierr);
  ierr = MatIncreaseOverlap(B,nd,is2,ov);CHKERRQ(ierr);
  if (useND) {
    IS *is;

    ierr = ISAllGatherDisjoint(is1[0],&is);CHKERRQ(ierr);
    ierr = ISDestroy(&is1[0]);CHKERRQ(ierr);
    ierr = PetscFree(is1);CHKERRQ(ierr);
    is1 = is;
  }
  /* Now see if the serial and parallel case have the same answers */
  for (i=0; i<nd; ++i) {
    ierr = ISEqual(is1[i],is2[i],&flg);CHKERRQ(ierr);
    if (!flg) {
      ierr = ISViewFromOptions(is1[i],NULL,"-err_view");CHKERRQ(ierr);
      ierr = ISViewFromOptions(is2[i],NULL,"-err_view");CHKERRQ(ierr);
      SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_PLIB,"proc:[%d], i=%D, flg =%d\n",rank,i,(int)flg);
    }
  }

  /* Free allocated memory */
  for (i=0; i<nd; ++i) {
    ierr = ISDestroy(&is1[i]);CHKERRQ(ierr);
    ierr = ISDestroy(&is2[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(is1);CHKERRQ(ierr);
  ierr = PetscFree(is2);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
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
