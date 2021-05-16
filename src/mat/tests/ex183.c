static char help[] = "Example of extracting an array of MPI submatrices from a given MPI matrix.\n"
  "This test can only be run in parallel.\n"
  "\n";

/*T
   Concepts: Mat^mat submatrix, parallel
   Processors: n
T*/

#include <petscmat.h>

int main(int argc, char **args)
{
  Mat             A,*submats;
  MPI_Comm        subcomm;
  PetscMPIInt     rank,size,subrank,subsize,color;
  PetscInt        m,n,N,bs,rstart,rend,i,j,k,total_subdomains,hash,nsubdomains=1;
  PetscInt        nis,*cols,gnsubdomains,gsubdomainnums[1],gsubdomainperm[1],s,gs;
  PetscInt        *rowindices,*colindices,idx,rep;
  PetscScalar     *vals;
  IS              rowis[1],colis[1];
  PetscViewer     viewer;
  PetscBool       permute_indices,flg;
  PetscErrorCode  ierr;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"ex183","Mat");CHKERRQ(ierr);
  m = 5;
  ierr = PetscOptionsInt("-m","Local matrix size","MatSetSizes",m,&m,&flg);CHKERRQ(ierr);
  total_subdomains = size-1;
  ierr = PetscOptionsInt("-total_subdomains","Number of submatrices where 0 < n < comm size","MatCreateSubMatricesMPI",total_subdomains,&total_subdomains,&flg);CHKERRQ(ierr);
  permute_indices = PETSC_FALSE;
  ierr = PetscOptionsBool("-permute_indices","Whether to permute indices before breaking them into subdomains","ISCreateGeneral",permute_indices,&permute_indices,&flg);CHKERRQ(ierr);
  hash = 7;
  ierr = PetscOptionsInt("-hash","Permutation factor, which has to be relatively prime to M = size*m (total matrix size)","ISCreateGeneral",hash,&hash,&flg);CHKERRQ(ierr);
  rep = 2;
  ierr = PetscOptionsInt("-rep","Number of times to carry out submatrix extractions; currently only 1 & 2 are supported",NULL,rep,&rep,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  if (total_subdomains > size) SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Number of subdomains %D must not exceed comm size %D",total_subdomains,size);
  if (total_subdomains < 1 || total_subdomains > size) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"number of subdomains must be > 0 and <= %D (comm size), got total_subdomains = %D",size,total_subdomains);
  if (rep != 1 && rep != 2) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid number of test repetitions: %D; must be 1 or 2",rep);

  viewer = PETSC_VIEWER_STDOUT_WORLD;
  /* Create logically sparse, but effectively dense matrix for easy verification of submatrix extraction correctness. */
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,m,m,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);
  ierr = MatGetSize(A,NULL,&N);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,NULL,&n);CHKERRQ(ierr);
  ierr = MatGetBlockSize(A,&bs);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(A,n,NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(A,n,NULL,N-n,NULL);CHKERRQ(ierr);
  ierr = MatSeqBAIJSetPreallocation(A,bs,n/bs,NULL);CHKERRQ(ierr);
  ierr = MatMPIBAIJSetPreallocation(A,bs,n/bs,NULL,(N-n)/bs,NULL);CHKERRQ(ierr);
  ierr = MatSeqSBAIJSetPreallocation(A,bs,n/bs,NULL);CHKERRQ(ierr);
  ierr = MatMPISBAIJSetPreallocation(A,bs,n/bs,NULL,(N-n)/bs,NULL);CHKERRQ(ierr);

  ierr = PetscMalloc2(N,&cols,N,&vals);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);
  for (j = 0; j < N; ++j) cols[j] = j;
  for (i=rstart; i<rend; i++) {
    for (j=0;j<N;++j) {
      vals[j] = i*10000+j;
    }
    ierr = MatSetValues(A,1,&i,N,cols,vals,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = PetscFree2(cols,vals);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = PetscViewerASCIIPrintf(viewer,"Initial matrix:\n");CHKERRQ(ierr);
  ierr = MatView(A,viewer);CHKERRQ(ierr);

  /*
     Create subcomms and ISs so that each rank participates in one IS.
     The IS either coalesces adjacent rank indices (contiguous),
     or selects indices by scrambling them using a hash.
  */
  k = size/total_subdomains + (size%total_subdomains>0); /* There are up to k ranks to a color */
  color = rank/k;
  ierr = MPI_Comm_split(PETSC_COMM_WORLD,color,rank,&subcomm);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(subcomm,&subsize);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(subcomm,&subrank);CHKERRMPI(ierr);
  ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);
  nis = 1;
  ierr = PetscMalloc2(rend-rstart,&rowindices,rend-rstart,&colindices);CHKERRQ(ierr);

  for (j = rstart; j < rend; ++j) {
    if (permute_indices) {
      idx = (j*hash);
    } else {
      idx = j;
    }
    rowindices[j-rstart] = idx%N;
    colindices[j-rstart] = (idx+m)%N;
  }
  ierr = ISCreateGeneral(subcomm,rend-rstart,rowindices,PETSC_COPY_VALUES,&rowis[0]);CHKERRQ(ierr);
  ierr = ISCreateGeneral(subcomm,rend-rstart,colindices,PETSC_COPY_VALUES,&colis[0]);CHKERRQ(ierr);
  ierr = ISSort(rowis[0]);CHKERRQ(ierr);
  ierr = ISSort(colis[0]);CHKERRQ(ierr);
  ierr = PetscFree2(rowindices,colindices);CHKERRQ(ierr);
  /*
    Now view the ISs.  To avoid deadlock when viewing a list of objects on different subcomms,
    we need to obtain the global numbers of our local objects and wait for the corresponding global
    number to be viewed.
  */
  ierr = PetscViewerASCIIPrintf(viewer,"Subdomains");CHKERRQ(ierr);
  if (permute_indices) {
    ierr = PetscViewerASCIIPrintf(viewer," (hash=%D)",hash);CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPrintf(viewer,":\n");CHKERRQ(ierr);
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);

  nsubdomains = 1;
  for (s = 0; s < nsubdomains; ++s) gsubdomainperm[s] = s;
  ierr = PetscObjectsListGetGlobalNumbering(PETSC_COMM_WORLD,1,(PetscObject*)rowis,&gnsubdomains,gsubdomainnums);CHKERRQ(ierr);
  ierr = PetscSortIntWithPermutation(nsubdomains,gsubdomainnums,gsubdomainperm);CHKERRQ(ierr);
  for (gs=0,s=0; gs < gnsubdomains;++gs) {
    if (s < nsubdomains) {
      PetscInt ss;
      ss = gsubdomainperm[s];
      if (gs == gsubdomainnums[ss]) { /* Global subdomain gs being viewed is my subdomain with local number ss. */
        PetscViewer subviewer = NULL;
        ierr = PetscViewerGetSubViewer(viewer,PetscObjectComm((PetscObject)rowis[ss]),&subviewer);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(subviewer,"Row IS %D\n",gs);CHKERRQ(ierr);
        ierr = ISView(rowis[ss],subviewer);CHKERRQ(ierr);
        ierr = PetscViewerFlush(subviewer);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(subviewer,"Col IS %D\n",gs);CHKERRQ(ierr);
        ierr = ISView(colis[ss],subviewer);CHKERRQ(ierr);
        ierr = PetscViewerRestoreSubViewer(viewer,PetscObjectComm((PetscObject)rowis[ss]),&subviewer);CHKERRQ(ierr);
        ++s;
      }
    }
    ierr = MPI_Barrier(PETSC_COMM_WORLD);CHKERRMPI(ierr);
  }
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  ierr = ISSort(rowis[0]);CHKERRQ(ierr);
  ierr = ISSort(colis[0]);CHKERRQ(ierr);
  nsubdomains = 1;
  ierr = MatCreateSubMatricesMPI(A,nsubdomains,rowis,colis,MAT_INITIAL_MATRIX,&submats);CHKERRQ(ierr);
  /*
    Now view the matrices.  To avoid deadlock when viewing a list of objects on different subcomms,
    we need to obtain the global numbers of our local objects and wait for the corresponding global
    number to be viewed.
  */
  ierr = PetscViewerASCIIPrintf(viewer,"Submatrices (repetition 1):\n");CHKERRQ(ierr);
  for (s = 0; s < nsubdomains; ++s) gsubdomainperm[s] = s;
  ierr = PetscObjectsListGetGlobalNumbering(PETSC_COMM_WORLD,1,(PetscObject*)submats,&gnsubdomains,gsubdomainnums);CHKERRQ(ierr);
  ierr = PetscSortIntWithPermutation(nsubdomains,gsubdomainnums,gsubdomainperm);CHKERRQ(ierr);
  for (gs=0,s=0; gs < gnsubdomains;++gs) {
    if (s < nsubdomains) {
      PetscInt ss;
      ss = gsubdomainperm[s];
      if (gs == gsubdomainnums[ss]) { /* Global subdomain gs being viewed is my subdomain with local number ss. */
        PetscViewer subviewer = NULL;
        ierr = PetscViewerGetSubViewer(viewer,PetscObjectComm((PetscObject)submats[ss]),&subviewer);CHKERRQ(ierr);
        ierr = MatView(submats[ss],subviewer);CHKERRQ(ierr);
        ierr = PetscViewerRestoreSubViewer(viewer,PetscObjectComm((PetscObject)submats[ss]),&subviewer);CHKERRQ(ierr);
        ++s;
      }
    }
    ierr = MPI_Barrier(PETSC_COMM_WORLD);CHKERRMPI(ierr);
  }
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  if (rep == 1) goto cleanup;
  nsubdomains = 1;
  ierr = MatCreateSubMatricesMPI(A,nsubdomains,rowis,colis,MAT_REUSE_MATRIX,&submats);CHKERRQ(ierr);
  /*
    Now view the matrices.  To avoid deadlock when viewing a list of objects on different subcomms,
    we need to obtain the global numbers of our local objects and wait for the corresponding global
    number to be viewed.
  */
  ierr = PetscViewerASCIIPrintf(viewer,"Submatrices (repetition 2):\n");CHKERRQ(ierr);
  for (s = 0; s < nsubdomains; ++s) gsubdomainperm[s] = s;
  ierr = PetscObjectsListGetGlobalNumbering(PETSC_COMM_WORLD,1,(PetscObject*)submats,&gnsubdomains,gsubdomainnums);CHKERRQ(ierr);
  ierr = PetscSortIntWithPermutation(nsubdomains,gsubdomainnums,gsubdomainperm);CHKERRQ(ierr);
  for (gs=0,s=0; gs < gnsubdomains;++gs) {
    if (s < nsubdomains) {
      PetscInt ss;
      ss = gsubdomainperm[s];
      if (gs == gsubdomainnums[ss]) { /* Global subdomain gs being viewed is my subdomain with local number ss. */
        PetscViewer subviewer = NULL;
        ierr = PetscViewerGetSubViewer(viewer,PetscObjectComm((PetscObject)submats[ss]),&subviewer);CHKERRQ(ierr);
        ierr = MatView(submats[ss],subviewer);CHKERRQ(ierr);
        ierr = PetscViewerRestoreSubViewer(viewer,PetscObjectComm((PetscObject)submats[ss]),&subviewer);CHKERRQ(ierr);
        ++s;
      }
    }
    ierr = MPI_Barrier(PETSC_COMM_WORLD);CHKERRMPI(ierr);
  }
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  cleanup:
  for (k=0;k<nsubdomains;++k) {
    ierr = MatDestroy(submats+k);CHKERRQ(ierr);
  }
  ierr = PetscFree(submats);CHKERRQ(ierr);
  for (k=0;k<nis;++k) {
    ierr = ISDestroy(rowis+k);CHKERRQ(ierr);
    ierr = ISDestroy(colis+k);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MPI_Comm_free(&subcomm);CHKERRMPI(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      nsize: 2
      args: -total_subdomains 1
      output_file: output/ex183_2_1.out

   test:
      suffix: 2
      nsize: 3
      args: -total_subdomains 2
      output_file: output/ex183_3_2.out

   test:
      suffix: 3
      nsize: 4
      args: -total_subdomains 2
      output_file: output/ex183_4_2.out

   test:
      suffix: 4
      nsize: 6
      args: -total_subdomains 2
      output_file: output/ex183_6_2.out

TEST*/
