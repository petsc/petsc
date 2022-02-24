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
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"ex183","Mat");CHKERRQ(ierr);
  m = 5;
  CHKERRQ(PetscOptionsInt("-m","Local matrix size","MatSetSizes",m,&m,&flg));
  total_subdomains = size-1;
  CHKERRQ(PetscOptionsInt("-total_subdomains","Number of submatrices where 0 < n < comm size","MatCreateSubMatricesMPI",total_subdomains,&total_subdomains,&flg));
  permute_indices = PETSC_FALSE;
  CHKERRQ(PetscOptionsBool("-permute_indices","Whether to permute indices before breaking them into subdomains","ISCreateGeneral",permute_indices,&permute_indices,&flg));
  hash = 7;
  CHKERRQ(PetscOptionsInt("-hash","Permutation factor, which has to be relatively prime to M = size*m (total matrix size)","ISCreateGeneral",hash,&hash,&flg));
  rep = 2;
  CHKERRQ(PetscOptionsInt("-rep","Number of times to carry out submatrix extractions; currently only 1 & 2 are supported",NULL,rep,&rep,&flg));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  PetscCheckFalse(total_subdomains > size,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Number of subdomains %" PetscInt_FMT " must not exceed comm size %d",total_subdomains,size);
  PetscCheckFalse(total_subdomains < 1 || total_subdomains > size,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"number of subdomains must be > 0 and <= %d (comm size), got total_subdomains = %" PetscInt_FMT,size,total_subdomains);
  PetscCheckFalse(rep != 1 && rep != 2,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid number of test repetitions: %" PetscInt_FMT "; must be 1 or 2",rep);

  viewer = PETSC_VIEWER_STDOUT_WORLD;
  /* Create logically sparse, but effectively dense matrix for easy verification of submatrix extraction correctness. */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,m,m,PETSC_DECIDE,PETSC_DECIDE));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));
  CHKERRQ(MatGetSize(A,NULL,&N));
  CHKERRQ(MatGetLocalSize(A,NULL,&n));
  CHKERRQ(MatGetBlockSize(A,&bs));
  CHKERRQ(MatSeqAIJSetPreallocation(A,n,NULL));
  CHKERRQ(MatMPIAIJSetPreallocation(A,n,NULL,N-n,NULL));
  CHKERRQ(MatSeqBAIJSetPreallocation(A,bs,n/bs,NULL));
  CHKERRQ(MatMPIBAIJSetPreallocation(A,bs,n/bs,NULL,(N-n)/bs,NULL));
  CHKERRQ(MatSeqSBAIJSetPreallocation(A,bs,n/bs,NULL));
  CHKERRQ(MatMPISBAIJSetPreallocation(A,bs,n/bs,NULL,(N-n)/bs,NULL));

  CHKERRQ(PetscMalloc2(N,&cols,N,&vals));
  CHKERRQ(MatGetOwnershipRange(A,&rstart,&rend));
  for (j = 0; j < N; ++j) cols[j] = j;
  for (i=rstart; i<rend; i++) {
    for (j=0;j<N;++j) {
      vals[j] = i*10000+j;
    }
    CHKERRQ(MatSetValues(A,1,&i,N,cols,vals,INSERT_VALUES));
  }
  CHKERRQ(PetscFree2(cols,vals));
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  CHKERRQ(PetscViewerASCIIPrintf(viewer,"Initial matrix:\n"));
  CHKERRQ(MatView(A,viewer));

  /*
     Create subcomms and ISs so that each rank participates in one IS.
     The IS either coalesces adjacent rank indices (contiguous),
     or selects indices by scrambling them using a hash.
  */
  k = size/total_subdomains + (size%total_subdomains>0); /* There are up to k ranks to a color */
  color = rank/k;
  CHKERRMPI(MPI_Comm_split(PETSC_COMM_WORLD,color,rank,&subcomm));
  CHKERRMPI(MPI_Comm_size(subcomm,&subsize));
  CHKERRMPI(MPI_Comm_rank(subcomm,&subrank));
  CHKERRQ(MatGetOwnershipRange(A,&rstart,&rend));
  nis = 1;
  CHKERRQ(PetscMalloc2(rend-rstart,&rowindices,rend-rstart,&colindices));

  for (j = rstart; j < rend; ++j) {
    if (permute_indices) {
      idx = (j*hash);
    } else {
      idx = j;
    }
    rowindices[j-rstart] = idx%N;
    colindices[j-rstart] = (idx+m)%N;
  }
  CHKERRQ(ISCreateGeneral(subcomm,rend-rstart,rowindices,PETSC_COPY_VALUES,&rowis[0]));
  CHKERRQ(ISCreateGeneral(subcomm,rend-rstart,colindices,PETSC_COPY_VALUES,&colis[0]));
  CHKERRQ(ISSort(rowis[0]));
  CHKERRQ(ISSort(colis[0]));
  CHKERRQ(PetscFree2(rowindices,colindices));
  /*
    Now view the ISs.  To avoid deadlock when viewing a list of objects on different subcomms,
    we need to obtain the global numbers of our local objects and wait for the corresponding global
    number to be viewed.
  */
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"Subdomains"));
  if (permute_indices) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer," (hash=%" PetscInt_FMT ")",hash));
  }
  CHKERRQ(PetscViewerASCIIPrintf(viewer,":\n"));
  CHKERRQ(PetscViewerFlush(viewer));

  nsubdomains = 1;
  for (s = 0; s < nsubdomains; ++s) gsubdomainperm[s] = s;
  CHKERRQ(PetscObjectsListGetGlobalNumbering(PETSC_COMM_WORLD,1,(PetscObject*)rowis,&gnsubdomains,gsubdomainnums));
  CHKERRQ(PetscSortIntWithPermutation(nsubdomains,gsubdomainnums,gsubdomainperm));
  for (gs=0,s=0; gs < gnsubdomains;++gs) {
    if (s < nsubdomains) {
      PetscInt ss;
      ss = gsubdomainperm[s];
      if (gs == gsubdomainnums[ss]) { /* Global subdomain gs being viewed is my subdomain with local number ss. */
        PetscViewer subviewer = NULL;
        CHKERRQ(PetscViewerGetSubViewer(viewer,PetscObjectComm((PetscObject)rowis[ss]),&subviewer));
        CHKERRQ(PetscViewerASCIIPrintf(subviewer,"Row IS %" PetscInt_FMT "\n",gs));
        CHKERRQ(ISView(rowis[ss],subviewer));
        CHKERRQ(PetscViewerFlush(subviewer));
        CHKERRQ(PetscViewerASCIIPrintf(subviewer,"Col IS %" PetscInt_FMT "\n",gs));
        CHKERRQ(ISView(colis[ss],subviewer));
        CHKERRQ(PetscViewerRestoreSubViewer(viewer,PetscObjectComm((PetscObject)rowis[ss]),&subviewer));
        ++s;
      }
    }
    CHKERRMPI(MPI_Barrier(PETSC_COMM_WORLD));
  }
  CHKERRQ(PetscViewerFlush(viewer));
  CHKERRQ(ISSort(rowis[0]));
  CHKERRQ(ISSort(colis[0]));
  nsubdomains = 1;
  CHKERRQ(MatCreateSubMatricesMPI(A,nsubdomains,rowis,colis,MAT_INITIAL_MATRIX,&submats));
  /*
    Now view the matrices.  To avoid deadlock when viewing a list of objects on different subcomms,
    we need to obtain the global numbers of our local objects and wait for the corresponding global
    number to be viewed.
  */
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"Submatrices (repetition 1):\n"));
  for (s = 0; s < nsubdomains; ++s) gsubdomainperm[s] = s;
  CHKERRQ(PetscObjectsListGetGlobalNumbering(PETSC_COMM_WORLD,1,(PetscObject*)submats,&gnsubdomains,gsubdomainnums));
  CHKERRQ(PetscSortIntWithPermutation(nsubdomains,gsubdomainnums,gsubdomainperm));
  for (gs=0,s=0; gs < gnsubdomains;++gs) {
    if (s < nsubdomains) {
      PetscInt ss;
      ss = gsubdomainperm[s];
      if (gs == gsubdomainnums[ss]) { /* Global subdomain gs being viewed is my subdomain with local number ss. */
        PetscViewer subviewer = NULL;
        CHKERRQ(PetscViewerGetSubViewer(viewer,PetscObjectComm((PetscObject)submats[ss]),&subviewer));
        CHKERRQ(MatView(submats[ss],subviewer));
        CHKERRQ(PetscViewerRestoreSubViewer(viewer,PetscObjectComm((PetscObject)submats[ss]),&subviewer));
        ++s;
      }
    }
    CHKERRMPI(MPI_Barrier(PETSC_COMM_WORLD));
  }
  CHKERRQ(PetscViewerFlush(viewer));
  if (rep == 1) goto cleanup;
  nsubdomains = 1;
  CHKERRQ(MatCreateSubMatricesMPI(A,nsubdomains,rowis,colis,MAT_REUSE_MATRIX,&submats));
  /*
    Now view the matrices.  To avoid deadlock when viewing a list of objects on different subcomms,
    we need to obtain the global numbers of our local objects and wait for the corresponding global
    number to be viewed.
  */
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"Submatrices (repetition 2):\n"));
  for (s = 0; s < nsubdomains; ++s) gsubdomainperm[s] = s;
  CHKERRQ(PetscObjectsListGetGlobalNumbering(PETSC_COMM_WORLD,1,(PetscObject*)submats,&gnsubdomains,gsubdomainnums));
  CHKERRQ(PetscSortIntWithPermutation(nsubdomains,gsubdomainnums,gsubdomainperm));
  for (gs=0,s=0; gs < gnsubdomains;++gs) {
    if (s < nsubdomains) {
      PetscInt ss;
      ss = gsubdomainperm[s];
      if (gs == gsubdomainnums[ss]) { /* Global subdomain gs being viewed is my subdomain with local number ss. */
        PetscViewer subviewer = NULL;
        CHKERRQ(PetscViewerGetSubViewer(viewer,PetscObjectComm((PetscObject)submats[ss]),&subviewer));
        CHKERRQ(MatView(submats[ss],subviewer));
        CHKERRQ(PetscViewerRestoreSubViewer(viewer,PetscObjectComm((PetscObject)submats[ss]),&subviewer));
        ++s;
      }
    }
    CHKERRMPI(MPI_Barrier(PETSC_COMM_WORLD));
  }
  CHKERRQ(PetscViewerFlush(viewer));
  cleanup:
  for (k=0;k<nsubdomains;++k) {
    CHKERRQ(MatDestroy(submats+k));
  }
  CHKERRQ(PetscFree(submats));
  for (k=0;k<nis;++k) {
    CHKERRQ(ISDestroy(rowis+k));
    CHKERRQ(ISDestroy(colis+k));
  }
  CHKERRQ(MatDestroy(&A));
  CHKERRMPI(MPI_Comm_free(&subcomm));
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
