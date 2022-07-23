static char help[] = "Example of extracting an array of MPI submatrices from a given MPI matrix.\n"
  "This test can only be run in parallel.\n"
  "\n";

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

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"ex183","Mat");
  m = 5;
  PetscCall(PetscOptionsInt("-m","Local matrix size","MatSetSizes",m,&m,&flg));
  total_subdomains = size-1;
  PetscCall(PetscOptionsInt("-total_subdomains","Number of submatrices where 0 < n < comm size","MatCreateSubMatricesMPI",total_subdomains,&total_subdomains,&flg));
  permute_indices = PETSC_FALSE;
  PetscCall(PetscOptionsBool("-permute_indices","Whether to permute indices before breaking them into subdomains","ISCreateGeneral",permute_indices,&permute_indices,&flg));
  hash = 7;
  PetscCall(PetscOptionsInt("-hash","Permutation factor, which has to be relatively prime to M = size*m (total matrix size)","ISCreateGeneral",hash,&hash,&flg));
  rep = 2;
  PetscCall(PetscOptionsInt("-rep","Number of times to carry out submatrix extractions; currently only 1 & 2 are supported",NULL,rep,&rep,&flg));
  PetscOptionsEnd();

  PetscCheck(total_subdomains <= size,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Number of subdomains %" PetscInt_FMT " must not exceed comm size %d",total_subdomains,size);
  PetscCheck(total_subdomains >= 1 && total_subdomains <= size,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"number of subdomains must be > 0 and <= %d (comm size), got total_subdomains = %" PetscInt_FMT,size,total_subdomains);
  PetscCheck(rep == 1 || rep == 2,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid number of test repetitions: %" PetscInt_FMT "; must be 1 or 2",rep);

  viewer = PETSC_VIEWER_STDOUT_WORLD;
  /* Create logically sparse, but effectively dense matrix for easy verification of submatrix extraction correctness. */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,m,m,PETSC_DECIDE,PETSC_DECIDE));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));
  PetscCall(MatGetSize(A,NULL,&N));
  PetscCall(MatGetLocalSize(A,NULL,&n));
  PetscCall(MatGetBlockSize(A,&bs));
  PetscCall(MatSeqAIJSetPreallocation(A,n,NULL));
  PetscCall(MatMPIAIJSetPreallocation(A,n,NULL,N-n,NULL));
  PetscCall(MatSeqBAIJSetPreallocation(A,bs,n/bs,NULL));
  PetscCall(MatMPIBAIJSetPreallocation(A,bs,n/bs,NULL,(N-n)/bs,NULL));
  PetscCall(MatSeqSBAIJSetPreallocation(A,bs,n/bs,NULL));
  PetscCall(MatMPISBAIJSetPreallocation(A,bs,n/bs,NULL,(N-n)/bs,NULL));

  PetscCall(PetscMalloc2(N,&cols,N,&vals));
  PetscCall(MatGetOwnershipRange(A,&rstart,&rend));
  for (j = 0; j < N; ++j) cols[j] = j;
  for (i=rstart; i<rend; i++) {
    for (j=0;j<N;++j) {
      vals[j] = i*10000+j;
    }
    PetscCall(MatSetValues(A,1,&i,N,cols,vals,INSERT_VALUES));
  }
  PetscCall(PetscFree2(cols,vals));
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  PetscCall(PetscViewerASCIIPrintf(viewer,"Initial matrix:\n"));
  PetscCall(MatView(A,viewer));

  /*
     Create subcomms and ISs so that each rank participates in one IS.
     The IS either coalesces adjacent rank indices (contiguous),
     or selects indices by scrambling them using a hash.
  */
  k = size/total_subdomains + (size%total_subdomains>0); /* There are up to k ranks to a color */
  color = rank/k;
  PetscCallMPI(MPI_Comm_split(PETSC_COMM_WORLD,color,rank,&subcomm));
  PetscCallMPI(MPI_Comm_size(subcomm,&subsize));
  PetscCallMPI(MPI_Comm_rank(subcomm,&subrank));
  PetscCall(MatGetOwnershipRange(A,&rstart,&rend));
  nis = 1;
  PetscCall(PetscMalloc2(rend-rstart,&rowindices,rend-rstart,&colindices));

  for (j = rstart; j < rend; ++j) {
    if (permute_indices) {
      idx = (j*hash);
    } else {
      idx = j;
    }
    rowindices[j-rstart] = idx%N;
    colindices[j-rstart] = (idx+m)%N;
  }
  PetscCall(ISCreateGeneral(subcomm,rend-rstart,rowindices,PETSC_COPY_VALUES,&rowis[0]));
  PetscCall(ISCreateGeneral(subcomm,rend-rstart,colindices,PETSC_COPY_VALUES,&colis[0]));
  PetscCall(ISSort(rowis[0]));
  PetscCall(ISSort(colis[0]));
  PetscCall(PetscFree2(rowindices,colindices));
  /*
    Now view the ISs.  To avoid deadlock when viewing a list of objects on different subcomms,
    we need to obtain the global numbers of our local objects and wait for the corresponding global
    number to be viewed.
  */
  PetscCall(PetscViewerASCIIPrintf(viewer,"Subdomains"));
  if (permute_indices) {
    PetscCall(PetscViewerASCIIPrintf(viewer," (hash=%" PetscInt_FMT ")",hash));
  }
  PetscCall(PetscViewerASCIIPrintf(viewer,":\n"));
  PetscCall(PetscViewerFlush(viewer));

  nsubdomains = 1;
  for (s = 0; s < nsubdomains; ++s) gsubdomainperm[s] = s;
  PetscCall(PetscObjectsListGetGlobalNumbering(PETSC_COMM_WORLD,1,(PetscObject*)rowis,&gnsubdomains,gsubdomainnums));
  PetscCall(PetscSortIntWithPermutation(nsubdomains,gsubdomainnums,gsubdomainperm));
  for (gs=0,s=0; gs < gnsubdomains;++gs) {
    if (s < nsubdomains) {
      PetscInt ss;
      ss = gsubdomainperm[s];
      if (gs == gsubdomainnums[ss]) { /* Global subdomain gs being viewed is my subdomain with local number ss. */
        PetscViewer subviewer = NULL;
        PetscCall(PetscViewerGetSubViewer(viewer,PetscObjectComm((PetscObject)rowis[ss]),&subviewer));
        PetscCall(PetscViewerASCIIPrintf(subviewer,"Row IS %" PetscInt_FMT "\n",gs));
        PetscCall(ISView(rowis[ss],subviewer));
        PetscCall(PetscViewerFlush(subviewer));
        PetscCall(PetscViewerASCIIPrintf(subviewer,"Col IS %" PetscInt_FMT "\n",gs));
        PetscCall(ISView(colis[ss],subviewer));
        PetscCall(PetscViewerRestoreSubViewer(viewer,PetscObjectComm((PetscObject)rowis[ss]),&subviewer));
        ++s;
      }
    }
    PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
  }
  PetscCall(PetscViewerFlush(viewer));
  PetscCall(ISSort(rowis[0]));
  PetscCall(ISSort(colis[0]));
  nsubdomains = 1;
  PetscCall(MatCreateSubMatricesMPI(A,nsubdomains,rowis,colis,MAT_INITIAL_MATRIX,&submats));
  /*
    Now view the matrices.  To avoid deadlock when viewing a list of objects on different subcomms,
    we need to obtain the global numbers of our local objects and wait for the corresponding global
    number to be viewed.
  */
  PetscCall(PetscViewerASCIIPrintf(viewer,"Submatrices (repetition 1):\n"));
  for (s = 0; s < nsubdomains; ++s) gsubdomainperm[s] = s;
  PetscCall(PetscObjectsListGetGlobalNumbering(PETSC_COMM_WORLD,1,(PetscObject*)submats,&gnsubdomains,gsubdomainnums));
  PetscCall(PetscSortIntWithPermutation(nsubdomains,gsubdomainnums,gsubdomainperm));
  for (gs=0,s=0; gs < gnsubdomains;++gs) {
    if (s < nsubdomains) {
      PetscInt ss;
      ss = gsubdomainperm[s];
      if (gs == gsubdomainnums[ss]) { /* Global subdomain gs being viewed is my subdomain with local number ss. */
        PetscViewer subviewer = NULL;
        PetscCall(PetscViewerGetSubViewer(viewer,PetscObjectComm((PetscObject)submats[ss]),&subviewer));
        PetscCall(MatView(submats[ss],subviewer));
        PetscCall(PetscViewerRestoreSubViewer(viewer,PetscObjectComm((PetscObject)submats[ss]),&subviewer));
        ++s;
      }
    }
    PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
  }
  PetscCall(PetscViewerFlush(viewer));
  if (rep == 1) goto cleanup;
  nsubdomains = 1;
  PetscCall(MatCreateSubMatricesMPI(A,nsubdomains,rowis,colis,MAT_REUSE_MATRIX,&submats));
  /*
    Now view the matrices.  To avoid deadlock when viewing a list of objects on different subcomms,
    we need to obtain the global numbers of our local objects and wait for the corresponding global
    number to be viewed.
  */
  PetscCall(PetscViewerASCIIPrintf(viewer,"Submatrices (repetition 2):\n"));
  for (s = 0; s < nsubdomains; ++s) gsubdomainperm[s] = s;
  PetscCall(PetscObjectsListGetGlobalNumbering(PETSC_COMM_WORLD,1,(PetscObject*)submats,&gnsubdomains,gsubdomainnums));
  PetscCall(PetscSortIntWithPermutation(nsubdomains,gsubdomainnums,gsubdomainperm));
  for (gs=0,s=0; gs < gnsubdomains;++gs) {
    if (s < nsubdomains) {
      PetscInt ss;
      ss = gsubdomainperm[s];
      if (gs == gsubdomainnums[ss]) { /* Global subdomain gs being viewed is my subdomain with local number ss. */
        PetscViewer subviewer = NULL;
        PetscCall(PetscViewerGetSubViewer(viewer,PetscObjectComm((PetscObject)submats[ss]),&subviewer));
        PetscCall(MatView(submats[ss],subviewer));
        PetscCall(PetscViewerRestoreSubViewer(viewer,PetscObjectComm((PetscObject)submats[ss]),&subviewer));
        ++s;
      }
    }
    PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
  }
  PetscCall(PetscViewerFlush(viewer));
  cleanup:
  for (k=0;k<nsubdomains;++k) {
    PetscCall(MatDestroy(submats+k));
  }
  PetscCall(PetscFree(submats));
  for (k=0;k<nis;++k) {
    PetscCall(ISDestroy(rowis+k));
    PetscCall(ISDestroy(colis+k));
  }
  PetscCall(MatDestroy(&A));
  PetscCallMPI(MPI_Comm_free(&subcomm));
  PetscCall(PetscFinalize());
  return 0;
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
