#include <petscsf.h>
#include <petsc/private/matimpl.h>


/*
 * Increase overlap for the sub-matrix across sub communicator
 * sub-matrix could be a graph or numerical matrix
 * */
PetscErrorCode  MatIncreaseOverlapSplit_Single(Mat mat,IS *is,PetscInt ov)
{
  PetscInt         i,nindx,*indices_sc,*indices_ov,localsize,*localsizes_sc,localsize_tmp;
  PetscInt         *indices_ov_rd,nroots,nleaves,*localoffsets,*indices_recv,*sources_sc,*sources_sc_rd;
  const PetscInt   *indices;
  PetscMPIInt      srank,ssize,issamecomm,k;
  IS               is_sc,allis_sc,partitioning;
  MPI_Comm         gcomm,scomm;
  PetscSF          sf;
  PetscSFNode      *remote;
  Mat              *smat;
  MatPartitioning  part;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  /*get a sub communicator before call individual MatIncreaseOverlap
   * since the sub communicator may be changed.
   * */
  ierr = PetscObjectGetComm((PetscObject)*is,&scomm);CHKERRQ(ierr);
  /*increase overlap on each individual subdomain*/
  ierr = (*mat->ops->increaseoverlap)(mat,1,is,ov);CHKERRQ(ierr);
  /*get a global communicator  */
  ierr = PetscObjectGetComm((PetscObject)mat,&gcomm);CHKERRQ(ierr);
  /*compare communicators */
  ierr = MPI_Comm_compare(gcomm,scomm,&issamecomm);CHKERRQ(ierr);
  /* if the sub-communicator is the same as the global communicator,
   * user does not want to use a sub-communicator
   * */
  if(issamecomm == MPI_IDENT) PetscFunctionReturn(0);
  /* if the sub-communicator is petsc_comm_self,
   * user also does not care the sub-communicator
   * */
  ierr = MPI_Comm_compare(scomm,PETSC_COMM_SELF,&issamecomm);CHKERRQ(ierr);
  if(issamecomm == MPI_IDENT) PetscFunctionReturn(0);
  /*local rank, size in a subcomm */
  ierr = MPI_Comm_rank(scomm,&srank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(scomm,&ssize);CHKERRQ(ierr);
  /*create a new IS based on subcomm
   * since the old IS is often petsc_comm_self
   * */
  ierr = ISGetLocalSize(*is,&nindx);CHKERRQ(ierr);
  ierr = PetscCalloc1(nindx,&indices_sc);CHKERRQ(ierr);
  ierr = ISGetIndices(*is,&indices);CHKERRQ(ierr);
  ierr = PetscMemcpy(indices_sc,indices,sizeof(PetscInt)*nindx);CHKERRQ(ierr);
  ierr = ISRestoreIndices(*is,&indices);CHKERRQ(ierr);
  /*we do not need any more*/
  ierr = ISDestroy(is);CHKERRQ(ierr);
  /*create a index set based on the sub communicator  */
  ierr = ISCreateGeneral(scomm,nindx,indices_sc,PETSC_OWN_POINTER,&is_sc);CHKERRQ(ierr);
  /*gather all indices within  the sub communicator*/
  ierr = ISAllGather(is_sc,&allis_sc);CHKERRQ(ierr);
  ierr = ISDestroy(&is_sc);CHKERRQ(ierr);
  /* gather local sizes */
  ierr = PetscMalloc1(ssize,&localsizes_sc);CHKERRQ(ierr);
  ierr = MPI_Gather(&nindx,1,MPIU_INT,localsizes_sc,1,MPIU_INT,0,scomm);CHKERRQ(ierr);
  /*only root does these computation */
  if(!srank){
   /*get local size for the big index set*/
   ierr = ISGetLocalSize(allis_sc,&localsize);CHKERRQ(ierr);
   ierr = PetscCalloc2(localsize,&indices_ov,localsize,&sources_sc);CHKERRQ(ierr);
   ierr = PetscCalloc2(localsize,&indices_ov_rd,localsize,&sources_sc_rd);CHKERRQ(ierr);
   ierr = ISGetIndices(allis_sc,&indices);CHKERRQ(ierr);
   ierr = PetscMemcpy(indices_ov,indices,sizeof(PetscInt)*localsize);CHKERRQ(ierr);
   ierr = ISRestoreIndices(allis_sc,&indices);CHKERRQ(ierr);

   ierr = ISDestroy(&allis_sc);CHKERRQ(ierr);
   /*assign corresponding sources */
   localsize_tmp = 0;
   for(k=0; k<ssize; k++){
     for(i=0; i<localsizes_sc[k]; i++){
       sources_sc[localsize_tmp++] = k;
     }
   }
   /*record where indices come from */
   ierr = PetscSortIntWithArray(localsize,indices_ov,sources_sc);CHKERRQ(ierr);
   localsize_tmp = 1;
   ierr = PetscMemzero(localsizes_sc,sizeof(PetscInt)*ssize);CHKERRQ(ierr);
   /*initialize the first entities*/
   if(localsize){
	 indices_ov_rd[0] = indices_ov[0];
	 sources_sc_rd[0] = sources_sc[0];
	 localsizes_sc[sources_sc[0]]++;
   }
   /*remove duplicate integers */
   for(i=1; i<localsize; i++){
	 if(indices_ov[i] != indices_ov[i-1]){
	   indices_ov_rd[localsize_tmp]   = indices_ov[i];
	   sources_sc_rd[localsize_tmp++] = sources_sc[i];
	   localsizes_sc[sources_sc[i]]++;
	 }
   }
   ierr = PetscFree2(indices_ov,sources_sc);CHKERRQ(ierr);
   ierr = PetscCalloc1(ssize+1,&localoffsets);CHKERRQ(ierr);
   for(k=0; k<ssize; k++){
	 localoffsets[k+1] = localoffsets[k] + localsizes_sc[i];
   }
   /*build a star forest to send data back */
   nleaves = localoffsets[ssize];
   ierr = PetscMemzero(localoffsets,(ssize+1)*sizeof(PetscInt));CHKERRQ(ierr);
   nroots  = localsizes_sc[srank];
   ierr = PetscCalloc1(nleaves,&remote);CHKERRQ(ierr);
   for(i=0; i<nleaves; i++){
	 remote[i].rank  = sources_sc[i];
	 remote[i].index = localoffsets[sources_sc[i]]++;
   }
  }else{
   ierr = ISDestroy(&allis_sc);CHKERRQ(ierr);
   nleaves = 0;
   indices_ov_rd = 0;
   sources_sc_rd = 0;
  }
  /*scatter sizes to everybody */
  ierr = MPI_Scatter(localsizes_sc,1, MPIU_INT,&nroots,1, MPIU_INT,0,scomm);CHKERRQ(ierr);
  ierr = PetscCalloc1(nroots,&indices_recv);CHKERRQ(ierr);
  /*set data back to every body */
  ierr = PetscSFCreate(scomm,&sf);CHKERRQ(ierr);
  ierr = PetscSFSetType(sf,PETSCSFBASIC);CHKERRQ(ierr);
  ierr = PetscSFSetFromOptions(sf);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(sf,nroots,nleaves,PETSC_NULL,PETSC_OWN_POINTER,remote,PETSC_OWN_POINTER);CHKERRQ(ierr);
  ierr = PetscSFReduceBegin(sf,MPIU_INT,&indices_ov_rd,indices_recv,MPI_REPLACE);CHKERRQ(ierr);
  ierr = PetscSFReduceEnd(sf,MPIU_INT,&indices_ov_rd,indices_recv,MPI_REPLACE);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);
  /* free memory */
  ierr = PetscFree2(indices_ov_rd,sources_sc_rd);CHKERRQ(ierr);
  /*create a index set*/
  ierr = ISCreateGeneral(scomm,nroots,indices_recv,PETSC_OWN_POINTER,&is_sc);CHKERRQ(ierr);
  /*create a index set for cols */
  ierr = ISAllGather(is_sc,&allis_sc);CHKERRQ(ierr);
  /*reparition */
  /*construct a parallel submatrix */
  ierr = PetscCalloc1(1,&smat);CHKERRQ(ierr);
  ierr = MatGetSubMatricesMPI(mat,1,&is_sc,&allis_sc,MAT_INITIAL_MATRIX,&smat);CHKERRQ(ierr);
  /* we do not need them any more */
  ierr = ISDestroy(&is_sc);CHKERRQ(ierr);
  ierr = ISDestroy(&allis_sc);CHKERRQ(ierr);
  /*create a partitioner to repartition the sub-matrix*/
  ierr = MatPartitioningCreate(scomm,&part);CHKERRQ(ierr);
  ierr = MatPartitioningSetAdjacency(part,smat[0]);CHKERRQ(ierr);
#if PETSC_HAVE_PARMETIS
  /* if there exists a ParMETIS installation, we try to use ParMETIS
   * because a repartition routine possibly work better
   * */
  ierr = MatPartitioningSetType(part,MATPARTITIONINGPARMETIS);CHKERRQ(ierr);
  /*try to use reparition function, instead of partition function */
  ierr = MatPartitioningParmetisSetRepartition(part);CHKERRQ(ierr);
#else
  /*we at least provide a default partitioner to rebalance the computation  */
  ierr = MatPartitioningSetType(part,MATPARTITIONINGAVERAGE);CHKERRQ(ierr);
#endif
  /*user can pick up any partitioner by using an option*/
  ierr = MatPartitioningSetFromOptions(part);CHKERRQ(ierr);
  /* apply partition */
  ierr = MatPartitioningApply(part,&partitioning);CHKERRQ(ierr);
  ierr = MatPartitioningDestroy(&part);CHKERRQ(ierr);
  ierr = PetscFree(smat);CHKERRQ(ierr);
  /* get local rows including  overlap */
  ierr = ISBuildTwoSided(partitioning,is);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


