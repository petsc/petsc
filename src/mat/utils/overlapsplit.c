/*
 * overlapsplit.c: increase the overlap of a 'big' subdomain across several processor cores
 *
 * Author: Fande Kong <fdkong.jd@gmail.com>
 */

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
  PetscMPIInt      srank,ssize,issamecomm,k,grank;
  IS               is_sc,allis_sc,partitioning;
  MPI_Comm         gcomm,dcomm,scomm;
  PetscSF          sf;
  PetscSFNode      *remote;
  Mat              *smat;
  MatPartitioning  part;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  /* get a sub communicator before call individual MatIncreaseOverlap
   * since the sub communicator may be changed.
   * */
  ierr = PetscObjectGetComm((PetscObject)(*is),&dcomm);CHKERRQ(ierr);
  /* make a copy before the original one is deleted */
  ierr = PetscCommDuplicate(dcomm,&scomm,NULL);CHKERRQ(ierr);
  /* get a global communicator, where mat should be a global matrix  */
  ierr = PetscObjectGetComm((PetscObject)mat,&gcomm);CHKERRQ(ierr);
  ierr = (*mat->ops->increaseoverlap)(mat,1,is,ov);CHKERRQ(ierr);
  ierr = MPI_Comm_compare(gcomm,scomm,&issamecomm);CHKERRQ(ierr);
  /* if the sub-communicator is the same as the global communicator,
   * user does not want to use a sub-communicator
   * */
  if(issamecomm == MPI_IDENT || issamecomm == MPI_CONGRUENT){
	ierr = PetscCommDestroy(&scomm);CHKERRQ(ierr);
	PetscFunctionReturn(0);
  }
  /* if the sub-communicator is petsc_comm_self,
   * user also does not care the sub-communicator
   * */
  ierr = MPI_Comm_compare(scomm,PETSC_COMM_SELF,&issamecomm);CHKERRQ(ierr);
  if (issamecomm == MPI_IDENT || issamecomm == MPI_CONGRUENT){
    ierr = PetscCommDestroy(&scomm);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = MPI_Comm_rank(scomm,&srank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(scomm,&ssize);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(gcomm,&grank);CHKERRQ(ierr);
  /* create a new IS based on sub-communicator
   * since the old IS is often based on petsc_comm_self
   * */
  ierr = ISGetLocalSize(*is,&nindx);CHKERRQ(ierr);
  ierr = PetscMalloc1(nindx,&indices_sc);CHKERRQ(ierr);
  ierr = ISGetIndices(*is,&indices);CHKERRQ(ierr);
  ierr = PetscArraycpy(indices_sc,indices,nindx);CHKERRQ(ierr);
  ierr = ISRestoreIndices(*is,&indices);CHKERRQ(ierr);
  /* we do not need any more */
  ierr = ISDestroy(is);CHKERRQ(ierr);
  /* create a index set based on the sub communicator  */
  ierr = ISCreateGeneral(scomm,nindx,indices_sc,PETSC_OWN_POINTER,&is_sc);CHKERRQ(ierr);
  /* gather all indices within  the sub communicator */
  ierr = ISAllGather(is_sc,&allis_sc);CHKERRQ(ierr);
  ierr = ISDestroy(&is_sc);CHKERRQ(ierr);
  /* gather local sizes */
  ierr = PetscMalloc1(ssize,&localsizes_sc);CHKERRQ(ierr);
  /* get individual local sizes for all index sets */
  ierr = MPI_Gather(&nindx,1,MPIU_INT,localsizes_sc,1,MPIU_INT,0,scomm);CHKERRQ(ierr);
  /* only root does these computations */
  if(!srank){
   /* get local size for the big index set */
   ierr = ISGetLocalSize(allis_sc,&localsize);CHKERRQ(ierr);
   ierr = PetscCalloc2(localsize,&indices_ov,localsize,&sources_sc);CHKERRQ(ierr);
   ierr = PetscCalloc2(localsize,&indices_ov_rd,localsize,&sources_sc_rd);CHKERRQ(ierr);
   ierr = ISGetIndices(allis_sc,&indices);CHKERRQ(ierr);
   ierr = PetscArraycpy(indices_ov,indices,localsize);CHKERRQ(ierr);
   ierr = ISRestoreIndices(allis_sc,&indices);CHKERRQ(ierr);
   ierr = ISDestroy(&allis_sc);CHKERRQ(ierr);
   /* assign corresponding sources */
   localsize_tmp = 0;
   for(k=0; k<ssize; k++){
     for(i=0; i<localsizes_sc[k]; i++){
       sources_sc[localsize_tmp++] = k;
     }
   }
   /* record where indices come from */
   ierr = PetscSortIntWithArray(localsize,indices_ov,sources_sc);CHKERRQ(ierr);
   /* count local sizes for reduced indices */
   ierr = PetscArrayzero(localsizes_sc,ssize);CHKERRQ(ierr);
   /* initialize the first entity */
   if (localsize){
     indices_ov_rd[0] = indices_ov[0];
     sources_sc_rd[0] = sources_sc[0];
     localsizes_sc[sources_sc[0]]++;
   }
   localsize_tmp = 1;
   /* remove duplicate integers */
   for (i=1; i<localsize; i++){
     if (indices_ov[i] != indices_ov[i-1]){
       indices_ov_rd[localsize_tmp]   = indices_ov[i];
       sources_sc_rd[localsize_tmp++] = sources_sc[i];
       localsizes_sc[sources_sc[i]]++;
     }
   }
   ierr = PetscFree2(indices_ov,sources_sc);CHKERRQ(ierr);
   ierr = PetscCalloc1(ssize+1,&localoffsets);CHKERRQ(ierr);
   for (k=0; k<ssize; k++){
     localoffsets[k+1] = localoffsets[k] + localsizes_sc[k];
   }
   nleaves = localoffsets[ssize];
   ierr    = PetscArrayzero(localoffsets,ssize+1);CHKERRQ(ierr);
   nroots  = localsizes_sc[srank];
   ierr    = PetscMalloc1(nleaves,&remote);CHKERRQ(ierr);
   for (i=0; i<nleaves; i++){
     remote[i].rank  = sources_sc_rd[i];
     remote[i].index = localoffsets[sources_sc_rd[i]]++;
   }
   ierr = PetscFree(localoffsets);CHKERRQ(ierr);
  } else {
   ierr = ISDestroy(&allis_sc);CHKERRQ(ierr);
   /* Allocate a 'zero' pointer to avoid using uninitialized variable  */
   ierr = PetscCalloc1(0,&remote);CHKERRQ(ierr);
   nleaves       = 0;
   indices_ov_rd = NULL;
   sources_sc_rd = NULL;
  }
  /* scatter sizes to everybody */
  ierr = MPI_Scatter(localsizes_sc,1, MPIU_INT,&nroots,1, MPIU_INT,0,scomm);CHKERRQ(ierr);
  ierr = PetscFree(localsizes_sc);CHKERRQ(ierr);
  ierr = PetscCalloc1(nroots,&indices_recv);CHKERRQ(ierr);
  /* set data back to every body */
  ierr = PetscSFCreate(scomm,&sf);CHKERRQ(ierr);
  ierr = PetscSFSetType(sf,PETSCSFBASIC);CHKERRQ(ierr);
  ierr = PetscSFSetFromOptions(sf);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(sf,nroots,nleaves,NULL,PETSC_OWN_POINTER,remote,PETSC_OWN_POINTER);CHKERRQ(ierr);
  ierr = PetscSFReduceBegin(sf,MPIU_INT,indices_ov_rd,indices_recv,MPIU_REPLACE);CHKERRQ(ierr);
  ierr = PetscSFReduceEnd(sf,MPIU_INT,indices_ov_rd,indices_recv,MPIU_REPLACE);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);
  ierr = PetscFree2(indices_ov_rd,sources_sc_rd);CHKERRQ(ierr);
  ierr = ISCreateGeneral(scomm,nroots,indices_recv,PETSC_OWN_POINTER,&is_sc);CHKERRQ(ierr);
  ierr = MatCreateSubMatricesMPI(mat,1,&is_sc,&is_sc,MAT_INITIAL_MATRIX,&smat);CHKERRQ(ierr);
  ierr = ISDestroy(&allis_sc);CHKERRQ(ierr);
  /* create a partitioner to repartition the sub-matrix */
  ierr = MatPartitioningCreate(scomm,&part);CHKERRQ(ierr);
  ierr = MatPartitioningSetAdjacency(part,smat[0]);CHKERRQ(ierr);
#if defined(PETSC_HAVE_PARMETIS)
  /* if there exists a ParMETIS installation, we try to use ParMETIS
   * because a repartition routine possibly work better
   * */
  ierr = MatPartitioningSetType(part,MATPARTITIONINGPARMETIS);CHKERRQ(ierr);
  /* try to use reparition function, instead of partition function */
  ierr = MatPartitioningParmetisSetRepartition(part);CHKERRQ(ierr);
#else
  /* we at least provide a default partitioner to rebalance the computation  */
  ierr = MatPartitioningSetType(part,MATPARTITIONINGAVERAGE);CHKERRQ(ierr);
#endif
  /* user can pick up any partitioner by using an option */
  ierr = MatPartitioningSetFromOptions(part);CHKERRQ(ierr);
  ierr = MatPartitioningApply(part,&partitioning);CHKERRQ(ierr);
  ierr = MatPartitioningDestroy(&part);CHKERRQ(ierr);
  ierr = MatDestroy(&(smat[0]));CHKERRQ(ierr);
  ierr = PetscFree(smat);CHKERRQ(ierr);
  /* get local rows including  overlap */
  ierr = ISBuildTwoSided(partitioning,is_sc,is);CHKERRQ(ierr);
  ierr = ISDestroy(&is_sc);CHKERRQ(ierr);
  ierr = ISDestroy(&partitioning);CHKERRQ(ierr);
  ierr = PetscCommDestroy(&scomm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


