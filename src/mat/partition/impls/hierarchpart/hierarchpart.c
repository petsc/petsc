
#include <../src/mat/impls/adj/mpi/mpiadj.h>    /*I "petscmat.h" I*/
#include <petscsf.h>
#include <petsc/private/matimpl.h>

/*
  It is a hierarchical partitioning. The partitioner has two goals:
  (1) Most of current partitioners fail at a large scale. The hierarchical partitioning
  strategy is trying to produce large number of subdomains when number of processor cores is large.
  (2) PCGASM needs one 'big' subdomain across multi-cores. The partitioner provides two
  consistent partitions, coarse parts and fine parts. A coarse part is a 'big' subdomain consisting
  of several small subdomains.
*/

PetscErrorCode MatPartitioningHierarchPart_DetermineDestination(MatPartitioning part, IS partitioning, PetscInt pstart, PetscInt pend, IS *destination);
PetscErrorCode MatPartitioningHierarchPart_AssembleSubdomain(Mat adj,IS destination,Mat *sadj, ISLocalToGlobalMapping *mapping);
PetscErrorCode MatPartitioningHierarchPart_ReassembleFineparts(Mat adj, IS fineparts, ISLocalToGlobalMapping mapping, IS *sfineparts);

typedef struct {
  char*                fineparttype; /* partitioner on fine level */
  char*                coarseparttype; /* partitioner on coarse level */
  PetscInt             Nfineparts; /* number of fine parts on each coarse subdomain*/
  PetscInt             Ncoarseparts; /* number of coarse parts */
  IS                   coarseparts; /* partitioning on coarse level */
  IS                   fineparts; /* partitioning on fine level */
} MatPartitioning_HierarchPart;

/*
   Uses a hierarchical partitioning strategy to partition the matrix in parallel.
   Use this interface to make the partitioner consistent with others
*/
#undef __FUNCT__
#define __FUNCT__ "MatPartitioningApply_HierarchPart"
static PetscErrorCode MatPartitioningApply_HierarchPart(MatPartitioning part,IS *partitioning)
{
  MatPartitioning_HierarchPart *hpart  = (MatPartitioning_HierarchPart*)part->data;
  const PetscInt               *fineparts_indices, *coarseparts_indices;
  PetscInt                     *parts_indices,i,j,mat_localsize;
  Mat                           mat    = part->adj,adj,sadj;
  PetscBool                     flg;
  PetscInt                      bs     = 1;
  MatPartitioning               finePart, coarsePart;
  PetscInt                     *coarse_vertex_weights = 0;
  PetscMPIInt                   size,rank;
  MPI_Comm                      comm,scomm;
  IS                            destination,fineparts_temp;
  ISLocalToGlobalMapping        mapping;
  PetscErrorCode                ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)part,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)mat,MATMPIADJ,&flg);CHKERRQ(ierr);
  if (flg) {
    adj = mat;
    ierr = PetscObjectReference((PetscObject)adj);CHKERRQ(ierr);
  }else {
    /* bs indicates if the converted matrix is "reduced" from the original and hence the
       resulting partition results need to be stretched to match the original matrix */
   ierr = MatConvert(mat,MATMPIADJ,MAT_INITIAL_MATRIX,&adj);CHKERRQ(ierr);
   if (adj->rmap->n > 0) bs = mat->rmap->n/adj->rmap->n;
  }
  /*local size of mat*/
  mat_localsize = adj->rmap->n;
  /* check parameters */
  /* how many small subdomains we want from a given 'big' suddomain */
  if(!hpart->Nfineparts) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG," must set number of small subdomains for each big subdomain \n");
  if(!hpart->Ncoarseparts && !part->n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE," did not either set number of coarse parts or total number of parts \n");
  if(part->n && part->n%hpart->Nfineparts!=0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,
		   " total number of parts %D can not be divided by number of fine parts %D\n",part->n,hpart->Nfineparts);
  if(part->n){
    hpart->Ncoarseparts = part->n/hpart->Nfineparts;
  }else{
	part->n = hpart->Ncoarseparts*hpart->Nfineparts;
  }
   /* we do not support this case currently, but this restriction should be
     * removed in the further
     * */
  if(hpart->Ncoarseparts>size) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP," we do not support number of coarse parts %D > size %D \n",hpart->Ncoarseparts,size);
  /*create a coarse partitioner */
  ierr = MatPartitioningCreate(comm,&coarsePart);CHKERRQ(ierr);
    /*if did not set partitioning type yet, use parmetis by default */
  if(!hpart->coarseparttype){
	ierr = MatPartitioningSetType(coarsePart,MATPARTITIONINGPARMETIS);CHKERRQ(ierr);
  }else{
	ierr = MatPartitioningSetType(coarsePart,hpart->coarseparttype);CHKERRQ(ierr);
  }
  ierr = MatPartitioningSetAdjacency(coarsePart,adj);CHKERRQ(ierr);
  ierr = MatPartitioningSetNParts(coarsePart, hpart->Ncoarseparts);CHKERRQ(ierr);
  /*copy over vertex weights */
  if(part->vertex_weights){
   ierr = PetscMalloc(sizeof(PetscInt)*mat_localsize,&coarse_vertex_weights);CHKERRQ(ierr);
   ierr = PetscMemcpy(coarse_vertex_weights,part->vertex_weights,sizeof(PetscInt)*mat_localsize);CHKERRQ(ierr);
   ierr = MatPartitioningSetVertexWeights(coarsePart,coarse_vertex_weights);CHKERRQ(ierr);
  }
   /*It looks nontrivial to support part weights */
  /*if(part->part_weights){
	ierr = PetscMalloc(sizeof(part->part_weights)*1,&coarse_partition_weights);CHKERRQ(ierr);
	ierr = PetscMemcpy(coarse_partition_weights,part->part_weights,sizeof(part->part_weights)*1);CHKERRQ(ierr);
	ierr = MatPartitioningSetPartitionWeights(coarsePart,coarse_partition_weights);CHKERRQ(ierr);
  }*/
  ierr = MatPartitioningApply(coarsePart,&hpart->coarseparts);CHKERRQ(ierr);
  ierr = MatPartitioningDestroy(&coarsePart);CHKERRQ(ierr);
#if 0
  ierr = ISView(hpart->coarseparts,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
#endif
  /* In the current implementation, destination should be the same as hpart->coarseparts,
   * and this interface is preserved to deal with the case hpart->coarseparts>size in the
   * future.
   * */
  ierr = MatPartitioningHierarchPart_DetermineDestination(part,hpart->coarseparts,0,hpart->Ncoarseparts,&destination);CHKERRQ(ierr);
#if 0
  ierr = ISView(destination,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
#endif
  /* create a sub-matrix*/
  ierr = MatPartitioningHierarchPart_AssembleSubdomain(adj,destination,&sadj,&mapping);CHKERRQ(ierr);
#if 0
  ierr = MatView(sadj,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = MPI_Barrier(comm);CHKERRQ(ierr);
#endif
  ierr = ISDestroy(&destination);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)sadj,&scomm);CHKERRQ(ierr);
  /*create a fine partitioner */
  ierr = MatPartitioningCreate(scomm,&finePart);CHKERRQ(ierr);
  /*if do not set partitioning type, use parmetis by default */
  if(!hpart->fineparttype){
    ierr = MatPartitioningSetType(finePart,MATPARTITIONINGPARMETIS);CHKERRQ(ierr);
  }else{
    ierr = MatPartitioningSetType(finePart,hpart->fineparttype);CHKERRQ(ierr);
  }
  ierr = MatPartitioningSetAdjacency(finePart,sadj);CHKERRQ(ierr);
  ierr = MatPartitioningSetNParts(finePart, hpart->Nfineparts);CHKERRQ(ierr);
  ierr = MatPartitioningApply(finePart,&fineparts_temp);CHKERRQ(ierr);
  ierr = MatDestroy(&sadj);CHKERRQ(ierr);
  ierr = MatPartitioningDestroy(&finePart);CHKERRQ(ierr);
#if 0
  ierr = ISView(fineparts_temp,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = MPI_Barrier(comm);CHKERRQ(ierr);
#endif
  ierr = MatPartitioningHierarchPart_ReassembleFineparts(adj,fineparts_temp,mapping,&hpart->fineparts);CHKERRQ(ierr);
  ierr = ISDestroy(&fineparts_temp);CHKERRQ(ierr);
#if 0
  ierr = ISView(hpart->fineparts,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MPI_Barrier(comm);CHKERRQ(ierr);
#endif
  ierr = ISLocalToGlobalMappingDestroy(&mapping);CHKERRQ(ierr);

  ierr = ISGetIndices(hpart->fineparts,&fineparts_indices);CHKERRQ(ierr);
  ierr = ISGetIndices(hpart->coarseparts,&coarseparts_indices);CHKERRQ(ierr);
  ierr = PetscMalloc1(bs*adj->rmap->n,&parts_indices);CHKERRQ(ierr);
  for(i=0; i<adj->rmap->n; i++){
    for(j=0; j<bs; j++){
      parts_indices[bs*i+j] = fineparts_indices[i]+coarseparts_indices[i]*hpart->Nfineparts;
    }
  }
  ierr = ISCreateGeneral(comm,bs*adj->rmap->n,parts_indices,PETSC_OWN_POINTER,partitioning);CHKERRQ(ierr);
#if 0
  ierr = ISView(*partitioning,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MPI_Barrier(comm);CHKERRQ(ierr);
  /*SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_INCOMP,"stop stop stop here \n");*/
#endif
  ierr = MatDestroy(&adj);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MatPartitioningHierarchPart_ReassembleFineparts"
PetscErrorCode MatPartitioningHierarchPart_ReassembleFineparts(Mat adj, IS fineparts, ISLocalToGlobalMapping mapping, IS *sfineparts)
{
  PetscInt            *local_indices, *global_indices,*owners,*sfineparts_indices,localsize,i;;
  const PetscInt      *ranges,*fineparts_indices;
  PetscMPIInt         rank;
  MPI_Comm            comm;
  PetscLayout         rmap;
  PetscSFNode        *remote;
  PetscSF             sf;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  /*get communicator */
  ierr = PetscObjectGetComm((PetscObject)adj,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = MatGetLayouts(adj,&rmap,PETSC_NULL);CHKERRQ(ierr);
  ierr = ISGetLocalSize(fineparts,&localsize);CHKERRQ(ierr);
  ierr = PetscCalloc2(localsize,&global_indices,localsize,&local_indices);CHKERRQ(ierr);
  for(i=0; i<localsize; i++){
	local_indices[i] = i;
  }
  /*global indices */
  ierr = ISLocalToGlobalMappingApply(mapping,localsize,local_indices,global_indices);CHKERRQ(ierr);
  ierr = PetscCalloc1(localsize,&owners);CHKERRQ(ierr);
  /*find owners for global indices */
  for(i=0; i<localsize; i++){
	ierr = PetscLayoutFindOwner(rmap,global_indices[i],&owners[i]);CHKERRQ(ierr);
  }
  /*ranges */
  ierr = PetscLayoutGetRanges(rmap,&ranges);CHKERRQ(ierr);
  ierr = PetscCalloc1(ranges[rank+1]-ranges[rank],&sfineparts_indices);CHKERRQ(ierr);
  ierr = ISGetIndices(fineparts,&fineparts_indices);CHKERRQ(ierr);
  /*create a SF to exchange data */
  ierr = PetscSFCreate(comm,&sf);CHKERRQ(ierr);
  ierr = PetscCalloc1(localsize,&remote);CHKERRQ(ierr);
  for(i=0; i<localsize; i++){
	remote[i].rank  = owners[i];
	remote[i].index = global_indices[i]-ranges[owners[i]];
  }
  ierr = PetscSFSetType(sf,PETSCSFBASIC);CHKERRQ(ierr);
  ierr = PetscSFSetFromOptions(sf);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(sf,localsize,localsize,PETSC_NULL,PETSC_OWN_POINTER,remote,PETSC_OWN_POINTER);CHKERRQ(ierr);
  ierr = PetscSFReduceBegin(sf,MPIU_INT,fineparts_indices,sfineparts_indices,MPIU_REPLACE);CHKERRQ(ierr);
  ierr = PetscSFReduceEnd(sf,MPIU_INT,fineparts_indices,sfineparts_indices,MPIU_REPLACE);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);
  ierr = ISRestoreIndices(fineparts,&fineparts_indices);CHKERRQ(ierr);
  /* comm self */
  ierr = ISCreateGeneral(comm,ranges[rank+1]-ranges[rank],sfineparts_indices,PETSC_OWN_POINTER,sfineparts);CHKERRQ(ierr);
  ierr = PetscFree2(global_indices,local_indices);CHKERRQ(ierr);
  ierr = PetscFree(owners);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPartitioningHierarchPart_AssembleSubdomain"
PetscErrorCode MatPartitioningHierarchPart_AssembleSubdomain(Mat adj,IS destination,Mat *sadj, ISLocalToGlobalMapping *mapping)
{
  PetscInt        *rows_send,*adjncy_send,*roffsets,*coffsets,*rsizes,*csizes,i,j,mat_localsize;
  PetscInt        rstart,rend,nto,*torsizes,*tocsizes,nfrom,*fromrsizes,*fromcsizes;
  PetscInt        *rows_recv,*adjncy_recv,nrows_recv,nzeros_recv,*ncols_send,*ncols_recv;
  PetscInt        *localperm,*si,*si_sizes,*sj,sj_size,location,k,m,dest_localsize,*reverseperm,*localperm_tmp;
  const PetscInt  *xadj, *adjncy,*dest_indices;
  MPI_Comm        comm;
  PetscMPIInt     size,rank,target_rank,*toranks,*fromranks,*fromperm;
  PetscLayout     rmap;
  PetscBool       done;
  PetscSF         sf;
  PetscSFNode     *iremote;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  /*communicator */
  ierr = PetscObjectGetComm((PetscObject)adj,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MatGetLayouts(adj,&rmap,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscLayoutGetRange(rmap,&rstart,&rend);CHKERRQ(ierr);
  /*offsets, depends on the number of processors  */
  ierr = PetscCalloc4(size+1,&roffsets,size+1,&coffsets,size,&rsizes,size,&csizes);CHKERRQ(ierr);
  /*retrieve data*/
  ierr = MatGetRowIJ(adj,0,PETSC_FALSE,PETSC_FALSE,&mat_localsize,&xadj,&adjncy,&done);CHKERRQ(ierr);
  ierr = ISGetLocalSize(destination,&dest_localsize);CHKERRQ(ierr);
  if(dest_localsize != mat_localsize) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"dest_localsize%d != mat_localsize %d \n",dest_localsize,mat_localsize);
  /*get indices */
  ierr = ISGetIndices(destination,&dest_indices);CHKERRQ(ierr);
  /*estimate the space*/
  for(i=0; i<mat_localsize; i++){
	if(dest_indices[i]<0) continue;
    rsizes[dest_indices[i]]++;
    csizes[dest_indices[i]] += xadj[i+1]-xadj[i];
  }
  /*offsets*/
  nto = 0;
  for(i=0; i<size; i++){
    roffsets[i+1] = roffsets[i]+rsizes[i];
    coffsets[i+1] = coffsets[i]+csizes[i];
    if(rsizes[i]>0) nto++;
  }
  ierr = PetscCalloc3(nto,&toranks,2*nto,&torsizes,2*nto,&tocsizes);CHKERRQ(ierr);
  /* send row and col sizes  */
  nto = 0;
  for(i=0; i<size; i++){
    if(rsizes[i]>0){
      toranks[nto]      = i;
      torsizes[2*nto]   = rsizes[i];
      torsizes[2*nto+1] = roffsets[i];
      tocsizes[2*nto]   = csizes[i];
      tocsizes[2*nto+1] = coffsets[i];
      nto++;
    }
  }
  ierr = PetscCalloc3(roffsets[size],&rows_send,roffsets[size],&ncols_send,coffsets[size],&adjncy_send);CHKERRQ(ierr);
  for(i=0; i<mat_localsize; i++){
	if(dest_indices[i]<0) continue;
    target_rank = dest_indices[i];
    rows_send[roffsets[target_rank]]     = i+rstart;
    ncols_send[roffsets[target_rank]++]  = xadj[i+1]-xadj[i];
    ierr = PetscMemcpy(adjncy_send+coffsets[target_rank],adjncy+xadj[i],sizeof(PetscInt)*(xadj[i+1]-xadj[i]));CHKERRQ(ierr);
    coffsets[target_rank] += xadj[i+1]-xadj[i];
  }
  ierr = PetscFree4(roffsets,coffsets,rsizes,csizes);CHKERRQ(ierr);
  /* rows */
  ierr = PetscCommBuildTwoSided(comm,2,MPIU_INT,nto,toranks,torsizes,&nfrom,&fromranks,&fromrsizes);CHKERRQ(ierr);
  ierr = PetscCalloc1(nfrom,&fromperm);CHKERRQ(ierr);
  for(i=0; i<nfrom; i++){
	fromperm[i] = i;
  }
  /*order MPI ranks so that rows and columns consist with each other*/
  ierr = PetscSortMPIIntWithArray(nfrom,fromranks,fromperm);CHKERRQ(ierr);
  nrows_recv   = 0;
  for(i=0; i<nfrom; i++){
	nrows_recv += fromrsizes[i*2];
  }
  ierr = PetscCalloc2(nrows_recv,&rows_recv,nrows_recv,&ncols_recv);CHKERRQ(ierr);
  ierr = PetscCalloc1(nrows_recv,&iremote);CHKERRQ(ierr);
  nrows_recv = 0;
  for(i=0; i<nfrom; i++){
    for(j=0; j<fromrsizes[2*fromperm[i]]; j++){
      iremote[nrows_recv].rank    = fromranks[i];
      iremote[nrows_recv++].index = fromrsizes[2*fromperm[i]+1]+j;
    }
  }
  /*create a sf to exchange rows */
  ierr = PetscSFCreate(comm,&sf);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(sf,nrows_recv,nrows_recv,PETSC_NULL,PETSC_OWN_POINTER,iremote,PETSC_OWN_POINTER);CHKERRQ(ierr);
  ierr = PetscSFSetType(sf,PETSCSFBASIC);CHKERRQ(ierr);
  ierr = PetscSFSetFromOptions(sf);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(sf,MPIU_INT,rows_send,rows_recv);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(sf,MPIU_INT,rows_send,rows_recv);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(sf,MPIU_INT,ncols_send,ncols_recv);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(sf,MPIU_INT,ncols_send,ncols_recv);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);
  ierr = PetscFree(fromranks);CHKERRQ(ierr);
  ierr = PetscFree(fromrsizes);CHKERRQ(ierr);
  ierr = PetscFree(fromperm);CHKERRQ(ierr);
  /*columns */
  ierr = PetscCommBuildTwoSided(comm,2,MPIU_INT,nto,toranks,tocsizes,&nfrom,&fromranks,&fromcsizes);CHKERRQ(ierr);
  ierr = PetscCalloc1(nfrom,&fromperm);CHKERRQ(ierr);
  for(i=0; i<nfrom; i++){
  	fromperm[i] = i;
  }
  /*order these so that rows and columns consist with each other*/
  ierr = PetscSortMPIIntWithArray(nfrom,fromranks,fromperm);CHKERRQ(ierr);
  nzeros_recv   = 0;
  for(i=0; i<nfrom; i++){
	nzeros_recv += fromcsizes[i*2];
  }
  ierr = PetscCalloc1(nzeros_recv,&adjncy_recv);CHKERRQ(ierr);
  ierr = PetscCalloc1(nzeros_recv,&iremote);CHKERRQ(ierr);
  nzeros_recv = 0;
  for(i=0; i<nfrom; i++){
    for(j=0; j<fromcsizes[2*fromperm[i]]; j++){
      iremote[nzeros_recv].rank    = fromranks[i];
      iremote[nzeros_recv++].index = fromcsizes[2*fromperm[i]+1]+j;
    }
  }
  /*create a sf to exchange columns */
  ierr = PetscSFCreate(comm,&sf);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(sf,nzeros_recv,nzeros_recv,PETSC_NULL,PETSC_OWN_POINTER,iremote,PETSC_OWN_POINTER);CHKERRQ(ierr);
  ierr = PetscSFSetType(sf,PETSCSFBASIC);CHKERRQ(ierr);
  ierr = PetscSFSetFromOptions(sf);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(sf,MPIU_INT,adjncy_send,adjncy_recv);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(sf,MPIU_INT,adjncy_send,adjncy_recv);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);
  ierr = PetscFree3(rows_send,ncols_send,adjncy_send);CHKERRQ(ierr);
  ierr = PetscFree3(toranks,torsizes,tocsizes);CHKERRQ(ierr);
  ierr = PetscFree(fromranks);CHKERRQ(ierr);
  ierr = PetscFree(fromcsizes);CHKERRQ(ierr);
  ierr = PetscFree(fromperm);CHKERRQ(ierr);
  ierr = PetscCalloc3(nrows_recv,&localperm,nrows_recv,&reverseperm,nrows_recv,&localperm_tmp);CHKERRQ(ierr);
  for(i=0; i<nrows_recv; i++){
	localperm[i] = i;
	reverseperm[i] = i;
  }
  ierr = PetscSortIntWithArray(nrows_recv,rows_recv,localperm);CHKERRQ(ierr);
  ierr = PetscMemcpy(localperm_tmp,localperm,sizeof(PetscInt)*nrows_recv);CHKERRQ(ierr);
  ierr = PetscSortIntWithArray(nrows_recv,localperm_tmp,reverseperm);CHKERRQ(ierr);
  /*create a mapping local to global */
  ierr = ISLocalToGlobalMappingCreate(comm,1,nrows_recv,rows_recv,PETSC_COPY_VALUES,mapping);CHKERRQ(ierr);
  ierr = PetscCalloc1(nrows_recv+1,&si);CHKERRQ(ierr);
  ierr = PetscCalloc1(nrows_recv,&si_sizes);CHKERRQ(ierr);
  ierr = PetscMemcpy(si_sizes,ncols_recv,sizeof(PetscInt)*nrows_recv);CHKERRQ(ierr);
#if 0
  ierr = PetscIntView(nrows_recv,ncols_recv,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MPI_Barrier(comm);CHKERRQ(ierr);
  ierr = PetscIntView(nrows_recv,rows_recv,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MPI_Barrier(comm);CHKERRQ(ierr);
  ierr = PetscIntView(nzeros_recv,adjncy_recv,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MPI_Barrier(comm);CHKERRQ(ierr);
  ierr = PetscIntView(nrows_recv,si_sizes,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MPI_Barrier(comm);CHKERRQ(ierr);
  ierr = PetscIntView(nrows_recv,reverseperm,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MPI_Barrier(comm);CHKERRQ(ierr);
#endif
  k       = 0;
  sj_size = 0;
  for(i=0; i<nrows_recv; i++){
	for(j=0; j<ncols_recv[i]; j++,k++){
	   ierr = PetscFindInt(adjncy_recv[k],nrows_recv,rows_recv,&location);CHKERRQ(ierr);
	   if(location<0){
		 adjncy_recv[k]              = -1;
		 si_sizes[reverseperm[i]]   -= 1;
	   }else{
		 adjncy_recv[k] = location;
		 sj_size++;
	   }
	}
  }
  for(i=0; i<nrows_recv; i++){
	si[i+1] = si[i]+si_sizes[i];
  }
#if 0
  ierr = PetscIntView(nrows_recv,si_sizes,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MPI_Barrier(comm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"sj_size %D \n",sj_size);CHKERRQ(ierr);
  ierr = MPI_Barrier(comm);CHKERRQ(ierr);
#endif
  ierr = PetscFree(si_sizes);CHKERRQ(ierr);
  ierr = PetscCalloc1(sj_size,&sj);CHKERRQ(ierr);
  k = 0;
  m = 0;
  for(i=0; i<nrows_recv; i++){
    for(j=0,m=0; j<ncols_recv[i]; j++,k++){
      if(adjncy_recv[k] < 0) continue;
      sj[si[reverseperm[i]]+m] = adjncy_recv[k];
      m++;
	}
  }
#if 0
  ierr = PetscIntView(nrows_recv+1,si,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MPI_Barrier(comm);CHKERRQ(ierr);
  ierr = PetscIntView(sj_size,sj,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MPI_Barrier(comm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"nrows_recv %D \n",nrows_recv);CHKERRQ(ierr);
  ierr = MPI_Barrier(comm);CHKERRQ(ierr);
#endif
  /* return the assembled submatrix */
  ierr = MatCreateMPIAdj(PETSC_COMM_SELF,nrows_recv,nrows_recv,si,sj,PETSC_NULL,sadj);CHKERRQ(ierr);
  ierr = MatRestoreRowIJ(adj,0,PETSC_FALSE,PETSC_FALSE,&mat_localsize,&xadj,&adjncy,&done);CHKERRQ(ierr);
  ierr = PetscFree2(rows_recv,ncols_recv);CHKERRQ(ierr);
  ierr = PetscFree3(localperm,reverseperm,localperm_tmp);CHKERRQ(ierr);
  ierr = PetscFree(adjncy_recv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPartitioningHierarchPart_DetermineDestination"
PetscErrorCode MatPartitioningHierarchPart_DetermineDestination(MatPartitioning part, IS partitioning, PetscInt pstart, PetscInt pend, IS *destination)
{
  MPI_Comm            comm;
  PetscMPIInt         rank,size,target;
  PetscInt            plocalsize,*dest_indices,i;
  const PetscInt     *part_indices;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  /*communicator*/
  ierr = PetscObjectGetComm((PetscObject)part,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if((pend-pstart)>size) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"range [%D, %D] should be smaller than or equal to size %D",pstart,pend,size);CHKERRQ(ierr);
  if(pstart>pend) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP," pstart %D should be smaller than pend %D",pstart,pend);CHKERRQ(ierr);
  /*local size*/
  ierr = ISGetLocalSize(partitioning,&plocalsize);CHKERRQ(ierr);
  ierr = PetscCalloc1(plocalsize,&dest_indices);CHKERRQ(ierr);
  ierr = ISGetIndices(partitioning,&part_indices);CHKERRQ(ierr);
  for(i=0; i<plocalsize; i++){
	/*compute target */
    target = part_indices[i]-pstart;
    /*mark out of range entity as -1*/
    if(part_indices[i]<pstart || part_indices[i]>pend) target = -1;
	dest_indices[i] = target;
  }
  /*return destination back*/
  ierr = ISCreateGeneral(comm,plocalsize,dest_indices,PETSC_OWN_POINTER,destination);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MatPartitioningView_HierarchPart"
PetscErrorCode MatPartitioningView_HierarchPart(MatPartitioning part,PetscViewer viewer)
{
  MatPartitioning_HierarchPart *hpart = (MatPartitioning_HierarchPart*)part->data;
  PetscErrorCode           ierr;
  PetscMPIInt              rank;
  PetscBool                iascii;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)part),&rank);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if(iascii){
	 ierr = PetscViewerASCIIPrintf(viewer," Fine partitioner %s \n",hpart->fineparttype);CHKERRQ(ierr);
	 ierr = PetscViewerASCIIPrintf(viewer," Coarse partitioner %s \n",hpart->coarseparttype);CHKERRQ(ierr);
	 ierr = PetscViewerASCIIPrintf(viewer," Number of coarse parts %D \n",hpart->Ncoarseparts);CHKERRQ(ierr);
	 ierr = PetscViewerASCIIPrintf(viewer," Number of fine parts %D \n",hpart->Nfineparts);CHKERRQ(ierr);
	 ierr = PetscViewerASCIISynchronizedAllow(viewer,PETSC_FALSE);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MatPartitioningHierarchpartGetFineparts"
PetscErrorCode MatPartitioningHierarchpartGetFineparts(MatPartitioning part,IS *fineparts)
{
  MatPartitioning_HierarchPart *hpart = (MatPartitioning_HierarchPart*)part->data;
  PetscErrorCode                ierr;

  PetscFunctionBegin;
  *fineparts = hpart->fineparts;
  ierr = PetscObjectReference((PetscObject)hpart->fineparts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPartitioningHierarchpartGetCoarseparts"
PetscErrorCode MatPartitioningHierarchpartGetCoarseparts(MatPartitioning part,IS *coarseparts)
{
  MatPartitioning_HierarchPart *hpart = (MatPartitioning_HierarchPart*)part->data;
  PetscErrorCode                ierr;

  PetscFunctionBegin;
  *coarseparts = hpart->coarseparts;
  ierr = PetscObjectReference((PetscObject)hpart->coarseparts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPartitioningHierarchpartSetNcoarseparts"
PetscErrorCode MatPartitioningHierarchpartSetNcoarseparts(MatPartitioning part, PetscInt Ncoarseparts)
{
  MatPartitioning_HierarchPart *hpart = (MatPartitioning_HierarchPart*)part->data;

  PetscFunctionBegin;
  hpart->Ncoarseparts = Ncoarseparts;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPartitioningHierarchpartSetNfineparts"
PetscErrorCode MatPartitioningHierarchpartSetNfineparts(MatPartitioning part, PetscInt Nfineparts)
{
  MatPartitioning_HierarchPart *hpart = (MatPartitioning_HierarchPart*)part->data;

  PetscFunctionBegin;
  hpart->Nfineparts = Nfineparts;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPartitioningSetFromOptions_HierarchPart"
PetscErrorCode MatPartitioningSetFromOptions_HierarchPart(PetscOptions *PetscOptionsObject,MatPartitioning part)
{
  MatPartitioning_HierarchPart *hpart = (MatPartitioning_HierarchPart*)part->data;
  PetscErrorCode ierr;
  char           value[1024];
  PetscBool      flag = PETSC_FALSE;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"Set hierarchical partitioning options");CHKERRQ(ierr);
  ierr = PetscOptionsString("-mat_partitioning_hierarchical_coarseparttype","coarse part type",PETSC_NULL,PETSC_NULL,value,1024,&flag);CHKERRQ(ierr);
  if(flag){
   ierr = PetscCalloc1(1024,&hpart->coarseparttype);CHKERRQ(ierr);
   ierr = PetscStrcpy(hpart->coarseparttype,value);CHKERRQ(ierr);
  }
  ierr = PetscOptionsString("-mat_partitioning_hierarchical_fineparttype","fine part type",PETSC_NULL,PETSC_NULL,value,1024,&flag);CHKERRQ(ierr);
  if(flag){
    ierr = PetscCalloc1(1024,&hpart->fineparttype);CHKERRQ(ierr);
    ierr = PetscStrcpy(hpart->fineparttype,value);CHKERRQ(ierr);
  }
  ierr = PetscOptionsInt("-mat_partitioning_hierarchical_Ncoarseparts","number of coarse parts",PETSC_NULL,0,&hpart->Ncoarseparts,&flag);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-mat_partitioning_hierarchical_Nfineparts","number of fine parts",PETSC_NULL,1,&hpart->Nfineparts,&flag);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MatPartitioningDestroy_HierarchPart"
PetscErrorCode MatPartitioningDestroy_HierarchPart(MatPartitioning part)
{
  MatPartitioning_HierarchPart *hpart = (MatPartitioning_HierarchPart*)part->data;
  PetscErrorCode           ierr;

  PetscFunctionBegin;
  if(hpart->coarseparttype) {ierr = PetscFree(hpart->coarseparttype);CHKERRQ(ierr);}
  if(hpart->fineparttype) {ierr = PetscFree(hpart->fineparttype);CHKERRQ(ierr);}
  ierr = ISDestroy(&hpart->fineparts);CHKERRQ(ierr);
  ierr = ISDestroy(&hpart->coarseparts);CHKERRQ(ierr);
  ierr = PetscFree(hpart);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*MC
   MATPARTITIONINGHIERARCHPART - Creates a partitioning context via hierarchical partitioning strategy.

   Collective on MPI_Comm

   Input Parameter:
.  part - the partitioning context

   Options Database Keys:

   Level: beginner

.keywords: Partitioning, create, context

.seealso: MatPartitioningSetType(), MatPartitioningType

M*/

#undef __FUNCT__
#define __FUNCT__ "MatPartitioningCreate_HierarchPart"
PETSC_EXTERN PetscErrorCode MatPartitioningCreate_HierarchPart(MatPartitioning part)
{
  PetscErrorCode                ierr;
  MatPartitioning_HierarchPart *hpart;

  PetscFunctionBegin;
  ierr       = PetscNewLog(part,&hpart);CHKERRQ(ierr);
  part->data = (void*)hpart;

  hpart->fineparttype       = 0; /* fine level partitioner */
  hpart->coarseparttype     = 0; /* coarse level partitioner */
  hpart->Nfineparts         = 1; /* we do not further partition coarse partition any more by default */
  hpart->Ncoarseparts       = 0; /* number of coarse parts (first level) */
  hpart->coarseparts        = 0;
  hpart->fineparts          = 0;

  part->ops->apply          = MatPartitioningApply_HierarchPart;
  part->ops->view           = MatPartitioningView_HierarchPart;
  part->ops->destroy        = MatPartitioningDestroy_HierarchPart;
  part->ops->setfromoptions = MatPartitioningSetFromOptions_HierarchPart;
  PetscFunctionReturn(0);
}


