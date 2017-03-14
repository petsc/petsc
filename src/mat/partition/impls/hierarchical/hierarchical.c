
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

PetscErrorCode MatPartitioningHierarchical_DetermineDestination(MatPartitioning part, IS partitioning, PetscInt pstart, PetscInt pend, IS *destination);
PetscErrorCode MatPartitioningHierarchical_AssembleSubdomain(Mat adj,IS destination,Mat *sadj, ISLocalToGlobalMapping *mapping);
PetscErrorCode MatPartitioningHierarchical_ReassembleFineparts(Mat adj, IS fineparts, ISLocalToGlobalMapping mapping, IS *sfineparts);

typedef struct {
  char*                fineparttype; /* partitioner on fine level */
  char*                coarseparttype; /* partitioner on coarse level */
  PetscInt             Nfineparts; /* number of fine parts on each coarse subdomain */
  PetscInt             Ncoarseparts; /* number of coarse parts */
  IS                   coarseparts; /* partitioning on coarse level */
  IS                   fineparts; /* partitioning on fine level */
} MatPartitioning_Hierarchical;

/*
   Uses a hierarchical partitioning strategy to partition the matrix in parallel.
   Use this interface to make the partitioner consistent with others
*/
static PetscErrorCode MatPartitioningApply_Hierarchical(MatPartitioning part,IS *partitioning)
{
  MatPartitioning_Hierarchical *hpart  = (MatPartitioning_Hierarchical*)part->data;
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
  /* local size of mat */
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
  ierr = MatPartitioningCreate(comm,&coarsePart);CHKERRQ(ierr);
    /* if did not set partitioning type yet, use parmetis by default */
  if (!hpart->coarseparttype){
#if defined(PETSC_HAVE_PARMETIS)
    ierr = MatPartitioningSetType(coarsePart,MATPARTITIONINGPARMETIS);CHKERRQ(ierr);
#else
    SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Requires PETSc be installed with ParMetis or run with -mat_partitioning_hierarchical_coarseparttype partitiontype"); 
#endif
  } else {
    ierr = MatPartitioningSetType(coarsePart,hpart->coarseparttype);CHKERRQ(ierr);
  }
  ierr = MatPartitioningSetAdjacency(coarsePart,adj);CHKERRQ(ierr);
  ierr = MatPartitioningSetNParts(coarsePart, hpart->Ncoarseparts);CHKERRQ(ierr);
  /* copy over vertex weights */
  if(part->vertex_weights){
   ierr = PetscMalloc1(mat_localsize,&coarse_vertex_weights);CHKERRQ(ierr);
   ierr = PetscMemcpy(coarse_vertex_weights,part->vertex_weights,sizeof(PetscInt)*mat_localsize);CHKERRQ(ierr);
   ierr = MatPartitioningSetVertexWeights(coarsePart,coarse_vertex_weights);CHKERRQ(ierr);
  }
  /* It looks nontrivial to support part weights,
   * I will return back to implement it when have
   * an idea.
   *  */
  ierr = MatPartitioningApply(coarsePart,&hpart->coarseparts);CHKERRQ(ierr);
  ierr = MatPartitioningDestroy(&coarsePart);CHKERRQ(ierr);
  /* In the current implementation, destination should be the same as hpart->coarseparts,
   * and this interface is preserved to deal with the case hpart->coarseparts>size in the
   * future.
   * */
  ierr = MatPartitioningHierarchical_DetermineDestination(part,hpart->coarseparts,0,hpart->Ncoarseparts,&destination);CHKERRQ(ierr);
  /* assemble a submatrix for partitioning subdomains  */
  ierr = MatPartitioningHierarchical_AssembleSubdomain(adj,destination,&sadj,&mapping);CHKERRQ(ierr);
  ierr = ISDestroy(&destination);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)sadj,&scomm);CHKERRQ(ierr);
  /* create a fine partitioner */
  ierr = MatPartitioningCreate(scomm,&finePart);CHKERRQ(ierr);
  /* if do not set partitioning type, use parmetis by default */
  if(!hpart->fineparttype){
#if defined(PETSC_HAVE_PARMETIS)
    ierr = MatPartitioningSetType(finePart,MATPARTITIONINGPARMETIS);CHKERRQ(ierr);
#else
    SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Requires PETSc be installed with ParMetis or run with -mat_partitioning_hierarchical_coarseparttype partitiontype"); 
#endif
  } else {
    ierr = MatPartitioningSetType(finePart,hpart->fineparttype);CHKERRQ(ierr);
  }
  ierr = MatPartitioningSetAdjacency(finePart,sadj);CHKERRQ(ierr);
  ierr = MatPartitioningSetNParts(finePart, hpart->Nfineparts);CHKERRQ(ierr);
  ierr = MatPartitioningApply(finePart,&fineparts_temp);CHKERRQ(ierr);
  ierr = MatDestroy(&sadj);CHKERRQ(ierr);
  ierr = MatPartitioningDestroy(&finePart);CHKERRQ(ierr);
  ierr = MatPartitioningHierarchical_ReassembleFineparts(adj,fineparts_temp,mapping,&hpart->fineparts);CHKERRQ(ierr);
  ierr = ISDestroy(&fineparts_temp);CHKERRQ(ierr);
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
  ierr = MatDestroy(&adj);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


PetscErrorCode MatPartitioningHierarchical_ReassembleFineparts(Mat adj, IS fineparts, ISLocalToGlobalMapping mapping, IS *sfineparts)
{
  PetscInt            *local_indices, *global_indices,*owners,*sfineparts_indices,localsize,i;
  const PetscInt      *ranges,*fineparts_indices;
  PetscMPIInt         rank;
  MPI_Comm            comm;
  PetscLayout         rmap;
  PetscSFNode        *remote;
  PetscSF             sf;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)adj,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = MatGetLayouts(adj,&rmap,NULL);CHKERRQ(ierr);
  ierr = ISGetLocalSize(fineparts,&localsize);CHKERRQ(ierr);
  ierr = PetscCalloc2(localsize,&global_indices,localsize,&local_indices);CHKERRQ(ierr);
  for(i=0; i<localsize; i++){
	local_indices[i] = i;
  }
  /* map local indices back to global so that we can permulate data globally */
  ierr = ISLocalToGlobalMappingApply(mapping,localsize,local_indices,global_indices);CHKERRQ(ierr);
  ierr = PetscCalloc1(localsize,&owners);CHKERRQ(ierr);
  /* find owners for global indices */
  for(i=0; i<localsize; i++){
	ierr = PetscLayoutFindOwner(rmap,global_indices[i],&owners[i]);CHKERRQ(ierr);
  }
  ierr = PetscLayoutGetRanges(rmap,&ranges);CHKERRQ(ierr);
  ierr = PetscCalloc1(ranges[rank+1]-ranges[rank],&sfineparts_indices);CHKERRQ(ierr);
  ierr = ISGetIndices(fineparts,&fineparts_indices);CHKERRQ(ierr);
  ierr = PetscSFCreate(comm,&sf);CHKERRQ(ierr);
  ierr = PetscCalloc1(localsize,&remote);CHKERRQ(ierr);
  for(i=0; i<localsize; i++){
	remote[i].rank  = owners[i];
	remote[i].index = global_indices[i]-ranges[owners[i]];
  }
  ierr = PetscSFSetType(sf,PETSCSFBASIC);CHKERRQ(ierr);
  /* not sure how to add prefix to sf */
  ierr = PetscSFSetFromOptions(sf);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(sf,localsize,localsize,NULL,PETSC_OWN_POINTER,remote,PETSC_OWN_POINTER);CHKERRQ(ierr);
  ierr = PetscSFReduceBegin(sf,MPIU_INT,fineparts_indices,sfineparts_indices,MPIU_REPLACE);CHKERRQ(ierr);
  ierr = PetscSFReduceEnd(sf,MPIU_INT,fineparts_indices,sfineparts_indices,MPIU_REPLACE);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);
  ierr = ISRestoreIndices(fineparts,&fineparts_indices);CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm,ranges[rank+1]-ranges[rank],sfineparts_indices,PETSC_OWN_POINTER,sfineparts);CHKERRQ(ierr);
  ierr = PetscFree2(global_indices,local_indices);CHKERRQ(ierr);
  ierr = PetscFree(owners);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


PetscErrorCode MatPartitioningHierarchical_AssembleSubdomain(Mat adj,IS destination,Mat *sadj, ISLocalToGlobalMapping *mapping)
{
  IS              irows,icols;
  PetscInt        irows_ln;
  PetscMPIInt     rank;
  const PetscInt *irows_indices;
  MPI_Comm        comm;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)adj,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  /* figure out where data comes from  */
  ierr = ISBuildTwoSided(destination,NULL,&irows);CHKERRQ(ierr);
  ierr = ISDuplicate(irows,&icols);CHKERRQ(ierr);
  ierr = ISGetLocalSize(irows,&irows_ln);CHKERRQ(ierr);
  ierr = ISGetIndices(irows,&irows_indices);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingCreate(comm,1,irows_ln,irows_indices,PETSC_COPY_VALUES,mapping);CHKERRQ(ierr);
  ierr = ISRestoreIndices(irows,&irows_indices);CHKERRQ(ierr);
  ierr = MatCreateSubMatrices(adj,1,&irows,&icols,MAT_INITIAL_MATRIX,&sadj);CHKERRQ(ierr);
  ierr = ISDestroy(&irows);CHKERRQ(ierr);
  ierr = ISDestroy(&icols);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


PetscErrorCode MatPartitioningHierarchical_DetermineDestination(MatPartitioning part, IS partitioning, PetscInt pstart, PetscInt pend, IS *destination)
{
  MPI_Comm            comm;
  PetscMPIInt         rank,size,target;
  PetscInt            plocalsize,*dest_indices,i;
  const PetscInt     *part_indices;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)part,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if((pend-pstart)>size) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"range [%D, %D] should be smaller than or equal to size %D",pstart,pend,size);CHKERRQ(ierr);
  if(pstart>pend) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP," pstart %D should be smaller than pend %D",pstart,pend);CHKERRQ(ierr);
  ierr = ISGetLocalSize(partitioning,&plocalsize);CHKERRQ(ierr);
  ierr = PetscCalloc1(plocalsize,&dest_indices);CHKERRQ(ierr);
  ierr = ISGetIndices(partitioning,&part_indices);CHKERRQ(ierr);
  for(i=0; i<plocalsize; i++){
	/* compute target */
    target = part_indices[i]-pstart;
    /* mark out of range entity as -1 */
    if(part_indices[i]<pstart || part_indices[i]>pend) target = -1;
	dest_indices[i] = target;
  }
  ierr = ISCreateGeneral(comm,plocalsize,dest_indices,PETSC_OWN_POINTER,destination);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


PetscErrorCode MatPartitioningView_Hierarchical(MatPartitioning part,PetscViewer viewer)
{
  MatPartitioning_Hierarchical *hpart = (MatPartitioning_Hierarchical*)part->data;
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
  }
  PetscFunctionReturn(0);
}


PetscErrorCode MatPartitioningHierarchicalGetFineparts(MatPartitioning part,IS *fineparts)
{
  MatPartitioning_Hierarchical *hpart = (MatPartitioning_Hierarchical*)part->data;
  PetscErrorCode                ierr;

  PetscFunctionBegin;
  *fineparts = hpart->fineparts;
  ierr = PetscObjectReference((PetscObject)hpart->fineparts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatPartitioningHierarchicalGetCoarseparts(MatPartitioning part,IS *coarseparts)
{
  MatPartitioning_Hierarchical *hpart = (MatPartitioning_Hierarchical*)part->data;
  PetscErrorCode                ierr;

  PetscFunctionBegin;
  *coarseparts = hpart->coarseparts;
  ierr = PetscObjectReference((PetscObject)hpart->coarseparts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatPartitioningHierarchicalSetNcoarseparts(MatPartitioning part, PetscInt Ncoarseparts)
{
  MatPartitioning_Hierarchical *hpart = (MatPartitioning_Hierarchical*)part->data;

  PetscFunctionBegin;
  hpart->Ncoarseparts = Ncoarseparts;
  PetscFunctionReturn(0);
}

PetscErrorCode MatPartitioningHierarchicalSetNfineparts(MatPartitioning part, PetscInt Nfineparts)
{
  MatPartitioning_Hierarchical *hpart = (MatPartitioning_Hierarchical*)part->data;

  PetscFunctionBegin;
  hpart->Nfineparts = Nfineparts;
  PetscFunctionReturn(0);
}

PetscErrorCode MatPartitioningSetFromOptions_Hierarchical(PetscOptionItems *PetscOptionsObject,MatPartitioning part)
{
  MatPartitioning_Hierarchical *hpart = (MatPartitioning_Hierarchical*)part->data;
  PetscErrorCode ierr;
  char           value[1024];
  PetscBool      flag = PETSC_FALSE;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"Set hierarchical partitioning options");CHKERRQ(ierr);
  ierr = PetscOptionsString("-mat_partitioning_hierarchical_coarseparttype","coarse part type",NULL,NULL,value,1024,&flag);CHKERRQ(ierr);
  if(flag){
   ierr = PetscCalloc1(1024,&hpart->coarseparttype);CHKERRQ(ierr);
   ierr = PetscStrcpy(hpart->coarseparttype,value);CHKERRQ(ierr);
  }
  ierr = PetscOptionsString("-mat_partitioning_hierarchical_fineparttype","fine part type",NULL,NULL,value,1024,&flag);CHKERRQ(ierr);
  if(flag){
    ierr = PetscCalloc1(1024,&hpart->fineparttype);CHKERRQ(ierr);
    ierr = PetscStrcpy(hpart->fineparttype,value);CHKERRQ(ierr);
  }
  ierr = PetscOptionsInt("-mat_partitioning_hierarchical_Ncoarseparts","number of coarse parts",NULL,0,&hpart->Ncoarseparts,&flag);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-mat_partitioning_hierarchical_Nfineparts","number of fine parts",NULL,1,&hpart->Nfineparts,&flag);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


PetscErrorCode MatPartitioningDestroy_Hierarchical(MatPartitioning part)
{
  MatPartitioning_Hierarchical *hpart = (MatPartitioning_Hierarchical*)part->data;
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

PETSC_EXTERN PetscErrorCode MatPartitioningCreate_Hierarchical(MatPartitioning part)
{
  PetscErrorCode                ierr;
  MatPartitioning_Hierarchical *hpart;

  PetscFunctionBegin;
  ierr       = PetscNewLog(part,&hpart);CHKERRQ(ierr);
  part->data = (void*)hpart;

  hpart->fineparttype       = 0; /* fine level partitioner */
  hpart->coarseparttype     = 0; /* coarse level partitioner */
  hpart->Nfineparts         = 1; /* we do not further partition coarse partition any more by default */
  hpart->Ncoarseparts       = 0; /* number of coarse parts (first level) */
  hpart->coarseparts        = 0;
  hpart->fineparts          = 0;

  part->ops->apply          = MatPartitioningApply_Hierarchical;
  part->ops->view           = MatPartitioningView_Hierarchical;
  part->ops->destroy        = MatPartitioningDestroy_Hierarchical;
  part->ops->setfromoptions = MatPartitioningSetFromOptions_Hierarchical;
  PetscFunctionReturn(0);
}
