
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

PetscErrorCode MatPartitioningHierarchical_DetermineDestination(MatPartitioning,IS,PetscInt,PetscInt,IS*);
PetscErrorCode MatPartitioningHierarchical_AssembleSubdomain(Mat,IS,IS,IS*,Mat*,ISLocalToGlobalMapping*);
PetscErrorCode MatPartitioningHierarchical_ReassembleFineparts(Mat,IS,ISLocalToGlobalMapping,IS*);

typedef struct {
  char*                fineparttype; /* partitioner on fine level */
  char*                coarseparttype; /* partitioner on coarse level */
  PetscInt             nfineparts; /* number of fine parts on each coarse subdomain */
  PetscInt             ncoarseparts; /* number of coarse parts */
  IS                   coarseparts; /* partitioning on coarse level */
  IS                   fineparts; /* partitioning on fine level */
  MatPartitioning      coarseMatPart; /* MatPartititioning on coarse level (first level) */
  MatPartitioning      fineMatPart; /* MatPartitioning on fine level (second level) */
  MatPartitioning      improver; /* Improve the quality of a partition */
} MatPartitioning_Hierarchical;

/*
   Uses a hierarchical partitioning strategy to partition the matrix in parallel.
   Use this interface to make the partitioner consistent with others
*/
static PetscErrorCode MatPartitioningApply_Hierarchical(MatPartitioning part,IS *partitioning)
{
  MatPartitioning_Hierarchical *hpart  = (MatPartitioning_Hierarchical*)part->data;
  const PetscInt               *fineparts_indices, *coarseparts_indices;
  PetscInt                     *fineparts_indices_tmp;
  PetscInt                     *parts_indices,i,j,mat_localsize, *offsets;
  Mat                           mat    = part->adj,adj,sadj;
  PetscReal                    *part_weights;
  PetscBool                     flg;
  PetscInt                      bs     = 1;
  PetscInt                     *coarse_vertex_weights = NULL;
  PetscMPIInt                   size,rank;
  MPI_Comm                      comm,scomm;
  IS                            destination,fineparts_temp, vweights, svweights;
  PetscInt                      nsvwegihts,*fp_vweights;
  const PetscInt                *svweights_indices;
  ISLocalToGlobalMapping        mapping;
  const char                    *prefix;
  PetscBool                     use_edge_weights;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)part,&comm));
  PetscCallMPI(MPI_Comm_size(comm,&size));
  PetscCallMPI(MPI_Comm_rank(comm,&rank));
  PetscCall(PetscObjectTypeCompare((PetscObject)mat,MATMPIADJ,&flg));
  if (flg) {
    adj = mat;
    PetscCall(PetscObjectReference((PetscObject)adj));
  }else {
    /* bs indicates if the converted matrix is "reduced" from the original and hence the
       resulting partition results need to be stretched to match the original matrix */
   PetscCall(MatConvert(mat,MATMPIADJ,MAT_INITIAL_MATRIX,&adj));
   if (adj->rmap->n > 0) bs = mat->rmap->n/adj->rmap->n;
  }
  /* local size of mat */
  mat_localsize = adj->rmap->n;
  /* check parameters */
  /* how many small subdomains we want from a given 'big' suddomain */
  PetscCheck(hpart->nfineparts,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG," must set number of small subdomains for each big subdomain ");
  PetscCheck(hpart->ncoarseparts || part->n,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE," did not either set number of coarse parts or total number of parts ");

  /* Partitioning the domain into one single subdomain is a trivial case, and we should just return  */
  if (part->n==1) {
    PetscCall(PetscCalloc1(bs*adj->rmap->n,&parts_indices));
    PetscCall(ISCreateGeneral(comm,bs*adj->rmap->n,parts_indices,PETSC_OWN_POINTER,partitioning));
    hpart->ncoarseparts = 1;
    hpart->nfineparts = 1;
    PetscCall(PetscStrallocpy("NONE",&hpart->coarseparttype));
    PetscCall(PetscStrallocpy("NONE",&hpart->fineparttype));
    PetscCall(MatDestroy(&adj));
    PetscFunctionReturn(0);
  }

  if (part->n) {
    hpart->ncoarseparts = part->n/hpart->nfineparts;

    if (part->n%hpart->nfineparts != 0) hpart->ncoarseparts++;
  }else{
    part->n = hpart->ncoarseparts*hpart->nfineparts;
  }

  PetscCall(PetscMalloc1(hpart->ncoarseparts+1, &offsets));
  PetscCall(PetscMalloc1(hpart->ncoarseparts, &part_weights));

  offsets[0] = 0;
  if (part->n%hpart->nfineparts != 0) offsets[1] = part->n%hpart->nfineparts;
  else offsets[1] = hpart->nfineparts;

  part_weights[0] = ((PetscReal)offsets[1])/part->n;

  for (i=2; i<=hpart->ncoarseparts; i++) {
    offsets[i] = hpart->nfineparts;
    part_weights[i-1] = ((PetscReal)offsets[i])/part->n;
  }

  offsets[0] = 0;
  for (i=1;i<=hpart->ncoarseparts; i++)
    offsets[i] += offsets[i-1];

  /* If these exists a mat partitioner, we should delete it */
  PetscCall(MatPartitioningDestroy(&hpart->coarseMatPart));
  PetscCall(MatPartitioningCreate(comm,&hpart->coarseMatPart));
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)part,&prefix));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)hpart->coarseMatPart,prefix));
  PetscCall(PetscObjectAppendOptionsPrefix((PetscObject)hpart->coarseMatPart,"hierarch_coarse_"));
    /* if did not set partitioning type yet, use parmetis by default */
  if (!hpart->coarseparttype) {
#if defined(PETSC_HAVE_PARMETIS)
    PetscCall(MatPartitioningSetType(hpart->coarseMatPart,MATPARTITIONINGPARMETIS));
    PetscCall(PetscStrallocpy(MATPARTITIONINGPARMETIS,&hpart->coarseparttype));
#elif defined(PETSC_HAVE_PTSCOTCH)
    PetscCall(MatPartitioningSetType(hpart->coarseMatPart,MATPARTITIONINGPTSCOTCH));
    PetscCall(PetscStrallocpy(MATPARTITIONINGPTSCOTCH,&hpart->coarseparttype));
#else
    SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Requires PETSc be installed with ParMetis or run with -mat_partitioning_hierarchical_coarseparttype partitiontype");
#endif
  } else {
    PetscCall(MatPartitioningSetType(hpart->coarseMatPart,hpart->coarseparttype));
  }
  PetscCall(MatPartitioningSetAdjacency(hpart->coarseMatPart,adj));
  PetscCall(MatPartitioningSetNParts(hpart->coarseMatPart, hpart->ncoarseparts));
  /* copy over vertex weights */
  if (part->vertex_weights) {
    PetscCall(PetscMalloc1(mat_localsize,&coarse_vertex_weights));
    PetscCall(PetscArraycpy(coarse_vertex_weights,part->vertex_weights,mat_localsize));
    PetscCall(MatPartitioningSetVertexWeights(hpart->coarseMatPart,coarse_vertex_weights));
  }
  /* Copy use_edge_weights flag from part to coarse part */
  PetscCall(MatPartitioningGetUseEdgeWeights(part,&use_edge_weights));
  PetscCall(MatPartitioningSetUseEdgeWeights(hpart->coarseMatPart,use_edge_weights));

  PetscCall(MatPartitioningSetPartitionWeights(hpart->coarseMatPart, part_weights));
  PetscCall(MatPartitioningApply(hpart->coarseMatPart,&hpart->coarseparts));

  /* Wrap the original vertex weights into an index set so that we can extract the corresponding
   * vertex weights for each big subdomain using ISCreateSubIS().
   * */
  if (part->vertex_weights) {
    PetscCall(ISCreateGeneral(comm,mat_localsize,part->vertex_weights,PETSC_COPY_VALUES,&vweights));
  }

  PetscCall(PetscCalloc1(mat_localsize, &fineparts_indices_tmp));
  for (i=0; i<hpart->ncoarseparts; i+=size) {
    /* Determine where we want to send big subdomains */
    PetscCall(MatPartitioningHierarchical_DetermineDestination(part,hpart->coarseparts,i,i+size,&destination));
    /* Assemble a submatrix and its vertex weights for partitioning subdomains  */
    PetscCall(MatPartitioningHierarchical_AssembleSubdomain(adj,part->vertex_weights? vweights:NULL,destination,part->vertex_weights? &svweights:NULL,&sadj,&mapping));
    /* We have to create a new array to hold vertex weights since coarse partitioner needs to own the vertex-weights array */
    if (part->vertex_weights) {
      PetscCall(ISGetLocalSize(svweights,&nsvwegihts));
      PetscCall(PetscMalloc1(nsvwegihts,&fp_vweights));
      PetscCall(ISGetIndices(svweights,&svweights_indices));
      PetscCall(PetscArraycpy(fp_vweights,svweights_indices,nsvwegihts));
      PetscCall(ISRestoreIndices(svweights,&svweights_indices));
      PetscCall(ISDestroy(&svweights));
    }

    PetscCall(ISDestroy(&destination));
    PetscCall(PetscObjectGetComm((PetscObject)sadj,&scomm));

    /*
     * If the number of big subdomains is smaller than the number of processor cores, the higher ranks do not
     * need to do partitioning
     * */
    if ((i+rank)<hpart->ncoarseparts) {
      PetscCall(MatPartitioningDestroy(&hpart->fineMatPart));
      /* create a fine partitioner */
      PetscCall(MatPartitioningCreate(scomm,&hpart->fineMatPart));
      PetscCall(PetscObjectSetOptionsPrefix((PetscObject)hpart->fineMatPart,prefix));
      PetscCall(PetscObjectAppendOptionsPrefix((PetscObject)hpart->fineMatPart,"hierarch_fine_"));
      /* if do not set partitioning type, use parmetis by default */
      if (!hpart->fineparttype) {
#if defined(PETSC_HAVE_PARMETIS)
        PetscCall(MatPartitioningSetType(hpart->fineMatPart,MATPARTITIONINGPARMETIS));
        PetscCall(PetscStrallocpy(MATPARTITIONINGPARMETIS,&hpart->fineparttype));
#elif defined(PETSC_HAVE_PTSCOTCH)
        PetscCall(MatPartitioningSetType(hpart->fineMatPart,MATPARTITIONINGPTSCOTCH));
        PetscCall(PetscStrallocpy(MATPARTITIONINGPTSCOTCH,&hpart->fineparttype));
#elif defined(PETSC_HAVE_CHACO)
        PetscCall(MatPartitioningSetType(hpart->fineMatPart,MATPARTITIONINGCHACO));
        PetscCall(PetscStrallocpy(MATPARTITIONINGCHACO,&hpart->fineparttype));
#elif defined(PETSC_HAVE_PARTY)
        PetscCall(MatPartitioningSetType(hpart->fineMatPart,MATPARTITIONINGPARTY));
        PetscCall(PetscStrallocpy(PETSC_HAVE_PARTY,&hpart->fineparttype));
#else
        SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Requires PETSc be installed with ParMetis or run with -mat_partitioning_hierarchical_coarseparttype partitiontype");
#endif
      } else {
        PetscCall(MatPartitioningSetType(hpart->fineMatPart,hpart->fineparttype));
      }
      PetscCall(MatPartitioningSetUseEdgeWeights(hpart->fineMatPart,use_edge_weights));
      PetscCall(MatPartitioningSetAdjacency(hpart->fineMatPart,sadj));
      PetscCall(MatPartitioningSetNParts(hpart->fineMatPart, offsets[rank+1+i]-offsets[rank+i]));
      if (part->vertex_weights) {
        PetscCall(MatPartitioningSetVertexWeights(hpart->fineMatPart,fp_vweights));
      }
      PetscCall(MatPartitioningApply(hpart->fineMatPart,&fineparts_temp));
    } else {
      PetscCall(ISCreateGeneral(scomm,0,NULL,PETSC_OWN_POINTER,&fineparts_temp));
    }

    PetscCall(MatDestroy(&sadj));

    /* Send partition back to the original owners */
    PetscCall(MatPartitioningHierarchical_ReassembleFineparts(adj,fineparts_temp,mapping,&hpart->fineparts));
    PetscCall(ISGetIndices(hpart->fineparts,&fineparts_indices));
    for (j=0;j<mat_localsize;j++)
      if (fineparts_indices[j] >=0) fineparts_indices_tmp[j] = fineparts_indices[j];

    PetscCall(ISRestoreIndices(hpart->fineparts,&fineparts_indices));
    PetscCall(ISDestroy(&hpart->fineparts));
    PetscCall(ISDestroy(&fineparts_temp));
    PetscCall(ISLocalToGlobalMappingDestroy(&mapping));
  }

  if (part->vertex_weights) {
    PetscCall(ISDestroy(&vweights));
  }

  PetscCall(ISCreateGeneral(comm,mat_localsize,fineparts_indices_tmp,PETSC_OWN_POINTER,&hpart->fineparts));
  PetscCall(ISGetIndices(hpart->fineparts,&fineparts_indices));
  PetscCall(ISGetIndices(hpart->coarseparts,&coarseparts_indices));
  PetscCall(PetscMalloc1(bs*adj->rmap->n,&parts_indices));
  /* Modify the local indices to the global indices by combing the coarse partition and the fine partitions */
  for (i=0; i<adj->rmap->n; i++) {
    for (j=0; j<bs; j++) {
      parts_indices[bs*i+j] = fineparts_indices[i]+offsets[coarseparts_indices[i]];
    }
  }
  PetscCall(ISRestoreIndices(hpart->fineparts,&fineparts_indices));
  PetscCall(ISRestoreIndices(hpart->coarseparts,&coarseparts_indices));
  PetscCall(PetscFree(offsets));
  PetscCall(ISCreateGeneral(comm,bs*adj->rmap->n,parts_indices,PETSC_OWN_POINTER,partitioning));
  PetscCall(MatDestroy(&adj));
  PetscFunctionReturn(0);
}

PetscErrorCode MatPartitioningHierarchical_ReassembleFineparts(Mat adj, IS fineparts, ISLocalToGlobalMapping mapping, IS *sfineparts)
{
  PetscInt            *local_indices, *global_indices,*sfineparts_indices,localsize,i;
  const PetscInt      *ranges,*fineparts_indices;
  PetscMPIInt         rank,*owners;
  MPI_Comm            comm;
  PetscLayout         rmap;
  PetscSFNode        *remote;
  PetscSF             sf;

  PetscFunctionBegin;
  PetscValidPointer(sfineparts, 4);
  PetscCall(PetscObjectGetComm((PetscObject)adj,&comm));
  PetscCallMPI(MPI_Comm_rank(comm,&rank));
  PetscCall(MatGetLayouts(adj,&rmap,NULL));
  PetscCall(ISGetLocalSize(fineparts,&localsize));
  PetscCall(PetscMalloc2(localsize,&global_indices,localsize,&local_indices));
  for (i=0; i<localsize; i++) {
    local_indices[i] = i;
  }
  /* map local indices back to global so that we can permulate data globally */
  PetscCall(ISLocalToGlobalMappingApply(mapping,localsize,local_indices,global_indices));
  PetscCall(PetscCalloc1(localsize,&owners));
  /* find owners for global indices */
  for (i=0; i<localsize; i++) {
    PetscCall(PetscLayoutFindOwner(rmap,global_indices[i],&owners[i]));
  }
  PetscCall(PetscLayoutGetRanges(rmap,&ranges));
  PetscCall(PetscMalloc1(ranges[rank+1]-ranges[rank],&sfineparts_indices));

  for (i=0; i<ranges[rank+1]-ranges[rank]; i++) {
    sfineparts_indices[i] = -1;
  }

  PetscCall(ISGetIndices(fineparts,&fineparts_indices));
  PetscCall(PetscSFCreate(comm,&sf));
  PetscCall(PetscMalloc1(localsize,&remote));
  for (i=0; i<localsize; i++) {
    remote[i].rank  = owners[i];
    remote[i].index = global_indices[i]-ranges[owners[i]];
  }
  PetscCall(PetscSFSetType(sf,PETSCSFBASIC));
  /* not sure how to add prefix to sf */
  PetscCall(PetscSFSetFromOptions(sf));
  PetscCall(PetscSFSetGraph(sf,localsize,localsize,NULL,PETSC_OWN_POINTER,remote,PETSC_OWN_POINTER));
  PetscCall(PetscSFReduceBegin(sf,MPIU_INT,fineparts_indices,sfineparts_indices,MPI_REPLACE));
  PetscCall(PetscSFReduceEnd(sf,MPIU_INT,fineparts_indices,sfineparts_indices,MPI_REPLACE));
  PetscCall(PetscSFDestroy(&sf));
  PetscCall(ISRestoreIndices(fineparts,&fineparts_indices));
  PetscCall(ISCreateGeneral(comm,ranges[rank+1]-ranges[rank],sfineparts_indices,PETSC_OWN_POINTER,sfineparts));
  PetscCall(PetscFree2(global_indices,local_indices));
  PetscCall(PetscFree(owners));
  PetscFunctionReturn(0);
}

PetscErrorCode MatPartitioningHierarchical_AssembleSubdomain(Mat adj,IS vweights, IS destination,IS *svweights,Mat *sadj,ISLocalToGlobalMapping *mapping)
{
  IS              irows,icols;
  PetscInt        irows_ln;
  PetscMPIInt     rank;
  const PetscInt *irows_indices;
  MPI_Comm        comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)adj,&comm));
  PetscCallMPI(MPI_Comm_rank(comm,&rank));
  /* figure out where data comes from  */
  PetscCall(ISBuildTwoSided(destination,NULL,&irows));
  PetscCall(ISDuplicate(irows,&icols));
  PetscCall(ISGetLocalSize(irows,&irows_ln));
  PetscCall(ISGetIndices(irows,&irows_indices));
  PetscCall(ISLocalToGlobalMappingCreate(comm,1,irows_ln,irows_indices,PETSC_COPY_VALUES,mapping));
  PetscCall(ISRestoreIndices(irows,&irows_indices));
  PetscCall(MatCreateSubMatrices(adj,1,&irows,&icols,MAT_INITIAL_MATRIX,&sadj));
  if (vweights && svweights) {
    PetscCall(ISCreateSubIS(vweights,irows,svweights));
  }
  PetscCall(ISDestroy(&irows));
  PetscCall(ISDestroy(&icols));
  PetscFunctionReturn(0);
}

PetscErrorCode MatPartitioningHierarchical_DetermineDestination(MatPartitioning part, IS partitioning, PetscInt pstart, PetscInt pend, IS *destination)
{
  MPI_Comm            comm;
  PetscMPIInt         rank,size,target;
  PetscInt            plocalsize,*dest_indices,i;
  const PetscInt     *part_indices;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)part,&comm));
  PetscCallMPI(MPI_Comm_rank(comm,&rank));
  PetscCallMPI(MPI_Comm_size(comm,&size));
  PetscCheck((pend-pstart)<=size,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"range [%" PetscInt_FMT ", %" PetscInt_FMT "] should be smaller than or equal to size %d",pstart,pend,size);
  PetscCheck(pstart<=pend,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP," pstart %" PetscInt_FMT " should be smaller than pend %" PetscInt_FMT,pstart,pend);
  PetscCall(ISGetLocalSize(partitioning,&plocalsize));
  PetscCall(PetscMalloc1(plocalsize,&dest_indices));
  PetscCall(ISGetIndices(partitioning,&part_indices));
  for (i=0; i<plocalsize; i++) {
    /* compute target */
    target = part_indices[i]-pstart;
    /* mark out of range entity as -1 */
    if (part_indices[i]<pstart || part_indices[i]>=pend) target = -1;
    dest_indices[i] = target;
  }
  PetscCall(ISCreateGeneral(comm,plocalsize,dest_indices,PETSC_OWN_POINTER,destination));
  PetscFunctionReturn(0);
}

PetscErrorCode MatPartitioningView_Hierarchical(MatPartitioning part,PetscViewer viewer)
{
  MatPartitioning_Hierarchical *hpart = (MatPartitioning_Hierarchical*)part->data;
  PetscMPIInt              rank;
  PetscBool                iascii;
  PetscViewer              sviewer;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)part),&rank));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer," Number of coarse parts: %" PetscInt_FMT "\n",hpart->ncoarseparts));
    PetscCall(PetscViewerASCIIPrintf(viewer," Coarse partitioner: %s\n",hpart->coarseparttype));
    if (hpart->coarseMatPart) {
      PetscCall(PetscViewerASCIIPushTab(viewer));
      PetscCall(MatPartitioningView(hpart->coarseMatPart,viewer));
      PetscCall(PetscViewerASCIIPopTab(viewer));
    }
    PetscCall(PetscViewerASCIIPrintf(viewer," Number of fine parts: %" PetscInt_FMT "\n",hpart->nfineparts));
    PetscCall(PetscViewerASCIIPrintf(viewer," Fine partitioner: %s\n",hpart->fineparttype));
    PetscCall(PetscViewerGetSubViewer(viewer,PETSC_COMM_SELF,&sviewer));
    if (rank == 0 && hpart->fineMatPart) {
      PetscCall(PetscViewerASCIIPushTab(viewer));
      PetscCall(MatPartitioningView(hpart->fineMatPart,sviewer));
      PetscCall(PetscViewerASCIIPopTab(viewer));
    }
    PetscCall(PetscViewerRestoreSubViewer(viewer,PETSC_COMM_SELF,&sviewer));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatPartitioningHierarchicalGetFineparts(MatPartitioning part,IS *fineparts)
{
  MatPartitioning_Hierarchical *hpart = (MatPartitioning_Hierarchical*)part->data;

  PetscFunctionBegin;
  *fineparts = hpart->fineparts;
  PetscCall(PetscObjectReference((PetscObject)hpart->fineparts));
  PetscFunctionReturn(0);
}

PetscErrorCode MatPartitioningHierarchicalGetCoarseparts(MatPartitioning part,IS *coarseparts)
{
  MatPartitioning_Hierarchical *hpart = (MatPartitioning_Hierarchical*)part->data;

  PetscFunctionBegin;
  *coarseparts = hpart->coarseparts;
  PetscCall(PetscObjectReference((PetscObject)hpart->coarseparts));
  PetscFunctionReturn(0);
}

PetscErrorCode MatPartitioningHierarchicalSetNcoarseparts(MatPartitioning part, PetscInt ncoarseparts)
{
  MatPartitioning_Hierarchical *hpart = (MatPartitioning_Hierarchical*)part->data;

  PetscFunctionBegin;
  hpart->ncoarseparts = ncoarseparts;
  PetscFunctionReturn(0);
}

PetscErrorCode MatPartitioningHierarchicalSetNfineparts(MatPartitioning part, PetscInt nfineparts)
{
  MatPartitioning_Hierarchical *hpart = (MatPartitioning_Hierarchical*)part->data;

  PetscFunctionBegin;
  hpart->nfineparts = nfineparts;
  PetscFunctionReturn(0);
}

PetscErrorCode MatPartitioningSetFromOptions_Hierarchical(PetscOptionItems *PetscOptionsObject,MatPartitioning part)
{
  MatPartitioning_Hierarchical *hpart = (MatPartitioning_Hierarchical*)part->data;
  char           value[1024];
  PetscBool      flag = PETSC_FALSE;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject,"Set hierarchical partitioning options");
  PetscCall(PetscOptionsString("-mat_partitioning_hierarchical_coarseparttype","coarse part type",NULL,NULL,value,sizeof(value),&flag));
  if (flag) {
   PetscCall(PetscStrallocpy(value,&hpart->coarseparttype));
  }
  PetscCall(PetscOptionsString("-mat_partitioning_hierarchical_fineparttype","fine part type",NULL,NULL,value,sizeof(value),&flag));
  if (flag) {
    PetscCall(PetscStrallocpy(value,&hpart->fineparttype));
  }
  PetscCall(PetscOptionsInt("-mat_partitioning_hierarchical_ncoarseparts","number of coarse parts",NULL,hpart->ncoarseparts,&hpart->ncoarseparts,&flag));
  PetscCall(PetscOptionsInt("-mat_partitioning_hierarchical_nfineparts","number of fine parts",NULL,hpart->nfineparts,&hpart->nfineparts,&flag));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(0);
}

PetscErrorCode MatPartitioningDestroy_Hierarchical(MatPartitioning part)
{
  MatPartitioning_Hierarchical *hpart = (MatPartitioning_Hierarchical*)part->data;

  PetscFunctionBegin;
  if (hpart->coarseparttype) PetscCall(PetscFree(hpart->coarseparttype));
  if (hpart->fineparttype) PetscCall(PetscFree(hpart->fineparttype));
  PetscCall(ISDestroy(&hpart->fineparts));
  PetscCall(ISDestroy(&hpart->coarseparts));
  PetscCall(MatPartitioningDestroy(&hpart->coarseMatPart));
  PetscCall(MatPartitioningDestroy(&hpart->fineMatPart));
  PetscCall(MatPartitioningDestroy(&hpart->improver));
  PetscCall(PetscFree(hpart));
  PetscFunctionReturn(0);
}

/*
   Improves the quality  of a partition
*/
static PetscErrorCode MatPartitioningImprove_Hierarchical(MatPartitioning part, IS *partitioning)
{
  MatPartitioning_Hierarchical *hpart = (MatPartitioning_Hierarchical*)part->data;
  Mat                           mat = part->adj, adj;
  PetscBool                    flg;
  const char                   *prefix;
#if defined(PETSC_HAVE_PARMETIS)
  PetscInt                     *vertex_weights;
#endif

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)mat,MATMPIADJ,&flg));
  if (flg) {
    adj = mat;
    PetscCall(PetscObjectReference((PetscObject)adj));
  }else {
    /* bs indicates if the converted matrix is "reduced" from the original and hence the
       resulting partition results need to be stretched to match the original matrix */
   PetscCall(MatConvert(mat,MATMPIADJ,MAT_INITIAL_MATRIX,&adj));
  }

  /* If there exists a mat partitioner, we should delete it */
  PetscCall(MatPartitioningDestroy(&hpart->improver));
  PetscCall(MatPartitioningCreate(PetscObjectComm((PetscObject)part),&hpart->improver));
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)part,&prefix));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)hpart->improver,prefix));
  PetscCall(PetscObjectAppendOptionsPrefix((PetscObject)hpart->improver,"hierarch_improver_"));
  /* Only parmetis supports to refine a partition */
#if defined(PETSC_HAVE_PARMETIS)
  PetscCall(MatPartitioningSetType(hpart->improver,MATPARTITIONINGPARMETIS));
  PetscCall(MatPartitioningSetAdjacency(hpart->improver,adj));
  PetscCall(MatPartitioningSetNParts(hpart->improver, part->n));
  /* copy over vertex weights */
  if (part->vertex_weights) {
    PetscCall(PetscMalloc1(adj->rmap->n,&vertex_weights));
    PetscCall(PetscArraycpy(vertex_weights,part->vertex_weights,adj->rmap->n));
    PetscCall(MatPartitioningSetVertexWeights(hpart->improver,vertex_weights));
  }
  PetscCall(MatPartitioningImprove(hpart->improver,partitioning));
  PetscCall(MatDestroy(&adj));
  PetscFunctionReturn(0);
#else
  SETERRQ(PetscObjectComm((PetscObject)adj),PETSC_ERR_SUP,"Requires PETSc be installed with ParMetis");
#endif
}

/*MC
   MATPARTITIONINGHIERARCH - Creates a partitioning context via hierarchical partitioning strategy.
   The graph is partitioned into a number of subgraphs, and each subgraph is further split into a few smaller
   subgraphs. The idea can be applied in a recursive manner. It is useful when you want to partition the graph
   into a large number of subgraphs (often more than 10K) since partitions obtained with existing partitioners
   such as ParMETIS and PTScotch are far from ideal. The hierarchical partitioning also tries to avoid off-node
   communication as much as possible for multi-core processor. Another user case for the hierarchical partitioning
   is to improve PCGASM convergence by generating multi-rank connected subdomain.

   Collective

   Input Parameter:
.  part - the partitioning context

   Options Database Keys:
+     -mat_partitioning_hierarchical_coarseparttype - partitioner type at the first level and parmetis is used by default
.     -mat_partitioning_hierarchical_fineparttype - partitioner type at the second level and parmetis is used by default
.     -mat_partitioning_hierarchical_ncoarseparts - number of subgraphs is required at the first level, which is often the number of compute nodes
-     -mat_partitioning_hierarchical_nfineparts - number of smaller subgraphs for each subgraph, which is often the number of cores per compute node

   Level: beginner

   References:
+  * - Fande Kong, Xiao-Chuan Cai, A highly scalable multilevel Schwarz method with boundary geometry preserving coarse spaces for 3D elasticity
      problems on domains with complex geometry,   SIAM Journal on Scientific Computing 38 (2), C73-C95, 2016
-  * - Fande Kong, Roy H. Stogner, Derek Gaston, John W. Peterson, Cody J. Permann, Andrew E. Slaughter, and Richard C. Martineau,
      A general-purpose hierarchical mesh partitioning method with node balancing strategies for large-scale numerical simulations,
      arXiv preprint arXiv:1809.02666CoRR, 2018.

.seealso: MatPartitioningSetType(), MatPartitioningType

M*/

PETSC_EXTERN PetscErrorCode MatPartitioningCreate_Hierarchical(MatPartitioning part)
{
  MatPartitioning_Hierarchical *hpart;

  PetscFunctionBegin;
  PetscCall(PetscNewLog(part,&hpart));
  part->data = (void*)hpart;

  hpart->fineparttype       = NULL; /* fine level (second) partitioner */
  hpart->coarseparttype     = NULL; /* coarse level (first) partitioner */
  hpart->nfineparts         = 1;    /* we do not further partition coarse partition any more by default */
  hpart->ncoarseparts       = 0;    /* number of coarse parts (first level) */
  hpart->coarseparts        = NULL;
  hpart->fineparts          = NULL;
  hpart->coarseMatPart      = NULL;
  hpart->fineMatPart        = NULL;

  part->ops->apply          = MatPartitioningApply_Hierarchical;
  part->ops->view           = MatPartitioningView_Hierarchical;
  part->ops->destroy        = MatPartitioningDestroy_Hierarchical;
  part->ops->setfromoptions = MatPartitioningSetFromOptions_Hierarchical;
  part->ops->improve        = MatPartitioningImprove_Hierarchical;
  PetscFunctionReturn(0);
}
