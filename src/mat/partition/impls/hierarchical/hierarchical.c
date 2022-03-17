
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
  PetscErrorCode                ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)part,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);
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
  PetscCheckFalse(!hpart->nfineparts,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG," must set number of small subdomains for each big subdomain ");
  PetscCheckFalse(!hpart->ncoarseparts && !part->n,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE," did not either set number of coarse parts or total number of parts ");

  /* Partitioning the domain into one single subdomain is a trivial case, and we should just return  */
  if (part->n==1) {
    ierr = PetscCalloc1(bs*adj->rmap->n,&parts_indices);CHKERRQ(ierr);
    ierr = ISCreateGeneral(comm,bs*adj->rmap->n,parts_indices,PETSC_OWN_POINTER,partitioning);CHKERRQ(ierr);
    hpart->ncoarseparts = 1;
    hpart->nfineparts = 1;
    ierr = PetscStrallocpy("NONE",&hpart->coarseparttype);CHKERRQ(ierr);
    ierr = PetscStrallocpy("NONE",&hpart->fineparttype);CHKERRQ(ierr);
    ierr = MatDestroy(&adj);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  if (part->n) {
    hpart->ncoarseparts = part->n/hpart->nfineparts;

    if (part->n%hpart->nfineparts != 0) hpart->ncoarseparts++;
  }else{
    part->n = hpart->ncoarseparts*hpart->nfineparts;
  }

  ierr = PetscMalloc1(hpart->ncoarseparts+1, &offsets);CHKERRQ(ierr);
  ierr = PetscMalloc1(hpart->ncoarseparts, &part_weights);CHKERRQ(ierr);

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
  ierr = MatPartitioningDestroy(&hpart->coarseMatPart);CHKERRQ(ierr);
  ierr = MatPartitioningCreate(comm,&hpart->coarseMatPart);CHKERRQ(ierr);
  ierr = PetscObjectGetOptionsPrefix((PetscObject)part,&prefix);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)hpart->coarseMatPart,prefix);CHKERRQ(ierr);
  ierr = PetscObjectAppendOptionsPrefix((PetscObject)hpart->coarseMatPart,"hierarch_coarse_");CHKERRQ(ierr);
    /* if did not set partitioning type yet, use parmetis by default */
  if (!hpart->coarseparttype) {
#if defined(PETSC_HAVE_PARMETIS)
    ierr = MatPartitioningSetType(hpart->coarseMatPart,MATPARTITIONINGPARMETIS);CHKERRQ(ierr);
    ierr = PetscStrallocpy(MATPARTITIONINGPARMETIS,&hpart->coarseparttype);CHKERRQ(ierr);
#elif defined(PETSC_HAVE_PTSCOTCH)
    ierr = MatPartitioningSetType(hpart->coarseMatPart,MATPARTITIONINGPTSCOTCH);CHKERRQ(ierr);
    ierr = PetscStrallocpy(MATPARTITIONINGPTSCOTCH,&hpart->coarseparttype);CHKERRQ(ierr);
#else
    SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Requires PETSc be installed with ParMetis or run with -mat_partitioning_hierarchical_coarseparttype partitiontype");
#endif
  } else {
    ierr = MatPartitioningSetType(hpart->coarseMatPart,hpart->coarseparttype);CHKERRQ(ierr);
  }
  ierr = MatPartitioningSetAdjacency(hpart->coarseMatPart,adj);CHKERRQ(ierr);
  ierr = MatPartitioningSetNParts(hpart->coarseMatPart, hpart->ncoarseparts);CHKERRQ(ierr);
  /* copy over vertex weights */
  if (part->vertex_weights) {
    ierr = PetscMalloc1(mat_localsize,&coarse_vertex_weights);CHKERRQ(ierr);
    ierr = PetscArraycpy(coarse_vertex_weights,part->vertex_weights,mat_localsize);CHKERRQ(ierr);
    ierr = MatPartitioningSetVertexWeights(hpart->coarseMatPart,coarse_vertex_weights);CHKERRQ(ierr);
  }
  /* Copy use_edge_weights flag from part to coarse part */
  ierr = MatPartitioningGetUseEdgeWeights(part,&use_edge_weights);CHKERRQ(ierr);
  ierr = MatPartitioningSetUseEdgeWeights(hpart->coarseMatPart,use_edge_weights);CHKERRQ(ierr);

  ierr = MatPartitioningSetPartitionWeights(hpart->coarseMatPart, part_weights);CHKERRQ(ierr);
  ierr = MatPartitioningApply(hpart->coarseMatPart,&hpart->coarseparts);CHKERRQ(ierr);

  /* Wrap the original vertex weights into an index set so that we can extract the corresponding
   * vertex weights for each big subdomain using ISCreateSubIS().
   * */
  if (part->vertex_weights) {
    ierr = ISCreateGeneral(comm,mat_localsize,part->vertex_weights,PETSC_COPY_VALUES,&vweights);CHKERRQ(ierr);
  }

  ierr = PetscCalloc1(mat_localsize, &fineparts_indices_tmp);CHKERRQ(ierr);
  for (i=0; i<hpart->ncoarseparts; i+=size) {
    /* Determine where we want to send big subdomains */
    ierr = MatPartitioningHierarchical_DetermineDestination(part,hpart->coarseparts,i,i+size,&destination);CHKERRQ(ierr);
    /* Assemble a submatrix and its vertex weights for partitioning subdomains  */
    ierr = MatPartitioningHierarchical_AssembleSubdomain(adj,part->vertex_weights? vweights:NULL,destination,part->vertex_weights? &svweights:NULL,&sadj,&mapping);CHKERRQ(ierr);
    /* We have to create a new array to hold vertex weights since coarse partitioner needs to own the vertex-weights array */
    if (part->vertex_weights) {
      ierr = ISGetLocalSize(svweights,&nsvwegihts);CHKERRQ(ierr);
      ierr = PetscMalloc1(nsvwegihts,&fp_vweights);CHKERRQ(ierr);
      ierr = ISGetIndices(svweights,&svweights_indices);CHKERRQ(ierr);
      ierr = PetscArraycpy(fp_vweights,svweights_indices,nsvwegihts);CHKERRQ(ierr);
      ierr = ISRestoreIndices(svweights,&svweights_indices);CHKERRQ(ierr);
      ierr = ISDestroy(&svweights);CHKERRQ(ierr);
    }

    ierr = ISDestroy(&destination);CHKERRQ(ierr);
    ierr = PetscObjectGetComm((PetscObject)sadj,&scomm);CHKERRQ(ierr);

    /*
     * If the number of big subdomains is smaller than the number of processor cores, the higher ranks do not
     * need to do partitioning
     * */
    if ((i+rank)<hpart->ncoarseparts) {
      ierr = MatPartitioningDestroy(&hpart->fineMatPart);CHKERRQ(ierr);
      /* create a fine partitioner */
      ierr = MatPartitioningCreate(scomm,&hpart->fineMatPart);CHKERRQ(ierr);
      ierr = PetscObjectSetOptionsPrefix((PetscObject)hpart->fineMatPart,prefix);CHKERRQ(ierr);
      ierr = PetscObjectAppendOptionsPrefix((PetscObject)hpart->fineMatPart,"hierarch_fine_");CHKERRQ(ierr);
      /* if do not set partitioning type, use parmetis by default */
      if (!hpart->fineparttype) {
#if defined(PETSC_HAVE_PARMETIS)
        ierr = MatPartitioningSetType(hpart->fineMatPart,MATPARTITIONINGPARMETIS);CHKERRQ(ierr);
        ierr = PetscStrallocpy(MATPARTITIONINGPARMETIS,&hpart->fineparttype);CHKERRQ(ierr);
#elif defined(PETSC_HAVE_PTSCOTCH)
        ierr = MatPartitioningSetType(hpart->fineMatPart,MATPARTITIONINGPTSCOTCH);CHKERRQ(ierr);
        ierr = PetscStrallocpy(MATPARTITIONINGPTSCOTCH,&hpart->fineparttype);CHKERRQ(ierr);
#elif defined(PETSC_HAVE_CHACO)
        ierr = MatPartitioningSetType(hpart->fineMatPart,MATPARTITIONINGCHACO);CHKERRQ(ierr);
        ierr = PetscStrallocpy(MATPARTITIONINGCHACO,&hpart->fineparttype);CHKERRQ(ierr);
#elif defined(PETSC_HAVE_PARTY)
        ierr = MatPartitioningSetType(hpart->fineMatPart,MATPARTITIONINGPARTY);CHKERRQ(ierr);
        ierr = PetscStrallocpy(PETSC_HAVE_PARTY,&hpart->fineparttype);CHKERRQ(ierr);
#else
        SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Requires PETSc be installed with ParMetis or run with -mat_partitioning_hierarchical_coarseparttype partitiontype");
#endif
      } else {
        ierr = MatPartitioningSetType(hpart->fineMatPart,hpart->fineparttype);CHKERRQ(ierr);
      }
      ierr = MatPartitioningSetUseEdgeWeights(hpart->fineMatPart,use_edge_weights);CHKERRQ(ierr);
      ierr = MatPartitioningSetAdjacency(hpart->fineMatPart,sadj);CHKERRQ(ierr);
      ierr = MatPartitioningSetNParts(hpart->fineMatPart, offsets[rank+1+i]-offsets[rank+i]);CHKERRQ(ierr);
      if (part->vertex_weights) {
        ierr = MatPartitioningSetVertexWeights(hpart->fineMatPart,fp_vweights);CHKERRQ(ierr);
      }
      ierr = MatPartitioningApply(hpart->fineMatPart,&fineparts_temp);CHKERRQ(ierr);
    } else {
      ierr = ISCreateGeneral(scomm,0,NULL,PETSC_OWN_POINTER,&fineparts_temp);CHKERRQ(ierr);
    }

    ierr = MatDestroy(&sadj);CHKERRQ(ierr);

    /* Send partition back to the original owners */
    ierr = MatPartitioningHierarchical_ReassembleFineparts(adj,fineparts_temp,mapping,&hpart->fineparts);CHKERRQ(ierr);
    ierr = ISGetIndices(hpart->fineparts,&fineparts_indices);CHKERRQ(ierr);
    for (j=0;j<mat_localsize;j++)
      if (fineparts_indices[j] >=0) fineparts_indices_tmp[j] = fineparts_indices[j];

    ierr = ISRestoreIndices(hpart->fineparts,&fineparts_indices);CHKERRQ(ierr);
    ierr = ISDestroy(&hpart->fineparts);CHKERRQ(ierr);
    ierr = ISDestroy(&fineparts_temp);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingDestroy(&mapping);CHKERRQ(ierr);
  }

  if (part->vertex_weights) {
    ierr = ISDestroy(&vweights);CHKERRQ(ierr);
  }

  ierr = ISCreateGeneral(comm,mat_localsize,fineparts_indices_tmp,PETSC_OWN_POINTER,&hpart->fineparts);CHKERRQ(ierr);
  ierr = ISGetIndices(hpart->fineparts,&fineparts_indices);CHKERRQ(ierr);
  ierr = ISGetIndices(hpart->coarseparts,&coarseparts_indices);CHKERRQ(ierr);
  ierr = PetscMalloc1(bs*adj->rmap->n,&parts_indices);CHKERRQ(ierr);
  /* Modify the local indices to the global indices by combing the coarse partition and the fine partitions */
  for (i=0; i<adj->rmap->n; i++) {
    for (j=0; j<bs; j++) {
      parts_indices[bs*i+j] = fineparts_indices[i]+offsets[coarseparts_indices[i]];
    }
  }
  ierr = ISRestoreIndices(hpart->fineparts,&fineparts_indices);CHKERRQ(ierr);
  ierr = ISRestoreIndices(hpart->coarseparts,&coarseparts_indices);CHKERRQ(ierr);
  ierr = PetscFree(offsets);CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm,bs*adj->rmap->n,parts_indices,PETSC_OWN_POINTER,partitioning);CHKERRQ(ierr);
  ierr = MatDestroy(&adj);CHKERRQ(ierr);
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
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  PetscValidPointer(sfineparts, 4);
  ierr = PetscObjectGetComm((PetscObject)adj,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);
  ierr = MatGetLayouts(adj,&rmap,NULL);CHKERRQ(ierr);
  ierr = ISGetLocalSize(fineparts,&localsize);CHKERRQ(ierr);
  ierr = PetscMalloc2(localsize,&global_indices,localsize,&local_indices);CHKERRQ(ierr);
  for (i=0; i<localsize; i++) {
    local_indices[i] = i;
  }
  /* map local indices back to global so that we can permulate data globally */
  ierr = ISLocalToGlobalMappingApply(mapping,localsize,local_indices,global_indices);CHKERRQ(ierr);
  ierr = PetscCalloc1(localsize,&owners);CHKERRQ(ierr);
  /* find owners for global indices */
  for (i=0; i<localsize; i++) {
    ierr = PetscLayoutFindOwner(rmap,global_indices[i],&owners[i]);CHKERRQ(ierr);
  }
  ierr = PetscLayoutGetRanges(rmap,&ranges);CHKERRQ(ierr);
  ierr = PetscMalloc1(ranges[rank+1]-ranges[rank],&sfineparts_indices);CHKERRQ(ierr);

  for (i=0; i<ranges[rank+1]-ranges[rank]; i++) {
    sfineparts_indices[i] = -1;
  }

  ierr = ISGetIndices(fineparts,&fineparts_indices);CHKERRQ(ierr);
  ierr = PetscSFCreate(comm,&sf);CHKERRQ(ierr);
  ierr = PetscMalloc1(localsize,&remote);CHKERRQ(ierr);
  for (i=0; i<localsize; i++) {
    remote[i].rank  = owners[i];
    remote[i].index = global_indices[i]-ranges[owners[i]];
  }
  ierr = PetscSFSetType(sf,PETSCSFBASIC);CHKERRQ(ierr);
  /* not sure how to add prefix to sf */
  ierr = PetscSFSetFromOptions(sf);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(sf,localsize,localsize,NULL,PETSC_OWN_POINTER,remote,PETSC_OWN_POINTER);CHKERRQ(ierr);
  ierr = PetscSFReduceBegin(sf,MPIU_INT,fineparts_indices,sfineparts_indices,MPI_REPLACE);CHKERRQ(ierr);
  ierr = PetscSFReduceEnd(sf,MPIU_INT,fineparts_indices,sfineparts_indices,MPI_REPLACE);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);
  ierr = ISRestoreIndices(fineparts,&fineparts_indices);CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm,ranges[rank+1]-ranges[rank],sfineparts_indices,PETSC_OWN_POINTER,sfineparts);CHKERRQ(ierr);
  ierr = PetscFree2(global_indices,local_indices);CHKERRQ(ierr);
  ierr = PetscFree(owners);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatPartitioningHierarchical_AssembleSubdomain(Mat adj,IS vweights, IS destination,IS *svweights,Mat *sadj,ISLocalToGlobalMapping *mapping)
{
  IS              irows,icols;
  PetscInt        irows_ln;
  PetscMPIInt     rank;
  const PetscInt *irows_indices;
  MPI_Comm        comm;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)adj,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);
  /* figure out where data comes from  */
  ierr = ISBuildTwoSided(destination,NULL,&irows);CHKERRQ(ierr);
  ierr = ISDuplicate(irows,&icols);CHKERRQ(ierr);
  ierr = ISGetLocalSize(irows,&irows_ln);CHKERRQ(ierr);
  ierr = ISGetIndices(irows,&irows_indices);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingCreate(comm,1,irows_ln,irows_indices,PETSC_COPY_VALUES,mapping);CHKERRQ(ierr);
  ierr = ISRestoreIndices(irows,&irows_indices);CHKERRQ(ierr);
  ierr = MatCreateSubMatrices(adj,1,&irows,&icols,MAT_INITIAL_MATRIX,&sadj);CHKERRQ(ierr);
  if (vweights && svweights) {
    ierr = ISCreateSubIS(vweights,irows,svweights);CHKERRQ(ierr);
  }
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
  ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  PetscCheckFalse((pend-pstart)>size,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"range [%" PetscInt_FMT ", %" PetscInt_FMT "] should be smaller than or equal to size %" PetscInt_FMT,pstart,pend,size);
  PetscCheckFalse(pstart>pend,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP," pstart %" PetscInt_FMT " should be smaller than pend %" PetscInt_FMT,pstart,pend);
  ierr = ISGetLocalSize(partitioning,&plocalsize);CHKERRQ(ierr);
  ierr = PetscMalloc1(plocalsize,&dest_indices);CHKERRQ(ierr);
  ierr = ISGetIndices(partitioning,&part_indices);CHKERRQ(ierr);
  for (i=0; i<plocalsize; i++) {
    /* compute target */
    target = part_indices[i]-pstart;
    /* mark out of range entity as -1 */
    if (part_indices[i]<pstart || part_indices[i]>=pend) target = -1;
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
  PetscViewer              sviewer;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)part),&rank);CHKERRMPI(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer," Number of coarse parts: %" PetscInt_FMT "\n",hpart->ncoarseparts);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer," Coarse partitioner: %s\n",hpart->coarseparttype);CHKERRQ(ierr);
    if (hpart->coarseMatPart) {
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = MatPartitioningView(hpart->coarseMatPart,viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer," Number of fine parts: %" PetscInt_FMT "\n",hpart->nfineparts);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer," Fine partitioner: %s\n",hpart->fineparttype);CHKERRQ(ierr);
    ierr = PetscViewerGetSubViewer(viewer,PETSC_COMM_SELF,&sviewer);CHKERRQ(ierr);
    if (rank == 0 && hpart->fineMatPart) {
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = MatPartitioningView(hpart->fineMatPart,sviewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
    ierr = PetscViewerRestoreSubViewer(viewer,PETSC_COMM_SELF,&sviewer);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  char           value[1024];
  PetscBool      flag = PETSC_FALSE;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"Set hierarchical partitioning options");CHKERRQ(ierr);
  ierr = PetscOptionsString("-mat_partitioning_hierarchical_coarseparttype","coarse part type",NULL,NULL,value,sizeof(value),&flag);CHKERRQ(ierr);
  if (flag) {
   ierr = PetscStrallocpy(value,&hpart->coarseparttype);CHKERRQ(ierr);
  }
  ierr = PetscOptionsString("-mat_partitioning_hierarchical_fineparttype","fine part type",NULL,NULL,value,sizeof(value),&flag);CHKERRQ(ierr);
  if (flag) {
    ierr = PetscStrallocpy(value,&hpart->fineparttype);CHKERRQ(ierr);
  }
  ierr = PetscOptionsInt("-mat_partitioning_hierarchical_ncoarseparts","number of coarse parts",NULL,hpart->ncoarseparts,&hpart->ncoarseparts,&flag);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-mat_partitioning_hierarchical_nfineparts","number of fine parts",NULL,hpart->nfineparts,&hpart->nfineparts,&flag);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatPartitioningDestroy_Hierarchical(MatPartitioning part)
{
  MatPartitioning_Hierarchical *hpart = (MatPartitioning_Hierarchical*)part->data;
  PetscErrorCode           ierr;

  PetscFunctionBegin;
  if (hpart->coarseparttype) {ierr = PetscFree(hpart->coarseparttype);CHKERRQ(ierr);}
  if (hpart->fineparttype) {ierr = PetscFree(hpart->fineparttype);CHKERRQ(ierr);}
  ierr = ISDestroy(&hpart->fineparts);CHKERRQ(ierr);
  ierr = ISDestroy(&hpart->coarseparts);CHKERRQ(ierr);
  ierr = MatPartitioningDestroy(&hpart->coarseMatPart);CHKERRQ(ierr);
  ierr = MatPartitioningDestroy(&hpart->fineMatPart);CHKERRQ(ierr);
  ierr = MatPartitioningDestroy(&hpart->improver);CHKERRQ(ierr);
  ierr = PetscFree(hpart);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   Improves the quality  of a partition
*/
static PetscErrorCode MatPartitioningImprove_Hierarchical(MatPartitioning part, IS *partitioning)
{
  PetscErrorCode               ierr;
  MatPartitioning_Hierarchical *hpart = (MatPartitioning_Hierarchical*)part->data;
  Mat                           mat = part->adj, adj;
  PetscBool                    flg;
  const char                   *prefix;
#if defined(PETSC_HAVE_PARMETIS)
  PetscInt                     *vertex_weights;
#endif

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)mat,MATMPIADJ,&flg);CHKERRQ(ierr);
  if (flg) {
    adj = mat;
    ierr = PetscObjectReference((PetscObject)adj);CHKERRQ(ierr);
  }else {
    /* bs indicates if the converted matrix is "reduced" from the original and hence the
       resulting partition results need to be stretched to match the original matrix */
   ierr = MatConvert(mat,MATMPIADJ,MAT_INITIAL_MATRIX,&adj);CHKERRQ(ierr);
  }

  /* If there exists a mat partitioner, we should delete it */
  ierr = MatPartitioningDestroy(&hpart->improver);CHKERRQ(ierr);
  ierr = MatPartitioningCreate(PetscObjectComm((PetscObject)part),&hpart->improver);CHKERRQ(ierr);
  ierr = PetscObjectGetOptionsPrefix((PetscObject)part,&prefix);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)hpart->improver,prefix);CHKERRQ(ierr);
  ierr = PetscObjectAppendOptionsPrefix((PetscObject)hpart->improver,"hierarch_improver_");CHKERRQ(ierr);
  /* Only parmetis supports to refine a partition */
#if defined(PETSC_HAVE_PARMETIS)
  ierr = MatPartitioningSetType(hpart->improver,MATPARTITIONINGPARMETIS);CHKERRQ(ierr);
  ierr = MatPartitioningSetAdjacency(hpart->improver,adj);CHKERRQ(ierr);
  ierr = MatPartitioningSetNParts(hpart->improver, part->n);CHKERRQ(ierr);
  /* copy over vertex weights */
  if (part->vertex_weights) {
    ierr = PetscMalloc1(adj->rmap->n,&vertex_weights);CHKERRQ(ierr);
    ierr = PetscArraycpy(vertex_weights,part->vertex_weights,adj->rmap->n);CHKERRQ(ierr);
    ierr = MatPartitioningSetVertexWeights(hpart->improver,vertex_weights);CHKERRQ(ierr);
  }
  ierr = MatPartitioningImprove(hpart->improver,partitioning);CHKERRQ(ierr);
  ierr = MatDestroy(&adj);CHKERRQ(ierr);
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
  PetscErrorCode                ierr;
  MatPartitioning_Hierarchical *hpart;

  PetscFunctionBegin;
  ierr       = PetscNewLog(part,&hpart);CHKERRQ(ierr);
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
