
#include <../src/mat/impls/adj/mpi/mpiadj.h>    /*I "petscmat.h" I*/

/*
   Currently using ParMetis-4.0.2
*/

#include <parmetis.h>

/*
      The first 5 elements of this structure are the input control array to Metis
*/
typedef struct {
  PetscInt  cuts;         /* number of cuts made (output) */
  PetscInt  foldfactor;
  PetscInt  parallel;     /* use parallel partitioner for coarse problem */
  PetscInt  indexing;     /* 0 indicates C indexing, 1 Fortran */
  PetscInt  printout;     /* indicates if one wishes Metis to print info */
  PetscBool repartition;
} MatPartitioning_Parmetis;

#define CHKERRQPARMETIS(n,func) do { \
    PetscCheckFalse(n == METIS_ERROR_INPUT,PETSC_COMM_SELF,PETSC_ERR_LIB,"ParMETIS error due to wrong inputs and/or options for %s",func); \
    else PetscCheckFalse(n == METIS_ERROR_MEMORY,PETSC_COMM_SELF,PETSC_ERR_LIB,"ParMETIS error due to insufficient memory in %s",func); \
    else PetscCheckFalse(n == METIS_ERROR,PETSC_COMM_SELF,PETSC_ERR_LIB,"ParMETIS general error in %s",func); \
  } while (0)

#define PetscStackCallParmetis_(name,func,args) do {    \
    PetscStackPush(name);                               \
    int status = func args;                             \
    PetscStackPop;                                      \
    CHKERRQPARMETIS(status,name);                       \
  } while (0)

#define PetscStackCallParmetis(func,args) PetscStackCallParmetis_(PetscStringize(func),func,args)

static PetscErrorCode MatPartitioningApply_Parmetis_Private(MatPartitioning part, PetscBool useND, PetscBool isImprove, IS *partitioning)
{
  MatPartitioning_Parmetis *pmetis = (MatPartitioning_Parmetis*)part->data;
  PetscInt                 *locals = NULL;
  Mat                      mat     = part->adj,amat,pmat;
  PetscBool                flg;
  PetscInt                 bs = 1;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part,MAT_PARTITIONING_CLASSID,1);
  PetscValidPointer(partitioning,4);
  CHKERRQ(PetscObjectTypeCompare((PetscObject)mat,MATMPIADJ,&flg));
  if (flg) {
    amat = mat;
    CHKERRQ(PetscObjectReference((PetscObject)amat));
  } else {
    /* bs indicates if the converted matrix is "reduced" from the original and hence the
       resulting partition results need to be stretched to match the original matrix */
    CHKERRQ(MatConvert(mat,MATMPIADJ,MAT_INITIAL_MATRIX,&amat));
    if (amat->rmap->n > 0) bs = mat->rmap->n/amat->rmap->n;
  }
  CHKERRQ(MatMPIAdjCreateNonemptySubcommMat(amat,&pmat));
  CHKERRMPI(MPI_Barrier(PetscObjectComm((PetscObject)part)));

  if (pmat) {
    MPI_Comm   pcomm,comm;
    Mat_MPIAdj *adj     = (Mat_MPIAdj*)pmat->data;
    PetscInt   *vtxdist = pmat->rmap->range;
    PetscInt   *xadj    = adj->i;
    PetscInt   *adjncy  = adj->j;
    PetscInt   *NDorder = NULL;
    PetscInt   itmp     = 0,wgtflag=0, numflag=0, ncon=1, nparts=part->n, options[24], i, j;
    real_t     *tpwgts,*ubvec,itr=0.1;

    CHKERRQ(PetscObjectGetComm((PetscObject)pmat,&pcomm));
    if (PetscDefined(USE_DEBUG)) {
      /* check that matrix has no diagonal entries */
      PetscInt rstart;
      CHKERRQ(MatGetOwnershipRange(pmat,&rstart,NULL));
      for (i=0; i<pmat->rmap->n; i++) {
        for (j=xadj[i]; j<xadj[i+1]; j++) {
          PetscCheckFalse(adjncy[j] == i+rstart,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Row %" PetscInt_FMT " has diagonal entry; Parmetis forbids diagonal entry",i+rstart);
        }
      }
    }

    CHKERRQ(PetscMalloc1(pmat->rmap->n,&locals));

    if (isImprove) {
      PetscInt       i;
      const PetscInt *part_indices;
      PetscValidHeaderSpecific(*partitioning,IS_CLASSID,4);
      CHKERRQ(ISGetIndices(*partitioning,&part_indices));
      for (i=0; i<pmat->rmap->n; i++) locals[i] = part_indices[i*bs];
      CHKERRQ(ISRestoreIndices(*partitioning,&part_indices));
      CHKERRQ(ISDestroy(partitioning));
    }

    if (adj->values && part->use_edge_weights && !part->vertex_weights) wgtflag = 1;
    if (part->vertex_weights && !adj->values) wgtflag = 2;
    if (part->vertex_weights && adj->values && part->use_edge_weights) wgtflag = 3;

    if (PetscLogPrintInfo) {itmp = pmetis->printout; pmetis->printout = 127;}
    CHKERRQ(PetscMalloc1(ncon*nparts,&tpwgts));
    for (i=0; i<ncon; i++) {
      for (j=0; j<nparts; j++) {
        if (part->part_weights) {
          tpwgts[i*nparts+j] = part->part_weights[i*nparts+j];
        } else {
          tpwgts[i*nparts+j] = 1./nparts;
        }
      }
    }
    CHKERRQ(PetscMalloc1(ncon,&ubvec));
    for (i=0; i<ncon; i++) ubvec[i] = 1.05;
    /* This sets the defaults */
    options[0] = 0;
    for (i=1; i<24; i++) options[i] = -1;
    /* Duplicate the communicator to be sure that ParMETIS attribute caching does not interfere with PETSc. */
    CHKERRMPI(MPI_Comm_dup(pcomm,&comm));
    if (useND) {
      PetscInt    *sizes, *seps, log2size, subd, *level;
      PetscMPIInt size;
      idx_t       mtype = PARMETIS_MTYPE_GLOBAL, rtype = PARMETIS_SRTYPE_2PHASE, p_nseps = 1, s_nseps = 1;
      real_t      ubfrac = 1.05;

      CHKERRMPI(MPI_Comm_size(comm,&size));
      CHKERRQ(PetscMalloc1(pmat->rmap->n,&NDorder));
      CHKERRQ(PetscMalloc3(2*size,&sizes,4*size,&seps,size,&level));
      PetscStackCallParmetis(ParMETIS_V32_NodeND,((idx_t*)vtxdist,(idx_t*)xadj,(idx_t*)adjncy,(idx_t*)part->vertex_weights,(idx_t*)&numflag,&mtype,&rtype,&p_nseps,&s_nseps,&ubfrac,NULL/* seed */,NULL/* dbglvl */,(idx_t*)NDorder,(idx_t*)(sizes),&comm));
      log2size = PetscLog2Real(size);
      subd = PetscPowInt(2,log2size);
      CHKERRQ(MatPartitioningSizesToSep_Private(subd,sizes,seps,level));
      for (i=0;i<pmat->rmap->n;i++) {
        PetscInt loc;

        CHKERRQ(PetscFindInt(NDorder[i],2*subd,seps,&loc));
        if (loc < 0) {
          loc = -(loc+1);
          if (loc%2) { /* part of subdomain */
            locals[i] = loc/2;
          } else {
            CHKERRQ(PetscFindInt(NDorder[i],2*(subd-1),seps+2*subd,&loc));
            loc = loc < 0 ? -(loc+1)/2 : loc/2;
            locals[i] = level[loc];
          }
        } else locals[i] = loc/2;
      }
      CHKERRQ(PetscFree3(sizes,seps,level));
    } else {
      if (pmetis->repartition) {
        PetscStackCallParmetis(ParMETIS_V3_AdaptiveRepart,((idx_t*)vtxdist,(idx_t*)xadj,(idx_t*)adjncy,(idx_t*)part->vertex_weights,(idx_t*)part->vertex_weights,(idx_t*)adj->values,(idx_t*)&wgtflag,(idx_t*)&numflag,(idx_t*)&ncon,(idx_t*)&nparts,tpwgts,ubvec,&itr,(idx_t*)options,(idx_t*)&pmetis->cuts,(idx_t*)locals,&comm));
      } else if (isImprove) {
        PetscStackCallParmetis(ParMETIS_V3_RefineKway,((idx_t*)vtxdist,(idx_t*)xadj,(idx_t*)adjncy,(idx_t*)part->vertex_weights,(idx_t*)adj->values,(idx_t*)&wgtflag,(idx_t*)&numflag,(idx_t*)&ncon,(idx_t*)&nparts,tpwgts,ubvec,(idx_t*)options,(idx_t*)&pmetis->cuts,(idx_t*)locals,&comm));
      } else {
        PetscStackCallParmetis(ParMETIS_V3_PartKway,((idx_t*)vtxdist,(idx_t*)xadj,(idx_t*)adjncy,(idx_t*)part->vertex_weights,(idx_t*)adj->values,(idx_t*)&wgtflag,(idx_t*)&numflag,(idx_t*)&ncon,(idx_t*)&nparts,tpwgts,ubvec,(idx_t*)options,(idx_t*)&pmetis->cuts,(idx_t*)locals,&comm));
      }
    }
    CHKERRMPI(MPI_Comm_free(&comm));

    CHKERRQ(PetscFree(tpwgts));
    CHKERRQ(PetscFree(ubvec));
    if (PetscLogPrintInfo) pmetis->printout = itmp;

    if (bs > 1) {
      PetscInt i,j,*newlocals;
      CHKERRQ(PetscMalloc1(bs*pmat->rmap->n,&newlocals));
      for (i=0; i<pmat->rmap->n; i++) {
        for (j=0; j<bs; j++) {
          newlocals[bs*i + j] = locals[i];
        }
      }
      CHKERRQ(PetscFree(locals));
      CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)part),bs*pmat->rmap->n,newlocals,PETSC_OWN_POINTER,partitioning));
    } else {
      CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)part),pmat->rmap->n,locals,PETSC_OWN_POINTER,partitioning));
    }
    if (useND) {
      IS ndis;

      if (bs > 1) {
        CHKERRQ(ISCreateBlock(PetscObjectComm((PetscObject)part),bs,pmat->rmap->n,NDorder,PETSC_OWN_POINTER,&ndis));
      } else {
        CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)part),pmat->rmap->n,NDorder,PETSC_OWN_POINTER,&ndis));
      }
      CHKERRQ(ISSetPermutation(ndis));
      CHKERRQ(PetscObjectCompose((PetscObject)(*partitioning),"_petsc_matpartitioning_ndorder",(PetscObject)ndis));
      CHKERRQ(ISDestroy(&ndis));
    }
  } else {
    CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)part),0,NULL,PETSC_COPY_VALUES,partitioning));
    if (useND) {
      IS ndis;

      if (bs > 1) {
        CHKERRQ(ISCreateBlock(PetscObjectComm((PetscObject)part),bs,0,NULL,PETSC_COPY_VALUES,&ndis));
      } else {
        CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)part),0,NULL,PETSC_COPY_VALUES,&ndis));
      }
      CHKERRQ(ISSetPermutation(ndis));
      CHKERRQ(PetscObjectCompose((PetscObject)(*partitioning),"_petsc_matpartitioning_ndorder",(PetscObject)ndis));
      CHKERRQ(ISDestroy(&ndis));
    }
  }
  CHKERRQ(MatDestroy(&pmat));
  CHKERRQ(MatDestroy(&amat));
  PetscFunctionReturn(0);
}

/*
   Uses the ParMETIS parallel matrix partitioner to compute a nested dissection ordering of the matrix in parallel
*/
static PetscErrorCode MatPartitioningApplyND_Parmetis(MatPartitioning part, IS *partitioning)
{
  PetscFunctionBegin;
  CHKERRQ(MatPartitioningApply_Parmetis_Private(part, PETSC_TRUE, PETSC_FALSE, partitioning));
  PetscFunctionReturn(0);
}

/*
   Uses the ParMETIS parallel matrix partitioner to partition the matrix in parallel
*/
static PetscErrorCode MatPartitioningApply_Parmetis(MatPartitioning part, IS *partitioning)
{
  PetscFunctionBegin;
  CHKERRQ(MatPartitioningApply_Parmetis_Private(part, PETSC_FALSE, PETSC_FALSE, partitioning));
  PetscFunctionReturn(0);
}

/*
   Uses the ParMETIS to improve the quality  of a partition
*/
static PetscErrorCode MatPartitioningImprove_Parmetis(MatPartitioning part, IS *partitioning)
{
  PetscFunctionBegin;
  CHKERRQ(MatPartitioningApply_Parmetis_Private(part, PETSC_FALSE, PETSC_TRUE, partitioning));
  PetscFunctionReturn(0);
}

PetscErrorCode MatPartitioningView_Parmetis(MatPartitioning part,PetscViewer viewer)
{
  MatPartitioning_Parmetis *pmetis = (MatPartitioning_Parmetis*)part->data;
  PetscMPIInt              rank;
  PetscBool                iascii;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)part),&rank));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    if (pmetis->parallel == 2) {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Using parallel coarse grid partitioner\n"));
    } else {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Using sequential coarse grid partitioner\n"));
    }
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Using %" PetscInt_FMT " fold factor\n",pmetis->foldfactor));
    CHKERRQ(PetscViewerASCIIPushSynchronized(viewer));
    CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer,"  [%d]Number of cuts found %" PetscInt_FMT "\n",rank,pmetis->cuts));
    CHKERRQ(PetscViewerFlush(viewer));
    CHKERRQ(PetscViewerASCIIPopSynchronized(viewer));
  }
  PetscFunctionReturn(0);
}

/*@
     MatPartitioningParmetisSetCoarseSequential - Use the sequential code to
         do the partitioning of the coarse grid.

  Logically Collective on MatPartitioning

  Input Parameter:
.  part - the partitioning context

   Level: advanced

@*/
PetscErrorCode  MatPartitioningParmetisSetCoarseSequential(MatPartitioning part)
{
  MatPartitioning_Parmetis *pmetis = (MatPartitioning_Parmetis*)part->data;

  PetscFunctionBegin;
  pmetis->parallel = 1;
  PetscFunctionReturn(0);
}

/*@
     MatPartitioningParmetisSetRepartition - Repartition
     current mesh to rebalance computation.

  Logically Collective on MatPartitioning

  Input Parameter:
.  part - the partitioning context

   Level: advanced

@*/
PetscErrorCode  MatPartitioningParmetisSetRepartition(MatPartitioning part)
{
  MatPartitioning_Parmetis *pmetis = (MatPartitioning_Parmetis*)part->data;

  PetscFunctionBegin;
  pmetis->repartition = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@
  MatPartitioningParmetisGetEdgeCut - Returns the number of edge cuts in the vertex partition.

  Input Parameter:
. part - the partitioning context

  Output Parameter:
. cut - the edge cut

   Level: advanced

@*/
PetscErrorCode  MatPartitioningParmetisGetEdgeCut(MatPartitioning part, PetscInt *cut)
{
  MatPartitioning_Parmetis *pmetis = (MatPartitioning_Parmetis*) part->data;

  PetscFunctionBegin;
  *cut = pmetis->cuts;
  PetscFunctionReturn(0);
}

PetscErrorCode MatPartitioningSetFromOptions_Parmetis(PetscOptionItems *PetscOptionsObject,MatPartitioning part)
{
  PetscBool      flag = PETSC_FALSE;

  PetscFunctionBegin;
  CHKERRQ(PetscOptionsHead(PetscOptionsObject,"Set ParMeTiS partitioning options"));
  CHKERRQ(PetscOptionsBool("-mat_partitioning_parmetis_coarse_sequential","Use sequential coarse partitioner","MatPartitioningParmetisSetCoarseSequential",flag,&flag,NULL));
  if (flag) {
    CHKERRQ(MatPartitioningParmetisSetCoarseSequential(part));
  }
  CHKERRQ(PetscOptionsBool("-mat_partitioning_parmetis_repartition","","MatPartitioningParmetisSetRepartition",flag,&flag,NULL));
  if (flag) {
    CHKERRQ(MatPartitioningParmetisSetRepartition(part));
  }
  CHKERRQ(PetscOptionsTail());
  PetscFunctionReturn(0);
}

PetscErrorCode MatPartitioningDestroy_Parmetis(MatPartitioning part)
{
  MatPartitioning_Parmetis *pmetis = (MatPartitioning_Parmetis*)part->data;

  PetscFunctionBegin;
  CHKERRQ(PetscFree(pmetis));
  PetscFunctionReturn(0);
}

/*MC
   MATPARTITIONINGPARMETIS - Creates a partitioning context via the external package PARMETIS.

   Collective

   Input Parameter:
.  part - the partitioning context

   Options Database Keys:
.  -mat_partitioning_parmetis_coarse_sequential - use sequential PARMETIS coarse partitioner

   Level: beginner

   Notes:
    See https://www-users.cs.umn.edu/~karypis/metis/

.seealso: MatPartitioningSetType(), MatPartitioningType

M*/

PETSC_EXTERN PetscErrorCode MatPartitioningCreate_Parmetis(MatPartitioning part)
{
  MatPartitioning_Parmetis *pmetis;

  PetscFunctionBegin;
  CHKERRQ(PetscNewLog(part,&pmetis));
  part->data = (void*)pmetis;

  pmetis->cuts       = 0;   /* output variable */
  pmetis->foldfactor = 150; /*folding factor */
  pmetis->parallel   = 2;   /* use parallel partitioner for coarse grid */
  pmetis->indexing   = 0;   /* index numbering starts from 0 */
  pmetis->printout   = 0;   /* print no output while running */
  pmetis->repartition      = PETSC_FALSE;

  part->ops->apply          = MatPartitioningApply_Parmetis;
  part->ops->applynd        = MatPartitioningApplyND_Parmetis;
  part->ops->improve        = MatPartitioningImprove_Parmetis;
  part->ops->view           = MatPartitioningView_Parmetis;
  part->ops->destroy        = MatPartitioningDestroy_Parmetis;
  part->ops->setfromoptions = MatPartitioningSetFromOptions_Parmetis;
  PetscFunctionReturn(0);
}

/*@
 MatMeshToVertexGraph -   This routine does not exist because ParMETIS does not provide the functionality.  Uses the ParMETIS package to
                       convert a Mat that represents a mesh to a Mat the represents the graph of the coupling
                       between vertices of the cells and is suitable for partitioning with the MatPartitioning object. Use this to partition
                       vertices of a mesh. More likely you should use MatMeshToCellGraph()

   Collective on Mat

   Input Parameters:
+     mesh - the graph that represents the mesh
-     ncommonnodes - mesh elements that share this number of common nodes are considered neighbors, use 2 for triangles and
                     quadrilaterials, 3 for tetrahedrals and 4 for hexahedrals

   Output Parameter:
.     dual - the dual graph

   Notes:
     Currently requires ParMetis to be installed and uses ParMETIS_V3_Mesh2Dual()

     The columns of each row of the Mat mesh are the global vertex numbers of the vertices of that rows cell. The number of rows in mesh is
     number of cells, the number of columns is the number of vertices.

   Level: advanced

.seealso: MatMeshToCellGraph(), MatCreateMPIAdj(), MatPartitioningCreate()

@*/
PetscErrorCode MatMeshToVertexGraph(Mat mesh,PetscInt ncommonnodes,Mat *dual)
{
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"ParMETIS does not provide this functionality");
}

/*@
     MatMeshToCellGraph -   Uses the ParMETIS package to convert a Mat that represents a mesh to a Mat the represents the graph of the coupling
                       between cells (the "dual" graph) and is suitable for partitioning with the MatPartitioning object. Use this to partition
                       cells of a mesh.

   Collective on Mat

   Input Parameters:
+     mesh - the graph that represents the mesh
-     ncommonnodes - mesh elements that share this number of common nodes are considered neighbors, use 2 for triangles and
                     quadrilaterials, 3 for tetrahedrals and 4 for hexahedrals

   Output Parameter:
.     dual - the dual graph

   Notes:
     Currently requires ParMetis to be installed and uses ParMETIS_V3_Mesh2Dual()

$     Each row of the mesh object represents a single cell in the mesh. For triangles it has 3 entries, quadrilaterials 4 entries,
$         tetrahedrals 4 entries and hexahedrals 8 entries. You can mix triangles and quadrilaterals in the same mesh, but cannot
$         mix  tetrahedrals and hexahedrals
$     The columns of each row of the Mat mesh are the global vertex numbers of the vertices of that row's cell.
$     The number of rows in mesh is number of cells, the number of columns is the number of vertices.

   Level: advanced

.seealso: MatMeshToVertexGraph(), MatCreateMPIAdj(), MatPartitioningCreate()

@*/
PetscErrorCode MatMeshToCellGraph(Mat mesh,PetscInt ncommonnodes,Mat *dual)
{
  PetscInt       *newxadj,*newadjncy;
  PetscInt       numflag=0;
  Mat_MPIAdj     *adj   = (Mat_MPIAdj*)mesh->data,*newadj;
  PetscBool      flg;
  MPI_Comm       comm;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)mesh,MATMPIADJ,&flg));
  PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_SUP,"Must use MPIAdj matrix type");

  CHKERRQ(PetscObjectGetComm((PetscObject)mesh,&comm));
  PetscStackCallParmetis(ParMETIS_V3_Mesh2Dual,((idx_t*)mesh->rmap->range,(idx_t*)adj->i,(idx_t*)adj->j,(idx_t*)&numflag,(idx_t*)&ncommonnodes,(idx_t**)&newxadj,(idx_t**)&newadjncy,&comm));
  CHKERRQ(MatCreateMPIAdj(PetscObjectComm((PetscObject)mesh),mesh->rmap->n,mesh->rmap->N,newxadj,newadjncy,NULL,dual));
  newadj = (Mat_MPIAdj*)(*dual)->data;

  newadj->freeaijwithfree = PETSC_TRUE; /* signal the matrix should be freed with system free since space was allocated by ParMETIS */
  PetscFunctionReturn(0);
}
