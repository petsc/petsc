
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

#define CHKERRQPARMETIS(n,func)                                             \
  if (n == METIS_ERROR_INPUT) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ParMETIS error due to wrong inputs and/or options for %s",func); \
  else if (n == METIS_ERROR_MEMORY) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ParMETIS error due to insufficient memory in %s",func); \
  else if (n == METIS_ERROR) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ParMETIS general error in %s",func); \

#define PetscStackCallParmetis(func,args) do {PetscStackPush(#func);status = func args;PetscStackPop; CHKERRQPARMETIS(status,#func);} while (0)

/*
   Uses the ParMETIS parallel matrix partitioner to partition the matrix in parallel
*/
static PetscErrorCode MatPartitioningApply_Parmetis(MatPartitioning part,IS *partitioning)
{
  MatPartitioning_Parmetis *pmetis = (MatPartitioning_Parmetis*)part->data;
  PetscErrorCode           ierr;
  PetscInt                 *locals = NULL;
  Mat                      mat     = part->adj,amat,pmat;
  PetscBool                flg;
  PetscInt                 bs = 1;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)mat,MATMPIADJ,&flg);CHKERRQ(ierr);
  if (flg) {
    amat = mat;
    PetscObjectReference((PetscObject)amat);CHKERRQ(ierr);
  } else {
    /* bs indicates if the converted matrix is "reduced" from the original and hence the
       resulting partition results need to be stretched to match the original matrix */
    ierr = MatConvert(mat,MATMPIADJ,MAT_INITIAL_MATRIX,&amat);CHKERRQ(ierr);
    if (amat->rmap->n > 0) bs = mat->rmap->n/amat->rmap->n;
  }
  ierr = MatMPIAdjCreateNonemptySubcommMat(amat,&pmat);CHKERRQ(ierr);
  ierr = MPI_Barrier(PetscObjectComm((PetscObject)part));CHKERRQ(ierr);

  if (pmat) {
    MPI_Comm   pcomm,comm;
    Mat_MPIAdj *adj     = (Mat_MPIAdj*)pmat->data;
    PetscInt   *vtxdist = pmat->rmap->range;
    PetscInt   *xadj    = adj->i;
    PetscInt   *adjncy  = adj->j;
    PetscInt   itmp     = 0,wgtflag=0, numflag=0, ncon=1, nparts=part->n, options[24], i, j;
    real_t     *tpwgts,*ubvec,itr=0.1;
    int        status;

    ierr = PetscObjectGetComm((PetscObject)pmat,&pcomm);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
    /* check that matrix has no diagonal entries */
    {
      PetscInt rstart;
      ierr = MatGetOwnershipRange(pmat,&rstart,NULL);CHKERRQ(ierr);
      for (i=0; i<pmat->rmap->n; i++) {
        for (j=xadj[i]; j<xadj[i+1]; j++) {
          if (adjncy[j] == i+rstart) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Row %d has diagonal entry; Parmetis forbids diagonal entry",i+rstart);
        }
      }
    }
#endif

    ierr = PetscMalloc1(amat->rmap->n,&locals);CHKERRQ(ierr);

    if (PetscLogPrintInfo) {itmp = pmetis->printout; pmetis->printout = 127;}
    ierr = PetscMalloc1(ncon*nparts,&tpwgts);CHKERRQ(ierr);
    for (i=0; i<ncon; i++) {
      for (j=0; j<nparts; j++) {
        if (part->part_weights) {
          tpwgts[i*nparts+j] = part->part_weights[i*nparts+j];
        } else {
          tpwgts[i*nparts+j] = 1./nparts;
        }
      }
    }
    ierr = PetscMalloc1(ncon,&ubvec);CHKERRQ(ierr);
    for (i=0; i<ncon; i++) {
      ubvec[i] = 1.05;
    }
    /* This sets the defaults */
    options[0] = 0;
    for (i=1; i<24; i++) {
      options[i] = -1;
    }
    /* Duplicate the communicator to be sure that ParMETIS attribute caching does not interfere with PETSc. */
    ierr   = MPI_Comm_dup(pcomm,&comm);CHKERRQ(ierr);
    if (pmetis->repartition){
      PetscStackCallParmetis(ParMETIS_V3_AdaptiveRepart,((idx_t*)vtxdist,(idx_t*)xadj,(idx_t*)adjncy,(idx_t*)part->vertex_weights,(idx_t*)part->vertex_weights,(idx_t*)adj->values,(idx_t*)&wgtflag,(idx_t*)&numflag,(idx_t*)&ncon,(idx_t*)&nparts,tpwgts,ubvec,&itr,(idx_t*)options,(idx_t*)&pmetis->cuts,(idx_t*)locals,&comm));
    } else{
      PetscStackCallParmetis(ParMETIS_V3_PartKway,((idx_t*)vtxdist,(idx_t*)xadj,(idx_t*)adjncy,(idx_t*)part->vertex_weights,(idx_t*)adj->values,(idx_t*)&wgtflag,(idx_t*)&numflag,(idx_t*)&ncon,(idx_t*)&nparts,tpwgts,ubvec,(idx_t*)options,(idx_t*)&pmetis->cuts,(idx_t*)locals,&comm));
    }
    ierr   = MPI_Comm_free(&comm);CHKERRQ(ierr);

    ierr = PetscFree(tpwgts);CHKERRQ(ierr);
    ierr = PetscFree(ubvec);CHKERRQ(ierr);
    if (PetscLogPrintInfo) pmetis->printout = itmp;

    if (bs > 1) {
      PetscInt i,j,*newlocals;
      ierr = PetscMalloc1(bs*amat->rmap->n,&newlocals);CHKERRQ(ierr);
      for (i=0; i<amat->rmap->n; i++) {
        for (j=0; j<bs; j++) {
          newlocals[bs*i + j] = locals[i];
        }
      }
      ierr = PetscFree(locals);CHKERRQ(ierr);
      ierr = ISCreateGeneral(PetscObjectComm((PetscObject)part),bs*amat->rmap->n,newlocals,PETSC_OWN_POINTER,partitioning);CHKERRQ(ierr);
    } else {
      ierr = ISCreateGeneral(PetscObjectComm((PetscObject)part),amat->rmap->n,locals,PETSC_OWN_POINTER,partitioning);CHKERRQ(ierr);
    }
  } else {
    ierr = ISCreateGeneral(PetscObjectComm((PetscObject)part),0,NULL,PETSC_COPY_VALUES,partitioning);CHKERRQ(ierr);
  }

  ierr = MatDestroy(&pmat);CHKERRQ(ierr);
  ierr = MatDestroy(&amat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatPartitioningView_Parmetis(MatPartitioning part,PetscViewer viewer)
{
  MatPartitioning_Parmetis *pmetis = (MatPartitioning_Parmetis*)part->data;
  PetscErrorCode           ierr;
  int                      rank;
  PetscBool                iascii;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)part),&rank);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    if (pmetis->parallel == 2) {
      ierr = PetscViewerASCIIPrintf(viewer,"  Using parallel coarse grid partitioner\n");CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  Using sequential coarse grid partitioner\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"  Using %d fold factor\n",pmetis->foldfactor);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushSynchronized(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"  [%d]Number of cuts found %d\n",rank,pmetis->cuts);CHKERRQ(ierr);
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopSynchronized(viewer);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  PetscBool      flag = PETSC_FALSE;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"Set ParMeTiS partitioning options");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-mat_partitioning_parmetis_coarse_sequential","Use sequential coarse partitioner","MatPartitioningParmetisSetCoarseSequential",flag,&flag,NULL);CHKERRQ(ierr);
  if (flag) {
    ierr = MatPartitioningParmetisSetCoarseSequential(part);CHKERRQ(ierr);
  }
  ierr = PetscOptionsBool("-mat_partitioning_parmetis_repartition","","MatPartitioningParmetisSetRepartition",flag,&flag,NULL);CHKERRQ(ierr);
  if(flag){
    ierr =  MatPartitioningParmetisSetRepartition(part);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


PetscErrorCode MatPartitioningDestroy_Parmetis(MatPartitioning part)
{
  MatPartitioning_Parmetis *pmetis = (MatPartitioning_Parmetis*)part->data;
  PetscErrorCode           ierr;

  PetscFunctionBegin;
  ierr = PetscFree(pmetis);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*MC
   MATPARTITIONINGPARMETIS - Creates a partitioning context via the external package PARMETIS.

   Collective on MPI_Comm

   Input Parameter:
.  part - the partitioning context

   Options Database Keys:
+  -mat_partitioning_parmetis_coarse_sequential - use sequential PARMETIS coarse partitioner

   Level: beginner

   Notes: See http://www-users.cs.umn.edu/~karypis/metis/

.keywords: Partitioning, create, context

.seealso: MatPartitioningSetType(), MatPartitioningType

M*/

PETSC_EXTERN PetscErrorCode MatPartitioningCreate_Parmetis(MatPartitioning part)
{
  PetscErrorCode           ierr;
  MatPartitioning_Parmetis *pmetis;

  PetscFunctionBegin;
  ierr       = PetscNewLog(part,&pmetis);CHKERRQ(ierr);
  part->data = (void*)pmetis;

  pmetis->cuts       = 0;   /* output variable */
  pmetis->foldfactor = 150; /*folding factor */
  pmetis->parallel   = 2;   /* use parallel partitioner for coarse grid */
  pmetis->indexing   = 0;   /* index numbering starts from 0 */
  pmetis->printout   = 0;   /* print no output while running */
  pmetis->repartition      = PETSC_FALSE;

  part->ops->apply          = MatPartitioningApply_Parmetis;
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

   Input Parameter:
+     mesh - the graph that represents the mesh
-     ncommonnodes - mesh elements that share this number of common nodes are considered neighbors, use 2 for triangules and
                     quadralaterials, 3 for tetrahedrals and 4 for hexahedrals

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
  PetscFunctionReturn(0);
}

/*@
     MatMeshToCellGraph -   Uses the ParMETIS package to convert a Mat that represents a mesh to a Mat the represents the graph of the coupling
                       between cells (the "dual" graph) and is suitable for partitioning with the MatPartitioning object. Use this to partition
                       cells of a mesh.

   Collective on Mat

   Input Parameter:
+     mesh - the graph that represents the mesh
-     ncommonnodes - mesh elements that share this number of common nodes are considered neighbors, use 2 for triangules and
                     quadralaterials, 3 for tetrahedrals and 4 for hexahedrals

   Output Parameter:
.     dual - the dual graph

   Notes:
     Currently requires ParMetis to be installed and uses ParMETIS_V3_Mesh2Dual()

$     Each row of the mesh object represents a single cell in the mesh. For triangles it has 3 entries, quadralaterials 4 entries,
$         tetrahedrals 4 entries and hexahedrals 8 entries. You can mix triangles and quadrilaterals in the same mesh, but cannot
$         mix  tetrahedrals and hexahedrals
$     The columns of each row of the Mat mesh are the global vertex numbers of the vertices of that row's cell.
$     The number of rows in mesh is number of cells, the number of columns is the number of vertices.


   Level: advanced

.seealso: MatMeshToVertexGraph(), MatCreateMPIAdj(), MatPartitioningCreate()


@*/
PetscErrorCode MatMeshToCellGraph(Mat mesh,PetscInt ncommonnodes,Mat *dual)
{
  PetscErrorCode ierr;
  PetscInt       *newxadj,*newadjncy;
  PetscInt       numflag=0;
  Mat_MPIAdj     *adj   = (Mat_MPIAdj*)mesh->data,*newadj;
  PetscBool      flg;
  int            status;
  MPI_Comm       comm;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)mesh,MATMPIADJ,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Must use MPIAdj matrix type");
  
  ierr = PetscObjectGetComm((PetscObject)mesh,&comm);CHKERRQ(ierr);
  PetscStackCallParmetis(ParMETIS_V3_Mesh2Dual,((idx_t*)mesh->rmap->range,(idx_t*)adj->i,(idx_t*)adj->j,(idx_t*)&numflag,(idx_t*)&ncommonnodes,(idx_t**)&newxadj,(idx_t**)&newadjncy,&comm));
  ierr   = MatCreateMPIAdj(PetscObjectComm((PetscObject)mesh),mesh->rmap->n,mesh->rmap->N,newxadj,newadjncy,NULL,dual);CHKERRQ(ierr);
  newadj = (Mat_MPIAdj*)(*dual)->data;

  newadj->freeaijwithfree = PETSC_TRUE; /* signal the matrix should be freed with system free since space was allocated by ParMETIS */
  PetscFunctionReturn(0);
}
