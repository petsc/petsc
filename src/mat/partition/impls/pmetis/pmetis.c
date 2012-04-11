 
#include <../src/mat/impls/adj/mpi/mpiadj.h>    /*I "petscmat.h" I*/

/* 
   Currently using ParMetis-4.0.2
*/

#include <parmetis.h>

/*
      The first 5 elements of this structure are the input control array to Metis
*/
typedef struct {
  PetscInt cuts;         /* number of cuts made (output) */
  PetscInt foldfactor;
  PetscInt parallel;     /* use parallel partitioner for coarse problem */
  PetscInt indexing;     /* 0 indicates C indexing, 1 Fortran */
  PetscInt printout;     /* indicates if one wishes Metis to print info */
} MatPartitioning_Parmetis;

#define CHKERRQPARMETIS(n) \
  if (n == METIS_ERROR_INPUT) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"ParMETIS error due to wrong inputs and/or options"); \
  else if (n == METIS_ERROR_MEMORY) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"ParMETIS error due to insufficient memory"); \
  else if (n == METIS_ERROR) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"ParMETIS general error"); \

/*
   Uses the ParMETIS parallel matrix partitioner to partition the matrix in parallel
*/
#undef __FUNCT__  
#define __FUNCT__ "MatPartitioningApply_Parmetis" 
static PetscErrorCode MatPartitioningApply_Parmetis(MatPartitioning part,IS *partitioning)
{
  MatPartitioning_Parmetis *parmetis = (MatPartitioning_Parmetis*)part->data;
  PetscErrorCode           ierr;
  PetscInt                 *locals = PETSC_NULL;
  Mat                      mat = part->adj,amat,pmat;
  PetscBool                flg;
  PetscInt                 bs = 1;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)mat,MATMPIADJ,&flg);CHKERRQ(ierr);
  if (flg) {
    amat = mat;
    PetscObjectReference((PetscObject)amat);CHKERRQ(ierr);
  } else {
    /* bs indicates if the converted matrix is "reduced" from the original and hence the 
       resulting partition results need to be stretched to match the original matrix */
    ierr = MatConvert(mat,MATMPIADJ,MAT_INITIAL_MATRIX,&amat);CHKERRQ(ierr);
    if (mat->rmap->n > 0) bs = amat->rmap->n/mat->rmap->n;
  }
  ierr = MatMPIAdjCreateNonemptySubcommMat(amat,&pmat);CHKERRQ(ierr);
  ierr = MPI_Barrier(((PetscObject)part)->comm);CHKERRQ(ierr);

  if (pmat) {
    MPI_Comm   pcomm    = ((PetscObject)pmat)->comm,comm_pmetis;
    Mat_MPIAdj *adj     = (Mat_MPIAdj*)pmat->data;
    PetscInt   *vtxdist = pmat->rmap->range;
    PetscInt   *xadj    = adj->i;
    PetscInt   *adjncy  = adj->j;
    PetscInt   itmp     = 0,wgtflag=0, numflag=0, ncon=1, nparts=part->n, options[24], i, j;
    real_t     *tpwgts,*ubvec;
    int        status;

#if defined(PETSC_USE_DEBUG)
    /* check that matrix has no diagonal entries */
    {
      PetscInt rstart;
      ierr = MatGetOwnershipRange(mat,&rstart,PETSC_NULL);CHKERRQ(ierr);
      for (i=0; i<mat->rmap->n; i++) {
        for (j=xadj[i]; j<xadj[i+1]; j++) {
          if (adjncy[j] == i+rstart) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Row %d has diagonal entry; Parmetis forbids diagonal entry",i+rstart);
        }
      }
    }
#endif

    ierr = PetscMalloc(amat->rmap->n*sizeof(PetscInt),&locals);CHKERRQ(ierr);

    if (PetscLogPrintInfo) {itmp = parmetis->printout; parmetis->printout = 127;}
    ierr = PetscMalloc(ncon*nparts*sizeof(real_t),&tpwgts);CHKERRQ(ierr);
    for (i=0; i<ncon; i++) {
      for (j=0; j<nparts; j++) {
        if (part->part_weights) {
          tpwgts[i*nparts+j] = part->part_weights[i*nparts+j];
        } else {
          tpwgts[i*nparts+j] = 1./nparts;
        }
      }
    }
    ierr = PetscMalloc(ncon*sizeof(real_t),&ubvec);CHKERRQ(ierr);
    for (i=0; i<ncon; i++) {
      ubvec[i] = 1.05;
    }
    /* This sets the defaults */
    options[0] = 0;
    for (i=1; i<24; i++) {
      options[i] = -1;
    }
    /* Duplicate the communicator to be sure that ParMETIS attribute caching does not interfere with PETSc. */
    ierr = MPI_Comm_dup(pcomm,&comm_pmetis);CHKERRQ(ierr);
    status = ParMETIS_V3_PartKway(vtxdist,xadj,adjncy,part->vertex_weights,adj->values,&wgtflag,&numflag,&ncon,&nparts,tpwgts,ubvec,options,&parmetis->cuts,locals,&comm_pmetis);CHKERRQPARMETIS(status);
    ierr = MPI_Comm_free(&comm_pmetis);CHKERRQ(ierr);

    ierr = PetscFree(tpwgts);CHKERRQ(ierr);
    ierr = PetscFree(ubvec);CHKERRQ(ierr);
    if (PetscLogPrintInfo) {parmetis->printout = itmp;}
  }

  if (bs > 1) {
    PetscInt i,j,*newlocals;
    ierr = PetscMalloc(bs*amat->rmap->n*sizeof(PetscInt),&newlocals);CHKERRQ(ierr);
    for (i=0; i<amat->rmap->n; i++) {
      for (j=0; j<bs; j++) {
        newlocals[bs*i + j] = locals[i];
      }
    }
    ierr = PetscFree(locals);CHKERRQ(ierr);
    ierr = ISCreateGeneral(((PetscObject)part)->comm,bs*amat->rmap->n,newlocals,PETSC_OWN_POINTER,partitioning);CHKERRQ(ierr);
  } else {
    ierr = ISCreateGeneral(((PetscObject)part)->comm,amat->rmap->n,locals,PETSC_OWN_POINTER,partitioning);CHKERRQ(ierr);
  }

  ierr = MatDestroy(&pmat);CHKERRQ(ierr);
  ierr = MatDestroy(&amat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatPartitioningView_Parmetis" 
PetscErrorCode MatPartitioningView_Parmetis(MatPartitioning part,PetscViewer viewer)
{
  MatPartitioning_Parmetis *parmetis = (MatPartitioning_Parmetis *)part->data;
  PetscErrorCode ierr;
  int rank;
  PetscBool                iascii;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(((PetscObject)part)->comm,&rank);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    if (parmetis->parallel == 2) {
      ierr = PetscViewerASCIIPrintf(viewer,"  Using parallel coarse grid partitioner\n");CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  Using sequential coarse grid partitioner\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"  Using %d fold factor\n",parmetis->foldfactor);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedAllow(viewer,PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"  [%d]Number of cuts found %d\n",rank,parmetis->cuts);CHKERRQ(ierr);
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedAllow(viewer,PETSC_FALSE);CHKERRQ(ierr);
  } else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Viewer type %s not supported for this Parmetis partitioner",((PetscObject)viewer)->type_name);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatPartitioningParmetisSetCoarseSequential"
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
  MatPartitioning_Parmetis *parmetis = (MatPartitioning_Parmetis *)part->data;

  PetscFunctionBegin;
  parmetis->parallel = 1;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatPartitioningParmetisGetEdgeCut"
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
  MatPartitioning_Parmetis *parmetis = (MatPartitioning_Parmetis *) part->data;

  PetscFunctionBegin;
  *cut = parmetis->cuts;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatPartitioningSetFromOptions_Parmetis" 
PetscErrorCode MatPartitioningSetFromOptions_Parmetis(MatPartitioning part)
{
  PetscErrorCode ierr;
  PetscBool      flag = PETSC_FALSE;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("Set ParMeTiS partitioning options");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-mat_partitioning_parmetis_coarse_sequential","Use sequential coarse partitioner","MatPartitioningParmetisSetCoarseSequential",flag,&flag,PETSC_NULL);CHKERRQ(ierr);
    if (flag) {
      ierr = MatPartitioningParmetisSetCoarseSequential(part);CHKERRQ(ierr);
    }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatPartitioningDestroy_Parmetis" 
PetscErrorCode MatPartitioningDestroy_Parmetis(MatPartitioning part)
{
  MatPartitioning_Parmetis *parmetis = (MatPartitioning_Parmetis *)part->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(parmetis);CHKERRQ(ierr);
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

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatPartitioningCreate_Parmetis" 
PetscErrorCode  MatPartitioningCreate_Parmetis(MatPartitioning part)
{
  PetscErrorCode ierr;
  MatPartitioning_Parmetis *parmetis;

  PetscFunctionBegin;
  ierr  = PetscNewLog(part,MatPartitioning_Parmetis,&parmetis);CHKERRQ(ierr);
  part->data                = (void*)parmetis;

  parmetis->cuts       = 0;   /* output variable */
  parmetis->foldfactor = 150; /*folding factor */
  parmetis->parallel   = 2;   /* use parallel partitioner for coarse grid */
  parmetis->indexing   = 0;   /* index numbering starts from 0 */
  parmetis->printout   = 0;   /* print no output while running */

  part->ops->apply          = MatPartitioningApply_Parmetis;
  part->ops->view           = MatPartitioningView_Parmetis;
  part->ops->destroy        = MatPartitioningDestroy_Parmetis;
  part->ops->setfromoptions = MatPartitioningSetFromOptions_Parmetis;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatMeshToVertexGraph"
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

#undef __FUNCT__  
#define __FUNCT__ "MatMeshToCellGraph"
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

     The columns of each row of the Mat mesh are the global vertex numbers of the vertices of that rows cell. The number of rows in mesh is 
     number of cells, the number of columns is the number of vertices.
   

   Level: advanced

.seealso: MatMeshToVertexGraph(), MatCreateMPIAdj(), MatPartitioningCreate()


@*/
PetscErrorCode MatMeshToCellGraph(Mat mesh,PetscInt ncommonnodes,Mat *dual)
{
  PetscErrorCode           ierr;
  PetscInt                 *newxadj,*newadjncy;
  PetscInt                 numflag=0;
  Mat_MPIAdj               *adj = (Mat_MPIAdj *)mesh->data,*newadj;
  PetscBool                flg;
  int                      status;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)mesh,MATMPIADJ,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Must use MPIAdj matrix type");

  CHKMEMQ;
  status = ParMETIS_V3_Mesh2Dual(mesh->rmap->range,adj->i,adj->j,&numflag,&ncommonnodes,&newxadj,&newadjncy,&((PetscObject)mesh)->comm);CHKERRQPARMETIS(status);
  CHKMEMQ;
  ierr = MatCreateMPIAdj(((PetscObject)mesh)->comm,mesh->rmap->n,mesh->rmap->N,newxadj,newadjncy,PETSC_NULL,dual);CHKERRQ(ierr);
  newadj = (Mat_MPIAdj *)(*dual)->data;
  newadj->freeaijwithfree = PETSC_TRUE; /* signal the matrix should be freed with system free since space was allocated by ParMETIS */
  PetscFunctionReturn(0);
}
