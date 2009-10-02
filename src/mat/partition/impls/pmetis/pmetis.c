#define PETSCMAT_DLL
 
#include "../src/mat/impls/adj/mpi/mpiadj.h"    /*I "petscmat.h" I*/

/* 
   Currently using ParMetis-2.0. The following include file has
   to be changed to par_kmetis.h for ParMetis-1.0
*/
EXTERN_C_BEGIN
#include "parmetis.h"
EXTERN_C_END

/*
      The first 5 elements of this structure are the input control array to Metis
*/
typedef struct {
  int cuts;         /* number of cuts made (output) */
  int foldfactor;
  int parallel;     /* use parallel partitioner for coarse problem */
  int indexing;     /* 0 indicates C indexing, 1 Fortran */
  int printout;     /* indicates if one wishes Metis to print info */
  MPI_Comm comm_pmetis;
} MatPartitioning_Parmetis;

/*
   Uses the ParMETIS parallel matrix partitioner to partition the matrix in parallel
*/
#undef __FUNCT__  
#define __FUNCT__ "MatPartitioningApply_Parmetis" 
static PetscErrorCode MatPartitioningApply_Parmetis(MatPartitioning part,IS *partitioning)
{
  MatPartitioning_Parmetis *parmetis = (MatPartitioning_Parmetis*)part->data;
  PetscErrorCode           ierr;
  int                      *locals,size,rank;
  int                      *vtxdist,*xadj,*adjncy,itmp = 0;
  int                      wgtflag=0, numflag=0, ncon=1, nparts=part->n, options[3],  i,j;
  Mat                      mat = part->adj;
  Mat_MPIAdj               *adj = (Mat_MPIAdj *)mat->data;
  PetscTruth               flg;
  float                    *tpwgts,*ubvec;
  PetscInt                 bs = 1,nold;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(((PetscObject)mat)->comm,&size);CHKERRQ(ierr);

  ierr = PetscTypeCompare((PetscObject)mat,MATMPIADJ,&flg);CHKERRQ(ierr);
  if (!flg) {
    /* bs indicates if the converted matrix is "reduced" from the original and hence the 
       resulting partition results need to be stretched to match the original matrix */
    nold = mat->rmap->n;
    ierr = MatConvert(mat,MATMPIADJ,MAT_INITIAL_MATRIX,&mat);CHKERRQ(ierr);
    bs   = nold/mat->rmap->n;
    adj  = (Mat_MPIAdj *)mat->data;
  }

  vtxdist = mat->rmap->range;
  xadj    = adj->i;
  adjncy  = adj->j;
  ierr    = MPI_Comm_rank(((PetscObject)part)->comm,&rank);CHKERRQ(ierr);
#if 0
  if (!(vtxdist[rank+1] - vtxdist[rank])) {
    SETERRQ(PETSC_ERR_LIB,"Does not support any processor with no entries");
  }
#endif
#if defined(PETSC_USE_DEBUG)
  /* check that matrix has no diagonal entries */
  {
    int rstart;
    ierr = MatGetOwnershipRange(mat,&rstart,PETSC_NULL);CHKERRQ(ierr);
    for (i=0; i<mat->rmap->n; i++) {
      for (j=xadj[i]; j<xadj[i+1]; j++) {
        if (adjncy[j] == i+rstart) SETERRQ1(PETSC_ERR_ARG_WRONG,"Row %d has diagonal entry; Parmetis forbids diagonal entry",i+rstart);
      }
    }
  }
#endif

  ierr = PetscMalloc((mat->rmap->n+1)*sizeof(int),&locals);CHKERRQ(ierr);

  if (PetscLogPrintInfo) {itmp = parmetis->printout; parmetis->printout = 127;}
  ierr = PetscMalloc(ncon*nparts*sizeof(float),&tpwgts);CHKERRQ(ierr);
  for (i=0; i<ncon; i++) {
    for (j=0; j<nparts; j++) {
      if (part->part_weights) {
	tpwgts[i*nparts+j] = part->part_weights[i*nparts+j];
      } else {
	tpwgts[i*nparts+j] = 1./nparts;
      }
    }
  }
  ierr = PetscMalloc(ncon*sizeof(float),&ubvec);CHKERRQ(ierr);
  for (i=0; i<ncon; i++) {
    ubvec[i] = 1.05;
  }
  options[0] = 0;
  /* ParMETIS has no error conditions ??? */
  ParMETIS_V3_PartKway(vtxdist,xadj,adjncy,part->vertex_weights,adj->values,&wgtflag,&numflag,&ncon,&nparts,tpwgts,ubvec,options,&parmetis->cuts,locals,&parmetis->comm_pmetis);
  ierr = PetscFree(tpwgts);CHKERRQ(ierr);
  ierr = PetscFree(ubvec);CHKERRQ(ierr);
  if (PetscLogPrintInfo) {parmetis->printout = itmp;}

  if (bs > 1) {
    PetscInt *newlocals;
    ierr = PetscMalloc(bs*mat->rmap->n*sizeof(PetscInt),&newlocals);CHKERRQ(ierr);
    for (i=0; i<mat->rmap->n; i++) {
      for (j=0; j<bs; j++) {
        newlocals[bs*i + j] = locals[i];
      }
    }
    ierr = ISCreateGeneral(((PetscObject)part)->comm,bs*mat->rmap->n,newlocals,partitioning);CHKERRQ(ierr);
    ierr = PetscFree(newlocals);CHKERRQ(ierr);
  } else {
    ierr = ISCreateGeneral(((PetscObject)part)->comm,mat->rmap->n,locals,partitioning);CHKERRQ(ierr);
  }
  ierr = PetscFree(locals);CHKERRQ(ierr);

  if (!flg) {
    ierr = MatDestroy(mat);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatPartitioningView_Parmetis" 
PetscErrorCode MatPartitioningView_Parmetis(MatPartitioning part,PetscViewer viewer)
{
  MatPartitioning_Parmetis *parmetis = (MatPartitioning_Parmetis *)part->data;
  PetscErrorCode ierr;
  int rank;
  PetscTruth               iascii;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(((PetscObject)part)->comm,&rank);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    if (parmetis->parallel == 2) {
      ierr = PetscViewerASCIIPrintf(viewer,"  Using parallel coarse grid partitioner\n");CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  Using sequential coarse grid partitioner\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"  Using %d fold factor\n",parmetis->foldfactor);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"  [%d]Number of cuts found %d\n",rank,parmetis->cuts);CHKERRQ(ierr);
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported for this Parmetis partitioner",((PetscObject)viewer)->type_name);
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatPartitioningParmetisSetCoarseSequential"
/*@
     MatPartitioningParmetisSetCoarseSequential - Use the sequential code to 
         do the partitioning of the coarse grid.

  Collective on MatPartitioning

  Input Parameter:
.  part - the partitioning context

   Level: advanced

@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatPartitioningParmetisSetCoarseSequential(MatPartitioning part)
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
PetscErrorCode PETSCMAT_DLLEXPORT MatPartitioningParmetisGetEdgeCut(MatPartitioning part, PetscInt *cut)
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
  PetscTruth     flag = PETSC_FALSE;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("Set ParMeTiS partitioning options");CHKERRQ(ierr);
  ierr = PetscOptionsTruth("-mat_partitioning_parmetis_coarse_sequential","Use sequential coarse partitioner","MatPartitioningParmetisSetCoarseSequential",flag,&flag,PETSC_NULL);CHKERRQ(ierr);
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
  ierr = MPI_Comm_free(&(parmetis->comm_pmetis));CHKERRQ(ierr);
  ierr = PetscFree(parmetis);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*MC
   MAT_PARTITIONING_PARMETIS - Creates a partitioning context via the external package PARMETIS.

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
PetscErrorCode PETSCMAT_DLLEXPORT MatPartitioningCreate_Parmetis(MatPartitioning part)
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

  ierr = MPI_Comm_dup(((PetscObject)part)->comm,&(parmetis->comm_pmetis));CHKERRQ(ierr);

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
  SETERRQ(PETSC_ERR_SUP,"ParMETIS does not provide this functionality");
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
  int                      *newxadj,*newadjncy;
  int                      numflag=0;
  Mat_MPIAdj               *adj = (Mat_MPIAdj *)mesh->data,*newadj;
  PetscTruth               flg;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)mesh,MATMPIADJ,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_ERR_SUP,"Must use MPIAdj matrix type");

  /* ParMETIS has no error conditions ??? */
  CHKMEMQ;
  ParMETIS_V3_Mesh2Dual(mesh->rmap->range,adj->i,adj->j,&numflag,&ncommonnodes,&newxadj,&newadjncy,&((PetscObject)mesh)->comm);
  CHKMEMQ;
  ierr = MatCreateMPIAdj(((PetscObject)mesh)->comm,mesh->rmap->n,mesh->rmap->N,newxadj,newadjncy,PETSC_NULL,dual);CHKERRQ(ierr);
  newadj = (Mat_MPIAdj *)(*dual)->data;
  newadj->freeaijwithfree = PETSC_TRUE; /* signal the matrix should be freed with system free since space was allocated by ParMETIS */
  PetscFunctionReturn(0);
}
