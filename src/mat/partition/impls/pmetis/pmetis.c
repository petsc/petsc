/*$Id: pmetis.c,v 1.46 2001/09/07 20:10:38 bsmith Exp $*/
 
#include "src/mat/impls/adj/mpi/mpiadj.h"    /*I "petscmat.h" I*/

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
static int MatPartitioningApply_Parmetis(MatPartitioning part,IS *partitioning)
{
  MatPartitioning_Parmetis *parmetis = (MatPartitioning_Parmetis*)part->data;
  int                      ierr,*locals,size,rank;
  int                      *vtxdist,*xadj,*adjncy,itmp = 0;
  int                      wgtflag=0, numflag=0, ncon=1, nparts=part->n, options[3], edgecut, i,j;
  Mat                      mat = part->adj,newmat;
  Mat_MPIAdj               *adj = (Mat_MPIAdj *)mat->data;
  PetscTruth               flg;
  float                    *tpwgts,*ubvec;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(mat->comm,&size);CHKERRQ(ierr);

  ierr = PetscTypeCompare((PetscObject)mat,MATMPIADJ,&flg);CHKERRQ(ierr);
  if (!flg) {
    ierr = MatConvert(mat,MATMPIADJ,&newmat);CHKERRQ(ierr);
    adj  = (Mat_MPIAdj *)newmat->data;
  }

  vtxdist = adj->rowners;
  xadj    = adj->i;
  adjncy  = adj->j;
  ierr    = MPI_Comm_rank(part->comm,&rank);CHKERRQ(ierr);
  if (!(vtxdist[rank+1] - vtxdist[rank])) {
    SETERRQ(1,"Does not support any processor with no entries");
  }
#if defined(PETSC_USE_BOPT_g)
  /* check that matrix has no diagonal entries */
  {
    int i,j,rstart;
    ierr = MatGetOwnershipRange(mat,&rstart,PETSC_NULL);CHKERRQ(ierr);
    for (i=0; i<mat->m; i++) {
      for (j=xadj[i]; j<xadj[i+1]; j++) {
        if (adjncy[j] == i+rstart) SETERRQ1(1,"Row %d has diagonal entry; Parmetis forbids diagonal entry",i+rstart);
      }
    }
  }
#endif

  ierr = PetscMalloc((mat->m+1)*sizeof(int),&locals);CHKERRQ(ierr);

  if (PetscLogPrintInfo) {itmp = parmetis->printout; parmetis->printout = 127;}
  ierr = PetscMalloc(ncon*nparts*sizeof(float),&tpwgts); CHKERRQ(ierr);
  for (i=0; i<ncon; i++) {
    for (j=0; j<nparts; j++) {
      if (part->part_weights) {
	tpwgts[i*nparts+j] = part->part_weights[i*nparts+j];
      } else {
	tpwgts[i*nparts+j] = 1./nparts;
      }
    }
  }
  ierr = PetscMalloc(ncon*sizeof(float),&ubvec); CHKERRQ(ierr);
  for (i=0; i<ncon; i++) {
    ubvec[i] = 1.05;
  }
  options[0] = 0;
  /* ParMETIS has no error conditions ??? */
  ParMETIS_V3_PartKway(vtxdist,xadj,adjncy,part->vertex_weights,adj->values,&wgtflag,&numflag,&ncon,&nparts,tpwgts,ubvec,
                       options,&edgecut,locals,&parmetis->comm_pmetis);
  ierr = PetscFree(tpwgts); CHKERRQ(ierr);
  ierr = PetscFree(ubvec); CHKERRQ(ierr);
  if (PetscLogPrintInfo) {parmetis->printout = itmp;}

  ierr = ISCreateGeneral(part->comm,mat->m,locals,partitioning);CHKERRQ(ierr);
  ierr = PetscFree(locals);CHKERRQ(ierr);

  if (!flg) {
    ierr = MatDestroy(newmat);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatPartitioningView_Parmetis" 
int MatPartitioningView_Parmetis(MatPartitioning part,PetscViewer viewer)
{
  MatPartitioning_Parmetis *parmetis = (MatPartitioning_Parmetis *)part->data;
  int                      ierr,rank;
  PetscTruth               isascii;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(part->comm,&rank);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    if (parmetis->parallel == 2) {
      ierr = PetscViewerASCIIPrintf(viewer,"  Using parallel coarse grid partitioner\n");CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  Using sequential coarse grid partitioner\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"  Using %d fold factor\n",parmetis->foldfactor);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"  [%d]Number of cuts found %d\n",rank,parmetis->cuts);CHKERRQ(ierr);
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  } else {
    SETERRQ1(1,"Viewer type %s not supported for this Parmetis partitioner",((PetscObject)viewer)->type_name);
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
int MatPartitioningParmetisSetCoarseSequential(MatPartitioning part)
{
  MatPartitioning_Parmetis *parmetis = (MatPartitioning_Parmetis *)part->data;

  PetscFunctionBegin;
  parmetis->parallel = 1;
  PetscFunctionReturn(0);
}
#undef __FUNCT__  
#define __FUNCT__ "MatPartitioningSetFromOptions_Parmetis" 
int MatPartitioningSetFromOptions_Parmetis(MatPartitioning part)
{
  int        ierr;
  PetscTruth flag;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("Set ParMeTiS partitioning options");CHKERRQ(ierr);
    ierr = PetscOptionsName("-mat_partitioning_parmetis_coarse_sequential","Use sequential coarse partitioner","MatPartitioningParmetisSetCoarseSequential",&flag);CHKERRQ(ierr);
    if (flag) {
      ierr = MatPartitioningParmetisSetCoarseSequential(part);CHKERRQ(ierr);
    }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatPartitioningDestroy_Parmetis" 
int MatPartitioningDestroy_Parmetis(MatPartitioning part)
{
  MatPartitioning_Parmetis *parmetis = (MatPartitioning_Parmetis *)part->data;
  int ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_free(&(parmetis->comm_pmetis));CHKERRQ(ierr);
  ierr = PetscFree(parmetis);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatPartitioningCreate_Parmetis" 
int MatPartitioningCreate_Parmetis(MatPartitioning part)
{
  int                      ierr;
  MatPartitioning_Parmetis *parmetis;

  PetscFunctionBegin;
  ierr  = PetscNew(MatPartitioning_Parmetis,&parmetis);CHKERRQ(ierr);

  parmetis->cuts       = 0;   /* output variable */
  parmetis->foldfactor = 150; /*folding factor */
  parmetis->parallel   = 2;   /* use parallel partitioner for coarse grid */
  parmetis->indexing   = 0;   /* index numbering starts from 0 */
  parmetis->printout   = 0;   /* print no output while running */

  ierr = MPI_Comm_dup(part->comm,&(parmetis->comm_pmetis));CHKERRQ(ierr);

  part->ops->apply          = MatPartitioningApply_Parmetis;
  part->ops->view           = MatPartitioningView_Parmetis;
  part->ops->destroy        = MatPartitioningDestroy_Parmetis;
  part->ops->setfromoptions = MatPartitioningSetFromOptions_Parmetis;
  part->data                = (void*)parmetis;
  PetscFunctionReturn(0);
}
EXTERN_C_END

