
#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: pmetis.c,v 1.24 1999/10/01 21:21:34 bsmith Exp bsmith $";
#endif
 
#include "petsc.h"
#if defined(PETSC_HAVE_PARMETIS)
#include "src/mat/impls/adj/mpi/mpiadj.h"    /*I "mat.h" I*/

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
} MatPartitioning_Parmetis;

/*
   Uses the ParMETIS parallel matrix partitioner to partition the matrix in parallel
*/
#undef __FUNC__  
#define __FUNC__ "MatPartitioningApply_Parmetis" 
static int MatPartitioningApply_Parmetis(MatPartitioning part, IS *partitioning)
{
  int                   ierr,*locals,size,rank;
  int                   *vtxdist, *xadj,*adjncy,itmp = 0;
  Mat                   mat = part->adj;
  Mat_MPIAdj            *adj = (Mat_MPIAdj *)mat->data;
  MatPartitioning_Parmetis *parmetis = (MatPartitioning_Parmetis*)part->data;

  PetscFunctionBegin;
  if (mat->type != MATMPIADJ) SETERRQ(PETSC_ERR_SUP,1,"Only MPIAdj matrix type supported");
  ierr = MPI_Comm_size(mat->comm,&size);CHKERRQ(ierr);
  if (part->n != size) {
    SETERRQ(PETSC_ERR_SUP,1,"Supports exactly one domain per processor");
  }

  vtxdist = adj->rowners;
  xadj    = adj->i;
  adjncy  = adj->j;
  ierr = MPI_Comm_rank(part->comm,&rank);CHKERRQ(ierr);
  if (vtxdist[rank+1] - vtxdist[rank] == 0) {
    SETERRQ(1,1,"Does not support any processor with no entries");
  }
  locals = (int *) PetscMalloc((adj->m+1)*sizeof(int));CHKPTRQ(locals);

  if (PLogPrintInfo) {itmp = parmetis->printout; parmetis->printout = 127;}
  PARKMETIS(vtxdist,xadj,0,adjncy,0,locals,(int*)parmetis,part->comm);
  if (PLogPrintInfo) {parmetis->printout = itmp;}

  ierr = ISCreateGeneral(part->comm,adj->m,locals,partitioning);CHKERRQ(ierr);
  ierr = PetscFree(locals);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ "MatPartitioningView_Parmetis" 
int MatPartitioningView_Parmetis(MatPartitioning part,Viewer viewer)
{
  MatPartitioning_Parmetis *parmetis = (MatPartitioning_Parmetis *)part->data;
  FILE                     *fd;
  int                      ierr,rank;
  int                      isascii;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(part->comm,&rank);CHKERRQ(ierr);
  isascii = PetscTypeCompare(viewer,ASCII_VIEWER);
  if (isascii) {
    ierr = ViewerASCIIGetPointer(viewer,&fd);CHKERRQ(ierr);
    if (parmetis->parallel == 2) {
      ierr = PetscFPrintf(part->comm,fd,"  Using parallel coarse grid partitioner\n");CHKERRQ(ierr);
    } else {
      ierr = PetscFPrintf(part->comm,fd,"  Using sequential coarse grid partitioner\n");CHKERRQ(ierr);
    }
    ierr = PetscFPrintf(part->comm,fd,"  Using %d fold factor\n",parmetis->foldfactor);CHKERRQ(ierr);
    ierr = PetscSynchronizedFPrintf(part->comm,fd,"  [%d]Number of cuts found %d\n",rank,parmetis->cuts);CHKERRQ(ierr);
    ierr = PetscSynchronizedFlush(part->comm);CHKERRQ(ierr);
  } else {
    SETERRQ1(1,1,"Viewer type %s not supported for this Parmetis partitioner",((PetscObject)viewer)->type_name);
  }

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatPartitioningParmetisSetCoarseSequential"
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

#undef __FUNC__  
#define __FUNC__ "MatPartitioningPrintHelp_Parmetis" 
int MatPartitioningPrintHelp_Parmetis(MatPartitioning part)
{
  int ierr;

  PetscFunctionBegin;
  ierr = (*PetscHelpPrintf)(part->comm,"ParMETIS options\n");CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(part->comm,"  -mat_partitioning_parmetis_coarse_sequential\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatPartitioningSetFromOptions_Parmetis" 
int MatPartitioningSetFromOptions_Parmetis(MatPartitioning part)
{
  int                   ierr,flag;

  PetscFunctionBegin;
  ierr = OptionsHasName(part->prefix,"-mat_partitioning_parmetis_coarse_sequential",&flag);CHKERRQ(ierr);
  if (flag) {
    ierr = MatPartitioningParmetisSetCoarseSequential(part);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ "MatPartitioningDestroy_Parmetis" 
int MatPartitioningDestroy_Parmetis(MatPartitioning part)
{
  MatPartitioning_Parmetis *parmetis = (MatPartitioning_Parmetis *)part->data;
  int ierr;

  PetscFunctionBegin;
  ierr = PetscFree(parmetis);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "MatPartitioningCreate_Parmetis" 
int MatPartitioningCreate_Parmetis(MatPartitioning part)
{
  MatPartitioning_Parmetis *parmetis;

  PetscFunctionBegin;
  parmetis = PetscNew(MatPartitioning_Parmetis);CHKPTRQ(parmetis);

  parmetis->cuts       = 0;   /* output variable */
  parmetis->foldfactor = 150; /*folding factor */
  parmetis->parallel   = 2;   /* use parallel partitioner for coarse grid */
  parmetis->indexing   = 0;   /* index numbering starts from 0 */
  parmetis->printout   = 0;   /* print no output while running */

  part->apply          = MatPartitioningApply_Parmetis;
  part->view           = MatPartitioningView_Parmetis;
  part->destroy        = MatPartitioningDestroy_Parmetis;
  part->printhelp      = MatPartitioningPrintHelp_Parmetis;
  part->setfromoptions = MatPartitioningSetFromOptions_Parmetis;
  part->data           = (void *) parmetis;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#else

/*
   Dummy function for compilers that don't like empty files.
*/
#undef __FUNC__  
#define __FUNC__ "MatPartitioningApply_Parmetis" 
int MatPartitioningApply_Parmetis(void)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#endif
