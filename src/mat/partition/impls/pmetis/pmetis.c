


#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: pmetis.c,v 1.2 1997/11/03 04:46:32 bsmith Exp bsmith $";
#endif
 
#include "petsc.h"

#if defined(HAVE_PARMETIS)
#include "src/mat/impls/adj/mpi/mpiadj.h"    /*I "mat.h" I*/
#include "par_kmetis.h"

/*
      The first 5 elements of this structure are the input control array to Metis
*/
typedef struct {
  int cuts;         /* number of cuts made (output) */
  int foldfactor;
  int parallel;     /* use parallel partitioner for coarse problem */
  int indexing;     /* 0 indicates C indexing, 1 Fortran */
  int printout;     /* indicates if one wishes Metis to print info */
} Partitioning_Parmetis;

/*
   Uses the ParMETIS parallel matrix partitioner to partition the matrix in parallel
*/
#undef __FUNC__  
#define __FUNC__ "PartitioningApply_Parmetis" 
static int PartitioningApply_Parmetis(Partitioning part, IS *partitioning)
{
  int                   ierr,*locals,size;
  int                   *vtxdist, *xadj,*adjncy,itmp = 0;
  Mat                   mat = part->adj;
  Mat_MPIAdj            *adj = (Mat_MPIAdj *)mat->data;
  Partitioning_Parmetis *parmetis = (Partitioning_Parmetis*)part->data;

  PetscFunctionBegin;
  if (mat->type != MATMPIADJ) SETERRQ(1,1,"Only MPIAdj matrix type supported");
  MPI_Comm_size(mat->comm,&size);
  if (part->n != size) {
    SETERRQ(1,1,"Supports exactly one domain per processor");
  }

  locals = (int *) PetscMalloc((adj->m+1)*sizeof(int));CHKPTRQ(locals);

  vtxdist = adj->rowners;
  xadj    = adj->i;
  adjncy  = adj->j;

  if (PLogPrintInfo) {itmp = parmetis->printout; parmetis->printout = 1;}
  PARKMETIS(vtxdist,xadj,0,adjncy,0,locals,(int*)parmetis,part->comm);
  if (PLogPrintInfo) {parmetis->printout = itmp;}

  ierr = ISCreateGeneral(part->comm,adj->m,locals,partitioning); CHKERRQ(ierr);
  PetscFree(locals);

  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ "PartitioningView_Parmetis" 
int PartitioningView_Parmetis(PetscObject obj,Viewer viewer)
{
  Partitioning          part = (Partitioning) obj;
  Partitioning_Parmetis *parmetis = (Partitioning_Parmetis *)part->data;
  ViewerType            vtype;
  FILE                  *fd;
  int                   ierr,rank;

  PetscFunctionBegin;
  MPI_Comm_rank(part->comm,&rank);
  ViewerGetType(viewer,&vtype);
  if (vtype  == ASCII_FILE_VIEWER || vtype == ASCII_FILES_VIEWER) {
    ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
    if (parmetis->parallel == 2) {
      PetscFPrintf(part->comm,fd,"  Using parallel coarse grid partitioner\n");
    } else {
      PetscFPrintf(part->comm,fd,"  Using sequential coarse grid partitioner\n");
    }
    PetscFPrintf(part->comm,fd,"  Using %d fold factor\n",parmetis->foldfactor);
    PetscSynchronizedFPrintf(part->comm,fd,"  [%d]Number of cuts found %d\n",rank,parmetis->cuts);
    PetscSynchronizedFlush(part->comm);
  }

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PartitioningParmetisSetCoarseSequential"
/*@
     PartitioningParmetisSetCoarseSequential - Use the sequential code to 
         do the partitioning of the coarse grid.

  Input Parameter:
.  part - the partitioning context

@*/
int PartitioningParmetisSetCoarseSequential(Partitioning part)
{
  Partitioning_Parmetis *parmetis = (Partitioning_Parmetis *)part->data;

  PetscFunctionBegin;
  parmetis->parallel = 1;
  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ "PartitioningPrintHelp_Parmetis" 
int PartitioningPrintHelp_Parmetis(Partitioning part)
{
  PetscFunctionBegin;
  PetscPrintf(part->comm,"ParMETIS options\n");
  PetscPrintf(part->comm,"  -partitioning_parmetis_coarse_sequential\n");
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PartitioningSetFromOptions_Parmetis" 
int PartitioningSetFromOptions_Parmetis(Partitioning part)
{
  int                   ierr,flag;

  PetscFunctionBegin;
  ierr = OptionsHasName(part->prefix,"-partitioning_parmetis_coarse_sequential",&flag);CHKERRQ(ierr);
  if (flag) {
    ierr = PartitioningParmetisSetCoarseSequential(part);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ "PartitioningDestroy_Parmetis" 
int PartitioningDestroy_Parmetis(PetscObject opart)
{
  Partitioning          part = (Partitioning) opart;
  Partitioning_Parmetis *parmetis = (Partitioning_Parmetis *)part->data;
  
  PetscFunctionBegin;
  PetscFree(parmetis);

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PartitioningCreate_Parmetis" 
int PartitioningCreate_Parmetis(Partitioning part)
{
  Partitioning_Parmetis *parmetis;

  PetscFunctionBegin;
  parmetis = PetscNew(Partitioning_Parmetis);CHKPTRQ(parmetis);

  parmetis->cuts       = 0;   /* output variable */
  parmetis->foldfactor = 150; /*folding factor */
  parmetis->parallel   = 2;   /* use parallel partitioner for coarse grid */
  parmetis->indexing   = 0;   /* index numbering starts from 0 */
  parmetis->printout   = 0;   /* print no output while running */

  part->apply          = PartitioningApply_Parmetis;
  part->view           = PartitioningView_Parmetis;
  part->destroy        = PartitioningDestroy_Parmetis;
  part->printhelp      = PartitioningPrintHelp_Parmetis;
  part->setfromoptions = PartitioningSetFromOptions_Parmetis;
  part->type           = PARTITIONING_PARMETIS;
  part->data           = (void *) parmetis;
  PetscFunctionReturn(0);
}

#else

/*
   Dummy function for compilers that don't like empty files.
*/
#undef __FUNC__  
#define __FUNC__ "PartitioningApply_Parmetis" 
int PartitioningApply_Parmetis()
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#endif
