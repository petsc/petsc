
#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ispai.c,v 1.1 1997/01/30 04:08:33 bsmith Exp bsmith $";
#endif

/*
      Provides an interface to the SPAI Sparse Approximate Inverse Preconditioner
   Code written by Stephen Barnard.
*/

#include "src/sles/pc/pcimpl.h"        /*I "pc.h" I*/

/*
    These are the SPAI include files
*/
#include "basics.h"
#include "command_line.h"
#include "vector.h"
#include "index_set.h"
#include "matrix.h"
#include "spai.h"
#include "mv_schedule.h"
#include "read_matrix.h"

extern int MatConvertToSPAI(Mat,matrix**);

typedef struct {
  matrix *B,*M;           /* B matrix in SPAI format, M the approximate inverse */
  double epsilon;         /* tolerance */
  int    nbsteps;         /* max number of "improvement" steps per line */
  int    maxapi;          /* upper limit on nonzeros in line of M */
  int    max;             /* max dimensions of is_I, q, etc. */
  int    maxnew;          /* max number of new entries per step */
  int    cache_size;      /* one of (1,2,3,4,5,6) indicting size of cache */

} PC_SPAI;

#undef __FUNC__  
#define __FUNC__ "PCSetUp_SPAI"
static int PCSetUp_SPAI(PC pc)
{
  PC_SPAI *ispai = (PC_SPAI *) pc->data;
  int      ierr;

  ierr = MatConvertToSPAI(pc->pmat,&ispai->B); CHKERRQ(ierr);

  /* construct SPAI preconditioner */
  /* FILE *messages */   /* file for warning messages */
  /* double epsilon */   /* tolerance */
  /* int nbsteps */      /* max number of "improvement" steps per line */
  /* int maxapi */       /* upper limit on nonzeros in line of M */
  /* int max */          /* max dimensions of is_I, q, etc. */
  /* int maxnew */       /* max number of new entries per step */
  /* int cache_size */   /* one of (1,2,3,4,5,6) indicting size of cache */
                         /* cache_size == 0 indicates no caching */

  ispai->M = spai(ispai->B, stderr, ispai->epsilon, ispai->nbsteps, ispai->maxapi,
                  ispai->max, ispai->maxnew, ispai->cache_size);
  if (!ispai->M) SETERRQ(1,1,"Unable to create SPAI preconditioner");

  mv_schedule(ispai->B);
  mv_schedule(ispai->M);

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCApply_SPAI"
static int PCApply_SPAI(PC pc,Vec x,Vec y)
{
  PC_SPAI *ispai = (PC_SPAI *) pc->data;
  int      ierr;
  vector  vx,vy;

  ierr   = VecGetSize(x,&vx.n); CHKERRQ(ierr);
  ierr   = VecGetLocalSize(x,&vx.mnl); CHKERRQ(ierr);
  vy.n   = vx.n;
  vy.mnl = vx.mnl;
  ierr   = VecGetArray(x,&vx.v); CHKERRQ(ierr);
  ierr   = VecGetArray(y,&vy.v); CHKERRQ(ierr);

  A_times_v(ispai->M,&vx,&vy); 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCDestroy_SPAI"
static int PCDestroy_SPAI(PC pc)
{
  PC_SPAI *ispai = (PC_SPAI *) pc->data;

  PetscFree(ispai);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCView_SPAI"
static int PCView_SPAI(PC pc,Viewer viewer)
{
  PC_SPAI    *ispai = (PC_SPAI *) pc->data;
  int        ierr;
  ViewerType vtype;

  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  if (PetscTypeCompare(vtype,ASCII_VIEWER)) {  
    ViewerASCIIPrintf(viewer,"  SPAI preconditioner\n");
    ViewerASCIIPrintf(viewer,"    epsilon %g\n",ispai->epsilon);
    ViewerASCIIPrintf(viewer,"    nbsteps %d\n",ispai->nbsteps);
    ViewerASCIIPrintf(viewer,"    maxapi %d\n",ispai->maxapi);
    ViewerASCIIPrintf(viewer,"    max %d\n",ispai->max);
    ViewerASCIIPrintf(viewer,"    maxnew %d\n",ispai->maxnew);
    ViewerASCIIPrintf(viewer,"    cache_size %d\n",ispai->cache_size);

  }
  PetscFunctionReturn(0);
}

/*
   PCSPAI must be initialized by a PCRegister()
*/
int PCSPAI;

#undef __FUNC__  
#define __FUNC__ "PCSPAISetEpsilon"
int PCSPAISetEpsilon(PC pc,double epsilon)
{
  PC_SPAI    *ispai = (PC_SPAI *) pc->data;
  if (pc->type != PCSPAI) PetscFunctionReturn(0);
  ispai->epsilon = epsilon;
  PetscFunctionReturn(0);
}
    
#undef __FUNC__  
#define __FUNC__ "PCSPAISetNBSteps"
int PCSPAISetNBSteps(PC pc,int nbsteps)
{
  PC_SPAI    *ispai = (PC_SPAI *) pc->data;
  if (pc->type != PCSPAI) PetscFunctionReturn(0);
  ispai->nbsteps = nbsteps;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCSPAISetMaxAPI"
int PCSPAISetMaxAPI(PC pc,int maxapi)
{
  PC_SPAI    *ispai = (PC_SPAI *) pc->data;
  if (pc->type != PCSPAI) PetscFunctionReturn(0);
  ispai->maxapi = maxapi;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCSPAISetMaxNew"
int PCSPAISetMaxNew(PC pc,int maxnew)
{
  PC_SPAI    *ispai = (PC_SPAI *) pc->data;
  if (pc->type != PCSPAI) PetscFunctionReturn(0);
  ispai->maxnew = maxnew;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCSPAISetCacheSize"
int PCSPAISetCacheSize(PC pc,int cache_size)
{
  PC_SPAI    *ispai = (PC_SPAI *) pc->data;
  if (pc->type != PCSPAI) PetscFunctionReturn(0);
  ispai->cache_size = cache_size;
  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ "PCSetFromOptions_SPAI"
static int PCSetFromOptions_SPAI(PC pc)
{
  int    ierr,flg,nbsteps,maxapi,maxnew,cache_size;
  double epsilon;

  ierr = OptionsGetDouble(pc->prefix,"-pc_spai_epsilon",&epsilon,&flg); CHKERRQ(ierr);
  if (flg) {
    ierr = PCSPAISetEpsilon(pc,epsilon); CHKERRQ(ierr);
  }
  ierr = OptionsGetInt(pc->prefix,"-pc_spai_nbsteps",&nbsteps,&flg); CHKERRQ(ierr);
  if (flg) {
    ierr = PCSPAISetNBSteps(pc,nbsteps); CHKERRQ(ierr);
  }
  ierr = OptionsGetInt(pc->prefix,"-pc_spai_maxapi",&maxapi,&flg); CHKERRQ(ierr);
  if (flg) {
    ierr = PCSPAISetMaxAPI(pc,maxapi); CHKERRQ(ierr);
  }
  ierr = OptionsGetInt(pc->prefix,"-pc_spai_maxnew",&maxnew,&flg); CHKERRQ(ierr);
  if (flg) {
    ierr = PCSPAISetMaxNew(pc,maxnew); CHKERRQ(ierr);
  } 
  ierr = OptionsGetInt(pc->prefix,"-pc_spai_cache_size",&cache_size,&flg); CHKERRQ(ierr);
  if (flg) {
    ierr = PCSPAISetCacheSize(pc,cache_size); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCPrintHelp_SPAI"
static int PCPrintHelp_SPAI(PC pc,char *p)
{
  PetscPrintf(pc->comm," Options for PCSPAI preconditioner:\n");
  PetscPrintf(pc->comm," %spc_spai_epsilon <epsilon> : (default .4)\n",p);
  PetscPrintf(pc->comm," %spc_spai_nbsteps <nbsteps> : (default 5)\n",p);
  PetscPrintf(pc->comm," %spc_spai_maxapi <maxapi>   : (default 60)\n",p);
  PetscPrintf(pc->comm," %spc_spai_maxnew <maxnew>   : (default 5)\n",p);
  PetscPrintf(pc->comm," %spc_spai_cache_size <cache_size> : (default 5)\n",p);
  PetscFunctionReturn(0);
}

/*
   PCCreate_SPAI - Creates the preconditioner context for the SPAI 
                   preconditioner written by Stephen Barnard.

*/
EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "PCCreate_SPAI"
int PCCreate_SPAI(PC pc)
{
  PC_SPAI *ispai;

  pc->destroy        = PCDestroy_SPAI;
  ispai              = PetscNew(PC_SPAI); CHKPTRQ(ispai);
  pc->data           = (void *) ispai;
  pc->apply          = PCApply_SPAI;
  pc->applyrich      = 0;
  pc->setup          = PCSetUp_SPAI;
  pc->type           = PCSPAI;
  pc->view           = PCView_SPAI;
  pc->setfrom        = PCSetFromOptions_SPAI;
  pc->name           = 0;
  pc->printhelp      = PCPrintHelp_SPAI;

  ispai->epsilon    = .4;  
  ispai->nbsteps    = 5;        
  ispai->maxapi     = 60;         
  ispai->max        = 1000;            
  ispai->maxnew     = 5;         
  ispai->cache_size = 5;     

  /*
       SPAI has GLOBAL variables numprocs and myid!!!!!!!!!!!!!!!!!!!!!!
  */
  MPI_Comm_size(pc->comm,&numprocs);
  MPI_Comm_rank(pc->comm,&myid);

  PetscFunctionReturn(0);
}
EXTERN_C_END

/*
   Converts from a PETSc matrix to an SPAI matrix 
*/
int MatConvertToSPAI(Mat A,matrix **B)
{
  matrix   *M;
  int      col,j;
  double   *vals;
  int      row_indx;
  int      col_pe,col_indx;
  int      len,i;
  clines   *rows;
  int      ierr,*cols;
  int      *num_row_ptr,n,mnl,nnl,rank,size,nz,rstart,rend;
  MPI_Comm comm;

  PetscObjectGetComm((PetscObject)A,&comm);
 
  MPI_Comm_size(comm,&size);
  MPI_Comm_rank(comm,&rank);
  ierr = MatGetSize(A,&n,&n); CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&mnl,&nnl); CHKERRQ(ierr);

  ierr = MPI_Barrier(MPI_COMM_WORLD);CHKERRQ(ierr);

  M = (matrix *) PetscMalloc(sizeof(matrix)); CHKPTRQ(M);

  M->n = n;

  M->mnls = (int *) PetscMalloc(sizeof(int)*size);CHKPTRQ(M->mnls);
  M->start_indices = (int *) PetscMalloc(sizeof(int)*size);CHKPTRQ(M->start_indices);

  ierr = MPI_Barrier(MPI_COMM_WORLD);CHKERRQ(ierr);
  ierr = MPI_Allgather(&mnl, 1, MPI_INT,M->mnls, 1, MPI_INT,MPI_COMM_WORLD);CHKERRQ(ierr);
  M->start_indices[0] = 0;
  for (i=1; i<size; i++) {
    M->start_indices[i] = M->start_indices[i-1] + M->mnls[i-1];
  }

  M->original_indices = (int *) PetscMalloc(sizeof(int)*M->mnls[rank]);
    CHKPTRQ(M->original_indices);
 
  for (i=0; i<M->mnls[rank]; i++) {
    M->original_indices[i] = M->start_indices[rank] + i;
  }

  /* row structure */
  M->lines = new_compressed_lines(M); CHKPTRQ(M->lines);
  M->rhs   = new_vector(M->n,M->mnls[rank]); CHKPTRQ(M->rhs);
  rows     = M->lines;

  /* count number of nonzeros */
  /* determine number of nonzeros in every row */
  num_row_ptr = (int *) PetscMalloc(mnl*sizeof(int)); CHKPTRQ(num_row_ptr);
  ierr = MatGetOwnershipRange(A,&rstart,&rend); CHKERRQ(ierr);
  for (i=rstart; i<rend; i++) {
    ierr = MatGetRow(A,i,&num_row_ptr[i-rstart],PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    ierr = MatRestoreRow(A,i,&num_row_ptr[i-rstart],PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  }

  /* allocate buffers */
  len = 0;
  for (i=0; i<n; i++) {
    if (len < num_row_ptr[i]) len = num_row_ptr[i];
  }
  /* get max length over all processors */
  ierr = MPI_Allreduce(&len,&M->maxnz,1,MPI_INT,MPI_MAX,comm);CHKERRQ(ierr);

  for ( i=rstart; i<rend; i++ ) {
    row_indx             = i-rstart;
    len                  = num_row_ptr[row_indx];
    rows->ptrs[row_indx] = (dist_ptr *) PetscMalloc(len*sizeof(dist_ptr));CHKPTRQ(rows->ptrs[row_indx]);
    rows->A[row_indx]    = (double *) PetscMalloc(len*sizeof(double));CHKPTRQ(rows->A[row_indx]);
    rows->adrs[row_indx] = (double **) PetscMalloc(len*sizeof(double *));CHKPTRQ(rows->adrs[row_indx]);
  }
  PetscFree(num_row_ptr);

  /* cp the matrix */
  for (i=rstart; i<rend; i++) {
    row_indx = i - rstart;
    ierr     = MatGetRow(A,i,&nz,&cols,&vals);CHKERRQ(ierr);
    for (j=0; j<nz; j++ ) {
      col = cols[j];
      
      compute_pe_and_index(M,col, &col_pe, &col_indx);

      len = rows->len[row_indx]++;
      rows->ptrs[row_indx][len].pe    = col_pe;
      rows->ptrs[row_indx][len].index = col_indx;
      rows->A[row_indx][len]          = vals[j];
    }
    ierr     = MatRestoreRow(A,i,&nz,&cols,&vals);CHKERRQ(ierr);
  }

  order_pointers(M);

  *B = M;
  PetscFunctionReturn(0);
}


