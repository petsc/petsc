#ifndef lint
static char vcid[] = "$Id: asm.c,v 1.1 1995/10/31 20:06:23 bsmith Exp bsmith $";
#endif
/*
   Defines a additive Schwarz preconditioner for any Mat implementation.

  Note each processor may have any number of subdomains. But in order to 
  deal easily with the VecScatter() we treat each processor as if it has the
  same number of subdomains.

       n - total number of true subdomains on all processors
       n_local_true - actual number on this processor
       n_local = maximum over all processors of n_local_true

*/
#include "pcimpl.h"
#include "sles.h"

typedef struct {
  int        n,n_local,n_local_true;
  SLES       *sles;                   /* linear solvers for each block */
  VecScatter *scat;                   /* mapping to subregion */
  Vec        *x,*y;
  IS         *is;                     /* index set that defines each subdomain */
  Mat        *mat,*pmat;              /* mat is not currently used */
} PC_ASM;

static int PCSetUp_ASM(PC pc)
{
  PC_ASM              *osm  = (PC_ASM *) pc->data;
  int                 i,ierr,m,n_local = osm->n_local,n_local_true = osm->n_local_true;
  MatGetSubMatrixCall scall = MAT_REUSE_MATRIX;
  IS                  isl;
  SLES                sles;
  KSP                 subksp;
  PC                  subpc;

  if (pc->setupcalled == 0) {
    if (osm->n == PETSC_DECIDE && osm->n_local_true == PETSC_DECIDE) { 
      /* no subdomains given, use one per processor */
      int  size,start,end;
      osm->n_local_true = osm->n_local = 1;
      MPI_Comm_size(pc->comm,&size);
      osm->n = size;
      MatGetOwnershipRange(pc->pmat,&start,&end);
      ierr = ISCreateStrideSeq(MPI_COMM_SELF,end-start,start,1,&isl); CHKERRQ(ierr);
      osm->is    = (IS *) PETSCMALLOC( sizeof(IS **) ); CHKPTRQ(osm->is);
      osm->is[0] = isl;
    }

    if (osm->n == PETSC_DECIDE) { /* determine global number of subdomains */
      MPI_Allreduce(&osm->n_local_true,&osm->n,1,MPI_INT,MPI_SUM,pc->comm);
      MPI_Allreduce(&osm->n_local_true,&osm->n_local,1,MPI_INT,MPI_MAX,pc->comm);
    }

    n_local      = osm->n_local;
    n_local_true = osm->n_local_true;

    osm->sles = (SLES *) PETSCMALLOC(n_local*sizeof(SLES **)); CHKPTRQ(osm->sles);
    osm->scat = (VecScatter *) PETSCMALLOC(n_local*sizeof(VecScatter **)); CHKPTRQ(osm->scat);
    osm->x    = (Vec *) PETSCMALLOC(2*n_local*sizeof(Vec **)); CHKPTRQ(osm->x);
    osm->y    = osm->x + n_local;

    /* create the local work vectors and scatter contexts */
    for ( i=0; i<n_local_true; i++ ) {
      ierr = ISGetSize(osm->is[i],&m); CHKERRQ(ierr);
      ierr = VecCreateSeq(MPI_COMM_SELF,m,&osm->x[i]); CHKERRQ(ierr);
      ierr = VecDuplicate(osm->x[i],&osm->y[i]); CHKERRQ(ierr);
      ierr = ISCreateStrideSeq(MPI_COMM_SELF,m,0,1,&isl); CHKERRQ(ierr);
      ierr = VecScatterCreate(pc->vec,osm->is[i],osm->x[i],isl,&osm->scat[i]); CHKERRQ(ierr);
      ierr = ISDestroy(isl); CHKERRQ(ierr);
    }
    for ( i=n_local_true; i<n_local; i++ ) {
      ierr = VecCreateSeq(MPI_COMM_SELF,0,&osm->x[i]); CHKERRQ(ierr);
      ierr = VecDuplicate(osm->x[i],&osm->y[i]); CHKERRQ(ierr);
      ierr = ISCreateStrideSeq(MPI_COMM_SELF,0,0,1,&isl); CHKERRQ(ierr);
      ierr = VecScatterCreate(pc->vec,isl,osm->x[i],isl,&osm->scat[i]); CHKERRQ(ierr);
      ierr = ISDestroy(isl); CHKERRQ(ierr);   
      ierr = MatCreateSeqAIJ(MPI_COMM_SELF,0,0,0,0,&osm->pmat[i]); CHKERRQ(ierr); 
    }

    /* create the local solvers */
    for ( i=0; i<n_local; i++ ) {
      ierr = SLESCreate(MPI_COMM_SELF,&sles); CHKERRQ(ierr);
      PLogObjectParent(pc,sles);
      ierr = SLESGetKSP(sles,&subksp); CHKERRQ(ierr);
      ierr = KSPSetMethod(subksp,KSPPREONLY); CHKERRQ(ierr);
      ierr = SLESGetPC(sles,&subpc); CHKERRQ(ierr);
      ierr = PCSetMethod(subpc,PCLU); CHKERRQ(ierr);
      ierr = SLESSetOptionsPrefix(sles,"-sub_"); CHKERRQ(ierr);
      ierr = SLESSetFromOptions(sles); CHKERRQ(ierr);
      osm->sles[i] = sles;
    }
    scall = MAT_INITIAL_MATRIX;
  }

  /* extract out the submatrices */
  ierr  = MatGetSubMatrices(pc->pmat,osm->n_local_true,osm->is,osm->is,scall,&osm->pmat); 
          CHKERRQ(ierr);

  /* loop over subdomains extracting them and putting them into local sles */
  for ( i=0; i<n_local_true; i++ ) {
    PLogObjectParent(pc,osm->pmat[i]);
    ierr = SLESSetOperators(osm->sles[i],osm->pmat[i],osm->pmat[i],pc->flag);CHKERRQ(ierr);
  }
  return 0;
}

static int PCApply_ASM(PC pc,Vec x,Vec y)
{
  PC_ASM *osm = (PC_ASM *) pc->data;
  int    i,n_local = osm->n_local,ierr,its;

  for ( i=0; i<n_local; i++ ) {
    ierr = VecScatterBegin(x,osm->x[i],INSERT_VALUES,SCATTER_ALL,osm->scat[i]);CHKERRQ(ierr);
  }
  for ( i=0; i<n_local; i++ ) {
    ierr = VecScatterEnd(x,osm->x[i],INSERT_VALUES,SCATTER_ALL,osm->scat[i]);CHKERRQ(ierr);
    ierr = SLESSolve(osm->sles[i],osm->x[i],osm->y[i],&its);CHKERRQ(ierr); 
  }
  for ( i=0; i<n_local; i++ ) {
    ierr = VecScatterBegin(osm->y[i],y,INSERT_VALUES,SCATTER_REVERSE,osm->scat[i]);
           CHKERRQ(ierr);
  }
  for ( i=0; i<n_local; i++ ) {
    ierr = VecScatterEnd(osm->y[i],y,INSERT_VALUES,SCATTER_REVERSE,osm->scat[i]);CHKERRQ(ierr);
  }
  return 0;
}

static int PCDestroy_ASM(PetscObject obj)
{
  PC     pc = (PC) obj;
  PC_ASM *osm = (PC_ASM *) pc->data;
  int    i,n_local = osm->n_local,ierr;

  for ( i=0; i<n_local; i++ ) {
    ierr = VecScatterDestroy(osm->scat[i]);
    ierr = VecDestroy(osm->x[i]);
    ierr = VecDestroy(osm->y[i]);
    ierr = MatDestroy(osm->pmat[i]);
    ierr = SLESDestroy(osm->sles[i]);
  }
  PETSCFREE(osm->sles);
  PETSCFREE(osm->scat);
  PETSCFREE(osm->x);
  PETSCFREE(osm->pmat);
  PETSCFREE(osm);
  return 0;
}

int PCCreate_ASM(PC pc)
{
  PC_ASM *osm = PETSCNEW(PC_ASM); CHKPTRQ(osm);

  PetscZero(osm,sizeof(PC_ASM)); 
  osm->n            = PETSC_DECIDE;
  osm->n_local_true = PETSC_DECIDE;

  pc->apply         = PCApply_ASM;
  pc->setup         = PCSetUp_ASM;
  pc->destroy       = PCDestroy_ASM;
  pc->type          = PCASM;
  pc->data          = (void *) osm;
  pc->view          = 0;
  return 0;
}

/*@

     PCASMSetSubdomains - Sets the subdomains for this processor for the 
           additive Schwarz preconditioner. Note: all or no processors in the
           pc must call this. 

  Input Parameters:
.   pc - the preconditioner context
.   n - the number of subdomains for this processor
.   is - the index sets that define the subdomains for this processor

@*/
int PCASMSetSubDomains(PC pc, int n, IS *is)
{
  PC_ASM *osm;
  PETSCVALIDHEADERSPECIFIC(pc,PC_COOKIE);
  if (pc->type != PCASM) return 0;  

  osm               = (PC_ASM *) pc->data;
  osm->n_local_true = n;
  osm->is           = is;
  return 0;
}
