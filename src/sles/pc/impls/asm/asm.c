#ifndef lint
static char vcid[] = "$Id: asm.c,v 1.4 1995/11/09 22:28:28 bsmith Exp curfman $";
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
#include "pcimpl.h"     /*I "pc.h" I*/
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
      osm->is    = (IS *) PetscMalloc( sizeof(IS **) ); CHKPTRQ(osm->is);
      osm->is[0] = isl;
    }

    if (osm->n == PETSC_DECIDE) { /* determine global number of subdomains */
      MPI_Allreduce(&osm->n_local_true,&osm->n,1,MPI_INT,MPI_SUM,pc->comm);
      MPI_Allreduce(&osm->n_local_true,&osm->n_local,1,MPI_INT,MPI_MAX,pc->comm);
    }

    n_local      = osm->n_local;
    n_local_true = osm->n_local_true;

    osm->sles = (SLES *) PetscMalloc(n_local*sizeof(SLES **)); CHKPTRQ(osm->sles);
    osm->scat = (VecScatter *) PetscMalloc(n_local*sizeof(VecScatter **)); CHKPTRQ(osm->scat);
    osm->x    = (Vec *) PetscMalloc(2*n_local*sizeof(Vec **)); CHKPTRQ(osm->x);
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
      ierr = MatCreateSeqAIJ(MPI_COMM_SELF,0,PetscNull,0,PetscNull,&osm->pmat[i]); CHKERRQ(ierr); 
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
  Scalar zero = 0.0;

  for ( i=0; i<n_local; i++ ) {
    ierr = VecScatterBegin(x,osm->x[i],INSERT_VALUES,SCATTER_ALL,osm->scat[i]);CHKERRQ(ierr);
  }
  ierr = VecSet(&zero,y); CHKERRQ(ierr);
  for ( i=0; i<n_local; i++ ) {
    ierr = VecScatterEnd(x,osm->x[i],INSERT_VALUES,SCATTER_ALL,osm->scat[i]);CHKERRQ(ierr);
    ierr = SLESSolve(osm->sles[i],osm->x[i],osm->y[i],&its);CHKERRQ(ierr); 
    ierr = VecScatterBegin(osm->y[i],y,ADD_VALUES,(ScatterMode)(SCATTER_ALL|SCATTER_REVERSE),
                           osm->scat[i]);CHKERRQ(ierr);
  }
  for ( i=0; i<n_local; i++ ) {
    ierr = VecScatterEnd(osm->y[i],y,ADD_VALUES,(ScatterMode)(SCATTER_ALL|SCATTER_REVERSE),
                         osm->scat[i]);CHKERRQ(ierr);
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
  PetscFree(osm->sles);
  PetscFree(osm->scat);
  PetscFree(osm->x);
  PetscFree(osm->pmat);
  PetscFree(osm);
  return 0;
}

int PCCreate_ASM(PC pc)
{
  PC_ASM *osm = PetscNew(PC_ASM); CHKPTRQ(osm);

  PetscMemzero(osm,sizeof(PC_ASM)); 
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
int PCASMSetSubdomains(PC pc, int n, IS *is)
{
  PC_ASM *osm;
  PETSCVALIDHEADERSPECIFIC(pc,PC_COOKIE);
  if (pc->type != PCASM) return 0;  

  osm               = (PC_ASM *) pc->data;
  osm->n_local_true = n;
  osm->is           = is;
  return 0;
}

/*@
     PCASMCreateSubdomains2D - Creates the index set for overlapping Schwarz 
       preconditioner for a two dimensional problem on a regular grid.

  Presently this is only good for sequential preconditioners.

  Input Parameters:
.   m, n - the number of mesh points in x and y direction
.   M, N - the number of subdomains in the x and y direction
.   dof - degrees of freedome per node
.   overlap - overlap in mesh lines

  Output Paramters:
.   Nsub - the number of subdomains created
.   is - the array of index sets defining the subdomains

@*/
int PCASMCreateSubdomains2D(int m,int n,int M,int N,int dof,int overlap,int *Nsub,IS **is)
{
  int i,j, height,width,ystart,xstart,yleft,yright,xleft,xright,loc_outter;
  int nidx,*idx,loc,ii,jj,ierr,count;

  if (dof != 1) SETERRQ(PETSC_ERR_SUP,"PCASMCreateSubdomains2D");

  *Nsub = N*M;
  *is = (IS *) PetscMalloc( (*Nsub)*sizeof(IS **) ); CHKPTRQ(is);
  ystart = 0;
  loc_outter = 0;
  for ( i=0; i<N; i++ ) {
    height = n/N + ((n % N) > i); /* height of subdomain */
    if (height < 2) SETERRA(1,"Too many M subdomains for m mesh");
    yleft  = ystart - overlap; if (yleft < 0) yleft = 0;
    yright = ystart + height + overlap; if (yright > n) yright = n;
    xstart = 0;
    for ( j=0; j<M; j++ ) {
      width = m/M + ((m % M) > j); /* width of subdomain */
      if (width < 2) SETERRA(1,"Too many M subdomains for m mesh");
      xleft  = xstart - overlap; if (xleft < 0) xleft = 0;
      xright = xstart + width + overlap; if (xright > m) xright = m;
      /*            
       printf("subdomain %d %d xstart %d end %d ystart %d end %d\n",i,j,xleft,xright,
              yleft,yright);
      */
      nidx   = (xright - xleft)*(yright - yleft);
      idx    = (int *) PetscMalloc( nidx*sizeof(int) ); CHKPTRQ(idx);
      loc    = 0;
      for ( ii=yleft; ii<yright; ii++ ) {
        count = m*ii + xleft;
        for ( jj=xleft; jj<xright; jj++ ) {
          idx[loc++] = count++;
        }
      }
      ierr = ISCreateSeq(MPI_COMM_SELF,nidx,idx,(*is)+loc_outter++); CHKERRQ(ierr);
      PetscFree(idx);
      /* ISView((*is)[loc_outter-1],0); */
      xstart += width;
    }
    ystart += height;
  }
  return 0;
}
