#ifndef lint
static char vcid[] = "$Id: asm.c,v 1.12 1996/01/19 22:58:03 balay Exp bsmith $";
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
  int        overlap;                 /* overlap requested by user */
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
  int                 start, start_val, end_val, size, sz;
  MatGetSubMatrixCall scall = MAT_REUSE_MATRIX;
  IS                  isl;
  SLES                sles;
  KSP                 subksp;
  PC                  subpc;
  char                *prefix;

  if (pc->setupcalled == 0) {
    if (osm->n == PETSC_DECIDE && osm->n_local_true == PETSC_DECIDE) { 
      /* no subdomains given, use one per processor */
      osm->n_local_true = osm->n_local = 1;
      MPI_Comm_size(pc->comm,&size);
      osm->n = size;
    } else if (osm->n == PETSC_DECIDE) { /* determine global number of subdomains */
      MPI_Allreduce(&osm->n_local_true,&osm->n,1,MPI_INT,MPI_SUM,pc->comm);
      MPI_Allreduce(&osm->n_local_true,&osm->n_local,1,MPI_INT,MPI_MAX,pc->comm);
    }
    n_local      = osm->n_local;
    n_local_true = osm->n_local_true;  
    if( !osm->is){ /* build the index sets */
      osm->is    = (IS *) PetscMalloc( n_local_true*sizeof(IS **) ); CHKPTRQ(osm->is);
      MatGetOwnershipRange(pc->pmat,&start_val,&end_val);
      sz    = end_val - start_val;
      start = start_val;
      for ( i=0; i<n_local_true; i++){
        size     = sz/n_local_true + (( sz % n_local_true) > i);
        ierr     = ISCreateStrideSeq(MPI_COMM_SELF,size,start,1,&isl); CHKERRQ(ierr);
        start    += size;
        osm->is[i] = isl;
      }
    }

    osm->sles = (SLES *) PetscMalloc(n_local*sizeof(SLES **)); CHKPTRQ(osm->sles);
    osm->scat = (VecScatter *) PetscMalloc(n_local*sizeof(VecScatter **));CHKPTRQ(osm->scat);
    osm->x    = (Vec *) PetscMalloc(2*n_local*sizeof(Vec **)); CHKPTRQ(osm->x);
    osm->y    = osm->x + n_local;

    /*  Extend the "overlapping" regions by a number of steps  */
    ierr = MatIncreaseOverlap(pc->pmat,n_local_true,osm->is,osm->overlap); CHKERRQ(ierr);

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
      ierr = MatCreateSeqAIJ(MPI_COMM_SELF,0,PETSC_NULL,0,PETSC_NULL,&osm->pmat[i]);CHKERRQ(ierr); 
    }

    /* 
       Create the local solvers, we create SLES objects even for "fake" local subdomains
       simply to simplify the loops in the code for the actual solves
    */
    for ( i=0; i<n_local; i++ ) {
      ierr = SLESCreate(MPI_COMM_SELF,&sles); CHKERRQ(ierr);
      PLogObjectParent(pc,sles);
      ierr = SLESGetKSP(sles,&subksp); CHKERRQ(ierr);
      ierr = KSPSetType(subksp,KSPPREONLY); CHKERRQ(ierr);
      ierr = SLESGetPC(sles,&subpc); CHKERRQ(ierr);
      ierr = PCSetType(subpc,PCLU); CHKERRQ(ierr);
      ierr = PCGetOptionsPrefix(pc,&prefix); CHKERRQ(ierr);
      ierr = SLESSetOptionsPrefix(sles,prefix); CHKERRQ(ierr);
      ierr = SLESAppendOptionsPrefix(sles,"sub_"); CHKERRQ(ierr);
      ierr = SLESSetFromOptions(sles); CHKERRQ(ierr);
      osm->sles[i] = sles;
    }
    scall = MAT_INITIAL_MATRIX;
  }

  /* extract out the submatrices */
  ierr = MatGetSubMatrices(pc->pmat,osm->n_local_true,osm->is,osm->is,scall,&osm->pmat);
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

static int PCPrintHelp_ASM(PC pc,char *p)
{
  MPIU_printf(pc->comm," Options for PCASM preconditioner:\n");
  MPIU_printf(pc->comm," %spc_asm_blocks blks: subdomain blocks per processor\n",p);
  MPIU_printf(pc->comm, " %spc_asm_overlap ovl: amount of overlap between subdomains\n",p); 
  MPIU_printf(pc->comm," %ssub : prefix to control options for individual blocks.\
 Add before the \n      usual KSP and PC option names (i.e., %ssub_ksp_type\
 <meth>)\n",p,p);
  return 0;
}

static int PCSetFromOptions_ASM(PC pc)
{
  int  blocks,flg, ovl,ierr;

  ierr = OptionsGetInt(pc->prefix,"-pc_asm_blocks",&blocks,&flg); CHKERRQ(ierr);
  if (flg) {ierr = PCASMSetSubdomains(pc,blocks,PETSC_NULL); CHKERRQ(ierr); }
  ierr = OptionsGetInt(pc->prefix,"-pc_asm_overlap", &ovl, &flg); CHKERRQ(ierr);
  if (flg) { ierr = PCASMSetOverlap( pc, ovl); CHKERRQ(ierr); }

  return 0;
}

int PCCreate_ASM(PC pc)
{
  PC_ASM *osm = PetscNew(PC_ASM); CHKPTRQ(osm);

  PetscMemzero(osm,sizeof(PC_ASM)); 
  osm->n            = PETSC_DECIDE;
  osm->n_local_true = PETSC_DECIDE;
  osm->overlap      = 0;

  pc->apply         = PCApply_ASM;
  pc->setup         = PCSetUp_ASM;
  pc->destroy       = PCDestroy_ASM;
  pc->type          = PCASM;
  pc->printhelp     = PCPrintHelp_ASM;
  pc->setfrom       = PCSetFromOptions_ASM;
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
         or PETSC_NULL for PETSc to determine
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

     PCASMSetOverlap - Sets the overlap between a pair of subdomains for the
           additive Schwarz preconditioner. Note: all or no processors in the
           pc must call this. 

  Input Parameters:
.   pc  - the preconditioner context
.   ovl - the amount of overlap 
          >= 0
@*/
int PCASMSetOverlap(PC pc, int ovl)
{
  PC_ASM *osm;
  PETSCVALIDHEADERSPECIFIC(pc,PC_COOKIE);
  if (pc->type != PCASM) return 0;  
  if (ovl < 0 ) SETERRQ(1,"PCASMSetOverlap: Negetive overlap value used");

  osm               = (PC_ASM *) pc->data;
  osm->overlap      = ovl;
  return 0;
}

/*@
     PCASMCreateSubdomains2D - Creates the index set for overlapping Schwarz 
       preconditioner for a two dimensional problem on a regular grid.

  Presently this is only good for sequential preconditioners.

  Input Parameters:
.   m, n - the number of mesh points in x and y direction
.   M, N - the number of subdomains in the x and y direction
.   dof - degrees of freedom per node
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
