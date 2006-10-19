#define PETSCKSP_DLL

/*
  This file defines a "solve the problem redundantly on each subgroup of processor" preconditioner.
*/
#include "private/pcimpl.h"     /*I "petscpc.h" I*/
#include "petscksp.h"

typedef struct {
  PC         pc;                   /* actual preconditioner used on each processor */
  Vec        xsub,ysub;            /* vectors of a subcommunicator to hold parallel vectors of pc->comm */
  Vec        xdup,ydup;            /* parallel vector that congregates xsub or ysub facilitating vector scattering */
  Mat        pmats;                /* matrix and optional preconditioner matrix belong to a subcommunicator */
  VecScatter scatterin,scatterout; /* scatter used to move all values to each processor group (subcommunicator) */
  PetscTruth useparallelmat;
  MPI_Comm   subcomm;              /* processors belong to a subcommunicator implement a PC in parallel */
  MPI_Comm   dupcomm;              /* processors belong to pc->comm with their rank remapped in the way 
                                      that vector xdup/ydup has contiguous rank while appending xsub/ysub along their colors */
  PetscInt   nsubcomm;             /* num of subcommunicators, which equals the num of redundant matrix systems */
  PetscInt   color;                /* color of processors in a subcommunicator */
} PC_Redundant;

#undef __FUNCT__  
#define __FUNCT__ "PCView_Redundant"
static PetscErrorCode PCView_Redundant(PC pc,PetscViewer viewer)
{
  PC_Redundant   *red = (PC_Redundant*)pc->data;
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  PetscTruth     iascii,isstring;
  PetscViewer    sviewer;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(pc->comm,&rank);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_STRING,&isstring);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  Redundant solver preconditioner: Actual PC follows\n");CHKERRQ(ierr);
    ierr = PetscViewerGetSingleton(viewer,&sviewer);CHKERRQ(ierr);
    if (!rank) {
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = PCView(red->pc,sviewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
    ierr = PetscViewerRestoreSingleton(viewer,&sviewer);CHKERRQ(ierr);
  } else if (isstring) {
    ierr = PetscViewerStringSPrintf(viewer," Redundant solver preconditioner");CHKERRQ(ierr);
    ierr = PetscViewerGetSingleton(viewer,&sviewer);CHKERRQ(ierr);
    if (!rank) {
      ierr = PCView(red->pc,sviewer);CHKERRQ(ierr);
    }
    ierr = PetscViewerRestoreSingleton(viewer,&sviewer);CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported for PC redundant",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#include "src/mat/matimpl.h"        /*I "petscmat.h" I*/
#include "private/vecimpl.h" 
#include "src/mat/impls/aij/mpi/mpiaij.h"   /*I "petscmat.h" I*/
#include "src/mat/impls/aij/seq/aij.h"      /*I "petscmat.h" I*/

typedef struct { /* used by MatGetRedundantMatrix() for reusing matredundant */
  PetscInt       nzlocal,nsends,nrecvs;
  PetscInt       *send_rank,*sbuf_nz,*sbuf_j,**rbuf_j;
  PetscScalar    *sbuf_a,**rbuf_a;
  PetscErrorCode (*MatDestroy)(Mat);
} Mat_Redundant;

#undef __FUNCT__  
#define __FUNCT__ "PetscObjectContainerDestroy_MatRedundant"
PetscErrorCode PetscObjectContainerDestroy_MatRedundant(void *ptr)
{
  PetscErrorCode       ierr;
  Mat_Redundant        *redund=(Mat_Redundant*)ptr;
  PetscInt             i;

  PetscFunctionBegin;
  ierr = PetscFree(redund->send_rank);CHKERRQ(ierr);
  ierr = PetscFree(redund->sbuf_j);CHKERRQ(ierr);
  ierr = PetscFree(redund->sbuf_a);CHKERRQ(ierr);
  for (i=0; i<redund->nrecvs; i++){
    ierr = PetscFree(redund->rbuf_j[i]);CHKERRQ(ierr);
    ierr = PetscFree(redund->rbuf_a[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree3(redund->sbuf_nz,redund->rbuf_j,redund->rbuf_a);CHKERRQ(ierr);
  ierr = PetscFree(redund);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_MatRedundant"
PetscErrorCode MatDestroy_MatRedundant(Mat A)
{
  PetscErrorCode       ierr;
  PetscObjectContainer container;
  Mat_Redundant        *redund=PETSC_NULL;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)A,"Mat_Redundant",(PetscObject *)&container);CHKERRQ(ierr);
  if (container) {
    ierr = PetscObjectContainerGetPointer(container,(void **)&redund);CHKERRQ(ierr);
  } else {
    SETERRQ(PETSC_ERR_PLIB,"Container does not exit");
  }
  A->ops->destroy = redund->MatDestroy;
  ierr = PetscObjectCompose((PetscObject)A,"Mat_Redundant",0);CHKERRQ(ierr);
  ierr = (*A->ops->destroy)(A);CHKERRQ(ierr);
  ierr = PetscObjectContainerDestroy(container);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetRedundantMatrix"
PetscErrorCode MatGetRedundantMatrix_AIJ(Mat mat,PetscInt nsubcomm,MPI_Comm subcomm,PetscInt mlocal_sub,MatReuse reuse,Mat *matredundant)
{
  PetscMPIInt    rank,size; 
  MPI_Comm       comm=mat->comm;
  PetscErrorCode ierr;
  PetscInt       nsends,nrecvs,i,prid=100,rownz_max;
  PetscMPIInt    *send_rank,*recv_rank;
  PetscInt       *rowrange=mat->rmap.range;
  Mat_MPIAIJ     *aij = (Mat_MPIAIJ*)mat->data;
  Mat            A=aij->A,B=aij->B,C=*matredundant;
  Mat_SeqAIJ     *a=(Mat_SeqAIJ*)A->data,*b=(Mat_SeqAIJ*)B->data;
  PetscScalar    *sbuf_a;
  PetscInt       nzlocal=a->nz+b->nz;
  PetscInt       j,cstart=mat->cmap.rstart,cend=mat->cmap.rend,row,nzA,nzB,ncols,*cworkA,*cworkB;
  PetscInt       rstart=mat->rmap.rstart,rend=mat->rmap.rend,*bmap=aij->garray,M,N;
  PetscInt       *cols,ctmp,lwrite,*rptr,l,*sbuf_j;
  PetscScalar    *vals,*aworkA,*aworkB;
  PetscMPIInt    tag1,tag2,tag3,imdex;
  MPI_Request    *s_waits1,*s_waits2,*s_waits3,*r_waits1,*r_waits2,*r_waits3;
  MPI_Status     recv_status,*send_status; 
  PetscInt       *sbuf_nz,*rbuf_nz,count;
  PetscInt       **rbuf_j;
  PetscScalar    **rbuf_a;
  Mat_Redundant  *redund=PETSC_NULL;
  PetscObjectContainer container;

  PetscFunctionBegin;
  ierr = PetscOptionsGetInt(PETSC_NULL,"-prid",&prid,PETSC_NULL);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);

  if (reuse == MAT_REUSE_MATRIX) {
    ierr = MatGetSize(C,&M,&N);CHKERRQ(ierr);
    if (M != N || M != mat->rmap.N) SETERRQ(PETSC_ERR_ARG_SIZ,"Cannot reuse matrix. Wrong global size");   
    ierr = MatGetLocalSize(C,&M,&N);CHKERRQ(ierr);    
    if (M != N || M != mlocal_sub) SETERRQ(PETSC_ERR_ARG_SIZ,"Cannot reuse matrix. Wrong local size");
    ierr = PetscObjectQuery((PetscObject)C,"Mat_Redundant",(PetscObject *)&container);CHKERRQ(ierr);
    if (container) {
      ierr = PetscObjectContainerGetPointer(container,(void **)&redund);CHKERRQ(ierr);
    } else {
      SETERRQ(PETSC_ERR_PLIB,"Container does not exit");
    }
    if (nzlocal != redund->nzlocal) SETERRQ(PETSC_ERR_ARG_SIZ,"Cannot reuse matrix. Wrong nzlocal");

    nsends    = redund->nsends;
    nrecvs    = redund->nrecvs;
    send_rank = redund->send_rank; recv_rank = send_rank + size; 
    sbuf_nz   = redund->sbuf_nz;     rbuf_nz = sbuf_nz + nsends;
    sbuf_j    = redund->sbuf_j;
    sbuf_a    = redund->sbuf_a;
    rbuf_j    = redund->rbuf_j;
    rbuf_a    = redund->rbuf_a;
  } 

  if (reuse == MAT_INITIAL_MATRIX){
    PetscMPIInt  subrank,subsize;
    PetscInt     nleftover,np_subcomm; 
    /* get the destination processors' id send_rank, nsends and nrecvs */
    ierr = MPI_Comm_rank(subcomm,&subrank);CHKERRQ(ierr);
    ierr = MPI_Comm_size(subcomm,&subsize);CHKERRQ(ierr);
    ierr = PetscMalloc((2*size+1)*sizeof(PetscMPIInt),&send_rank);
    recv_rank = send_rank + size;
    np_subcomm = size/nsubcomm;
    nleftover  = size - nsubcomm*np_subcomm;
    nsends = 0; nrecvs = 0;
    for (i=0; i<size; i++){ /* i=rank*/
      if (subrank == i/nsubcomm && rank != i){ /* my_subrank == other's subrank */
        send_rank[nsends] = i; nsends++;
        recv_rank[nrecvs++] = i;
      } 
    }
    if (rank >= size - nleftover){/* this proc is a leftover processor */
      i = size-nleftover-1; 
      j = 0;
      while (j < nsubcomm - nleftover){
        send_rank[nsends++] = i;
        i--; j++;
      }
    }

    if (nleftover && subsize == size/nsubcomm && subrank==subsize-1){ /* this proc recvs from leftover processors */
      for (i=0; i<nleftover; i++){
        recv_rank[nrecvs++] = size-nleftover+i;
      }
    } 

    /* allocate sbuf_j, sbuf_a */
    i = nzlocal + rowrange[rank+1] - rowrange[rank] + 2;
    ierr = PetscMalloc(i*sizeof(PetscInt),&sbuf_j);CHKERRQ(ierr);
    ierr = PetscMalloc((nzlocal+1)*sizeof(PetscScalar),&sbuf_a);CHKERRQ(ierr);
  } /* endof if (reuse == MAT_INITIAL_MATRIX) */
 
  /* copy mat's local entries into the buffers */
  if (reuse == MAT_INITIAL_MATRIX){
    rownz_max = 0;
    rptr = sbuf_j;
    cols = sbuf_j + rend-rstart + 1;
    vals = sbuf_a;
    rptr[0] = 0;
    for (i=0; i<rend-rstart; i++){
      row = i + rstart;
      nzA    = a->i[i+1] - a->i[i]; nzB = b->i[i+1] - b->i[i];
      ncols  = nzA + nzB;
      cworkA = a->j + a->i[i]; cworkB = b->j + b->i[i]; 
      aworkA = a->a + a->i[i]; aworkB = b->a + b->i[i];
      /* load the column indices for this row into cols */
      lwrite = 0;
      for (l=0; l<nzB; l++) {
        if ((ctmp = bmap[cworkB[l]]) < cstart){
          vals[lwrite]   = aworkB[l];
          cols[lwrite++] = ctmp;
        }
      }
      for (l=0; l<nzA; l++){
        vals[lwrite]   = aworkA[l];
        cols[lwrite++] = cstart + cworkA[l];
      }
      for (l=0; l<nzB; l++) {
        if ((ctmp = bmap[cworkB[l]]) >= cend){
          vals[lwrite]   = aworkB[l];
          cols[lwrite++] = ctmp;
        }
      }
      vals += ncols;
      cols += ncols;
      rptr[i+1] = rptr[i] + ncols;  
      if (rownz_max < ncols) rownz_max = ncols;
    } 
    if (rptr[rend-rstart] != a->nz + b->nz) SETERRQ4(1, "rptr[%d] %d != %d + %d",rend-rstart,rptr[rend-rstart+1],a->nz,b->nz);
  } else { /* only copy matrix values into sbuf_a */
    rptr = sbuf_j;
    vals = sbuf_a;
    rptr[0] = 0;
    for (i=0; i<rend-rstart; i++){
      row = i + rstart;
      nzA    = a->i[i+1] - a->i[i]; nzB = b->i[i+1] - b->i[i];
      ncols  = nzA + nzB;
      cworkA = a->j + a->i[i]; cworkB = b->j + b->i[i]; 
      aworkA = a->a + a->i[i]; aworkB = b->a + b->i[i];
      lwrite = 0;
      for (l=0; l<nzB; l++) {
        if ((ctmp = bmap[cworkB[l]]) < cstart) vals[lwrite++] = aworkB[l];      
      }
      for (l=0; l<nzA; l++) vals[lwrite++] = aworkA[l];  
      for (l=0; l<nzB; l++) {
        if ((ctmp = bmap[cworkB[l]]) >= cend) vals[lwrite++] = aworkB[l];
      }
      vals += ncols;
      rptr[i+1] = rptr[i] + ncols; 
    } 
  } /* endof if (reuse == MAT_INITIAL_MATRIX) */

  /* send nzlocal to others, and recv other's nzlocal */
  /*--------------------------------------------------*/
  if (reuse == MAT_INITIAL_MATRIX){
    ierr = PetscMalloc2(3*(nsends + nrecvs)+1,MPI_Request,&s_waits3,nsends+1,MPI_Status,&send_status);CHKERRQ(ierr);
    s_waits2 = s_waits3 + nsends;
    s_waits1 = s_waits2 + nsends;
    r_waits1 = s_waits1 + nsends;
    r_waits2 = r_waits1 + nrecvs;
    r_waits3 = r_waits2 + nrecvs;
  } else {
    ierr = PetscMalloc2(nsends + nrecvs +1,MPI_Request,&s_waits3,nsends+1,MPI_Status,&send_status);CHKERRQ(ierr);
    r_waits3 = s_waits3 + nsends;
  }

  ierr = PetscObjectGetNewTag((PetscObject)mat,&tag3);CHKERRQ(ierr); 
  if (reuse == MAT_INITIAL_MATRIX){
    /* get new tags to keep the communication clean */
    ierr = PetscObjectGetNewTag((PetscObject)mat,&tag1);CHKERRQ(ierr); 
    ierr = PetscObjectGetNewTag((PetscObject)mat,&tag2);CHKERRQ(ierr);
    ierr = PetscMalloc3(nsends+nrecvs+1,PetscInt,&sbuf_nz,nrecvs,PetscInt*,&rbuf_j,nrecvs,PetscScalar*,&rbuf_a);CHKERRQ(ierr);
    rbuf_nz = sbuf_nz + nsends;
    
    /* post receives of other's nzlocal */
    for (i=0; i<nrecvs; i++){
      ierr = MPI_Irecv(rbuf_nz+i,1,MPIU_INT,MPI_ANY_SOURCE,tag1,comm,r_waits1+i);CHKERRQ(ierr);
    }  
    /* send nzlocal to others */
    for (i=0; i<nsends; i++){
      sbuf_nz[i] = nzlocal;
      ierr = MPI_Isend(sbuf_nz+i,1,MPIU_INT,send_rank[i],tag1,comm,s_waits1+i);CHKERRQ(ierr);
    }
    /* wait on receives of nzlocal; allocate space for rbuf_j, rbuf_a */
    count = nrecvs;
    while (count) {
      ierr = MPI_Waitany(nrecvs,r_waits1,&imdex,&recv_status);CHKERRQ(ierr);
      recv_rank[imdex] = recv_status.MPI_SOURCE;
      /* allocate rbuf_a and rbuf_j; then post receives of rbuf_j */
      ierr = PetscMalloc((rbuf_nz[imdex]+1)*sizeof(PetscScalar),&rbuf_a[imdex]);CHKERRQ(ierr);

      i = rowrange[recv_status.MPI_SOURCE+1] - rowrange[recv_status.MPI_SOURCE]; /* number of expected mat->i */
      rbuf_nz[imdex] += i + 2;
      ierr = PetscMalloc(rbuf_nz[imdex]*sizeof(PetscInt),&rbuf_j[imdex]);CHKERRQ(ierr);
      ierr = MPI_Irecv(rbuf_j[imdex],rbuf_nz[imdex],MPIU_INT,recv_status.MPI_SOURCE,tag2,comm,r_waits2+imdex);CHKERRQ(ierr);
      count--;
    }
    /* wait on sends of nzlocal */
    if (nsends) {ierr = MPI_Waitall(nsends,s_waits1,send_status);CHKERRQ(ierr);}
    /* send mat->i,j to others, and recv from other's */
    /*------------------------------------------------*/
    for (i=0; i<nsends; i++){
      j = nzlocal + rowrange[rank+1] - rowrange[rank] + 1;
      ierr = MPI_Isend(sbuf_j,j,MPIU_INT,send_rank[i],tag2,comm,s_waits2+i);CHKERRQ(ierr);
    }
    /* wait on receives of mat->i,j */
    /*------------------------------*/
    count = nrecvs;
    while (count) {
      ierr = MPI_Waitany(nrecvs,r_waits2,&imdex,&recv_status);CHKERRQ(ierr);
      if (recv_rank[imdex] != recv_status.MPI_SOURCE) SETERRQ2(1, "recv_rank %d != MPI_SOURCE %d",recv_rank[imdex],recv_status.MPI_SOURCE);
      count--;
    }
    /* wait on sends of mat->i,j */
    /*---------------------------*/
    if (nsends) {
      ierr = MPI_Waitall(nsends,s_waits2,send_status);CHKERRQ(ierr);
    }
  } /* endof if (reuse == MAT_INITIAL_MATRIX) */

  /* post receives, send and receive mat->a */
  /*----------------------------------------*/
  for (imdex=0; imdex<nrecvs; imdex++) {
    ierr = MPI_Irecv(rbuf_a[imdex],rbuf_nz[imdex],MPIU_SCALAR,recv_rank[imdex],tag3,comm,r_waits3+imdex);CHKERRQ(ierr);
  }
  for (i=0; i<nsends; i++){
    ierr = MPI_Isend(sbuf_a,nzlocal,MPIU_SCALAR,send_rank[i],tag3,comm,s_waits3+i);CHKERRQ(ierr);
  }
  count = nrecvs;
  while (count) {
    ierr = MPI_Waitany(nrecvs,r_waits3,&imdex,&recv_status);CHKERRQ(ierr);
    if (recv_rank[imdex] != recv_status.MPI_SOURCE) SETERRQ2(1, "recv_rank %d != MPI_SOURCE %d",recv_rank[imdex],recv_status.MPI_SOURCE);
    count--;
  }
  if (nsends) {
    ierr = MPI_Waitall(nsends,s_waits3,send_status);CHKERRQ(ierr);
  }

  ierr = PetscFree2(s_waits3,send_status);CHKERRQ(ierr);
  
  /* create redundant matrix */
  /*-------------------------*/
  if (reuse == MAT_INITIAL_MATRIX){
    /* compute rownz_max for preallocation */
    for (imdex=0; imdex<nrecvs; imdex++){
      j = rowrange[recv_rank[imdex]+1] - rowrange[recv_rank[imdex]];
      rptr = rbuf_j[imdex];
      for (i=0; i<j; i++){
        ncols = rptr[i+1] - rptr[i];
        if (rownz_max < ncols) rownz_max = ncols;
      }
    }
    
    ierr = MatCreate(subcomm,&C);CHKERRQ(ierr);
    ierr = MatSetSizes(C,mlocal_sub,mlocal_sub,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
    ierr = MatSetFromOptions(C);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(C,rownz_max,PETSC_NULL);CHKERRQ(ierr);
    ierr = MatMPIAIJSetPreallocation(C,rownz_max,PETSC_NULL,rownz_max,PETSC_NULL);CHKERRQ(ierr);
  } else {
    C = *matredundant;
  }

  /* insert local matrix entries */
  rptr = sbuf_j;
  cols = sbuf_j + rend-rstart + 1;
  vals = sbuf_a;
  for (i=0; i<rend-rstart; i++){
    row   = i + rstart;
    ncols = rptr[i+1] - rptr[i];
    ierr = MatSetValues(C,1,&row,ncols,cols,vals,INSERT_VALUES);CHKERRQ(ierr);
    vals += ncols;
    cols += ncols;
  }
  /* insert received matrix entries */
  for (imdex=0; imdex<nrecvs; imdex++){    
    rstart = rowrange[recv_rank[imdex]];
    rend   = rowrange[recv_rank[imdex]+1];
    rptr = rbuf_j[imdex];
    cols = rbuf_j[imdex] + rend-rstart + 1;
    vals = rbuf_a[imdex];
    for (i=0; i<rend-rstart; i++){
      row   = i + rstart;
      ncols = rptr[i+1] - rptr[i];
      ierr = MatSetValues(C,1,&row,ncols,cols,vals,INSERT_VALUES);CHKERRQ(ierr);
      vals += ncols;
      cols += ncols;
    }
  }
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatGetSize(C,&M,&N);CHKERRQ(ierr);
  if (M != mat->rmap.N || N != mat->cmap.N) SETERRQ2(PETSC_ERR_ARG_INCOMP,"redundant mat size %d != input mat size %d",M,mat->rmap.N);
  if (reuse == MAT_INITIAL_MATRIX){
    PetscObjectContainer container;
    *matredundant = C;
    /* create a supporting struct and attach it to C for reuse */
    ierr = PetscNew(Mat_Redundant,&redund);CHKERRQ(ierr);
    ierr = PetscObjectContainerCreate(PETSC_COMM_SELF,&container);CHKERRQ(ierr);
    ierr = PetscObjectContainerSetPointer(container,redund);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)C,"Mat_Redundant",(PetscObject)container);CHKERRQ(ierr);
    ierr = PetscObjectContainerSetUserDestroy(container,PetscObjectContainerDestroy_MatRedundant);CHKERRQ(ierr);
    
    redund->nzlocal = nzlocal;
    redund->nsends  = nsends;
    redund->nrecvs  = nrecvs;
    redund->send_rank = send_rank;
    redund->sbuf_nz = sbuf_nz;
    redund->sbuf_j  = sbuf_j;
    redund->sbuf_a  = sbuf_a;
    redund->rbuf_j  = rbuf_j;
    redund->rbuf_a  = rbuf_a;

    redund->MatDestroy = C->ops->destroy;
    C->ops->destroy    = MatDestroy_MatRedundant;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCSetUp_Redundant"
static PetscErrorCode PCSetUp_Redundant(PC pc)
{
  PC_Redundant   *red  = (PC_Redundant*)pc->data;
  PetscErrorCode ierr;
  PetscInt       mstart,mend,mlocal,m;
  PetscMPIInt    size;
  IS             isl;
  MatReuse       reuse = MAT_INITIAL_MATRIX;
  MatStructure   str   = DIFFERENT_NONZERO_PATTERN;
  MPI_Comm       comm;
  Vec            vec;

  PetscInt    mlocal_sub;
  PetscMPIInt subsize,subrank;
  PetscInt    rstart_sub,rend_sub,mloc_sub;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)pc->pmat,&comm);CHKERRQ(ierr);
  ierr = MatGetVecs(pc->pmat,&vec,0);CHKERRQ(ierr);
  ierr = PCSetFromOptions(red->pc);CHKERRQ(ierr);
  ierr = VecGetSize(vec,&m);CHKERRQ(ierr);
  if (!pc->setupcalled) {
    /* create working vectors xsub/ysub and xdup/ydup */
    ierr = VecGetLocalSize(vec,&mlocal);CHKERRQ(ierr);  
    ierr = VecGetOwnershipRange(vec,&mstart,&mend);CHKERRQ(ierr);

    /* get local size of xsub/ysub */    
    ierr = MPI_Comm_size(red->subcomm,&subsize);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(red->subcomm,&subrank);CHKERRQ(ierr);
    rstart_sub = pc->pmat->rmap.range[red->nsubcomm*subrank]; /* rstart in xsub/ysub */    
    if (subrank+1 < subsize){
      rend_sub = pc->pmat->rmap.range[red->nsubcomm*(subrank+1)];
    } else {
      rend_sub = m; 
    }
    mloc_sub = rend_sub - rstart_sub;
    ierr = VecCreateMPI(red->subcomm,mloc_sub,PETSC_DECIDE,&red->ysub);CHKERRQ(ierr);
    /* create xsub with empty local arrays, because xdup's arrays will be placed into it */
    ierr = VecCreateMPIWithArray(red->subcomm,mloc_sub,PETSC_DECIDE,PETSC_NULL,&red->xsub);CHKERRQ(ierr);

    /* create xdup and ydup. ydup has empty local arrays because ysub's arrays will be place into it. 
       Note: we use communicator dupcomm, not pc->comm! */      
    ierr = VecCreateMPI(red->dupcomm,mloc_sub,PETSC_DECIDE,&red->xdup);CHKERRQ(ierr);
    ierr = VecCreateMPIWithArray(red->dupcomm,mloc_sub,PETSC_DECIDE,PETSC_NULL,&red->ydup);CHKERRQ(ierr);
  
    /* create vec scatters */
    if (!red->scatterin){
      IS       is1,is2;
      PetscInt *idx1,*idx2,i,j,k; 
      ierr = PetscMalloc(2*red->nsubcomm*mlocal*sizeof(PetscInt),&idx1);CHKERRQ(ierr);
      idx2 = idx1 + red->nsubcomm*mlocal;
      j = 0;
      for (k=0; k<red->nsubcomm; k++){
        for (i=mstart; i<mend; i++){
          idx1[j]   = i;
          idx2[j++] = i + m*k;
        }
      }
      ierr = ISCreateGeneral(comm,red->nsubcomm*mlocal,idx1,&is1);CHKERRQ(ierr);
      ierr = ISCreateGeneral(comm,red->nsubcomm*mlocal,idx2,&is2);CHKERRQ(ierr);      
      ierr = VecScatterCreate(vec,is1,red->xdup,is2,&red->scatterin);CHKERRQ(ierr);
      ierr = ISDestroy(is1);CHKERRQ(ierr);
      ierr = ISDestroy(is2);CHKERRQ(ierr);

      ierr = ISCreateStride(comm,mlocal,mstart+ red->color*m,1,&is1);CHKERRQ(ierr);
      ierr = ISCreateStride(comm,mlocal,mstart,1,&is2);CHKERRQ(ierr);
      ierr = VecScatterCreate(red->xdup,is1,vec,is2,&red->scatterout);CHKERRQ(ierr);      
      ierr = ISDestroy(is1);CHKERRQ(ierr);
      ierr = ISDestroy(is2);CHKERRQ(ierr);
      ierr = PetscFree(idx1);CHKERRQ(ierr);
    }
  }
  ierr = VecDestroy(vec);CHKERRQ(ierr);

  /* if pmatrix set by user is sequential then we do not need to gather the parallel matrix */
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size == 1) {
    red->useparallelmat = PETSC_FALSE;
  }

  if (red->useparallelmat) {
    if (pc->setupcalled == 1 && pc->flag == DIFFERENT_NONZERO_PATTERN) {
      /* destroy old matrices */
      if (red->pmats) {
        //ierr = MatDestroyMatrices(1,&red->pmats);CHKERRQ(ierr);
        ierr = MatDestroy(red->pmats);CHKERRQ(ierr);
      }
    } else if (pc->setupcalled == 1) {
      reuse = MAT_REUSE_MATRIX;
      str   = SAME_NONZERO_PATTERN;
    }
       
    /* grab the parallel matrix and put it into processors of a subcomminicator */ 
    /*--------------------------------------------------------------------------*/
    ierr = VecGetLocalSize(red->ysub,&mlocal_sub);CHKERRQ(ierr);  
    ierr = MatGetRedundantMatrix_AIJ(pc->pmat,red->nsubcomm,red->subcomm,mlocal_sub,reuse,&red->pmats);CHKERRQ(ierr);
    /* tell PC of the subcommunicator its operators */
    ierr = PCSetOperators(red->pc,red->pmats,red->pmats,str);CHKERRQ(ierr);
  } else {
    ierr = PCSetOperators(red->pc,pc->mat,pc->pmat,pc->flag);CHKERRQ(ierr);
  }
  ierr = PCSetFromOptions(red->pc);CHKERRQ(ierr);
  ierr = PCSetUp(red->pc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCApply_Redundant"
static PetscErrorCode PCApply_Redundant(PC pc,Vec x,Vec y)
{
  PC_Redundant   *red = (PC_Redundant*)pc->data;
  PetscErrorCode ierr;
  PetscScalar    *array;

  PetscFunctionBegin;
  /* scatter x to xdup */
  ierr = VecScatterBegin(x,red->xdup,INSERT_VALUES,SCATTER_FORWARD,red->scatterin);CHKERRQ(ierr);
  ierr = VecScatterEnd(x,red->xdup,INSERT_VALUES,SCATTER_FORWARD,red->scatterin);CHKERRQ(ierr);
  
  /* place xdup's local array into xsub */
  ierr = VecGetArray(red->xdup,&array);CHKERRQ(ierr);
  ierr = VecPlaceArray(red->xsub,(const PetscScalar*)array);CHKERRQ(ierr);

  /* apply preconditioner on each processor */
  ierr = PCApply(red->pc,red->xsub,red->ysub);CHKERRQ(ierr);
  ierr = VecResetArray(red->xsub);CHKERRQ(ierr);
  ierr = VecRestoreArray(red->xdup,&array);CHKERRQ(ierr);
 
  /* place ysub's local array into ydup */
  ierr = VecGetArray(red->ysub,&array);CHKERRQ(ierr);
  ierr = VecPlaceArray(red->ydup,(const PetscScalar*)array);CHKERRQ(ierr);

  /* scatter ydup to y */
  ierr = VecScatterBegin(red->ydup,y,INSERT_VALUES,SCATTER_FORWARD,red->scatterout);CHKERRQ(ierr);
  ierr = VecScatterEnd(red->ydup,y,INSERT_VALUES,SCATTER_FORWARD,red->scatterout);CHKERRQ(ierr);
  ierr = VecResetArray(red->ydup);CHKERRQ(ierr);
  ierr = VecRestoreArray(red->ysub,&array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCDestroy_Redundant"
static PetscErrorCode PCDestroy_Redundant(PC pc)
{
  PC_Redundant   *red = (PC_Redundant*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (red->scatterin)  {ierr = VecScatterDestroy(red->scatterin);CHKERRQ(ierr);}
  if (red->scatterout) {ierr = VecScatterDestroy(red->scatterout);CHKERRQ(ierr);}
  if (red->ysub)       {ierr = VecDestroy(red->ysub);CHKERRQ(ierr);}
  if (red->xsub)       {ierr = VecDestroy(red->xsub);CHKERRQ(ierr);}
  if (red->xdup)       {ierr = VecDestroy(red->xdup);CHKERRQ(ierr);}
  if (red->ydup)       {ierr = VecDestroy(red->ydup);CHKERRQ(ierr);}
  if (red->pmats) {
    ierr = MatDestroy(red->pmats);CHKERRQ(ierr);
  }


  ierr = PCDestroy(red->pc);CHKERRQ(ierr);
  ierr = PetscFree(red);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCSetFromOptions_Redundant"
static PetscErrorCode PCSetFromOptions_Redundant(PC pc)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCRedundantSetScatter_Redundant"
PetscErrorCode PETSCKSP_DLLEXPORT PCRedundantSetScatter_Redundant(PC pc,VecScatter in,VecScatter out)
{
  PC_Redundant   *red = (PC_Redundant*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  red->scatterin  = in; 
  red->scatterout = out;
  ierr = PetscObjectReference((PetscObject)in);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)out);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "PCRedundantSetScatter"
/*@
   PCRedundantSetScatter - Sets the scatter used to copy values into the
     redundant local solve and the scatter to move them back into the global
     vector.

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context
.  in - the scatter to move the values in
-  out - the scatter to move them out

   Level: advanced

.keywords: PC, redundant solve
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCRedundantSetScatter(PC pc,VecScatter in,VecScatter out)
{
  PetscErrorCode ierr,(*f)(PC,VecScatter,VecScatter);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  PetscValidHeaderSpecific(in,VEC_SCATTER_COOKIE,2);
  PetscValidHeaderSpecific(out,VEC_SCATTER_COOKIE,3);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCRedundantSetScatter_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,in,out);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}  

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCRedundantGetPC_Redundant"
PetscErrorCode PETSCKSP_DLLEXPORT PCRedundantGetPC_Redundant(PC pc,PC *innerpc)
{
  PC_Redundant *red = (PC_Redundant*)pc->data;

  PetscFunctionBegin;
  *innerpc = red->pc;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "PCRedundantGetPC"
/*@
   PCRedundantGetPC - Gets the sequential PC created by the redundant PC.

   Not Collective

   Input Parameter:
.  pc - the preconditioner context

   Output Parameter:
.  innerpc - the sequential PC 

   Level: advanced

.keywords: PC, redundant solve
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCRedundantGetPC(PC pc,PC *innerpc)
{
  PetscErrorCode ierr,(*f)(PC,PC*);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  PetscValidPointer(innerpc,2);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCRedundantGetPC_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,innerpc);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}  

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCRedundantGetOperators_Redundant"
PetscErrorCode PETSCKSP_DLLEXPORT PCRedundantGetOperators_Redundant(PC pc,Mat *mat,Mat *pmat)
{
  PC_Redundant *red = (PC_Redundant*)pc->data;

  PetscFunctionBegin;
  if (mat)  *mat  = red->pmats;
  if (pmat) *pmat = red->pmats;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "PCRedundantGetOperators"
/*@
   PCRedundantGetOperators - gets the sequential matrix and preconditioner matrix

   Not Collective

   Input Parameter:
.  pc - the preconditioner context

   Output Parameters:
+  mat - the matrix
-  pmat - the (possibly different) preconditioner matrix

   Level: advanced

.keywords: PC, redundant solve
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCRedundantGetOperators(PC pc,Mat *mat,Mat *pmat)
{
  PetscErrorCode ierr,(*f)(PC,Mat*,Mat*);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  if (mat)  PetscValidPointer(mat,2);
  if (pmat) PetscValidPointer(pmat,3); 
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCRedundantGetOperators_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,mat,pmat);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}  

/* -------------------------------------------------------------------------------------*/
/*MC
     PCREDUNDANT - Runs a preconditioner for the entire problem on each processor


     Options for the redundant preconditioners can be set with -redundant_pc_xxx

   Level: intermediate


.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PCRedundantSetScatter(),
           PCRedundantGetPC(), PCRedundantGetOperators()

M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCCreate_Redundant"
PetscErrorCode PETSCKSP_DLLEXPORT PCCreate_Redundant(PC pc)
{
  PetscErrorCode ierr;
  PC_Redundant   *red;
  const char     *prefix;

  PetscInt       nsubcomm,np_subcomm,nleftover,i,j,color;
  PetscMPIInt    rank,size,subrank,*subsize;
  MPI_Comm       subcomm;
  PetscMPIInt    duprank;
  PetscMPIInt    rank_dup,size_dup;
  MPI_Comm       dupcomm;

  PetscFunctionBegin;
  ierr = PetscNew(PC_Redundant,&red);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(pc,sizeof(PC_Redundant));CHKERRQ(ierr);
  red->useparallelmat   = PETSC_TRUE;
  
  ierr = MPI_Comm_rank(pc->comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(pc->comm,&size);CHKERRQ(ierr);
  nsubcomm = size;
  ierr = PetscOptionsGetInt(PETSC_NULL,"-nsubcomm",&nsubcomm,PETSC_NULL);CHKERRQ(ierr);
  if (nsubcomm > size) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE, "Num of subcommunicators %D cannot be larger than MPI_Comm_size %D",nsubcomm,size);

  /* get size of each subcommunicators */
  ierr = PetscMalloc((1+nsubcomm)*sizeof(PetscMPIInt),&subsize);CHKERRQ(ierr);
  np_subcomm = size/nsubcomm;
  nleftover  = size - nsubcomm*np_subcomm;
  for (i=0; i<nsubcomm; i++){
    subsize[i] = np_subcomm;
    if (i<nleftover) subsize[i]++;
  }

  /* find color for this proc */
  color   = rank%nsubcomm;
  subrank = rank/nsubcomm;

  ierr = MPI_Comm_split(pc->comm,color,subrank,&subcomm);CHKERRQ(ierr);
  red->subcomm  = subcomm; 
  red->color    = color;
  red->nsubcomm = nsubcomm;

  j = 0; duprank = 0;
  for (i=0; i<nsubcomm; i++){
    if (j == color){
      duprank += subrank;
      break;
    }
    duprank += subsize[i]; j++;
  }
  /*
  ierr = PetscSynchronizedPrintf(pc->comm, "[%d] color %d, subrank %d, duprank %d\n",rank,color,subrank,duprank);
  ierr = PetscSynchronizedFlush(pc->comm);CHKERRQ(ierr);
  */
 
  ierr = MPI_Comm_split(pc->comm,0,duprank,&dupcomm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(dupcomm,&rank_dup);CHKERRQ(ierr);
  ierr = MPI_Comm_size(dupcomm,&size_dup);CHKERRQ(ierr);
  /*
  ierr = PetscSynchronizedPrintf(pc->comm, "[%d] duprank %d\n",rank,duprank);
  ierr = PetscSynchronizedFlush(pc->comm);CHKERRQ(ierr);
  */
  red->dupcomm = dupcomm;
  ierr = PetscFree(subsize);CHKERRQ(ierr);

  /* create the sequential PC that each processor has copy of */
  ierr = PCCreate(subcomm,&red->pc);CHKERRQ(ierr);
  ierr = PCSetType(red->pc,PCLU);CHKERRQ(ierr);
  ierr = PCGetOptionsPrefix(pc,&prefix);CHKERRQ(ierr);
  ierr = PCSetOptionsPrefix(red->pc,prefix);CHKERRQ(ierr);
  ierr = PCAppendOptionsPrefix(red->pc,"redundant_");CHKERRQ(ierr);

  pc->ops->apply             = PCApply_Redundant;
  pc->ops->applytranspose    = 0;
  pc->ops->setup             = PCSetUp_Redundant;
  pc->ops->destroy           = PCDestroy_Redundant;
  pc->ops->setfromoptions    = PCSetFromOptions_Redundant;
  pc->ops->setuponblocks     = 0;
  pc->ops->view              = PCView_Redundant;
  pc->ops->applyrichardson   = 0;

  pc->data     = (void*)red;     

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCRedundantSetScatter_C","PCRedundantSetScatter_Redundant",
                    PCRedundantSetScatter_Redundant);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCRedundantGetPC_C","PCRedundantGetPC_Redundant",
                    PCRedundantGetPC_Redundant);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCRedundantGetOperators_C","PCRedundantGetOperators_Redundant",
                    PCRedundantGetOperators_Redundant);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END
