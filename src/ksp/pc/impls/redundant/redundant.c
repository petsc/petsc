#define PETSCKSP_DLL

/*
  This file defines a "solve the problem redundantly on each subgroup of processor" preconditioner.
*/
#include "private/pcimpl.h"     /*I "petscpc.h" I*/
#include "petscksp.h"

#undef CONTIGUOUS_COLOR
#define INTER_COLOR

typedef struct {
  PC         pc;                   /* actual preconditioner used on each processor */
  Vec        xsub,ysub;            /* vectors of a subcommunicator to hold parallel vectors of pc->comm */
  Vec        xdup,ydup;            /* parallel vector that congregates xsub or ysub facilitating vector scattering */
  Mat        *pmats;               /* matrix and optional preconditioner matrix */
  Mat        pmats_sub;            /* matrix and optional preconditioner matrix */
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

#undef __FUNCT__  
#define __FUNCT__ "MatGetRedundantMatrix"
PetscErrorCode MatGetRedundantMatrix_AIJ(PC pc,Mat mat,MPI_Comm subcomm,PetscInt mlocal_sub,Mat *matredundant)
{
  PetscMPIInt    rank,size,subrank,subsize;
  MPI_Comm       comm=mat->comm;
  PetscErrorCode ierr;
  PC_Redundant   *red=(PC_Redundant*)pc->data;
  PetscInt       nsubcomm=red->nsubcomm,nsends,nrecvs,i,prid=100,itmp;
  PetscMPIInt    *send_rank,*recv_rank;
  PetscInt       *rowrange=pc->pmat->rmap.range,mlocal_max,nzlocal;
  Mat_MPIAIJ     *aij = (Mat_MPIAIJ*)mat->data;
  Mat            A=aij->A,B=aij->B;
  Mat_SeqAIJ     *a=(Mat_SeqAIJ*)A->data,*b=(Mat_SeqAIJ*)B->data;
  Mat            C;

  PetscInt       nleftover,np_subcomm,j; 
  PetscInt       nz_A,nz_B,*sbuf_j;
  PetscScalar    *sbuf_a;
  PetscInt       cstart=mat->cmap.rstart,cend=mat->cmap.rend,row,nzA,nzB,ncols,*cworkA,*cworkB;
  PetscInt       rstart=mat->rmap.rstart,rend=mat->rmap.rend,*bmap=aij->garray,M,N;
  PetscInt       *cols,ctmp,lwrite,*rptr,l;
  PetscScalar    *vals,*aworkA,*aworkB;

  PetscMPIInt    tag1,tag2,tag3,imdex;
  MPI_Request    *s_waits1,*s_waits2,*s_waits3,*r_waits1,*r_waits2,*r_waits3;
  MPI_Status     recv_status,*send_status; 
  PetscInt       *sbuf_nz,*rbuf_nz,count;

  PetscInt     **rbuf_j;
  PetscScalar  **rbuf_a;

  PetscFunctionBegin;
  ierr = PetscOptionsGetInt(PETSC_NULL,"-prid",&prid,PETSC_NULL);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(subcomm,&subrank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(subcomm,&subsize);CHKERRQ(ierr);
  /*
  ierr = PetscSynchronizedPrintf(comm, "[%d] subrank %d, subsize %d\n",rank,subrank,subsize);
  ierr = PetscSynchronizedFlush(comm);CHKERRQ(ierr);
  */
  /* get the destination processors */
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
  if (prid == rank){
    printf("[%d] sends to ",rank);
    for (i=0; i<nsends; i++) printf(" [%d],",send_rank[i]);
    printf("  \n");                      
  }
  /*
  ierr = PetscSynchronizedPrintf(comm, "[%d] nsends %d, nrecvs %d\n",rank,nsends,nrecvs);
  ierr = PetscSynchronizedFlush(comm);CHKERRQ(ierr);
  ierr = MPI_Barrier(comm);CHKERRQ(ierr);
  */

  /* get this processor's nzlocal=nz_A+nz_B */
  nz_A = a->nz; nz_B = b->nz;
  nzlocal = nz_A + nz_B;

  /* allocate sbuf_j, sbuf_a, then copy mat's local entries into the buffers */
  itmp = nzlocal + rowrange[rank+1] - rowrange[rank] + 2;
  ierr = PetscMalloc(itmp*sizeof(PetscInt),&sbuf_j);CHKERRQ(ierr);
  ierr = PetscMalloc((nzlocal+1)*sizeof(PetscScalar),&sbuf_a);CHKERRQ(ierr);
 
    rptr = sbuf_j;
    cols = sbuf_j + rend-rstart + 1;
    vals = sbuf_a;
    rptr[0] = 0;
    for (i=0; i<rend-rstart; i++){
      row = i + rstart;
      if (rank == prid) printf(" \n row %d: ",row);
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
          if (rank == prid) printf(" (%d,%g)",ctmp,aworkB[l]);
        }
      }
      for (l=0; l<nzA; l++){
        vals[lwrite]   = aworkA[l];
        cols[lwrite++] = cstart + cworkA[l];
        if (rank == prid) printf(" (%d,%g)",cstart + cworkA[l],aworkA[l]);
      }
      for (l=0; l<nzB; l++) {
        if ((ctmp = bmap[cworkB[l]]) >= cend){
          vals[lwrite]   = aworkB[l];
          cols[lwrite++] = ctmp;
          if (rank == prid) printf(" (%d,%g)",ctmp,aworkB[l]);
        }
      }
      /* insert local matrix values into C */
      //ierr = MatSetValues(C,1,&row,ncols,cols,vals,INSERT_VALUES);CHKERRQ(ierr);

      vals += ncols;
      cols += ncols;
      rptr[i+1] = rptr[i] + ncols;    
    }
    /*
    ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    */
    if (rptr[rend-rstart] != a->nz + b->nz) SETERRQ4(1, "rptr[%d] %d != %d + %d",rend-rstart,rptr[rend-rstart+1],a->nz,b->nz);

  /* send nzlocal to others, and recv other's nzlocal */
  /*--------------------------------------------------*/
  ierr = PetscMalloc3(nsends+nrecvs+1,PetscInt,&sbuf_nz,nrecvs,PetscInt*,&rbuf_j,nrecvs,PetscScalar*,&rbuf_a);CHKERRQ(ierr);
  rbuf_nz = sbuf_nz + nsends;

  ierr = PetscMalloc2(3*(nsends + nrecvs)+1,MPI_Request,&s_waits1,nsends+1,MPI_Status,&send_status);CHKERRQ(ierr);
  s_waits2 = s_waits1 + nsends;
  s_waits3 = s_waits2 + nsends;
  r_waits1 = s_waits3 + nsends;
  r_waits2 = r_waits1 + nrecvs;
  r_waits3 = r_waits2 + nrecvs;

  /* get some new tags to keep the communication clean */
  ierr = PetscObjectGetNewTag((PetscObject)A,&tag1);CHKERRQ(ierr); 
  ierr = PetscObjectGetNewTag((PetscObject)A,&tag2);CHKERRQ(ierr);
  ierr = PetscObjectGetNewTag((PetscObject)A,&tag3);CHKERRQ(ierr);

  /* post receives of other's nzlocal */
  for (i=0; i<nrecvs; i++){
    ierr = MPI_Irecv(rbuf_nz+i,1,MPIU_INT,MPI_ANY_SOURCE,tag1,comm,r_waits1+i);CHKERRQ(ierr);
  }
  
  /* send nzlocal to others */
  for (i=0; i<nsends; i++){
    sbuf_nz[i] = nzlocal;
    ierr = MPI_Isend(sbuf_nz+i,1,MPIU_INT,send_rank[i],tag1,comm,s_waits1+i);CHKERRQ(ierr);
    if (prid == rank) printf(" [%d] sends nz %d to [%d]\n",rank,nzlocal,send_rank[i]);
  }

  /* wait on receives of nzlocal; allocate space for rbuf_j, rbuf_a */
  count = nrecvs;
  while (count) {
    ierr = MPI_Waitany(nrecvs,r_waits1,&imdex,&recv_status);CHKERRQ(ierr);
    recv_rank[imdex] = recv_status.MPI_SOURCE;
    /* allocate rbuf_a and rbuf_j; then post receives of rbuf_a and rbuf_j */
    ierr = PetscMalloc((rbuf_nz[imdex]+1)*sizeof(PetscScalar),&rbuf_a[imdex]);CHKERRQ(ierr);
    ierr = MPI_Irecv(rbuf_a[imdex],rbuf_nz[imdex],MPIU_SCALAR,recv_status.MPI_SOURCE,tag3,comm,r_waits3+imdex);CHKERRQ(ierr);

    itmp = rowrange[recv_status.MPI_SOURCE+1] - rowrange[recv_status.MPI_SOURCE]; /* number of expected mat->i */
    rbuf_nz[imdex] += itmp+2;
    ierr = PetscMalloc(rbuf_nz[imdex]*sizeof(PetscInt),&rbuf_j[imdex]);CHKERRQ(ierr);
    ierr = MPI_Irecv(rbuf_j[imdex],rbuf_nz[imdex],MPIU_INT,recv_status.MPI_SOURCE,tag2,comm,r_waits2+imdex);CHKERRQ(ierr);
    count--;
  }

  /* wait on sends of nzlocal */
  if (nsends) {ierr = MPI_Waitall(nsends,s_waits1,send_status);CHKERRQ(ierr);}

  /* send mat->i,j and mat->a to others, and recv from other's */
  /*-----------------------------------------------------------*/
  for (i=0; i<nsends; i++){
    ierr = MPI_Isend(sbuf_a,nzlocal,MPIU_SCALAR,send_rank[i],tag3,comm,s_waits3+i);CHKERRQ(ierr);
    itmp = nzlocal + rowrange[rank+1] - rowrange[rank] + 1;
    ierr = MPI_Isend(sbuf_j,itmp,MPIU_INT,send_rank[i],tag2,comm,s_waits2+i);CHKERRQ(ierr);
  }

  /* wait on receives of mat->i,j and mat->a */
  /*-----------------------------------------*/
  count = nrecvs;
  while (count) {
    ierr = MPI_Waitany(nrecvs,r_waits3,&imdex,&recv_status);CHKERRQ(ierr);
    if (recv_rank[imdex] != recv_status.MPI_SOURCE) SETERRQ2(1, "recv_rank %d != MPI_SOURCE %d",recv_rank[imdex],recv_status.MPI_SOURCE);
    ierr = MPI_Waitany(nrecvs,r_waits2,&imdex,&recv_status);CHKERRQ(ierr);
    count--;
  }

  /* wait on sends of mat->i,j and mat->a */
  /*--------------------------------------*/
  if (nsends) {
    ierr = MPI_Waitall(nsends,s_waits3,send_status);CHKERRQ(ierr);
    ierr = MPI_Waitall(nsends,s_waits2,send_status);CHKERRQ(ierr);
  }
  ierr = PetscFree(sbuf_nz);CHKERRQ(ierr);
  ierr = PetscFree2(s_waits1,send_status);CHKERRQ(ierr);
  
  /* create redundant matrix */
  /*-------------------------*/
  ierr = MatCreate(subcomm,&C);CHKERRQ(ierr);
  ierr = MatSetSizes(C,mlocal_sub,mlocal_sub,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetFromOptions(C);CHKERRQ(ierr);
  
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
  *matredundant = C;

  /* free space */
  ierr = PetscFree(send_rank);CHKERRQ(ierr);
  ierr = PetscFree(sbuf_j);CHKERRQ(ierr);
  ierr = PetscFree(sbuf_a);CHKERRQ(ierr);
  for (i=0; i<nrecvs; i++){
    ierr = PetscFree(rbuf_j[i]);CHKERRQ(ierr);
    ierr = PetscFree(rbuf_a[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree3(sbuf_nz,rbuf_j,rbuf_a);CHKERRQ(ierr);
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

  PetscMPIInt rank,size_sub,itmp;
  PetscInt    mlocal_sub;
  PetscMPIInt subsize,subrank;
  PetscInt    rstart_sub,rend_sub,mloc_sub;
  Mat         Aredundant;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)pc->pmat,&comm);CHKERRQ(ierr);
  ierr = MatGetVecs(pc->pmat,&vec,0);CHKERRQ(ierr);
  ierr = PCSetFromOptions(red->pc);CHKERRQ(ierr);
  ierr = VecGetSize(vec,&m);CHKERRQ(ierr);
  if (!pc->setupcalled) {
    /* create working vectors xsub/ysub and xdup/ydup */
    ierr = VecGetLocalSize(vec,&mlocal);CHKERRQ(ierr);  
    ierr = VecGetOwnershipRange(vec,&mstart,&mend);CHKERRQ(ierr);

#ifdef INTER_COLOR
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
#endif
#ifdef CONTIGUOUS_COLOR
    ierr = VecCreateMPI(red->subcomm,PETSC_DECIDE,m,&red->ysub);CHKERRQ(ierr);   
    ierr = VecGetLocalSize(red->ysub,&mloc_sub);CHKERRQ(ierr);
    ierr = VecCreateMPIWithArray(red->subcomm,mloc_sub,m,PETSC_NULL,&red->xsub);CHKERRQ(ierr);
#endif

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
        ierr = MatDestroyMatrices(1,&red->pmats);CHKERRQ(ierr);
      }
    } else if (pc->setupcalled == 1) {
      reuse = MAT_REUSE_MATRIX;
      str   = SAME_NONZERO_PATTERN;
    }
       
    /* ================== matrix ============= */
    ierr = VecGetLocalSize(red->ysub,&mlocal_sub);CHKERRQ(ierr);
    ierr = MatGetRedundantMatrix_AIJ(pc,pc->pmat,red->subcomm,mlocal_sub,&Aredundant);CHKERRQ(ierr);
    
    /* grab the parallel matrix and put it into processors of a subcomminicator */
    ierr = ISCreateStride(PETSC_COMM_SELF,m,0,1,&isl);CHKERRQ(ierr);
    ierr = MatGetSubMatrices(pc->pmat,1,&isl,&isl,reuse,&red->pmats);CHKERRQ(ierr);
    ierr = ISDestroy(isl);CHKERRQ(ierr);

    /* ------- temporarily set a mpi matrix pmats_sub- provided by user! --*/
    ierr = MatCreate(red->subcomm,&red->pmats_sub);CHKERRQ(ierr);
    ierr = MatSetSizes(red->pmats_sub,mlocal_sub,mlocal_sub,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
    ierr = MatSetFromOptions(red->pmats_sub);CHKERRQ(ierr);
    {
      PetscInt          Istart,Iend,ncols,i;
      const PetscInt    *cols;
      const PetscScalar *vals;
      PetscTruth flg;
      ierr = MatGetOwnershipRange(red->pmats_sub,&Istart,&Iend);CHKERRQ(ierr);
      for (i=Istart; i<Iend; i++) {
        ierr = MatGetRow(red->pmats[0],i,&ncols,&cols,&vals);CHKERRQ(ierr);
        ierr = MatSetValues(red->pmats_sub,1,&i,ncols,cols,vals,INSERT_VALUES);CHKERRQ(ierr);
        ierr = MatRestoreRow(red->pmats[0],i,&ncols,&cols,&vals);CHKERRQ(ierr);
      }   
      ierr = MatAssemblyBegin(red->pmats_sub,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(red->pmats_sub,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    
      ierr = MatEqual(red->pmats_sub,Aredundant,&flg);CHKERRQ(ierr);
      if (!flg) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"pmats_sub !=Aredundant ");  
      ierr = MatDestroy(red->pmats_sub);
    }
    red->pmats_sub = Aredundant;

    /* tell PC of the subcommunicator its operators */
    ierr = PCSetOperators(red->pc,red->pmats_sub,red->pmats_sub,str);CHKERRQ(ierr);
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
  if (red->pmats_sub) {
    ierr = MatDestroy(red->pmats_sub);CHKERRQ(ierr);
  }

  if (red->pmats) {
    ierr = MatDestroyMatrices(1,&red->pmats);CHKERRQ(ierr);
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
  if (mat)  *mat  = red->pmats[0];
  if (pmat) *pmat = red->pmats[0];
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
#ifdef INTER_COLOR
  color   = rank%nsubcomm;
  subrank = rank/nsubcomm;
#endif

#ifdef CONTIGUOUS_COLOR
  color = 0; subrank = 0; i = 0; j=0; 
  while (i < size){
    if (rank == i) break; /* my color is found */
    if (j >= subsize[color]-1){ /* next subcomm */
      j = -1; subrank = -1; color++;
    }
    i++; j++; subrank++;
  }
#endif

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
