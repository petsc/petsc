
#ifndef lint
static char vcid[] = "$Id: mscat.c,v 1.5 1995/09/11 18:49:22 bsmith Exp bsmith $";
#endif

/*
     Code to scatter between matrices, at the moment this is very limited
  one can only scatter from MPIAIJ matrices to AIJ matrices
*/

#include "matimpl.h"    /*I "mat.h"  I*/
#include "src/mat/impls/aij/mpi/mpiaij.h"

/*
   MatScatterBegin_SeqRows - sends rows of matrix in scatter
*/
static int MatScatterBegin_SeqRows(Mat x,Mat y,InsertMode addv,
                                MatScatterCtx ctx)
{
  return 0;
}
static int MatScatterEnd_SeqRows(Mat x,Mat y,InsertMode addv,
                              MatScatterCtx ctx)
{
  MatAssemblyBegin(y,FINAL_ASSEMBLY);
  MatAssemblyEnd(y,FINAL_ASSEMBLY);
  return 0;
}

static int MatScatterCtxDestroy_SeqRows(PetscObject obj)
{
  MatScatterCtx             ctx = (MatScatterCtx) obj;
  MatScatterCtx_MPISendRows *rows = 
                               (MatScatterCtx_MPISendRows *)ctx->send;

  PETSCFREE(rows->procs); PETSCFREE(rows->rows);
  PETSCFREE(rows); PETSCHEADERDESTROY(ctx);
  return 0;
}
/*
    MatScatterCtxCreate - Creates scatter context for matrix matrix 
             scatter. This is so limited it is not yet made public.

.seealso: MatScatterBegin(), MatScatterEnd()
*/
int MatScatterCtxCreate(Mat X,IS xr,IS xc,Mat Y,IS yr,IS yc,
                        MatScatterCtx* ctx)
{
  Mat_MPIAIJ        *x = (Mat_MPIAIJ*) X->data;
  int               nrows,*rows,ncols,*cols,ierr,first,step, idx; 
  int               numtids;
  int               *owners = x->rowners,*owner,*nprocs,*procs,*work;
  int               mytid;
  int               nreceives,nmax,*rvalues,i,j,tag = X->tag,*starts;
  int               count;
  int               *svalues,nsends,*values,imdex,n,*nnrows;
  MPI_Request       *recv_waits,*send_waits;
  MPI_Status        recv_status,*send_status;
  MPI_Comm          comm = X->comm;
  MatScatterCtx     scat;
  MatScatterCtx_MPISendRows *mpisend;
  MatScatterCtx_MPIRecvRows *mpirecv;
  PETSCVALIDHEADERSPECIFIC(X,MAT_COOKIE); PETSCVALIDHEADERSPECIFIC(Y,MAT_COOKIE);

  MPI_Comm_size(comm,&numtids);
  MPI_Comm_rank(comm,&mytid);

  if (X->type != (int) MATMPIAIJ) {
    SETERRQ(1,"MatScatterCtxCreate:only supports scatter from MPIAIJ");
  }
  if (Y->type != (int) MATSEQAIJ) {
    SETERRQ(1,"MatScatterCtxCreate:only supports scatter to AIJ");
  }

  ierr = ISGetSize(xr,&nrows); CHKERRQ(ierr);
  ierr = ISGetSize(xc,&ncols); CHKERRQ(ierr);

  /* check if one is looking for all columns or rows */
  if (((PetscObject)xc)->type == (int) ISSTRIDESEQ) {
    ISStrideGetInfo(xc,&first,&step); 
    if (first == 0 && ncols == x->N) {cols = 0;}
    else {ierr = ISGetIndices(xc,&cols); CHKERRQ(ierr);}
  }
  else {ierr = ISGetIndices(xc,&cols); CHKERRQ(ierr);}
  if (((PetscObject)xr)->type == (int) ISSTRIDESEQ) {
    ISStrideGetInfo(xr,&first,&step); 
    if (first == 0 && nrows == x->M) {rows = 0;}
    else {ierr = ISGetIndices(xr,&rows); CHKERRQ(ierr);}
  }
  else {ierr = ISGetIndices(xr,&rows); CHKERRQ(ierr);}

  if (!rows && !cols) 
    SETERRQ(1,"MatScatterCtxCreate:not yet for all matrix");
  if (rows && cols) SETERRQ(1,"MatScatterCtxCreate:must all row or col");

  /* BIG assumption: all processors either want rows or columns */

  if (rows) { /* do the case of certain rows */
    /* loop over rows determining who owns each */
    procs = (int*) PETSCMALLOC((nrows+2*numtids)*sizeof(int));
    CHKPTRQ(procs);
    nprocs = procs + numtids; owner = nprocs + numtids; 
    PETSCMEMSET(procs,0,2*numtids*sizeof(int));
    for ( i=0; i<nrows; i++ ) {
      idx = rows[i];
      for ( j=0; j<numtids; j++ ) {
        if (idx >= owners[j] && idx < owners[j+1]) {
          nprocs[j]++; procs[j] = 1; owner[i] = j; break;
        }
      }
    }
    nsends = 0; for (i=0;i<numtids;i++) nsends += procs[i];

    /* inform other processors of number of messages and max length*/
    work = (int *) PETSCMALLOC( numtids*sizeof(int) ); CHKPTRQ(work);
    MPI_Allreduce((void *) procs,(void *) work,numtids,MPI_INT,MPI_SUM,
                  comm);
    nreceives = work[mytid]; 
    MPI_Allreduce((void *) nprocs,(void *) work,numtids,MPI_INT,MPI_MAX,
                  comm);
    nmax = work[mytid];
    PETSCFREE(work);
fprintf(stderr,"nrows %d nreceives %d nmax %d nsends %d\n",nrows,nreceives,nmax,nsends);

    /* post receives: 
       1) each message will consist of ordered pairs 
     (global index,value) we store the global index as a double 
     to simplify the message passing. 
       2) since we don't know how long each individual message is we 
     allocate the largest needed buffer for each receive. Potentially 
     this is a lot of wasted space.

       This could be done better.
    */
    rvalues = (int *) PETSCMALLOC((nreceives+1)*(nmax+1)*sizeof(int));
    CHKPTRQ(rvalues);
    recv_waits = (MPI_Request *)PETSCMALLOC(
                                    (nreceives+1)*sizeof(MPI_Request));
    CHKPTRQ(recv_waits);
    for ( i=0; i<nreceives; i++ ) {
      MPI_Irecv((void *)(rvalues+nmax*i),nmax,MPI_INT,MPI_ANY_SOURCE,
                 tag,comm,recv_waits+i);
    }
    /* do sends:
      1) starts[i] gives the starting index in svalues for stuff
         going to the ith processor
    */
    svalues = (int *) PETSCMALLOC( (nrows+1)*sizeof(int) );
    CHKPTRQ(svalues);
    send_waits = (MPI_Request *) PETSCMALLOC( 
                                        (nsends+1)*sizeof(MPI_Request));
    CHKPTRQ(send_waits);
    starts = (int *) PETSCMALLOC( numtids*sizeof(int) ); CHKPTRQ(starts);
    starts[0] = 0; 
    for ( i=1; i<numtids; i++ ) { starts[i] = starts[i-1]+nprocs[i-1];} 
    for ( i=0; i<nrows; i++ ) {
      svalues[starts[owner[i]]++] = rows[i];
    }
    starts[0] = 0;
    for ( i=1; i<numtids; i++ ) { starts[i] = starts[i-1]+nprocs[i-1];} 
    count = 0;
    for ( i=0; i<numtids; i++ ) {
printf("procs %d %d\n",i,procs[i]);
      if (procs[i]) {
        MPI_Isend((void*)(svalues+starts[i]),nprocs[i],MPI_INT,i,tag,
                  comm,send_waits+count++);
      }
    }
    PETSCFREE(starts); PETSCFREE(procs);

    mpisend = PETSCNEW(MatScatterCtx_MPISendRows); CHKPTRQ(mpisend);
    mpirecv = PETSCNEW(MatScatterCtx_MPIRecvRows); CHKPTRQ(mpirecv);
    PETSCHEADERCREATE(scat,_MatScatterCtx,MAT_SCATTER_COOKIE,0,comm);
    scat->send    = (void *) mpisend;
    scat->begin   = MatScatterBegin_SeqRows;
    scat->end     = MatScatterEnd_SeqRows;
    scat->destroy = MatScatterCtxDestroy_SeqRows;

    mpisend->n = nreceives;
    mpisend->procs = (int *) PETSCMALLOC( (2*nreceives+1)*sizeof(int));
    CHKPTRQ(mpisend->procs);
    mpisend->starts = mpisend->procs + nreceives; 

    /* wait on receives */
    nnrows = (int *) PETSCMALLOC(nreceives*sizeof(int)); CHKPTRQ(nnrows);
    count = nreceives;
    while (count) {
      MPI_Waitany(nreceives,recv_waits,&imdex,&recv_status);
      /* unpack receives into our local space */
      values = rvalues + imdex*nmax;
      MPI_Get_count(&recv_status,MPI_INT,&n);
      count--;
      mpisend->procs[imdex] = recv_status.MPI_SOURCE;
      nnrows[imdex] = n;
fprintf(stderr,"received %d slots %d\n",n,values[0]);
    }
    PETSCFREE(recv_waits);
    
    /* allocate space to list of rows each processor needs */
    for ( i=0; i<nreceives; i++ ) {
      count += nnrows[i];
    }
    mpisend->rows = (int *) PETSCMALLOC( (count+1)*sizeof(int) ); 
    CHKPTRQ(mpisend->rows);
    count = 0;
    for ( i=0; i<nreceives; i++ ) {
      mpisend->starts[i] = count;
      for ( j=0; j<nnrows[i]; j++ ) {
        mpisend->rows[count++] = rvalues[i*nmax+j];
      }
    }  
    mpisend->starts[nreceives] = count;
    PETSCFREE(rvalues);  PETSCFREE(nnrows);

    /* wait on posted sends */
    if (nsends) {
      send_status = (MPI_Status *)PETSCMALLOC(
                                         (nsends+1)*sizeof(MPI_Status));
      CHKPTRQ(send_status);
      MPI_Waitall(nsends,send_waits,send_status);
      PETSCFREE(send_status);
    }
    PETSCFREE(send_waits); PETSCFREE(svalues);
  }
  else {
    scat = 0;
  }

  if (rows) {ierr = ISRestoreIndices(xr,&rows); CHKERRQ(ierr);}
  if (cols) {ierr = ISRestoreIndices(xc,&cols); CHKERRQ(ierr);}
  *ctx = scat;
  return 0;
}

/*
   MatScatterBegin - Begins scattering from one matrix to another.


.keywords: matrix, scatter, gather, begin

.seealso: MatScatterCtxCreate(), MatScatterEnd()
*/
int MatScatterBegin(Mat x,Mat y,InsertMode addv,MatScatterCtx inctx)
{
  PETSCVALIDHEADERSPECIFIC(x,MAT_COOKIE); PETSCVALIDHEADERSPECIFIC(y,MAT_COOKIE);
  PETSCVALIDHEADERSPECIFIC(inctx,MAT_SCATTER_COOKIE);
  if (inctx->inuse) 
    SETERRQ(1,"MatScatterBegin:Scatter ctx already in use");
  return (*(inctx)->begin)(x,y,addv,inctx);
}

/*
   MatScatterEnd - Ends scattering from one Matrix to another.
   Call after first calling MatScatterBegin().

.keywords: matrix, scatter, gather, end

.seealso: MatScatterBegin(), MatScatterCtxCreate()
*/
int MatScatterEnd(Mat x,Mat y,InsertMode addv,MatScatterCtx ctx)
{
  PETSCVALIDHEADERSPECIFIC(x,MAT_COOKIE); PETSCVALIDHEADERSPECIFIC(y,MAT_COOKIE);
  PETSCVALIDHEADERSPECIFIC(ctx,MAT_SCATTER_COOKIE);
  ctx->inuse = 0;
  if ((ctx)->end) return (*(ctx)->end)(x,y,addv,ctx);
  else return 0;
}

/*
    MatScatterCtxDestroy - Destroys scatter context.
              This is so limited it is not yet made public.

.seealso: MatScatterBegin(), MatScatterEnd(), MatScatterCtxCreate()
*/
int MatScatterCtxDestroy(MatScatterCtx ctx)
{
  int ierr;
  ierr = (*(ctx)->destroy)((PetscObject)ctx); CHKERRQ(ierr);
  return 0;
}






