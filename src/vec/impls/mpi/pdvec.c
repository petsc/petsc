/* $Id: pdvec.c,v 1.44 1996/03/10 17:26:49 bsmith Exp bsmith $ */

/*
     Code for some of the parallel vector primatives.
*/
#include "pinclude/pviewer.h"
#include "sysio.h"

static int VecGetOwnershipRange_MPI(Vec v,int *low,int* high) 
{
  Vec_MPI *x = (Vec_MPI *) v->data;
  *low  = x->ownership[x->rank];
  *high = x->ownership[x->rank+1];
  return 0;
}

static int VecDestroy_MPI(PetscObject obj )
{
  Vec     v = (Vec ) obj;
  Vec_MPI *x = (Vec_MPI *) v->data;
#if defined(PETSC_LOG)
  PLogObjectState(obj,"Length=%d",x->N);
#endif  
  if (x->stash.array) PetscFree(x->stash.array);
  PetscFree(x->array);
  PetscFree(v->data);
  PLogObjectDestroy(v);
  PetscHeaderDestroy(v);
  return 0;
}

static int VecView_MPI_File(Vec xin, Viewer ptr )
{
  Vec_MPI     *x = (Vec_MPI *) xin->data;
  int         i,rank,ierr;
  FILE        *fd;

  ierr = ViewerFileGetPointer(ptr,&fd); CHKERRQ(ierr);

  MPI_Comm_rank(xin->comm,&rank); 
  MPIU_Seq_begin(xin->comm,1);
  fprintf(fd,"Processor [%d] \n",rank);
  for ( i=0; i<x->n; i++ ) {
#if defined(PETSC_COMPLEX)
    if (imag(x->array[i]) != 0.0) {
      fprintf(fd,"%g + %g i\n",real(x->array[i]),imag(x->array[i]));
    }
    else {
      fprintf(fd,"%g\n",real(x->array[i]));
    }
#else
    fprintf(fd,"%g\n",x->array[i]);
#endif
  }
  fflush(fd);
  MPIU_Seq_end(xin->comm,1);
  return 0;
}

static int VecView_MPI_Files(Vec xin, Viewer ptr )
{
  Vec_MPI     *x = (Vec_MPI *) xin->data;
  int         i,rank,len, work = x->n,n,j,size,ierr;
  MPI_Status  status;
  FILE        *fd;
  Scalar      *values;

  ierr = ViewerFileGetPointer(ptr,&fd); CHKERRQ(ierr);
  /* determine maximum message to arrive */
  MPI_Comm_rank(xin->comm,&rank);
  MPI_Reduce(&work,&len,1,MPI_INT,MPI_MAX,0,xin->comm);
  MPI_Comm_size(xin->comm,&size);

  if (!rank) {
    values = (Scalar *) PetscMalloc( len*sizeof(Scalar) ); CHKPTRQ(values);
    fprintf(fd,"Processor [%d]\n",rank);
    for ( i=0; i<x->n; i++ ) {
#if defined(PETSC_COMPLEX)
      if (imag(x->array[i]) != 0.0) {
        fprintf(fd,"%g + %g i\n",real(x->array[i]),imag(x->array[i]));
      }
      else {
        fprintf(fd,"%g\n",real(x->array[i]));
      }
#else
      fprintf(fd,"%g\n",x->array[i]);
#endif

    }
    /* receive and print messages */
    for ( j=1; j<size; j++ ) {
      MPI_Recv(values,len,MPIU_SCALAR,j,47,xin->comm,&status);
      MPI_Get_count(&status,MPIU_SCALAR,&n);          
      fprintf(fd,"Processor [%d]\n",j);
      for ( i=0; i<n; i++ ) {
#if defined(PETSC_COMPLEX)
        if (imag(values[i]) != 0.0) {
          fprintf(fd,"%g + %g i\n",real(values[i]),imag(values[i]));
        }
        else {
          fprintf(fd,"%g\n",real(values[i]));
        }
#else
        fprintf(fd,"%g\n",values[i]);
#endif
      }          
    }
    PetscFree(values);
    fflush(fd);
  }
  else {
    /* send values */
    MPI_Send(x->array,x->n,MPIU_SCALAR,0,47,xin->comm);
  }
  return 0;
}

static int VecView_MPI_Binary(Vec xin, Viewer ptr )
{
  Vec_MPI     *x = (Vec_MPI *) xin->data;
  int         rank,ierr,len, work = x->n,n,j,size, fdes;
  MPI_Status  status;
  Scalar      *values;

  ierr = ViewerFileGetDescriptor_Private(ptr,&fdes); CHKERRQ(ierr);

  /* determine maximum message to arrive */
  MPI_Comm_rank(xin->comm,&rank);
  MPI_Reduce(&work,&len,1,MPI_INT,MPI_MAX,0,xin->comm);
  MPI_Comm_size(xin->comm,&size);

  if (!rank) {
    ierr = SYWrite(fdes,&xin->cookie,1,SYINT,0); CHKERRQ(ierr);
    ierr = SYWrite(fdes,&x->N,1,SYINT,0); CHKERRQ(ierr);
    ierr = SYWrite(fdes,x->array,x->n,SYSCALAR,0); 
    CHKERRQ(ierr);

    values = (Scalar *) PetscMalloc( len*sizeof(Scalar) ); CHKPTRQ(values);
    /* receive and print messages */
    for ( j=1; j<size; j++ ) {
      MPI_Recv(values,len,MPIU_SCALAR,j,47,xin->comm,&status);
      MPI_Get_count(&status,MPIU_SCALAR,&n);          
      ierr = SYWrite(fdes,values,n,SYSCALAR,0); 
    }
    PetscFree(values);
  }
  else {
    /* send values */
    MPI_Send(x->array,x->n,MPIU_SCALAR,0,47,xin->comm);
  }
  return 0;
}

static int VecView_MPI_Draw(Vec xin, Viewer v )
{
  Vec_MPI     *x = (Vec_MPI *) xin->data;
  int         i,rank,size,ierr,start,end;
  MPI_Status  status;
  double      coors[4],ymin,ymax,xmin,xmax,tmp;
  Draw        draw;
  PetscTruth  isnull;

  ViewerDrawGetDraw(v,&draw);
  ierr = DrawIsNull(draw,&isnull); CHKERRQ(ierr); if (isnull) return 0;
  MPI_Comm_size(xin->comm,&size); 

  xmin = 1.e20; xmax = -1.e20;
  for ( i=0; i<x->n; i++ ) {
#if defined(PETSC_COMPLEX)
    if (real(x->array[i]) < xmin) xmin = real(x->array[i]);
    else if (real(x->array[i]) > xmax) xmax = real(x->array[i]);
#else
    if (x->array[i] < xmin) xmin = x->array[i];
    else if (x->array[i] > xmax) xmax = x->array[i];
#endif
  }
  MPI_Reduce(&xmin,&ymin,1,MPI_DOUBLE,MPI_MIN,0,xin->comm);
  MPI_Reduce(&xmax,&ymax,1,MPI_DOUBLE,MPI_MAX,0,xin->comm);
  MPI_Comm_rank(xin->comm,&rank);
  if (!rank) {
    DrawAxis axis;
    DrawClear(draw); DrawFlush(draw);
    ierr = DrawAxisCreate(draw,&axis); CHKERRQ(ierr);
    PLogObjectParent(draw,axis);
    ierr = DrawAxisSetLimits(axis,0.0,(double) x->N,ymin,ymax); CHKERRQ(ierr);
    ierr = DrawAxisDraw(axis); CHKERRQ(ierr);
    DrawAxisDestroy(axis);
    DrawGetCoordinates(draw,coors,coors+1,coors+2,coors+3);
  }
  MPI_Bcast(coors,4,MPI_DOUBLE,0,xin->comm);
  if (rank) DrawSetCoordinates(draw,coors[0],coors[1],coors[2],coors[3]);
  /* draw local part of vector */
  VecGetOwnershipRange(xin,&start,&end);
  if (rank < size-1) { /*send value to right */
    MPI_Send(&x->array[x->n-1],1,MPI_DOUBLE,rank+1,xin->tag,xin->comm);
  }
  for ( i=1; i<x->n; i++ ) {
#if !defined(PETSC_COMPLEX)
    DrawLine(draw,(double)(i-1+start),x->array[i-1],(double)(i+start),
                   x->array[i],DRAW_RED);
#else
    DrawLine(draw,(double)(i-1+start),real(x->array[i-1]),(double)(i+start),
                   real(x->array[i]),DRAW_RED);
#endif
  }
  if (rank) { /* receive value from right */
    MPI_Recv(&tmp,1,MPI_DOUBLE,rank-1,xin->tag,xin->comm,&status);
#if !defined(PETSC_COMPLEX)
    DrawLine(draw,(double)start-1,tmp,(double)start,x->array[0],DRAW_RED);
#else
    DrawLine(draw,(double)start-1,tmp,(double)start,real(x->array[0]),DRAW_RED);
#endif
  }
  ierr = DrawSyncFlush(draw); CHKERRQ(ierr);
  ierr = DrawPause(draw); CHKERRQ(ierr);
  return 0;
}

static int VecView_MPI_LG(Vec xin, DrawLG lg )
{
  Vec_MPI     *x = (Vec_MPI *) xin->data;
  int         i,rank,size, N = x->N,*lens;
  Draw        draw;
  double      *xx,*yy;

  MPI_Comm_rank(xin->comm,&rank);
  MPI_Comm_size(xin->comm,&size);
  if (!rank) {
    DrawLGReset(lg);
    xx = (double *) PetscMalloc( 2*N*sizeof(double) ); CHKPTRQ(xx);
    for ( i=0; i<N; i++ ) {xx[i] = (double) i;}
    yy = xx + N;
    lens = (int *) PetscMalloc(size*sizeof(int)); CHKPTRQ(lens);
    for (i=0; i<size; i++ ) {
      lens[i] = x->ownership[i+1] - x->ownership[i];
    }
#if !defined(PETSC_COMPLEX)
    MPI_Gatherv(x->array,x->n,MPI_DOUBLE,yy,lens,x->ownership,MPI_DOUBLE,0,xin->comm);
#else
    {
      double *xr;
      xr = (double *) PetscMalloc( x->n*sizeof(double) ); CHKPTRQ(xr);
      for ( i=0; i<x->n; i++ ) {
        xr[i] = real(x->array[i]);
      }
      MPI_Gatherv(xr,x->n,MPI_DOUBLE,yy,lens,x->ownership,MPI_DOUBLE,0,xin->comm);
      PetscFree(xr);
    }
#endif
    PetscFree(lens);
    DrawLGAddPoints(lg,N,&xx,&yy);
    PetscFree(xx);
    DrawLGDraw(lg);
  }
  else {
#if !defined(PETSC_COMPLEX)
    MPI_Gatherv(x->array,x->n,MPI_DOUBLE,0,0,0,MPI_DOUBLE,0,xin->comm);
#else
    {
      double *xr;
      xr = (double *) PetscMalloc( x->n*sizeof(double) ); CHKPTRQ(xr);
      for ( i=0; i<x->n; i++ ) {
        xr[i] = real(x->array[i]);
      }
      MPI_Gatherv(xr,x->n,MPI_DOUBLE,0,0,0,MPI_DOUBLE,0,xin->comm);
      PetscFree(xr);
    }
#endif
  }
  DrawLGGetDraw(lg,&draw);
  DrawSyncFlush(draw);
  DrawPause(draw);
  return 0;
}

static int VecView_MPI_Matlab(Vec xin, Viewer viewer )
{
  Vec_MPI     *x = (Vec_MPI *) xin->data;
  int         i,rank,size, N = x->N,*lens;
  double      *xx;

#if defined(PETSC_COMPLEX)
  SETERRQ(1,"VecView_MPI_Matlab:Complex not done");
#else
  MPI_Comm_rank(xin->comm,&rank);
  MPI_Comm_size(xin->comm,&size);
  if (!rank) {
    xx = (double *) PetscMalloc( N*sizeof(double) ); CHKPTRQ(xx);
    lens = (int *) PetscMalloc(size*sizeof(int)); CHKPTRQ(lens);
    for (i=0; i<size; i++ ) {
      lens[i] = x->ownership[i+1] - x->ownership[i];
    }
    MPI_Gatherv(x->array,x->n,MPI_DOUBLE,xx,lens,x->ownership,MPI_DOUBLE,0,xin->comm);
    PetscFree(lens);
    ViewerMatlabPutArray_Private(viewer,N,1,xx);
    PetscFree(xx);
  }
  else {
    MPI_Gatherv(x->array,x->n,MPI_DOUBLE,0,0,0,MPI_DOUBLE,0,xin->comm);
  }
  return 0;
#endif
}

static int VecView_MPI(PetscObject obj,Viewer viewer)
{
  Vec         xin = (Vec) obj;
  ViewerType  vtype;
  int         ierr;

  if (!viewer) { /* so that viewers may be used from debuggers */
    viewer = STDOUT_VIEWER_SELF;
  }

  if (((PetscObject)viewer)->cookie == LG_COOKIE){
    return VecView_MPI_LG(xin,(DrawLG) viewer);
  }

  ierr = ViewerGetType(viewer,&vtype);
  if (vtype == ASCII_FILE_VIEWER){
    return VecView_MPI_File(xin,viewer);
  }
  else if (vtype == ASCII_FILES_VIEWER){
    return VecView_MPI_Files(xin,viewer);
  }
  else if (vtype == MATLAB_VIEWER) {
    return VecView_MPI_Matlab(xin,viewer);
  } 
  else if (vtype == BINARY_FILE_VIEWER) {
    return VecView_MPI_Binary(xin,viewer);
  }
  else if (vtype == DRAW_VIEWER) {
    return VecView_MPI_Draw(xin,viewer);
  }
  return 0;
}

static int VecGetSize_MPI(Vec xin,int *N)
{
  Vec_MPI  *x = (Vec_MPI *)xin->data;
  *N = x->N;
  return 0;
}
/*
      Uses a slow search to determine if item is already cached. 
   Could keep cache list sorted at all times.
*/
static int VecSetValues_MPI(Vec xin, int ni, int *ix, Scalar* y,InsertMode addv)
{
  Vec_MPI  *x = (Vec_MPI *)xin->data;
  int        rank = x->rank, *owners = x->ownership, start = owners[rank];
  int        end = owners[rank+1], i;
  Scalar     *xx = x->array;

  if (x->insertmode == INSERT_VALUES && addv == ADD_VALUES) { SETERRQ(1,
   "VecSetValues_MPI:You have already inserted values; you cannot now add");
  }
  else if (x->insertmode == ADD_VALUES && addv == INSERT_VALUES) { SETERRQ(1,
   "VecSetValues_MPI:You have already added values; you cannot now insert");
  }
  x->insertmode = addv;

  for ( i=0; i<ni; i++ ) {
    if ( ix[i] >= start && ix[i] < end) {
      if (addv == INSERT_VALUES) xx[ix[i]-start] = y[i];
      else                      xx[ix[i]-start] += y[i];
    }
    else {
      if (ix[i] < 0 || ix[i] > x->N) SETERRQ(1,"VecSetValues_MPI:Out of range");
        if (x->stash.n == x->stash.nmax) {/* cache is full */
          int    *idx, nmax = x->stash.nmax;
          Scalar *array;
          array = (Scalar *) PetscMalloc( (nmax+200)*sizeof(Scalar) + 
                                     (nmax+200)*sizeof(int) ); CHKPTRQ(array);
          PLogObjectMemory(xin,200*(sizeof(Scalar) + sizeof(int)));
          idx = (int *) (array + nmax + 200);
          PetscMemcpy(array,x->stash.array,nmax*sizeof(Scalar));
          PetscMemcpy(idx,x->stash.idx,nmax*sizeof(int));
          if (x->stash.array) PetscFree(x->stash.array);
          x->stash.array = array; x->stash.idx = idx;
          x->stash.nmax += 200;
        }
        x->stash.array[x->stash.n] = y[i];
        x->stash.idx[x->stash.n++] = ix[i];
    }
  }
  return 0;
}

/*
   Since nsends or nreceives may be zero we add 1 in certain mallocs
to make sure we never malloc an empty one.      
*/
static int VecAssemblyBegin_MPI(Vec xin)
{
  Vec_MPI    *x = (Vec_MPI *)xin->data;
  int         rank = x->rank, *owners = x->ownership, size = x->size;
  int         *nprocs,i,j,idx,*procs,nsends,nreceives,nmax,*work;
  int         *owner,*starts,count,tag = xin->tag;
  InsertMode  addv;
  Scalar      *rvalues,*svalues;
  MPI_Comm    comm = xin->comm;
  MPI_Request *send_waits,*recv_waits;

  /* make sure all processors are either in INSERTMODE or ADDMODE */
  MPI_Allreduce(&x->insertmode,&addv,1,MPI_INT,MPI_BOR,comm);
  if (addv == (ADD_VALUES|INSERT_VALUES)) { SETERRQ(1,
    "VecAssemblyBegin_MPI:Some processors inserted values while others added");
  }
  x->insertmode = addv; /* in case this processor had no cache */

  /*  first count number of contributors to each processor */
  nprocs = (int *) PetscMalloc( 2*size*sizeof(int) ); CHKPTRQ(nprocs);
  PetscMemzero(nprocs,2*size*sizeof(int)); procs = nprocs + size;
  owner = (int *) PetscMalloc( (x->stash.n+1)*sizeof(int) ); CHKPTRQ(owner);
  for ( i=0; i<x->stash.n; i++ ) {
    idx = x->stash.idx[i];
    for ( j=0; j<size; j++ ) {
      if (idx >= owners[j] && idx < owners[j+1]) {
        nprocs[j]++; procs[j] = 1; owner[i] = j; break;
      }
    }
  }
  nsends = 0;  for ( i=0; i<size; i++ ) { nsends += procs[i];} 

  /* inform other processors of number of messages and max length*/
  work = (int *) PetscMalloc( size*sizeof(int) ); CHKPTRQ(work);
  MPI_Allreduce(procs, work,size,MPI_INT,MPI_SUM,comm);
  nreceives = work[rank]; 
  MPI_Allreduce(nprocs,work,size,MPI_INT,MPI_MAX,comm);
  nmax = work[rank];
  PetscFree(work);

  /* post receives: 
       1) each message will consist of ordered pairs 
     (global index,value) we store the global index as a double 
     to simply the message passing. 
       2) since we don't know how long each individual message is we 
     allocate the largest needed buffer for each receive. Potentially 
     this is a lot of wasted space.
       This could be done better.
  */
  rvalues = (Scalar *) PetscMalloc(2*(nreceives+1)*(nmax+1)*sizeof(Scalar));
  CHKPTRQ(rvalues);
  recv_waits = (MPI_Request *) PetscMalloc((nreceives+1)*sizeof(MPI_Request));
  CHKPTRQ(recv_waits);
  for ( i=0; i<nreceives; i++ ) {
    MPI_Irecv(rvalues+2*nmax*i,2*nmax,MPIU_SCALAR,MPI_ANY_SOURCE,tag,
              comm,recv_waits+i);
  }

  /* do sends:
      1) starts[i] gives the starting index in svalues for stuff going to 
         the ith processor
  */
  svalues = (Scalar *) PetscMalloc(2*(x->stash.n+1)*sizeof(Scalar));CHKPTRQ(svalues);
  send_waits = (MPI_Request *) PetscMalloc( (nsends+1)*sizeof(MPI_Request));
  CHKPTRQ(send_waits);
  starts = (int *) PetscMalloc( size*sizeof(int) ); CHKPTRQ(starts);
  starts[0] = 0; 
  for ( i=1; i<size; i++ ) { starts[i] = starts[i-1] + nprocs[i-1];} 
  for ( i=0; i<x->stash.n; i++ ) {
    svalues[2*starts[owner[i]]]       = (Scalar)  x->stash.idx[i];
    svalues[2*(starts[owner[i]]++)+1] =  x->stash.array[i];
  }
  PetscFree(owner);
  starts[0] = 0;
  for ( i=1; i<size; i++ ) { starts[i] = starts[i-1] + nprocs[i-1];} 
  count = 0;
  for ( i=0; i<size; i++ ) {
    if (procs[i]) {
      MPI_Isend(svalues+2*starts[i],2*nprocs[i],MPIU_SCALAR,i,tag,
                comm,send_waits+count++);
    }
  }
  PetscFree(starts); PetscFree(nprocs);

  /* Free cache space */
  x->stash.nmax = x->stash.n = 0;
  if (x->stash.array){ PetscFree(x->stash.array); x->stash.array = 0;}

  x->svalues    = svalues;       x->rvalues = rvalues;
  x->nsends     = nsends;         x->nrecvs = nreceives;
  x->send_waits = send_waits; x->recv_waits = recv_waits;
  x->rmax       = nmax;
  
  return 0;
}

static int VecAssemblyEnd_MPI(Vec vec)
{
  Vec_MPI     *x = (Vec_MPI *)vec->data;
  MPI_Status  *send_status,recv_status;
  int         imdex,base,nrecvs = x->nrecvs, count = nrecvs, i, n;
  Scalar      *values;

  base = x->ownership[x->rank];

  /*  wait on receives */
  while (count) {
    MPI_Waitany(nrecvs,x->recv_waits,&imdex,&recv_status);
    /* unpack receives into our local space */
    values = x->rvalues + 2*imdex*x->rmax;
    MPI_Get_count(&recv_status,MPIU_SCALAR,&n);
    n = n/2;
    if (x->insertmode == ADD_VALUES) {
      for ( i=0; i<n; i++ ) {
        x->array[((int) PetscReal(values[2*i])) - base] += values[2*i+1];
      }
    }
    else if (x->insertmode == INSERT_VALUES) {
      for ( i=0; i<n; i++ ) {
        x->array[((int) PetscReal(values[2*i])) - base] = values[2*i+1];
      }
    }
    else { SETERRQ(1,
      "VecAssemblyEnd_MPI:Insert mode is not set correctly; corrupted vector");
    }
    count--;
  }
  PetscFree(x->recv_waits); PetscFree(x->rvalues);
 
  /* wait on sends */
  if (x->nsends) {
    send_status = (MPI_Status *) PetscMalloc( x->nsends*sizeof(MPI_Status) );
    CHKPTRQ(send_status);
    MPI_Waitall(x->nsends,x->send_waits,send_status);
    PetscFree(send_status);
  }
  PetscFree(x->send_waits); PetscFree(x->svalues);

  x->insertmode = NOT_SET_VALUES;
  return 0;
}

