
static int VeiPDestroyVector(PetscObject obj )
{
  Vec       v = (Vec ) obj;
  DvPVector *x = (DvPVector *) v->data;
  if (x->stash.array) FREE(x->stash.array);
  FREE(v->data); FREE(v);
  return 0;
}

static int VeiDVPview( PetscObject obj, Viewer ptr )
{
  Vec xin = (Vec) obj;
  DvPVector *x = (DvPVector *) xin->data;
  int i,j,mytid;

  MPI_Comm_rank(x->comm,&mytid); 

  MPE_Seq_begin(x->comm,1);
    printf("Processor [%d] \n",mytid);
    for ( i=0; i<x->n; i++ ) {
#if defined(PETSC_COMPLEX)
      printf("%g + %g i\n",real(x->array[i]),imag(x->array[i]));
#else
      printf("%g \n",x->array[i]);
#endif
    }
    fflush(stdout);
  MPE_Seq_end(x->comm,1);
  return 0;
}

static int VeiPgsize(Vec xin,int *N)
{
  DvPVector  *x = (DvPVector *)xin->data;
  *N = x->N;
  return 0;
}
/*
      Uses a slow search to determine if item is already cached. 
   Could keep cache list sorted at all times.
*/
static int VeiPDVinsertvalues(Vec xin, int ni, int *ix, Scalar* y,
                                   InsertMode addv )
{
  DvPVector  *x = (DvPVector *)xin->data;
  int        mytid = x->mytid, *owners = x->ownership, start = owners[mytid];
  int        end = owners[mytid+1], i, j, alreadycached;
  Scalar     *xx = x->array;

#if defined(PETSC_DEBUG)
  if (x->insertmode == InsertValues && addv == AddValues) {
    SETERR(1,"You have already inserted vector values, you cannot now add");
  }
  else if (x->insertmode == AddValues && addv == InsertValues) {
    SETERR(1,"You have already added vector values, you cannot now insert");
  }
#endif
  x->insertmode = addv;

  for ( i=0; i<ni; i++ ) {
    if ( ix[i] >= start && ix[i] < end) {
      if (addv == InsertValues) xx[ix[i]-start] = y[i];
      else                      xx[ix[i]-start] += y[i];
    }
    else {
#if defined(PETSC_DEBUG)
      if (ix[i] < 0 || ix[i] > x->N) SETERR(1,"Index out of range");
#endif
      /* check if this index has already been cached */
      alreadycached = 0;
      for ( j=0; j<x->stash.n; j++ ) {
        if (x->stash.idx[j] == ix[i]) {
          if (addv == InsertValues) x->stash.array[j] = y[i];
          else                      x->stash.array[j] += y[i];
          alreadycached = 1; 
          break;
        }
      }
      if (!alreadycached) {
        if (x->stash.n == x->stash.nmax) {/* cache is full */
          int    *idx, nmax = x->stash.nmax;
          Scalar *array;
          array = (Scalar *) MALLOC( (nmax+10)*sizeof(Scalar) + 
                                     (nmax+10)*sizeof(int) ); CHKPTR(array);
          idx = (int *) (array + nmax + 10);
          MEMCPY(array,x->stash.array,nmax*sizeof(Scalar));
          MEMCPY(idx,x->stash.idx,nmax*sizeof(int));
          if (x->stash.array) FREE(x->stash.array);
          x->stash.array = array; x->stash.idx = idx;
          x->stash.nmax += 10;
        }
        x->stash.array[x->stash.n] = y[i];
        x->stash.idx[x->stash.n++] = ix[i];
      }
    }
  }
  return 0;
}

/*
   Since nsends or nreceives may be zero we add 1 in certain mallocs
to make sure we never malloc an empty one.      
*/
static int VeiDVPBeginAssembly(Vec xin)
{
  DvPVector   *x = (DvPVector *)xin->data;
  int         mytid = x->mytid, *owners = x->ownership, numtids = x->numtids;
  int         *nprocs,i,j,n,idx,*procs,nsends,nreceives,nmax,*work;
  int         *owner,*starts,count,tag = 22;
  InsertMode  addv;
  Scalar      *rvalues,*svalues;
  MPI_Comm    comm = x->comm;
  MPI_Request *send_waits,*recv_waits;

  /* make sure all processors are either in INSERTMODE or ADDMODE */
  MPI_Allreduce((void *) &x->insertmode,(void *) &addv,numtids,MPI_INT,
                MPI_BOR,comm);
  if (addv == (AddValues|InsertValues)) {
    SETERR(1,"Some processors have inserted while others have added");
  }
  x->insertmode = addv; /* in case this processor had no cache */

  /*  first count number of contributors to each processor */
  nprocs = (int *) MALLOC( 2*numtids*sizeof(int) ); CHKPTR(nprocs);
  MEMSET(nprocs,0,2*numtids*sizeof(int)); procs = nprocs + numtids;
  owner = (int *) MALLOC( (x->stash.n+1)*sizeof(int) ); CHKPTR(owner);
  for ( i=0; i<x->stash.n; i++ ) {
    idx = x->stash.idx[i];
    for ( j=0; j<numtids; j++ ) {
      if (idx >= owners[j] && idx < owners[j+1]) {
        nprocs[j]++; procs[j] = 1; owner[i] = j; break;
      }
    }
  }
  nsends = 0;  for ( i=0; i<numtids; i++ ) { nsends += procs[i];} 

  /* inform other processors of number of messages and max length*/
  work = (int *) MALLOC( numtids*sizeof(int) ); CHKPTR(work);
  MPI_Allreduce((void *) procs,(void *) work,numtids,MPI_INT,MPI_SUM,comm);
  nreceives = work[mytid]; 
  MPI_Allreduce((void *) nprocs,(void *) work,numtids,MPI_INT,MPI_MAX,comm);
  nmax = work[mytid];
  FREE(work);

  /* post receives: 
       1) each message will consist of ordered pairs 
     (global index,value) we store the global index as a double 
     to simply the message passing. 
       2) since we don't know how long each individual message is we 
     allocate the largest needed buffer for each receive. Potentially 
     this is a lot of wasted space.
       This could be done better.
  */
  rvalues = (Scalar *) MALLOC(2*(nreceives+1)*nmax*sizeof(Scalar));
  CHKPTR(rvalues);
  recv_waits = (MPI_Request *) MALLOC((nreceives+1)*sizeof(MPI_Request));
  CHKPTR(recv_waits);
  for ( i=0; i<nreceives; i++ ) {
    MPI_Irecv((void *)(rvalues+2*nmax*i),2*nmax,MPI_SCALAR,MPI_ANY_SOURCE,tag,
              comm,recv_waits+i);
  }

  /* do sends:
      1) starts[i] gives the starting index in svalues for stuff going to 
         the ith processor
  */
  svalues = (Scalar *) MALLOC( 2*(x->stash.n+1)*sizeof(Scalar) );
  CHKPTR(svalues);
  send_waits = (MPI_Request *) MALLOC( (nsends+1)*sizeof(MPI_Request));
  CHKPTR(send_waits);
  starts = (int *) MALLOC( numtids*sizeof(int) ); CHKPTR(starts);
  starts[0] = 0; 
  for ( i=1; i<numtids; i++ ) { starts[i] = starts[i-1] + nprocs[i-1];} 
  for ( i=0; i<x->stash.n; i++ ) {
    svalues[2*starts[owner[i]]]       = (Scalar)  x->stash.idx[i];
    svalues[2*(starts[owner[i]]++)+1] =  x->stash.array[i];
  }
  FREE(owner);
  starts[0] = 0;
  for ( i=1; i<numtids; i++ ) { starts[i] = starts[i-1] + nprocs[i-1];} 
  count = 0;
  for ( i=0; i<numtids; i++ ) {
    if (procs[i]) {
      MPI_Isend((void*)(svalues+2*starts[i]),2*nprocs[i],MPI_SCALAR,i,tag,
                comm,send_waits+count++);
    }
  }
  FREE(starts); FREE(nprocs);

  /* Free cache space */
  x->stash.nmax = x->stash.n = 0;
  if (x->stash.array){ FREE(x->stash.array); x->stash.array = 0;}

  x->svalues    = svalues;       x->rvalues = rvalues;
  x->nsends     = nsends;         x->nrecvs = nreceives;
  x->send_waits = send_waits; x->recv_waits = recv_waits;
  x->rmax       = nmax;
  
  return 0;
}

static int VeiDVPEndAssembly(Vec vec)
{
  DvPVector   *x = (DvPVector *)vec->data;
  MPI_Status  *send_status,recv_status;
  int         index,idx,base,nrecvs = x->nrecvs, count = nrecvs, i, n;
  Scalar      *values;

  base = x->ownership[x->mytid];

  /*  wait on receives */
  while (count) {
    MPI_Waitany(nrecvs,x->recv_waits,&index,&recv_status);
    /* unpack receives into our local space */
    values = x->rvalues + 2*index*x->rmax;
    MPI_Get_count(&recv_status,MPI_SCALAR,&n);
    n = n/2;
    if (x->insertmode == AddValues) {
      for ( i=0; i<n; i++ ) {
        x->array[((int) PETSCREAL(values[2*i])) - base] += values[2*i+1];
      }
    }
    else if (x->insertmode == InsertValues) {
      for ( i=0; i<n; i++ ) {
        x->array[((int) PETSCREAL(values[2*i])) - base] = values[2*i+1];
      }
    }
    else {
        SETERR(1,"Insert mode is not set correct; corrupt vector");
    }
    count--;
  }
  FREE(x->recv_waits); FREE(x->rvalues);
 
  /* wait on sends */
  if (x->nsends) {
    send_status = (MPI_Status *) MALLOC( x->nsends*sizeof(MPI_Status) );
    CHKPTR(send_status);
    MPI_Waitall(x->nsends,x->send_waits,send_status);
    FREE(send_status);
  }
  FREE(x->send_waits); FREE(x->svalues);

  x->insertmode = NotSetValues;
  return 0;
}

