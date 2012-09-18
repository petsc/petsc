
/*
    The memory scalable AO application ordering routines. These store the
  local orderings on each processor.
*/

#include <../src/dm/ao/aoimpl.h>          /*I  "petscao.h"   I*/

typedef struct {
  PetscInt     *app_loc;   /* app_loc[i] is the partner for the ith local PETSc slot */
  PetscInt     *petsc_loc; /* petsc_loc[j] is the partner for the jth local app slot */
  PetscLayout  map;        /* determines the local sizes of ao */
} AO_MemoryScalable;

/*
       All processors have the same data so processor 1 prints it
*/
#undef __FUNCT__
#define __FUNCT__ "AOView_MemoryScalable"
PetscErrorCode AOView_MemoryScalable(AO ao,PetscViewer viewer)
{
  PetscErrorCode    ierr;
  PetscMPIInt       rank,size;
  AO_MemoryScalable *aomems = (AO_MemoryScalable*)ao->data;
  PetscBool         iascii;
  PetscMPIInt       tag_app,tag_petsc;
  PetscLayout       map = aomems->map;
  PetscInt          *app,*app_loc,*petsc,*petsc_loc,len,i,j;
  MPI_Status        status;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (!iascii) SETERRQ1(((PetscObject)viewer)->comm,PETSC_ERR_SUP,"Viewer type %s not supported for AO MemoryScalable",((PetscObject)viewer)->type_name);

  ierr = MPI_Comm_rank(((PetscObject)ao)->comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(((PetscObject)ao)->comm,&size);CHKERRQ(ierr);

  ierr   = PetscObjectGetNewTag((PetscObject)ao,&tag_app);CHKERRQ(ierr);
  ierr   = PetscObjectGetNewTag((PetscObject)ao,&tag_petsc);CHKERRQ(ierr);

  if (!rank){
    ierr = PetscViewerASCIIPrintf(viewer,"Number of elements in ordering %D\n",ao->N);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,  "PETSc->App  App->PETSc\n");CHKERRQ(ierr);

    ierr = PetscMalloc2(map->N,PetscInt,&app,map->N,PetscInt,&petsc);CHKERRQ(ierr);
    len = map->n;
    /* print local AO */
    ierr = PetscViewerASCIIPrintf(viewer,"Process [%D]\n",rank);
    for (i=0; i<len; i++){
      ierr = PetscViewerASCIIPrintf(viewer,"%3D  %3D    %3D  %3D\n",i,aomems->app_loc[i],i,aomems->petsc_loc[i]);CHKERRQ(ierr);
    }

    /* recv and print off-processor's AO */
    for (i=1; i<size; i++) {
      len = map->range[i+1] - map->range[i];
      app_loc   = app  + map->range[i];
      petsc_loc = petsc+ map->range[i];
      ierr = MPI_Recv(app_loc,(PetscMPIInt)len,MPIU_INT,i,tag_app,((PetscObject)ao)->comm,&status);CHKERRQ(ierr);
      ierr = MPI_Recv(petsc_loc,(PetscMPIInt)len,MPIU_INT,i,tag_petsc,((PetscObject)ao)->comm,&status);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"Process [%D]\n",i);
      for (j=0; j<len; j++) {
        ierr = PetscViewerASCIIPrintf(viewer,"%3D  %3D    %3D  %3D\n",map->range[i]+j,app_loc[j],map->range[i]+j,petsc_loc[j]);CHKERRQ(ierr);
      }
    }
    ierr = PetscFree2(app,petsc);CHKERRQ(ierr);

  } else {
    /* send values */
    ierr = MPI_Send((void*)aomems->app_loc,map->n,MPIU_INT,0,tag_app,((PetscObject)ao)->comm);CHKERRQ(ierr);
    ierr = MPI_Send((void*)aomems->petsc_loc,map->n,MPIU_INT,0,tag_petsc,((PetscObject)ao)->comm);CHKERRQ(ierr);
  }
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "AODestroy_MemoryScalable"
PetscErrorCode AODestroy_MemoryScalable(AO ao)
{
  AO_MemoryScalable *aomems = (AO_MemoryScalable*)ao->data;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscFree2(aomems->app_loc,aomems->petsc_loc);CHKERRQ(ierr);
  ierr = PetscLayoutDestroy(&aomems->map);CHKERRQ(ierr);
  ierr = PetscFree(aomems);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   Input Parameters:
+   ao - the application ordering context
.   n  - the number of integers in ia[]
.   ia - the integers; these are replaced with their mapped value
-   maploc - app_loc or petsc_loc in struct "AO_MemoryScalable"

   Output Parameter:
.   ia - the mapped interges
 */
#undef __FUNCT__
#define __FUNCT__ "AOMap_MemoryScalable_private"
PetscErrorCode AOMap_MemoryScalable_private(AO ao,PetscInt n,PetscInt *ia,PetscInt *maploc)
{
  PetscErrorCode    ierr;
  AO_MemoryScalable *aomems = (AO_MemoryScalable*)ao->data;
  MPI_Comm          comm=((PetscObject)ao)->comm;
  PetscMPIInt       rank,size,tag1,tag2;
  PetscInt          *owner,*start,*nprocs,nsends,nreceives;
  PetscInt          nmax,count,*sindices,*rindices,i,j,idx,lastidx,*sindices2,*rindices2;
  PetscInt          *owners = aomems->map->range;
  MPI_Request       *send_waits,*recv_waits,*send_waits2,*recv_waits2;
  MPI_Status        recv_status;
  PetscMPIInt       nindices,source,widx;
  PetscInt          *rbuf,*sbuf;
  MPI_Status        *send_status,*send_status2;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);

  /*  first count number of contributors to each processor */
  ierr   = PetscMalloc2(2*size,PetscInt,&nprocs,size,PetscInt,&start);CHKERRQ(ierr);
  ierr   = PetscMemzero(nprocs,2*size*sizeof(PetscInt));CHKERRQ(ierr);
  ierr   = PetscMalloc(n*sizeof(PetscInt),&owner);CHKERRQ(ierr);
  ierr   = PetscMemzero(owner,n*sizeof(PetscInt));CHKERRQ(ierr);

  j       = 0;
  lastidx = -1;
  for (i=0; i<n; i++) {
    /* if indices are NOT locally sorted, need to start search at the beginning */
    if (lastidx > (idx = ia[i])) j = 0;
    lastidx = idx;
    for (; j<size; j++) {
      if (idx >= owners[j] && idx < owners[j+1]) {
        nprocs[2*j]++;     /* num of indices to be sent */
        nprocs[2*j+1] = 1; /* send to proc[j] */
        owner[i] = j;
        break;
      }
    }
  }
  nprocs[2*rank]=nprocs[2*rank+1]=0; /* do not receive from self! */
  nsends = 0;
  for (i=0; i<size; i++) nsends += nprocs[2*i+1];

  /* inform other processors of number of messages and max length*/
  ierr = PetscMaxSum(comm,nprocs,&nmax,&nreceives);CHKERRQ(ierr);

  /* allocate arrays */
  ierr = PetscObjectGetNewTag((PetscObject)ao,&tag1);CHKERRQ(ierr);
  ierr = PetscObjectGetNewTag((PetscObject)ao,&tag2);CHKERRQ(ierr);

  ierr = PetscMalloc2(nreceives*nmax,PetscInt,&rindices,nreceives,MPI_Request,&recv_waits);CHKERRQ(ierr);
  ierr = PetscMalloc2(nsends*nmax,PetscInt,&rindices2,nsends,MPI_Request,&recv_waits2);CHKERRQ(ierr);

  ierr = PetscMalloc3(n,PetscInt,&sindices,nsends,MPI_Request,&send_waits,nsends,MPI_Status,&send_status);CHKERRQ(ierr);
  ierr = PetscMalloc3(n,PetscInt,&sindices2,nreceives,MPI_Request,&send_waits2,nreceives,MPI_Status,&send_status2);CHKERRQ(ierr);

  /* post 1st receives: receive others requests
     since we don't know how long each individual message is we
     allocate the largest needed buffer for each receive. Potentially
     this is a lot of wasted space.
  */
  for (i=0,count=0; i<nreceives; i++) {
    ierr = MPI_Irecv(rindices+nmax*i,nmax,MPIU_INT,MPI_ANY_SOURCE,tag1,comm,recv_waits+count++);CHKERRQ(ierr);
  }

  /* do 1st sends:
      1) starts[i] gives the starting index in svalues for stuff going to
         the ith processor
  */
  start[0] = 0;
  for (i=1; i<size; i++) start[i] = start[i-1] + nprocs[2*i-2];
  for (i=0; i<n; i++) {
    j = owner[i];
    if (j != rank){
      sindices[start[j]++]  = ia[i];
    } else { /* compute my own map */
      if (ia[i] >= owners[rank] && ia[i] < owners[rank+1] ) {
        ia[i] = maploc[ia[i]-owners[rank]];
      } else {
        ia[i] = -1 ; /* ia[i] is not in the range of 0 and N-1, maps it to -1 */
      }
    }
  }

  start[0] = 0;
  for (i=1; i<size; i++) { start[i] = start[i-1] + nprocs[2*i-2];}
  for (i=0,count=0; i<size; i++) {
    if (nprocs[2*i+1]) {
      /* send my request to others */
      ierr = MPI_Isend(sindices+start[i],nprocs[2*i],MPIU_INT,i,tag1,comm,send_waits+count);CHKERRQ(ierr);
      /* post receive for the answer of my request */
      ierr = MPI_Irecv(sindices2+start[i],nprocs[2*i],MPIU_INT,i,tag2,comm,recv_waits2+count);CHKERRQ(ierr);
      count++;
    }
  }
  if (nsends != count) SETERRQ2(comm,PETSC_ERR_SUP,"nsends %d != count %d",nsends,count);

  /* wait on 1st sends */
  if (nsends) {
    ierr = MPI_Waitall(nsends,send_waits,send_status);CHKERRQ(ierr);
  }

  /* 1st recvs: other's requests */
  for (j=0; j< nreceives; j++){
    ierr = MPI_Waitany(nreceives,recv_waits,&widx,&recv_status);CHKERRQ(ierr); /* idx: index of handle for operation that completed */
    ierr = MPI_Get_count(&recv_status,MPIU_INT,&nindices);CHKERRQ(ierr);
    rbuf = rindices+nmax*widx; /* global index */
    source = recv_status.MPI_SOURCE;

    /* compute mapping */
    sbuf = rbuf;
    for (i=0; i<nindices; i++)sbuf[i] = maploc[rbuf[i]-owners[rank]];

    /* send mapping back to the sender */
    ierr = MPI_Isend(sbuf,nindices,MPIU_INT,source,tag2,comm,send_waits2+widx);CHKERRQ(ierr);
  }

  /* wait on 2nd sends */
  if (nreceives) {
    ierr = MPI_Waitall(nreceives,send_waits2,send_status2);CHKERRQ(ierr);
  }

  /* 2nd recvs: for the answer of my request */
  for (j=0; j< nsends; j++){
    ierr = MPI_Waitany(nsends,recv_waits2,&widx,&recv_status);CHKERRQ(ierr);
    ierr = MPI_Get_count(&recv_status,MPIU_INT,&nindices);CHKERRQ(ierr);
    source = recv_status.MPI_SOURCE;
    /* pack output ia[] */
    rbuf=sindices2+start[source];
    count=0;
    for (i=0; i<n; i++){
      if (source == owner[i]) ia[i] = rbuf[count++];
    }
  }

  /* free arrays */
  ierr = PetscFree2(nprocs,start);CHKERRQ(ierr);
  ierr = PetscFree(owner);CHKERRQ(ierr);
  ierr = PetscFree2(rindices,recv_waits);CHKERRQ(ierr);
  ierr = PetscFree2(rindices2,recv_waits2);CHKERRQ(ierr);
  ierr = PetscFree3(sindices,send_waits,send_status);CHKERRQ(ierr);
  ierr = PetscFree3(sindices2,send_waits2,send_status2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "AOPetscToApplication_MemoryScalable"
PetscErrorCode AOPetscToApplication_MemoryScalable(AO ao,PetscInt n,PetscInt *ia)
{
  PetscErrorCode    ierr;
  AO_MemoryScalable *aomems = (AO_MemoryScalable*)ao->data;
  PetscInt          *app_loc = aomems->app_loc;

  PetscFunctionBegin;
  ierr = AOMap_MemoryScalable_private(ao,n,ia,app_loc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "AOApplicationToPetsc_MemoryScalable"
PetscErrorCode AOApplicationToPetsc_MemoryScalable(AO ao,PetscInt n,PetscInt *ia)
{
  PetscErrorCode    ierr;
  AO_MemoryScalable *aomems = (AO_MemoryScalable*)ao->data;
  PetscInt          *petsc_loc = aomems->petsc_loc;

  PetscFunctionBegin;
  ierr = AOMap_MemoryScalable_private(ao,n,ia,petsc_loc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static struct _AOOps AOOps_MemoryScalable = {
       AOView_MemoryScalable,
       AODestroy_MemoryScalable,
       AOPetscToApplication_MemoryScalable,
       AOApplicationToPetsc_MemoryScalable,
       0,
       0,
       0,
       0
};

#undef __FUNCT__
#define __FUNCT__ "AOCreateMemoryScalable_private"
PetscErrorCode  AOCreateMemoryScalable_private(MPI_Comm comm,PetscInt napp,const PetscInt from_array[],const PetscInt to_array[],AO ao, PetscInt *aomap_loc)
{
  PetscErrorCode    ierr;
  AO_MemoryScalable *aomems=(AO_MemoryScalable*)ao->data;
  PetscLayout       map=aomems->map;
  PetscInt          n_local = map->n,i,j;
  PetscMPIInt       rank,size,tag;
  PetscInt          *owner,*start,*nprocs,nsends,nreceives;
  PetscInt          nmax,count,*sindices,*rindices,idx,lastidx;
  PetscInt          *owners = aomems->map->range;
  MPI_Request       *send_waits,*recv_waits;
  MPI_Status        recv_status;
  PetscMPIInt       nindices,widx;
  PetscInt          *rbuf;
  PetscInt          n=napp,ip,ia;
  MPI_Status        *send_status;

  PetscFunctionBegin;
  ierr = PetscMemzero(aomap_loc,n_local*sizeof(PetscInt));CHKERRQ(ierr);

  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);

  /*  first count number of contributors (of from_array[]) to each processor */
  ierr = PetscMalloc(2*size*sizeof(PetscInt),&nprocs);CHKERRQ(ierr);
  ierr = PetscMemzero(nprocs,2*size*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMalloc(n*sizeof(PetscInt),&owner);CHKERRQ(ierr);

  j       = 0;
  lastidx = -1;
  for (i=0; i<n; i++) {
    /* if indices are NOT locally sorted, need to start search at the beginning */
    if (lastidx > (idx = from_array[i])) j = 0;
    lastidx = idx;
    for (; j<size; j++) {
      if (idx >= owners[j] && idx < owners[j+1]) {
        nprocs[2*j]  += 2; /* num of indices to be sent - in pairs (ip,ia) */
        nprocs[2*j+1] = 1; /* send to proc[j] */
        owner[i] = j;
        break;
      }
    }
  }
  nprocs[2*rank]=nprocs[2*rank+1]=0; /* do not receive from self! */
  nsends = 0;
  for (i=0; i<size; i++) nsends += nprocs[2*i+1];

  /* inform other processors of number of messages and max length*/
  ierr = PetscMaxSum(comm,nprocs,&nmax,&nreceives);CHKERRQ(ierr);

  /* allocate arrays */
  ierr = PetscObjectGetNewTag((PetscObject)ao,&tag);CHKERRQ(ierr);
  ierr = PetscMalloc2(nreceives*nmax,PetscInt,&rindices,nreceives,MPI_Request,&recv_waits);CHKERRQ(ierr);
  ierr = PetscMalloc3(2*n,PetscInt,&sindices,nsends,MPI_Request,&send_waits,nsends,MPI_Status,&send_status);CHKERRQ(ierr);
  ierr = PetscMalloc(size*sizeof(PetscInt),&start);CHKERRQ(ierr);

  /* post receives: */
  for (i=0; i<nreceives; i++) {
    ierr = MPI_Irecv(rindices+nmax*i,nmax,MPIU_INT,MPI_ANY_SOURCE,tag,comm,recv_waits+i);CHKERRQ(ierr);
  }

  /* do sends:
      1) starts[i] gives the starting index in svalues for stuff going to
         the ith processor
  */
  start[0] = 0;
  for (i=1; i<size; i++) start[i] = start[i-1] + nprocs[2*i-2];
  for (i=0; i<n; i++) {
    j = owner[i];
    if (j != rank){
      ip = from_array[i];
      ia = to_array[i];
      sindices[start[j]++]  = ip; sindices[start[j]++]  = ia;
    } else { /* compute my own map */
      ip = from_array[i] -owners[rank];
      ia = to_array[i];
      aomap_loc[ip] = ia;
    }
  }

  start[0] = 0;
  for (i=1; i<size; i++) { start[i] = start[i-1] + nprocs[2*i-2];}
  for (i=0,count=0; i<size; i++) {
    if (nprocs[2*i+1]) {
      ierr = MPI_Isend(sindices+start[i],nprocs[2*i],MPIU_INT,i,tag,comm,send_waits+count);CHKERRQ(ierr);
      count++;
    }
  }
  if (nsends != count) SETERRQ2(comm,PETSC_ERR_SUP,"nsends %d != count %d",nsends,count);

  /* wait on sends */
  if (nsends) {
    ierr = MPI_Waitall(nsends,send_waits,send_status);CHKERRQ(ierr);
  }

  /* recvs */
  count=0;
  for (j= nreceives; j>0; j--){
    ierr = MPI_Waitany(nreceives,recv_waits,&widx,&recv_status);CHKERRQ(ierr);
    ierr = MPI_Get_count(&recv_status,MPIU_INT,&nindices);CHKERRQ(ierr);
    rbuf = rindices+nmax*widx; /* global index */

    /* compute local mapping */
    for (i=0; i<nindices; i+=2){ /* pack aomap_loc */
      ip = rbuf[i] - owners[rank]; /* local index */
      ia = rbuf[i+1];
      aomap_loc[ip] = ia;
    }
    count++;
  }

  ierr = PetscFree(start);CHKERRQ(ierr);
  ierr = PetscFree3(sindices,send_waits,send_status);CHKERRQ(ierr);
  ierr = PetscFree2(rindices,recv_waits);CHKERRQ(ierr);
  ierr = PetscFree(owner);CHKERRQ(ierr);
  ierr = PetscFree(nprocs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "AOCreate_MemoryScalable"
PetscErrorCode AOCreate_MemoryScalable(AO ao)
{
  PetscErrorCode    ierr;
  IS                isapp=ao->isapp,ispetsc=ao->ispetsc;
  const PetscInt    *mypetsc,*myapp;
  PetscInt          napp,n_local,N,i,start,*petsc;
  MPI_Comm          comm;
  AO_MemoryScalable *aomems;
  PetscLayout       map;
  PetscMPIInt       *lens,size,rank,*disp;

  PetscFunctionBegin;
  /* create special struct aomems */
  ierr = PetscNewLog(ao, AO_MemoryScalable, &aomems);CHKERRQ(ierr);
  ao->data = (void*) aomems;
  ierr = PetscMemcpy(ao->ops,&AOOps_MemoryScalable,sizeof(struct _AOOps));CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)ao,AOMEMORYSCALABLE);CHKERRQ(ierr);

  /* transmit all local lengths of isapp to all processors */
  ierr = PetscObjectGetComm((PetscObject)isapp,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = PetscMalloc2(size,PetscMPIInt, &lens,size,PetscMPIInt,&disp);CHKERRQ(ierr);
  ierr = ISGetLocalSize(isapp,&napp);CHKERRQ(ierr);
  ierr = MPI_Allgather(&napp, 1, MPI_INT, lens, 1, MPI_INT, comm);CHKERRQ(ierr);

  N = 0;
  for (i = 0; i < size; i++) {
    disp[i] = N;
    N      += lens[i];
  }

  /* If ispetsc is 0 then use "natural" numbering */
  if (napp) {
    if (!ispetsc) {
      start = disp[rank];
      ierr  = PetscMalloc((napp+1) * sizeof(PetscInt), &petsc);CHKERRQ(ierr);
      for (i=0; i<napp; i++) petsc[i] = start + i;
    } else {
      ierr = ISGetIndices(ispetsc,&mypetsc);CHKERRQ(ierr);
      petsc = (PetscInt*)mypetsc;
    }
  }

  /* create a map with global size N - used to determine the local sizes of ao - shall we use local napp instead of N? */
  ierr = PetscLayoutCreate(comm,&map);CHKERRQ(ierr);
  map->bs = 1;
  map->N  = N;
  ierr = PetscLayoutSetUp(map);CHKERRQ(ierr);

  ao->N       = N;
  ao->n       = map->n;
  aomems->map = map;

  /* create distributed indices app_loc: petsc->app and petsc_loc: app->petsc */
  n_local = map->n;
  ierr = PetscMalloc2(n_local,PetscInt, &aomems->app_loc,n_local,PetscInt,&aomems->petsc_loc);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(ao,2*n_local*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemzero(aomems->app_loc,n_local*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemzero(aomems->petsc_loc,n_local*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = ISGetIndices(isapp,&myapp);CHKERRQ(ierr);

  ierr = AOCreateMemoryScalable_private(comm,napp,petsc,myapp,ao,aomems->app_loc);CHKERRQ(ierr);
  ierr = AOCreateMemoryScalable_private(comm,napp,myapp,petsc,ao,aomems->petsc_loc);CHKERRQ(ierr);

  ierr = ISRestoreIndices(isapp,&myapp);CHKERRQ(ierr);
  if (napp){
    if (ispetsc){
      ierr = ISRestoreIndices(ispetsc,&mypetsc);CHKERRQ(ierr);
    } else {
      ierr = PetscFree(petsc);CHKERRQ(ierr);
    }
  }
  ierr  = PetscFree2(lens,disp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "AOCreateMemoryScalable"
/*@C
   AOCreateMemoryScalable - Creates a memory scalable application ordering using two integer arrays.

   Collective on MPI_Comm

   Input Parameters:
+  comm - MPI communicator that is to share AO
.  napp - size of integer arrays
.  myapp - integer array that defines an ordering
-  mypetsc - integer array that defines another ordering (may be PETSC_NULL to
             indicate the natural ordering, that is 0,1,2,3,...)

   Output Parameter:
.  aoout - the new application ordering

   Level: beginner

    Notes: The arrays myapp and mypetsc must contain the all the integers 0 to napp-1 with no duplicates; that is there cannot be any "holes"
           in the indices. Use AOCreateMapping() or AOCreateMappingIS() if you wish to have "holes" in the indices.
           Comparing with AOCreateBasic(), this routine trades memory with message communication.

.keywords: AO, create

.seealso: AOCreateMemoryScalableIS(), AODestroy(), AOPetscToApplication(), AOApplicationToPetsc()
@*/
PetscErrorCode AOCreateMemoryScalable(MPI_Comm comm,PetscInt napp,const PetscInt myapp[],const PetscInt mypetsc[],AO *aoout)
{
  PetscErrorCode ierr;
  IS             isapp,ispetsc;
  const PetscInt *app=myapp,*petsc=mypetsc;

  PetscFunctionBegin;
  ierr = ISCreateGeneral(comm,napp,app,PETSC_USE_POINTER,&isapp);CHKERRQ(ierr);
  if (mypetsc){
    ierr = ISCreateGeneral(comm,napp,petsc,PETSC_USE_POINTER,&ispetsc);CHKERRQ(ierr);
  } else {
    ispetsc = PETSC_NULL;
  }
  ierr = AOCreateMemoryScalableIS(isapp,ispetsc,aoout);CHKERRQ(ierr);
  ierr = ISDestroy(&isapp);CHKERRQ(ierr);
  if (mypetsc){
    ierr = ISDestroy(&ispetsc);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "AOCreateMemoryScalableIS"
/*@C
   AOCreateMemoryScalableIS - Creates a memory scalable application ordering using two index sets.

   Collective on IS

   Input Parameters:
+  isapp - index set that defines an ordering
-  ispetsc - index set that defines another ordering (may be PETSC_NULL to use the
             natural ordering)

   Output Parameter:
.  aoout - the new application ordering

   Level: beginner

    Notes: The index sets isapp and ispetsc must contain the all the integers 0 to napp-1 (where napp is the length of the index sets) with no duplicates;
           that is there cannot be any "holes".
           Comparing with AOCreateBasicIS(), this routine trades memory with message communication.
.keywords: AO, create

.seealso: AOCreateMemoryScalable(),  AODestroy()
@*/
PetscErrorCode  AOCreateMemoryScalableIS(IS isapp,IS ispetsc,AO *aoout)
{
  PetscErrorCode ierr;
  MPI_Comm       comm;
  AO             ao;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)isapp,&comm);CHKERRQ(ierr);
  ierr = AOCreate(comm,&ao);CHKERRQ(ierr);
  ierr = AOSetIS(ao,isapp,ispetsc);CHKERRQ(ierr);
  ierr = AOSetType(ao,AOMEMORYSCALABLE);CHKERRQ(ierr);
  *aoout = ao;
  PetscFunctionReturn(0);
}
