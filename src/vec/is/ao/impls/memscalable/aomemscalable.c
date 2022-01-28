
/*
    The memory scalable AO application ordering routines. These store the
  orderings on each processor for that processor's range of values
*/

#include <../src/vec/is/ao/aoimpl.h>          /*I  "petscao.h"   I*/

typedef struct {
  PetscInt    *app_loc;    /* app_loc[i] is the partner for the ith local PETSc slot */
  PetscInt    *petsc_loc;  /* petsc_loc[j] is the partner for the jth local app slot */
  PetscLayout map;         /* determines the local sizes of ao */
} AO_MemoryScalable;

/*
       All processors ship the data to process 0 to be printed; note that this is not scalable because
       process 0 allocates space for all the orderings entry across all the processes
*/
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
  PetscAssertFalse(!iascii,PetscObjectComm((PetscObject)viewer),PETSC_ERR_SUP,"Viewer type %s not supported for AO MemoryScalable",((PetscObject)viewer)->type_name);

  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)ao),&rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)ao),&size);CHKERRMPI(ierr);

  ierr = PetscObjectGetNewTag((PetscObject)ao,&tag_app);CHKERRQ(ierr);
  ierr = PetscObjectGetNewTag((PetscObject)ao,&tag_petsc);CHKERRQ(ierr);

  if (rank == 0) {
    ierr = PetscViewerASCIIPrintf(viewer,"Number of elements in ordering %" PetscInt_FMT "\n",ao->N);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,  "PETSc->App  App->PETSc\n");CHKERRQ(ierr);

    ierr = PetscMalloc2(map->N,&app,map->N,&petsc);CHKERRQ(ierr);
    len  = map->n;
    /* print local AO */
    ierr = PetscViewerASCIIPrintf(viewer,"Process [%d]\n",rank);CHKERRQ(ierr);
    for (i=0; i<len; i++) {
      ierr = PetscViewerASCIIPrintf(viewer,"%3" PetscInt_FMT "  %3" PetscInt_FMT "    %3" PetscInt_FMT "  %3" PetscInt_FMT "\n",i,aomems->app_loc[i],i,aomems->petsc_loc[i]);CHKERRQ(ierr);
    }

    /* recv and print off-processor's AO */
    for (i=1; i<size; i++) {
      len       = map->range[i+1] - map->range[i];
      app_loc   = app  + map->range[i];
      petsc_loc = petsc+ map->range[i];
      ierr      = MPI_Recv(app_loc,(PetscMPIInt)len,MPIU_INT,i,tag_app,PetscObjectComm((PetscObject)ao),&status);CHKERRMPI(ierr);
      ierr      = MPI_Recv(petsc_loc,(PetscMPIInt)len,MPIU_INT,i,tag_petsc,PetscObjectComm((PetscObject)ao),&status);CHKERRMPI(ierr);
      ierr      = PetscViewerASCIIPrintf(viewer,"Process [%" PetscInt_FMT "]\n",i);CHKERRQ(ierr);
      for (j=0; j<len; j++) {
        ierr = PetscViewerASCIIPrintf(viewer,"%3" PetscInt_FMT "  %3" PetscInt_FMT "    %3" PetscInt_FMT "  %3" PetscInt_FMT "\n",map->range[i]+j,app_loc[j],map->range[i]+j,petsc_loc[j]);CHKERRQ(ierr);
      }
    }
    ierr = PetscFree2(app,petsc);CHKERRQ(ierr);

  } else {
    /* send values */
    ierr = MPI_Send((void*)aomems->app_loc,map->n,MPIU_INT,0,tag_app,PetscObjectComm((PetscObject)ao));CHKERRMPI(ierr);
    ierr = MPI_Send((void*)aomems->petsc_loc,map->n,MPIU_INT,0,tag_petsc,PetscObjectComm((PetscObject)ao));CHKERRMPI(ierr);
  }
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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
PetscErrorCode AOMap_MemoryScalable_private(AO ao,PetscInt n,PetscInt *ia,const PetscInt *maploc)
{
  PetscErrorCode    ierr;
  AO_MemoryScalable *aomems = (AO_MemoryScalable*)ao->data;
  MPI_Comm          comm;
  PetscMPIInt       rank,size,tag1,tag2;
  PetscInt          *owner,*start,*sizes,nsends,nreceives;
  PetscInt          nmax,count,*sindices,*rindices,i,j,idx,lastidx,*sindices2,*rindices2;
  const PetscInt    *owners = aomems->map->range;
  MPI_Request       *send_waits,*recv_waits,*send_waits2,*recv_waits2;
  MPI_Status        recv_status;
  PetscMPIInt       nindices,source,widx;
  PetscInt          *rbuf,*sbuf;
  MPI_Status        *send_status,*send_status2;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)ao,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);

  /*  first count number of contributors to each processor */
  ierr = PetscMalloc1(size,&start);CHKERRQ(ierr);
  ierr = PetscCalloc2(2*size,&sizes,n,&owner);CHKERRQ(ierr);

  j       = 0;
  lastidx = -1;
  for (i=0; i<n; i++) {
    if (ia[i] < 0) owner[i] = -1; /* mark negative entries (which are not to be mapped) with a special negative value */
    if (ia[i] >= ao->N) owner[i] = -2; /* mark out of range entries with special negative value */
    else {
      /* if indices are NOT locally sorted, need to start search at the beginning */
      if (lastidx > (idx = ia[i])) j = 0;
      lastidx = idx;
      for (; j<size; j++) {
        if (idx >= owners[j] && idx < owners[j+1]) {
          sizes[2*j]++;     /* num of indices to be sent */
          sizes[2*j+1] = 1; /* send to proc[j] */
          owner[i]     = j;
          break;
        }
      }
    }
  }
  sizes[2*rank]=sizes[2*rank+1]=0; /* do not receive from self! */
  nsends        = 0;
  for (i=0; i<size; i++) nsends += sizes[2*i+1];

  /* inform other processors of number of messages and max length*/
  ierr = PetscMaxSum(comm,sizes,&nmax,&nreceives);CHKERRQ(ierr);

  /* allocate arrays */
  ierr = PetscObjectGetNewTag((PetscObject)ao,&tag1);CHKERRQ(ierr);
  ierr = PetscObjectGetNewTag((PetscObject)ao,&tag2);CHKERRQ(ierr);

  ierr = PetscMalloc2(nreceives*nmax,&rindices,nreceives,&recv_waits);CHKERRQ(ierr);
  ierr = PetscMalloc2(nsends*nmax,&rindices2,nsends,&recv_waits2);CHKERRQ(ierr);

  ierr = PetscMalloc3(n,&sindices,nsends,&send_waits,nsends,&send_status);CHKERRQ(ierr);
  ierr = PetscMalloc3(n,&sindices2,nreceives,&send_waits2,nreceives,&send_status2);CHKERRQ(ierr);

  /* post 1st receives: receive others requests
     since we don't know how long each individual message is we
     allocate the largest needed buffer for each receive. Potentially
     this is a lot of wasted space.
  */
  for (i=0,count=0; i<nreceives; i++) {
    ierr = MPI_Irecv(rindices+nmax*i,nmax,MPIU_INT,MPI_ANY_SOURCE,tag1,comm,recv_waits+count++);CHKERRMPI(ierr);
  }

  /* do 1st sends:
      1) starts[i] gives the starting index in svalues for stuff going to
         the ith processor
  */
  start[0] = 0;
  for (i=1; i<size; i++) start[i] = start[i-1] + sizes[2*i-2];
  for (i=0; i<n; i++) {
    j = owner[i];
    if (j == -1) continue; /* do not remap negative entries in ia[] */
    else if (j == -2) { /* out of range entries get mapped to -1 */
      ia[i] = -1;
      continue;
    } else if (j != rank) {
      sindices[start[j]++]  = ia[i];
    } else { /* compute my own map */
      ia[i] = maploc[ia[i]-owners[rank]];
    }
  }

  start[0] = 0;
  for (i=1; i<size; i++) start[i] = start[i-1] + sizes[2*i-2];
  for (i=0,count=0; i<size; i++) {
    if (sizes[2*i+1]) {
      /* send my request to others */
      ierr = MPI_Isend(sindices+start[i],sizes[2*i],MPIU_INT,i,tag1,comm,send_waits+count);CHKERRMPI(ierr);
      /* post receive for the answer of my request */
      ierr = MPI_Irecv(sindices2+start[i],sizes[2*i],MPIU_INT,i,tag2,comm,recv_waits2+count);CHKERRMPI(ierr);
      count++;
    }
  }
  PetscAssertFalse(nsends != count,comm,PETSC_ERR_SUP,"nsends %" PetscInt_FMT " != count %" PetscInt_FMT,nsends,count);

  /* wait on 1st sends */
  if (nsends) {
    ierr = MPI_Waitall(nsends,send_waits,send_status);CHKERRMPI(ierr);
  }

  /* 1st recvs: other's requests */
  for (j=0; j< nreceives; j++) {
    ierr   = MPI_Waitany(nreceives,recv_waits,&widx,&recv_status);CHKERRMPI(ierr); /* idx: index of handle for operation that completed */
    ierr   = MPI_Get_count(&recv_status,MPIU_INT,&nindices);CHKERRMPI(ierr);
    rbuf   = rindices+nmax*widx; /* global index */
    source = recv_status.MPI_SOURCE;

    /* compute mapping */
    sbuf = rbuf;
    for (i=0; i<nindices; i++) sbuf[i] = maploc[rbuf[i]-owners[rank]];

    /* send mapping back to the sender */
    ierr = MPI_Isend(sbuf,nindices,MPIU_INT,source,tag2,comm,send_waits2+widx);CHKERRMPI(ierr);
  }

  /* wait on 2nd sends */
  if (nreceives) {
    ierr = MPI_Waitall(nreceives,send_waits2,send_status2);CHKERRMPI(ierr);
  }

  /* 2nd recvs: for the answer of my request */
  for (j=0; j< nsends; j++) {
    ierr   = MPI_Waitany(nsends,recv_waits2,&widx,&recv_status);CHKERRMPI(ierr);
    ierr   = MPI_Get_count(&recv_status,MPIU_INT,&nindices);CHKERRMPI(ierr);
    source = recv_status.MPI_SOURCE;
    /* pack output ia[] */
    rbuf  = sindices2+start[source];
    count = 0;
    for (i=0; i<n; i++) {
      if (source == owner[i]) ia[i] = rbuf[count++];
    }
  }

  /* free arrays */
  ierr = PetscFree(start);CHKERRQ(ierr);
  ierr = PetscFree2(sizes,owner);CHKERRQ(ierr);
  ierr = PetscFree2(rindices,recv_waits);CHKERRQ(ierr);
  ierr = PetscFree2(rindices2,recv_waits2);CHKERRQ(ierr);
  ierr = PetscFree3(sindices,send_waits,send_status);CHKERRQ(ierr);
  ierr = PetscFree3(sindices2,send_waits2,send_status2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode AOPetscToApplication_MemoryScalable(AO ao,PetscInt n,PetscInt *ia)
{
  PetscErrorCode    ierr;
  AO_MemoryScalable *aomems  = (AO_MemoryScalable*)ao->data;
  PetscInt          *app_loc = aomems->app_loc;

  PetscFunctionBegin;
  ierr = AOMap_MemoryScalable_private(ao,n,ia,app_loc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode AOApplicationToPetsc_MemoryScalable(AO ao,PetscInt n,PetscInt *ia)
{
  PetscErrorCode    ierr;
  AO_MemoryScalable *aomems    = (AO_MemoryScalable*)ao->data;
  PetscInt          *petsc_loc = aomems->petsc_loc;

  PetscFunctionBegin;
  ierr = AOMap_MemoryScalable_private(ao,n,ia,petsc_loc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static struct _AOOps AOOps_MemoryScalable = {
  PetscDesignatedInitializer(view,AOView_MemoryScalable),
  PetscDesignatedInitializer(destroy,AODestroy_MemoryScalable),
  PetscDesignatedInitializer(petsctoapplication,AOPetscToApplication_MemoryScalable),
  PetscDesignatedInitializer(applicationtopetsc,AOApplicationToPetsc_MemoryScalable),
};

PetscErrorCode  AOCreateMemoryScalable_private(MPI_Comm comm,PetscInt napp,const PetscInt from_array[],const PetscInt to_array[],AO ao, PetscInt *aomap_loc)
{
  PetscErrorCode    ierr;
  AO_MemoryScalable *aomems = (AO_MemoryScalable*)ao->data;
  PetscLayout       map     = aomems->map;
  PetscInt          n_local = map->n,i,j;
  PetscMPIInt       rank,size,tag;
  PetscInt          *owner,*start,*sizes,nsends,nreceives;
  PetscInt          nmax,count,*sindices,*rindices,idx,lastidx;
  PetscInt          *owners = aomems->map->range;
  MPI_Request       *send_waits,*recv_waits;
  MPI_Status        recv_status;
  PetscMPIInt       nindices,widx;
  PetscInt          *rbuf;
  PetscInt          n=napp,ip,ia;
  MPI_Status        *send_status;

  PetscFunctionBegin;
  ierr = PetscArrayzero(aomap_loc,n_local);CHKERRQ(ierr);

  ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);

  /*  first count number of contributors (of from_array[]) to each processor */
  ierr = PetscCalloc1(2*size,&sizes);CHKERRQ(ierr);
  ierr = PetscMalloc1(n,&owner);CHKERRQ(ierr);

  j       = 0;
  lastidx = -1;
  for (i=0; i<n; i++) {
    /* if indices are NOT locally sorted, need to start search at the beginning */
    if (lastidx > (idx = from_array[i])) j = 0;
    lastidx = idx;
    for (; j<size; j++) {
      if (idx >= owners[j] && idx < owners[j+1]) {
        sizes[2*j]  += 2; /* num of indices to be sent - in pairs (ip,ia) */
        sizes[2*j+1] = 1; /* send to proc[j] */
        owner[i]     = j;
        break;
      }
    }
  }
  sizes[2*rank]=sizes[2*rank+1]=0; /* do not receive from self! */
  nsends        = 0;
  for (i=0; i<size; i++) nsends += sizes[2*i+1];

  /* inform other processors of number of messages and max length*/
  ierr = PetscMaxSum(comm,sizes,&nmax,&nreceives);CHKERRQ(ierr);

  /* allocate arrays */
  ierr = PetscObjectGetNewTag((PetscObject)ao,&tag);CHKERRQ(ierr);
  ierr = PetscMalloc2(nreceives*nmax,&rindices,nreceives,&recv_waits);CHKERRQ(ierr);
  ierr = PetscMalloc3(2*n,&sindices,nsends,&send_waits,nsends,&send_status);CHKERRQ(ierr);
  ierr = PetscMalloc1(size,&start);CHKERRQ(ierr);

  /* post receives: */
  for (i=0; i<nreceives; i++) {
    ierr = MPI_Irecv(rindices+nmax*i,nmax,MPIU_INT,MPI_ANY_SOURCE,tag,comm,recv_waits+i);CHKERRMPI(ierr);
  }

  /* do sends:
      1) starts[i] gives the starting index in svalues for stuff going to
         the ith processor
  */
  start[0] = 0;
  for (i=1; i<size; i++) start[i] = start[i-1] + sizes[2*i-2];
  for (i=0; i<n; i++) {
    j = owner[i];
    if (j != rank) {
      ip                   = from_array[i];
      ia                   = to_array[i];
      sindices[start[j]++] = ip;
      sindices[start[j]++] = ia;
    } else { /* compute my own map */
      ip            = from_array[i] - owners[rank];
      ia            = to_array[i];
      aomap_loc[ip] = ia;
    }
  }

  start[0] = 0;
  for (i=1; i<size; i++) start[i] = start[i-1] + sizes[2*i-2];
  for (i=0,count=0; i<size; i++) {
    if (sizes[2*i+1]) {
      ierr = MPI_Isend(sindices+start[i],sizes[2*i],MPIU_INT,i,tag,comm,send_waits+count);CHKERRMPI(ierr);
      count++;
    }
  }
  PetscAssertFalse(nsends != count,comm,PETSC_ERR_SUP,"nsends %" PetscInt_FMT " != count %" PetscInt_FMT,nsends,count);

  /* wait on sends */
  if (nsends) {
    ierr = MPI_Waitall(nsends,send_waits,send_status);CHKERRMPI(ierr);
  }

  /* recvs */
  count=0;
  for (j= nreceives; j>0; j--) {
    ierr = MPI_Waitany(nreceives,recv_waits,&widx,&recv_status);CHKERRMPI(ierr);
    ierr = MPI_Get_count(&recv_status,MPIU_INT,&nindices);CHKERRMPI(ierr);
    rbuf = rindices+nmax*widx; /* global index */

    /* compute local mapping */
    for (i=0; i<nindices; i+=2) { /* pack aomap_loc */
      ip            = rbuf[i] - owners[rank]; /* local index */
      ia            = rbuf[i+1];
      aomap_loc[ip] = ia;
    }
    count++;
  }

  ierr = PetscFree(start);CHKERRQ(ierr);
  ierr = PetscFree3(sindices,send_waits,send_status);CHKERRQ(ierr);
  ierr = PetscFree2(rindices,recv_waits);CHKERRQ(ierr);
  ierr = PetscFree(owner);CHKERRQ(ierr);
  ierr = PetscFree(sizes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode AOCreate_MemoryScalable(AO ao)
{
  PetscErrorCode    ierr;
  IS                isapp=ao->isapp,ispetsc=ao->ispetsc;
  const PetscInt    *mypetsc,*myapp;
  PetscInt          napp,n_local,N,i,start,*petsc,*lens,*disp;
  MPI_Comm          comm;
  AO_MemoryScalable *aomems;
  PetscLayout       map;
  PetscMPIInt       size,rank;

  PetscFunctionBegin;
  PetscAssertFalse(!isapp,PetscObjectComm((PetscObject)ao),PETSC_ERR_ARG_WRONGSTATE,"AOSetIS() must be called before AOSetType()");
  /* create special struct aomems */
  ierr     = PetscNewLog(ao,&aomems);CHKERRQ(ierr);
  ao->data = (void*) aomems;
  ierr     = PetscMemcpy(ao->ops,&AOOps_MemoryScalable,sizeof(struct _AOOps));CHKERRQ(ierr);
  ierr     = PetscObjectChangeTypeName((PetscObject)ao,AOMEMORYSCALABLE);CHKERRQ(ierr);

  /* transmit all local lengths of isapp to all processors */
  ierr = PetscObjectGetComm((PetscObject)isapp,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
  ierr = PetscMalloc2(size,&lens,size,&disp);CHKERRQ(ierr);
  ierr = ISGetLocalSize(isapp,&napp);CHKERRQ(ierr);
  ierr = MPI_Allgather(&napp, 1, MPIU_INT, lens, 1, MPIU_INT, comm);CHKERRMPI(ierr);

  N = 0;
  for (i = 0; i < size; i++) {
    disp[i] = N;
    N      += lens[i];
  }

  /* If ispetsc is 0 then use "natural" numbering */
  if (napp) {
    if (!ispetsc) {
      start = disp[rank];
      ierr  = PetscMalloc1(napp+1, &petsc);CHKERRQ(ierr);
      for (i=0; i<napp; i++) petsc[i] = start + i;
    } else {
      ierr  = ISGetIndices(ispetsc,&mypetsc);CHKERRQ(ierr);
      petsc = (PetscInt*)mypetsc;
    }
  } else {
    petsc = NULL;
  }

  /* create a map with global size N - used to determine the local sizes of ao - shall we use local napp instead of N? */
  ierr    = PetscLayoutCreate(comm,&map);CHKERRQ(ierr);
  map->bs = 1;
  map->N  = N;
  ierr    = PetscLayoutSetUp(map);CHKERRQ(ierr);

  ao->N       = N;
  ao->n       = map->n;
  aomems->map = map;

  /* create distributed indices app_loc: petsc->app and petsc_loc: app->petsc */
  n_local = map->n;
  ierr    = PetscCalloc2(n_local, &aomems->app_loc,n_local,&aomems->petsc_loc);CHKERRQ(ierr);
  ierr    = PetscLogObjectMemory((PetscObject)ao,2*n_local*sizeof(PetscInt));CHKERRQ(ierr);
  ierr    = ISGetIndices(isapp,&myapp);CHKERRQ(ierr);

  ierr = AOCreateMemoryScalable_private(comm,napp,petsc,myapp,ao,aomems->app_loc);CHKERRQ(ierr);
  ierr = AOCreateMemoryScalable_private(comm,napp,myapp,petsc,ao,aomems->petsc_loc);CHKERRQ(ierr);

  ierr = ISRestoreIndices(isapp,&myapp);CHKERRQ(ierr);
  if (napp) {
    if (ispetsc) {
      ierr = ISRestoreIndices(ispetsc,&mypetsc);CHKERRQ(ierr);
    } else {
      ierr = PetscFree(petsc);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree2(lens,disp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   AOCreateMemoryScalable - Creates a memory scalable application ordering using two integer arrays.

   Collective

   Input Parameters:
+  comm - MPI communicator that is to share AO
.  napp - size of integer arrays
.  myapp - integer array that defines an ordering
-  mypetsc - integer array that defines another ordering (may be NULL to
             indicate the natural ordering, that is 0,1,2,3,...)

   Output Parameter:
.  aoout - the new application ordering

   Level: beginner

    Notes:
    The arrays myapp and mypetsc must contain the all the integers 0 to napp-1 with no duplicates; that is there cannot be any "holes"
           in the indices. Use AOCreateMapping() or AOCreateMappingIS() if you wish to have "holes" in the indices.
           Comparing with AOCreateBasic(), this routine trades memory with message communication.

.seealso: AOCreateMemoryScalableIS(), AODestroy(), AOPetscToApplication(), AOApplicationToPetsc()
@*/
PetscErrorCode AOCreateMemoryScalable(MPI_Comm comm,PetscInt napp,const PetscInt myapp[],const PetscInt mypetsc[],AO *aoout)
{
  PetscErrorCode ierr;
  IS             isapp,ispetsc;
  const PetscInt *app=myapp,*petsc=mypetsc;

  PetscFunctionBegin;
  ierr = ISCreateGeneral(comm,napp,app,PETSC_USE_POINTER,&isapp);CHKERRQ(ierr);
  if (mypetsc) {
    ierr = ISCreateGeneral(comm,napp,petsc,PETSC_USE_POINTER,&ispetsc);CHKERRQ(ierr);
  } else {
    ispetsc = NULL;
  }
  ierr = AOCreateMemoryScalableIS(isapp,ispetsc,aoout);CHKERRQ(ierr);
  ierr = ISDestroy(&isapp);CHKERRQ(ierr);
  if (mypetsc) {
    ierr = ISDestroy(&ispetsc);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
   AOCreateMemoryScalableIS - Creates a memory scalable application ordering using two index sets.

   Collective on IS

   Input Parameters:
+  isapp - index set that defines an ordering
-  ispetsc - index set that defines another ordering (may be NULL to use the
             natural ordering)

   Output Parameter:
.  aoout - the new application ordering

   Level: beginner

    Notes:
    The index sets isapp and ispetsc must contain the all the integers 0 to napp-1 (where napp is the length of the index sets) with no duplicates;
           that is there cannot be any "holes".
           Comparing with AOCreateBasicIS(), this routine trades memory with message communication.
.seealso: AOCreateMemoryScalable(),  AODestroy()
@*/
PetscErrorCode  AOCreateMemoryScalableIS(IS isapp,IS ispetsc,AO *aoout)
{
  PetscErrorCode ierr;
  MPI_Comm       comm;
  AO             ao;

  PetscFunctionBegin;
  ierr   = PetscObjectGetComm((PetscObject)isapp,&comm);CHKERRQ(ierr);
  ierr   = AOCreate(comm,&ao);CHKERRQ(ierr);
  ierr   = AOSetIS(ao,isapp,ispetsc);CHKERRQ(ierr);
  ierr   = AOSetType(ao,AOMEMORYSCALABLE);CHKERRQ(ierr);
  ierr   = AOViewFromOptions(ao,NULL,"-ao_view");CHKERRQ(ierr);
  *aoout = ao;
  PetscFunctionReturn(0);
}
