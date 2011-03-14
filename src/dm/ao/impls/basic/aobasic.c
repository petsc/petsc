
/*
    The most basic AO application ordering routines. These store the 
  entire orderings on each processor.
*/

#include <../src/dm/ao/aoimpl.h>          /*I  "petscao.h"   I*/

typedef struct {
  PetscInt N;
  PetscInt *app,*petsc;  /* app[i] is the partner for the ith PETSc slot */
                         /* petsc[j] is the partner for the jth app slot */
  /* for destributed AO */
  PetscInt    *app_loc,*petsc_loc;
  PetscLayout map;
} AO_Basic;

/*
       All processors have the same data so processor 1 prints it
*/
#undef __FUNCT__  
#define __FUNCT__ "AOView_Basic" 
PetscErrorCode AOView_Basic(AO ao,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  PetscInt       i;
  AO_Basic       *aodebug = (AO_Basic*)ao->data;
  PetscBool      iascii;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(((PetscObject)ao)->comm,&rank);CHKERRQ(ierr);
  if (!rank){
    ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
    if (iascii) { 
      ierr = PetscViewerASCIIPrintf(viewer,"Number of elements in ordering %D\n",aodebug->N);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,  "PETSc->App  App->PETSc\n");CHKERRQ(ierr);
      for (i=0; i<aodebug->N; i++) {
        ierr = PetscViewerASCIIPrintf(viewer,"%3D  %3D    %3D  %3D\n",i,aodebug->app[i],i,aodebug->petsc[i]);CHKERRQ(ierr);
      }
    } else {
      SETERRQ1(((PetscObject)viewer)->comm,PETSC_ERR_SUP,"Viewer type %s not supported for AO basic",((PetscObject)viewer)->type_name);
    }
  }
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AOView_BasicMemoryScalable" 
PetscErrorCode AOView_BasicMemoryScalable(AO ao,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank,size;
  PetscInt       i,j;
  AO_Basic       *aobasic = (AO_Basic*)ao->data;
  PetscBool      iascii;
  PetscMPIInt    tag_app,tag_petsc;
  PetscLayout    map = aobasic->map;
  PetscInt       *app,*app_loc,*petsc,*petsc_loc,len;
  MPI_Status     status;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (!iascii) SETERRQ1(((PetscObject)viewer)->comm,PETSC_ERR_SUP,"Viewer type %s not supported for AO basic",((PetscObject)viewer)->type_name);

  ierr = MPI_Comm_rank(((PetscObject)ao)->comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(((PetscObject)ao)->comm,&size);CHKERRQ(ierr);
  
  ierr   = PetscObjectGetNewTag((PetscObject)ao,&tag_app);CHKERRQ(ierr);
  ierr   = PetscObjectGetNewTag((PetscObject)ao,&tag_petsc);CHKERRQ(ierr);

  if (!rank){
    ierr = PetscViewerASCIIPrintf(viewer,"Number of elements in ordering %D\n",aobasic->N);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,  "PETSc->App  App->PETSc\n");CHKERRQ(ierr);

    ierr = PetscMalloc2(map->N,PetscInt,&app,map->N,PetscInt,&petsc);CHKERRQ(ierr);
    len = map->n;
    /* print local AO */
    ierr = PetscViewerASCIIPrintf(viewer,"Process [%D]\n",rank);
    for (i=0; i<len; i++){
      ierr = PetscViewerASCIIPrintf(viewer,"%3D  %3D    %3D  %3D\n",i,aobasic->app_loc[i],i,aobasic->petsc_loc[i]);CHKERRQ(ierr);
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
    ierr = MPI_Send((void*)aobasic->app_loc,map->n,MPIU_INT,0,tag_app,((PetscObject)ao)->comm);CHKERRQ(ierr);
    ierr = MPI_Send((void*)aobasic->petsc_loc,map->n,MPIU_INT,0,tag_petsc,((PetscObject)ao)->comm);CHKERRQ(ierr);
  }
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AODestroy_Basic" 
PetscErrorCode AODestroy_Basic(AO ao)
{
  AO_Basic       *aobasic = (AO_Basic*)ao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree2(aobasic->app,aobasic->petsc);CHKERRQ(ierr);
  ierr = PetscFree(ao->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AODestroy_BasicMemoryScalable" 
PetscErrorCode AODestroy_BasicMemoryScalable(AO ao)
{
  AO_Basic       *aobasic = (AO_Basic*)ao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree2(aobasic->app_loc,aobasic->petsc_loc);CHKERRQ(ierr);
  ierr = PetscLayoutDestroy(aobasic->map);CHKERRQ(ierr);
 
  ierr = AODestroy_Basic(ao);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AOBasicGetIndices_Private" 
PetscErrorCode AOBasicGetIndices_Private(AO ao,PetscInt **app,PetscInt **petsc)
{
  AO_Basic *basic = (AO_Basic*)ao->data;

  PetscFunctionBegin;
  if (app)   *app   = basic->app;
  if (petsc) *petsc = basic->petsc;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AOMap_BasicMemoryScalable_private"  
PetscErrorCode AOMap_BasicMemoryScalable_private(AO ao,PetscInt n,PetscInt *ia,PetscInt *maploc)
{
  PetscErrorCode ierr;
  AO_Basic       *aobasic = (AO_Basic*)ao->data;
  PetscInt       *app_loc = maploc; //aobasic->app_loc;  
  MPI_Comm       comm=((PetscObject)ao)->comm;
  PetscMPIInt    rank,size,tag1,tag2;
  PetscInt       *owner,*start,*nprocs,nsends,nreceives;
  PetscInt       nmax,count,*sindices,*rindices,i,j,idx,lastidx,*sindices2,*rindices2;
  PetscInt       *owners = aobasic->map->range;
  MPI_Request    *send_waits,*recv_waits,*send_waits2,*recv_waits2;
  MPI_Status     recv_status;
  PetscMPIInt    nindices,source;
  PetscInt       *rbuf,*sbuf;
  
  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);

  /*  first count number of contributors to each processor */
  ierr   = PetscMalloc(2*size*sizeof(PetscInt),&nprocs);CHKERRQ(ierr);
  ierr   = PetscMemzero(nprocs,2*size*sizeof(PetscInt));CHKERRQ(ierr);
  ierr   = PetscMalloc(n*sizeof(PetscInt),&owner);CHKERRQ(ierr);
  
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
  //printf("[%d] nmax=sum(num of indices) %d, nreceives %d (include self) \n",rank,nmax,nreceives); 

  /* post 1st receives: 
     since we don't know how long each individual message is we 
     allocate the largest needed buffer for each receive. Potentially 
     this is a lot of wasted space.
  */
  ierr = PetscObjectGetNewTag((PetscObject)ao,&tag1);CHKERRQ(ierr);
  ierr = PetscObjectGetNewTag((PetscObject)ao,&tag2);CHKERRQ(ierr);

  ierr = PetscMalloc(nreceives*nmax*sizeof(PetscInt),&rindices);CHKERRQ(ierr);
  ierr = PetscMalloc(nreceives*sizeof(MPI_Request),&recv_waits);CHKERRQ(ierr);

  ierr = PetscMalloc(nsends*nmax*sizeof(PetscInt),&rindices2);CHKERRQ(ierr);
  ierr = PetscMalloc(nsends*sizeof(MPI_Request),&recv_waits2);CHKERRQ(ierr);

  for (i=0,count=0; i<nreceives; i++) {  
    ierr = MPI_Irecv(rindices+nmax*i,nmax,MPIU_INT,MPI_ANY_SOURCE,tag1,comm,recv_waits+count++);CHKERRQ(ierr);
  }

  /* do 1st sends:
      1) starts[i] gives the starting index in svalues for stuff going to 
         the ith processor
  */
  ierr = PetscMalloc(n*sizeof(PetscInt),&sindices);CHKERRQ(ierr);
  ierr = PetscMalloc(n*sizeof(PetscInt),&sindices2);CHKERRQ(ierr);
  ierr = PetscMalloc(nsends*sizeof(MPI_Request),&send_waits);CHKERRQ(ierr);
  ierr = PetscMalloc(nreceives*sizeof(MPI_Request),&send_waits2);CHKERRQ(ierr);

  ierr = PetscMalloc(size*sizeof(PetscInt),&start);CHKERRQ(ierr);
  
  start[0] = 0;
  for (i=1; i<size; i++) start[i] = start[i-1] + nprocs[2*i-2];
  for (i=0; i<n; i++) {
    j = owner[i];
    if (j != rank){
      sindices[start[j]++]  = ia[i];
    } else { /* compute my own map */
      ia[i] = app_loc[ia[i]-owners[rank]];
    }
  }

  start[0] = 0;
  for (i=1; i<size; i++) { start[i] = start[i-1] + nprocs[2*i-2];} 
  for (i=0,count=0; i<size; i++) {
    if (nprocs[2*i+1]) {
      ierr = MPI_Isend(sindices+start[i],nprocs[2*i],MPIU_INT,i,tag1,comm,send_waits+count);CHKERRQ(ierr);
      ierr = MPI_Irecv(sindices2+start[i],nprocs[2*i],MPIU_INT,i,tag2,comm,recv_waits2+count);CHKERRQ(ierr);
      count++;
    }
  }
  if (nsends != count) SETERRQ2(comm,PETSC_ERR_SUP,"nsends %d != count %d",nsends,count);

  /* wait on 1st sends */
  if (nsends) {
    MPI_Status     *send_status;
    ierr = PetscMalloc(nsends*sizeof(MPI_Status),&send_status);CHKERRQ(ierr);
    ierr = MPI_Waitall(nsends,send_waits,send_status);CHKERRQ(ierr);
    ierr = PetscFree(send_status);CHKERRQ(ierr);
  }
  
  /* 1st recvs */
  count=0;
  for (j= nreceives; j>0; j--){
    ierr = MPI_Waitany(nreceives,recv_waits,&i,&recv_status);CHKERRQ(ierr);
    ierr = MPI_Get_count(&recv_status,MPIU_INT,&nindices);CHKERRQ(ierr);
    rbuf = rindices+nmax*i; /* global index */
    source = recv_status.MPI_SOURCE;
    /* compute mapping */
    sbuf = rbuf;
    for (i=0; i<nindices; i++)sbuf[i] = app_loc[rbuf[i]-owners[rank]];

    /* send mapping */ 
    ierr = MPI_Isend(sbuf,nindices,MPIU_INT,source,tag2,comm,send_waits2+count);CHKERRQ(ierr);  
    count++;
  }

  /* wait on 2nd sends */
  if (nreceives) {
    MPI_Status     *send_status;
    ierr = PetscMalloc(nreceives*sizeof(MPI_Status),&send_status);CHKERRQ(ierr);
    ierr = MPI_Waitall(nreceives,send_waits2,send_status);CHKERRQ(ierr);
    ierr = PetscFree(send_status);CHKERRQ(ierr);
  }

  /* 2nd recvs */
  for (j= nsends; j>0; j--){
    ierr = MPI_Waitany(nsends,recv_waits2,&i,&recv_status);CHKERRQ(ierr);
    ierr = MPI_Get_count(&recv_status,MPIU_INT,&nindices);CHKERRQ(ierr);
    source = recv_status.MPI_SOURCE;
    /* pack output ia[] */
    rbuf=sindices2+start[source];
    count=0;
    for (i=0; i<n; i++){
      if (source == owner[i]) ia[i] = rbuf[count++]; 
    }
    //printf("[%d] 2nd recv %d maps from [%d]: %d %d\n",rank,nindices,source,rbuf[0],rbuf[1]);    
  }

  ierr = PetscFree(start);CHKERRQ(ierr);
  ierr = PetscFree(nprocs);CHKERRQ(ierr);
  ierr = PetscFree(owner);CHKERRQ(ierr);
  ierr = PetscFree(rindices);CHKERRQ(ierr);  ierr = PetscFree(rindices2);CHKERRQ(ierr); 
  ierr = PetscFree(recv_waits);CHKERRQ(ierr);ierr = PetscFree(recv_waits2);CHKERRQ(ierr);
  ierr = PetscFree(sindices);CHKERRQ(ierr);  ierr = PetscFree(sindices2);CHKERRQ(ierr);
  ierr = PetscFree(send_waits);CHKERRQ(ierr);ierr = PetscFree(send_waits2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AOPetscToApplication_BasicMemoryScalable"  
PetscErrorCode AOPetscToApplication_BasicMemoryScalable(AO ao,PetscInt n,PetscInt *ia)
{
  PetscErrorCode ierr;
  AO_Basic       *aobasic = (AO_Basic*)ao->data;
  PetscInt       *app_loc = aobasic->app_loc;

  PetscFunctionBegin;
  ierr = AOMap_BasicMemoryScalable_private(ao,n,ia,app_loc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AOPetscToApplication_Basic"  
PetscErrorCode AOPetscToApplication_Basic(AO ao,PetscInt n,PetscInt *ia)
{
  PetscInt i;
  AO_Basic *aodebug = (AO_Basic*)ao->data;

  PetscFunctionBegin;
  for (i=0; i<n; i++) {
    if (ia[i] >= 0) {ia[i] = aodebug->app[ia[i]];}
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AOApplicationToPetsc_BasicMemoryScalable" 
PetscErrorCode AOApplicationToPetsc_BasicMemoryScalable(AO ao,PetscInt n,PetscInt *ia)
{
  PetscErrorCode ierr;
  AO_Basic       *aobasic = (AO_Basic*)ao->data;
  PetscInt       *petsc_loc = aobasic->petsc_loc;

  PetscFunctionBegin;
  ierr = AOMap_BasicMemoryScalable_private(ao,n,ia,petsc_loc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AOApplicationToPetsc_Basic" 
PetscErrorCode AOApplicationToPetsc_Basic(AO ao,PetscInt n,PetscInt *ia)
{
  PetscInt i;
  AO_Basic *aodebug = (AO_Basic*)ao->data;

  PetscFunctionBegin;
  for (i=0; i<n; i++) {
    if (ia[i] >= 0) {ia[i] = aodebug->petsc[ia[i]];}
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AOPetscToApplicationPermuteInt_Basic"
PetscErrorCode AOPetscToApplicationPermuteInt_Basic(AO ao, PetscInt block, PetscInt *array)
{
  AO_Basic       *aodebug = (AO_Basic *) ao->data;
  PetscInt       *temp;
  PetscInt       i, j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc(aodebug->N*block * sizeof(PetscInt), &temp);CHKERRQ(ierr);
  for(i = 0; i < aodebug->N; i++) {
    for(j = 0; j < block; j++) temp[i*block+j] = array[aodebug->petsc[i]*block+j];
  }
  ierr = PetscMemcpy(array, temp, aodebug->N*block * sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscFree(temp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AOApplicationToPetscPermuteInt_Basic"
PetscErrorCode AOApplicationToPetscPermuteInt_Basic(AO ao, PetscInt block, PetscInt *array)
{
  AO_Basic       *aodebug = (AO_Basic *) ao->data;
  PetscInt       *temp;
  PetscInt       i, j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc(aodebug->N*block * sizeof(PetscInt), &temp);CHKERRQ(ierr);
  for(i = 0; i < aodebug->N; i++) {
    for(j = 0; j < block; j++) temp[i*block+j] = array[aodebug->app[i]*block+j];
  }
  ierr = PetscMemcpy(array, temp, aodebug->N*block * sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscFree(temp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AOPetscToApplicationPermuteReal_Basic"
PetscErrorCode AOPetscToApplicationPermuteReal_Basic(AO ao, PetscInt block, PetscReal *array)
{
  AO_Basic       *aodebug = (AO_Basic *) ao->data;
  PetscReal      *temp;
  PetscInt       i, j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc(aodebug->N*block * sizeof(PetscReal), &temp);CHKERRQ(ierr);
  for(i = 0; i < aodebug->N; i++) {
    for(j = 0; j < block; j++) temp[i*block+j] = array[aodebug->petsc[i]*block+j];
  }
  ierr = PetscMemcpy(array, temp, aodebug->N*block * sizeof(PetscReal));CHKERRQ(ierr);
  ierr = PetscFree(temp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AOApplicationToPetscPermuteReal_Basic"
PetscErrorCode AOApplicationToPetscPermuteReal_Basic(AO ao, PetscInt block, PetscReal *array)
{
  AO_Basic       *aodebug = (AO_Basic *) ao->data;
  PetscReal      *temp;
  PetscInt       i, j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc(aodebug->N*block * sizeof(PetscReal), &temp);CHKERRQ(ierr);
  for(i = 0; i < aodebug->N; i++) {
    for(j = 0; j < block; j++) temp[i*block+j] = array[aodebug->app[i]*block+j];
  }
  ierr = PetscMemcpy(array, temp, aodebug->N*block * sizeof(PetscReal));CHKERRQ(ierr);
  ierr = PetscFree(temp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static struct _AOOps AOops = {AOView_Basic,
                              AODestroy_Basic,
                              AOPetscToApplication_Basic,
                              AOApplicationToPetsc_Basic,
                              AOPetscToApplicationPermuteInt_Basic,
                              AOApplicationToPetscPermuteInt_Basic,
                              AOPetscToApplicationPermuteReal_Basic,
                              AOApplicationToPetscPermuteReal_Basic};

#undef __FUNCT__  
#define __FUNCT__ "AOCreateBasic" 
/*@C
   AOCreateBasic - Creates a basic application ordering using two integer arrays.

   Collective on MPI_Comm

   Input Parameters:
+  comm - MPI communicator that is to share AO
.  napp - size of integer arrays
.  myapp - integer array that defines an ordering
-  mypetsc - integer array that defines another ordering (may be PETSC_NULL to 
             indicate the natural ordering, that is 0,1,2,3,...)

   Output Parameter:
.  aoout - the new application ordering

   Options Database Key:
.   -ao_view - call AOView() at the conclusion of AOCreateBasic()

   Level: beginner

    Notes: the arrays myapp and mypetsc must contain the all the integers 0 to napp-1 with no duplicates; that is there cannot be any "holes"  
           in the indices. Use AOCreateMapping() or AOCreateMappingIS() if you wish to have "holes" in the indices.

.keywords: AO, create

.seealso: AOCreateBasicIS(), AODestroy(), AOPetscToApplication(), AOApplicationToPetsc()
@*/
PetscErrorCode  AOCreateBasic(MPI_Comm comm,PetscInt napp,const PetscInt myapp[],const PetscInt mypetsc[],AO *aoout)
{
  AO_Basic       *aobasic;
  AO             ao;
  PetscMPIInt    *lens,size,rank,nnapp,*disp;
  PetscInt       *allpetsc,*allapp,ip,ia,N,i,*petsc,start;
  PetscBool      opt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(aoout,5);
  *aoout = 0;
#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = AOInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif

  ierr = PetscHeaderCreate(ao, _p_AO, struct _AOOps, AO_CLASSID, AO_BASIC, "AO", comm, AODestroy, AOView);CHKERRQ(ierr);
  ierr = PetscNewLog(ao, AO_Basic, &aobasic);CHKERRQ(ierr);

  ierr = PetscMemcpy(ao->ops, &AOops, sizeof(AOops));CHKERRQ(ierr);
  ao->data = (void*) aobasic;

  /* transmit all lengths to all processors */
  ierr  = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  ierr  = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr  = PetscMalloc2(size,PetscMPIInt, &lens,size,PetscMPIInt,&disp);CHKERRQ(ierr);
  nnapp = napp;
  ierr  = MPI_Allgather(&nnapp, 1, MPI_INT, lens, 1, MPI_INT, comm);CHKERRQ(ierr);
  N    =  0;
  for(i = 0; i < size; i++) {
    disp[i] = N;
    N += lens[i];
  }
  aobasic->N = N;

  /*
     If mypetsc is 0 then use "natural" numbering 
  */
  if (napp && !mypetsc) {
    start = disp[rank];
    ierr  = PetscMalloc((napp+1) * sizeof(PetscInt), &petsc);CHKERRQ(ierr);
    for (i=0; i<napp; i++) {
      petsc[i] = start + i;
    }
  } else {
    petsc = (PetscInt*)mypetsc;
  }

  /* get all indices on all processors */
  ierr   = PetscMalloc2(N,PetscInt, &allpetsc,N,PetscInt,&allapp);CHKERRQ(ierr);
  ierr   = MPI_Allgatherv(petsc, napp, MPIU_INT, allpetsc, lens, disp, MPIU_INT, comm);CHKERRQ(ierr);
  ierr   = MPI_Allgatherv((void*)myapp, napp, MPIU_INT, allapp, lens, disp, MPIU_INT, comm);CHKERRQ(ierr);
  ierr   = PetscFree2(lens,disp);CHKERRQ(ierr);

#if defined(PETSC_USE_DEBUG)
  {
    PetscInt *sorted;
    ierr = PetscMalloc(N*sizeof(PetscInt),&sorted);CHKERRQ(ierr);

    ierr = PetscMemcpy(sorted,allpetsc,N*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscSortInt(N,sorted);CHKERRQ(ierr);
    for (i=0; i<N; i++) {
      if (sorted[i] != i) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"PETSc ordering requires a permutation of numbers 0 to N-1\n it is missing %D has %D",i,sorted[i]);
    }

    ierr = PetscMemcpy(sorted,allapp,N*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscSortInt(N,sorted);CHKERRQ(ierr);
    for (i=0; i<N; i++) {
      if (sorted[i] != i) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Application ordering requires a permutation of numbers 0 to N-1\n it is missing %D has %D",i,sorted[i]);
    }

    ierr = PetscFree(sorted);CHKERRQ(ierr);
  }
#endif

  /* generate a list of application and PETSc node numbers */
  ierr = PetscMalloc2(N,PetscInt, &aobasic->app,N,PetscInt,&aobasic->petsc);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(ao,2*N*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemzero(aobasic->app, N*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemzero(aobasic->petsc, N*sizeof(PetscInt));CHKERRQ(ierr);
  for(i = 0; i < N; i++) {
    ip = allpetsc[i];
    ia = allapp[i];
    /* check there are no duplicates */
    if (aobasic->app[ip]) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Duplicate in PETSc ordering at position %d. Already mapped to %d, not %d.", i, aobasic->app[ip]-1, ia);
    aobasic->app[ip] = ia + 1;
    if (aobasic->petsc[ia]) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Duplicate in Application ordering at position %d. Already mapped to %d, not %d.", i, aobasic->petsc[ia]-1, ip);
    aobasic->petsc[ia] = ip + 1;
  }
  if (!mypetsc) {
    ierr = PetscFree(petsc);CHKERRQ(ierr);
  }
  ierr = PetscFree2(allpetsc,allapp);CHKERRQ(ierr);
  /* shift indices down by one */
  for(i = 0; i < N; i++) {
    aobasic->app[i]--;
    aobasic->petsc[i]--;
  }

  opt = PETSC_FALSE;
  ierr = PetscOptionsGetBool(PETSC_NULL, "-ao_view", &opt,PETSC_NULL);CHKERRQ(ierr);
  if (opt) {
    ierr = AOView(ao, PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  }

  *aoout = ao;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AOCreateBasicMemoryScalable" 
PetscErrorCode  AOCreateBasicMemoryScalable(MPI_Comm comm,PetscInt napp,const PetscInt myapp[],const PetscInt mypetsc[],AO *aoout)
{
  PetscErrorCode ierr;
  AO             ao;
  AO_Basic       *aobasic;
  PetscLayout    map;
  PetscInt       n_local,i,j;
  PetscBool      opt;

  PetscFunctionBegin;
  // not done yet!
  ierr = AOCreateBasic(comm,napp,myapp,mypetsc,&ao);CHKERRQ(ierr);
  ao->ops->view               = AOView_BasicMemoryScalable;
  ao->ops->destroy            = AODestroy_BasicMemoryScalable;
  ao->ops->petsctoapplication = AOPetscToApplication_BasicMemoryScalable;
  ao->ops->applicationtopetsc = AOApplicationToPetsc_BasicMemoryScalable;
  aobasic = (AO_Basic*)ao->data;
   
  ierr = PetscLayoutCreate(comm,&map);CHKERRQ(ierr);
  map->bs = 1;
  map->N  = aobasic->N;
  
  ierr = PetscLayoutSetUp(map);CHKERRQ(ierr);
  aobasic->map = map;
  
  n_local = map->n;
  ierr   = PetscMalloc2(n_local,PetscInt, &aobasic->app_loc,n_local,PetscInt,&aobasic->petsc_loc);CHKERRQ(ierr);
  // copy app and petsc to app_loc and petsc_loc
  j = map->rstart;
  for (i=0; i<n_local; i++){
    aobasic->app_loc[i]   = aobasic->app[j];
    aobasic->petsc_loc[i] = aobasic->petsc[j];
    j++;
  }

  opt = PETSC_FALSE;
  ierr = PetscOptionsGetBool(PETSC_NULL, "-ao_view", &opt,PETSC_NULL);CHKERRQ(ierr);
  if (opt) {
    ierr = AOView_BasicMemoryScalable(ao, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  
  *aoout = ao;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AOCreateBasicMemoryScalableIS" 
PetscErrorCode  AOCreateBasicMemoryScalableIS(IS isapp,IS ispetsc,AO *aoout)
{
  PetscErrorCode ierr;
  const PetscInt *mypetsc = 0,*myapp;
  PetscInt       napp,npetsc;
  MPI_Comm       comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)isapp,&comm);CHKERRQ(ierr);
  ierr = ISGetLocalSize(isapp,&napp);CHKERRQ(ierr);
  if (ispetsc) {
    ierr = ISGetLocalSize(ispetsc,&npetsc);CHKERRQ(ierr);
    if (napp != npetsc) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local IS lengths must match");
    ierr = ISGetIndices(ispetsc,&mypetsc);CHKERRQ(ierr);
  }
  ierr = ISGetIndices(isapp,&myapp);CHKERRQ(ierr);

  ierr = AOCreateBasicMemoryScalable(comm,napp,myapp,mypetsc,aoout);CHKERRQ(ierr);

  ierr = ISRestoreIndices(isapp,&myapp);CHKERRQ(ierr);
  if (ispetsc) {
    ierr = ISRestoreIndices(ispetsc,&mypetsc);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AOCreateBasicIS" 
/*@C
   AOCreateBasicIS - Creates a basic application ordering using two index sets.

   Collective on IS

   Input Parameters:
+  isapp - index set that defines an ordering
-  ispetsc - index set that defines another ordering (may be PETSC_NULL to use the
             natural ordering)

   Output Parameter:
.  aoout - the new application ordering

   Options Database Key:
-   -ao_view - call AOView() at the conclusion of AOCreateBasicIS()

   Level: beginner

    Notes: the index sets isapp and ispetsc must contain the all the integers 0 to napp-1 (where napp is the length of the index sets) with no duplicates; 
           that is there cannot be any "holes"  

.keywords: AO, create

.seealso: AOCreateBasic(),  AODestroy()
@*/
PetscErrorCode  AOCreateBasicIS(IS isapp,IS ispetsc,AO *aoout)
{
  PetscErrorCode ierr;
  const PetscInt *mypetsc = 0,*myapp;
  PetscInt       napp,npetsc;
  MPI_Comm       comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)isapp,&comm);CHKERRQ(ierr);
  ierr = ISGetLocalSize(isapp,&napp);CHKERRQ(ierr);
  if (ispetsc) {
    ierr = ISGetLocalSize(ispetsc,&npetsc);CHKERRQ(ierr);
    if (napp != npetsc) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local IS lengths must match");
    ierr = ISGetIndices(ispetsc,&mypetsc);CHKERRQ(ierr);
  }
  ierr = ISGetIndices(isapp,&myapp);CHKERRQ(ierr);

  ierr = AOCreateBasic(comm,napp,myapp,mypetsc,aoout);CHKERRQ(ierr);

  ierr = ISRestoreIndices(isapp,&myapp);CHKERRQ(ierr);
  if (ispetsc) {
    ierr = ISRestoreIndices(ispetsc,&mypetsc);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

