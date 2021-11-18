
/*
    The most basic AO application ordering routines. These store the
  entire orderings on each processor.
*/

#include <../src/vec/is/ao/aoimpl.h>          /*I  "petscao.h"   I*/

typedef struct {
  PetscInt *app;     /* app[i] is the partner for the ith PETSc slot */
  PetscInt *petsc;   /* petsc[j] is the partner for the jth app slot */
} AO_Basic;

/*
       All processors have the same data so processor 1 prints it
*/
PetscErrorCode AOView_Basic(AO ao,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  PetscInt       i;
  AO_Basic       *aobasic = (AO_Basic*)ao->data;
  PetscBool      iascii;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)ao),&rank);CHKERRMPI(ierr);
  if (rank == 0) {
    ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
    if (iascii) {
      ierr = PetscViewerASCIIPrintf(viewer,"Number of elements in ordering %" PetscInt_FMT "\n",ao->N);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,  "PETSc->App  App->PETSc\n");CHKERRQ(ierr);
      for (i=0; i<ao->N; i++) {
        ierr = PetscViewerASCIIPrintf(viewer,"%3" PetscInt_FMT "  %3" PetscInt_FMT "    %3" PetscInt_FMT "  %3" PetscInt_FMT "\n",i,aobasic->app[i],i,aobasic->petsc[i]);CHKERRQ(ierr);
      }
    }
  }
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode AODestroy_Basic(AO ao)
{
  AO_Basic       *aobasic = (AO_Basic*)ao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree2(aobasic->app,aobasic->petsc);CHKERRQ(ierr);
  ierr = PetscFree(aobasic);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode AOBasicGetIndices_Private(AO ao,PetscInt **app,PetscInt **petsc)
{
  AO_Basic *basic = (AO_Basic*)ao->data;

  PetscFunctionBegin;
  if (app)   *app   = basic->app;
  if (petsc) *petsc = basic->petsc;
  PetscFunctionReturn(0);
}

PetscErrorCode AOPetscToApplication_Basic(AO ao,PetscInt n,PetscInt *ia)
{
  PetscInt i,N=ao->N;
  AO_Basic *aobasic = (AO_Basic*)ao->data;

  PetscFunctionBegin;
  for (i=0; i<n; i++) {
    if (ia[i] >= 0 && ia[i] < N) {
      ia[i] = aobasic->app[ia[i]];
    } else {
      ia[i] = -1;
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode AOApplicationToPetsc_Basic(AO ao,PetscInt n,PetscInt *ia)
{
  PetscInt i,N=ao->N;
  AO_Basic *aobasic = (AO_Basic*)ao->data;

  PetscFunctionBegin;
  for (i=0; i<n; i++) {
    if (ia[i] >= 0 && ia[i] < N) {
      ia[i] = aobasic->petsc[ia[i]];
    } else {
      ia[i] = -1;
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode AOPetscToApplicationPermuteInt_Basic(AO ao, PetscInt block, PetscInt *array)
{
  AO_Basic       *aobasic = (AO_Basic*) ao->data;
  PetscInt       *temp;
  PetscInt       i, j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc1(ao->N*block, &temp);CHKERRQ(ierr);
  for (i = 0; i < ao->N; i++) {
    for (j = 0; j < block; j++) temp[i*block+j] = array[aobasic->petsc[i]*block+j];
  }
  ierr = PetscArraycpy(array, temp, ao->N*block);CHKERRQ(ierr);
  ierr = PetscFree(temp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode AOApplicationToPetscPermuteInt_Basic(AO ao, PetscInt block, PetscInt *array)
{
  AO_Basic       *aobasic = (AO_Basic*) ao->data;
  PetscInt       *temp;
  PetscInt       i, j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc1(ao->N*block, &temp);CHKERRQ(ierr);
  for (i = 0; i < ao->N; i++) {
    for (j = 0; j < block; j++) temp[i*block+j] = array[aobasic->app[i]*block+j];
  }
  ierr = PetscArraycpy(array, temp, ao->N*block);CHKERRQ(ierr);
  ierr = PetscFree(temp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode AOPetscToApplicationPermuteReal_Basic(AO ao, PetscInt block, PetscReal *array)
{
  AO_Basic       *aobasic = (AO_Basic*) ao->data;
  PetscReal      *temp;
  PetscInt       i, j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc1(ao->N*block, &temp);CHKERRQ(ierr);
  for (i = 0; i < ao->N; i++) {
    for (j = 0; j < block; j++) temp[i*block+j] = array[aobasic->petsc[i]*block+j];
  }
  ierr = PetscArraycpy(array, temp, ao->N*block);CHKERRQ(ierr);
  ierr = PetscFree(temp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode AOApplicationToPetscPermuteReal_Basic(AO ao, PetscInt block, PetscReal *array)
{
  AO_Basic       *aobasic = (AO_Basic*) ao->data;
  PetscReal      *temp;
  PetscInt       i, j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc1(ao->N*block, &temp);CHKERRQ(ierr);
  for (i = 0; i < ao->N; i++) {
    for (j = 0; j < block; j++) temp[i*block+j] = array[aobasic->app[i]*block+j];
  }
  ierr = PetscArraycpy(array, temp, ao->N*block);CHKERRQ(ierr);
  ierr = PetscFree(temp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static struct _AOOps AOOps_Basic = {
  AOView_Basic,
  AODestroy_Basic,
  AOPetscToApplication_Basic,
  AOApplicationToPetsc_Basic,
  AOPetscToApplicationPermuteInt_Basic,
  AOApplicationToPetscPermuteInt_Basic,
  AOPetscToApplicationPermuteReal_Basic,
  AOApplicationToPetscPermuteReal_Basic
};

PETSC_EXTERN PetscErrorCode AOCreate_Basic(AO ao)
{
  AO_Basic       *aobasic;
  PetscMPIInt    size,rank,count,*lens,*disp;
  PetscInt       napp,*allpetsc,*allapp,ip,ia,N,i,*petsc=NULL,start;
  PetscErrorCode ierr;
  IS             isapp=ao->isapp,ispetsc=ao->ispetsc;
  MPI_Comm       comm;
  const PetscInt *myapp,*mypetsc=NULL;

  PetscFunctionBegin;
  /* create special struct aobasic */
  ierr     = PetscNewLog(ao,&aobasic);CHKERRQ(ierr);
  ao->data = (void*) aobasic;
  ierr     = PetscMemcpy(ao->ops,&AOOps_Basic,sizeof(struct _AOOps));CHKERRQ(ierr);
  ierr     = PetscObjectChangeTypeName((PetscObject)ao,AOBASIC);CHKERRQ(ierr);

  ierr = ISGetLocalSize(isapp,&napp);CHKERRQ(ierr);
  ierr = ISGetIndices(isapp,&myapp);CHKERRQ(ierr);

  ierr = PetscMPIIntCast(napp,&count);CHKERRQ(ierr);

  /* transmit all lengths to all processors */
  ierr = PetscObjectGetComm((PetscObject)isapp,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
  ierr = PetscMalloc2(size, &lens,size,&disp);CHKERRQ(ierr);
  ierr = MPI_Allgather(&count, 1, MPI_INT, lens, 1, MPI_INT, comm);CHKERRMPI(ierr);
  N    =  0;
  for (i = 0; i < size; i++) {
    ierr = PetscMPIIntCast(N,disp+i);CHKERRQ(ierr); /* = sum(lens[j]), j< i */
    N   += lens[i];
  }
  ao->N = N;
  ao->n = N;

  /* If mypetsc is 0 then use "natural" numbering */
  if (napp) {
    if (!ispetsc) {
      start = disp[rank];
      ierr  = PetscMalloc1(napp+1, &petsc);CHKERRQ(ierr);
      for (i=0; i<napp; i++) petsc[i] = start + i;
    } else {
      ierr  = ISGetIndices(ispetsc,&mypetsc);CHKERRQ(ierr);
      petsc = (PetscInt*)mypetsc;
    }
  }

  /* get all indices on all processors */
  ierr = PetscMalloc2(N,&allpetsc,N,&allapp);CHKERRQ(ierr);
  ierr = MPI_Allgatherv(petsc, count, MPIU_INT, allpetsc, lens, disp, MPIU_INT, comm);CHKERRMPI(ierr);
  ierr = MPI_Allgatherv((void*)myapp, count, MPIU_INT, allapp, lens, disp, MPIU_INT, comm);CHKERRMPI(ierr);
  ierr = PetscFree2(lens,disp);CHKERRQ(ierr);

  if (PetscDefined(USE_DEBUG)) {
    PetscInt *sorted;
    ierr = PetscMalloc1(N,&sorted);CHKERRQ(ierr);

    ierr = PetscArraycpy(sorted,allpetsc,N);CHKERRQ(ierr);
    ierr = PetscSortInt(N,sorted);CHKERRQ(ierr);
    for (i=0; i<N; i++) {
      if (sorted[i] != i) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"PETSc ordering requires a permutation of numbers 0 to N-1\n it is missing %" PetscInt_FMT " has %" PetscInt_FMT "",i,sorted[i]);
    }

    ierr = PetscArraycpy(sorted,allapp,N);CHKERRQ(ierr);
    ierr = PetscSortInt(N,sorted);CHKERRQ(ierr);
    for (i=0; i<N; i++) {
      if (sorted[i] != i) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Application ordering requires a permutation of numbers 0 to N-1\n it is missing %" PetscInt_FMT " has %" PetscInt_FMT "",i,sorted[i]);
    }

    ierr = PetscFree(sorted);CHKERRQ(ierr);
  }

  /* generate a list of application and PETSc node numbers */
  ierr = PetscCalloc2(N, &aobasic->app,N,&aobasic->petsc);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory((PetscObject)ao,2*N*sizeof(PetscInt));CHKERRQ(ierr);
  for (i = 0; i < N; i++) {
    ip = allpetsc[i];
    ia = allapp[i];
    /* check there are no duplicates */
    if (PetscUnlikely(aobasic->app[ip])) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Duplicate in PETSc ordering at position %" PetscInt_FMT ". Already mapped to %" PetscInt_FMT ", not %" PetscInt_FMT ".", i, aobasic->app[ip]-1, ia);
    aobasic->app[ip] = ia + 1;
    if (PetscUnlikely(aobasic->petsc[ia])) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Duplicate in Application ordering at position %" PetscInt_FMT ". Already mapped to %" PetscInt_FMT ", not %" PetscInt_FMT ".", i, aobasic->petsc[ia]-1, ip);
    aobasic->petsc[ia] = ip + 1;
  }
  if (napp && !mypetsc) {
    ierr = PetscFree(petsc);CHKERRQ(ierr);
  }
  ierr = PetscFree2(allpetsc,allapp);CHKERRQ(ierr);
  /* shift indices down by one */
  for (i = 0; i < N; i++) {
    aobasic->app[i]--;
    aobasic->petsc[i]--;
  }

  ierr = ISRestoreIndices(isapp,&myapp);CHKERRQ(ierr);
  if (napp) {
    if (ispetsc) {
      ierr = ISRestoreIndices(ispetsc,&mypetsc);CHKERRQ(ierr);
    } else {
      ierr = PetscFree(petsc);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*@C
   AOCreateBasic - Creates a basic application ordering using two integer arrays.

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
    the arrays myapp and mypetsc must contain the all the integers 0 to napp-1 with no duplicates; that is there cannot be any "holes"
           in the indices. Use AOCreateMapping() or AOCreateMappingIS() if you wish to have "holes" in the indices.

.seealso: AOCreateBasicIS(), AODestroy(), AOPetscToApplication(), AOApplicationToPetsc()
@*/
PetscErrorCode  AOCreateBasic(MPI_Comm comm,PetscInt napp,const PetscInt myapp[],const PetscInt mypetsc[],AO *aoout)
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
  ierr = AOCreateBasicIS(isapp,ispetsc,aoout);CHKERRQ(ierr);
  ierr = ISDestroy(&isapp);CHKERRQ(ierr);
  if (mypetsc) {
    ierr = ISDestroy(&ispetsc);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
   AOCreateBasicIS - Creates a basic application ordering using two index sets.

   Collective on IS

   Input Parameters:
+  isapp - index set that defines an ordering
-  ispetsc - index set that defines another ordering (may be NULL to use the
             natural ordering)

   Output Parameter:
.  aoout - the new application ordering

   Level: beginner

    Notes:
    the index sets isapp and ispetsc must contain the all the integers 0 to napp-1 (where napp is the length of the index sets) with no duplicates;
           that is there cannot be any "holes"

.seealso: AOCreateBasic(),  AODestroy()
@*/
PetscErrorCode AOCreateBasicIS(IS isapp,IS ispetsc,AO *aoout)
{
  PetscErrorCode ierr;
  MPI_Comm       comm;
  AO             ao;

  PetscFunctionBegin;
  ierr   = PetscObjectGetComm((PetscObject)isapp,&comm);CHKERRQ(ierr);
  ierr   = AOCreate(comm,&ao);CHKERRQ(ierr);
  ierr   = AOSetIS(ao,isapp,ispetsc);CHKERRQ(ierr);
  ierr   = AOSetType(ao,AOBASIC);CHKERRQ(ierr);
  ierr   = AOViewFromOptions(ao,NULL,"-ao_view");CHKERRQ(ierr);
  *aoout = ao;
  PetscFunctionReturn(0);
}

