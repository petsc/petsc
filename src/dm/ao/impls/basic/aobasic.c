/*
    The most basic AO application ordering routines. These store the 
  entire orderings on each processor.
*/

#include "src/dm/ao/aoimpl.h"          /*I  "petscao.h"   I*/
#include "petscsys.h"

typedef struct {
  PetscInt N;
  PetscInt *app,*petsc;  /* app[i] is the partner for the ith PETSc slot */
                         /* petsc[j] is the partner for the jth app slot */
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
  PetscTruth     iascii;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(ao->comm,&rank);CHKERRQ(ierr);
  if (!rank){
    ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
    if (iascii) { 
      ierr = PetscViewerASCIIPrintf(viewer,"Number of elements in ordering %D\n",aodebug->N);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,  "PETSc->App  App->PETSc\n");CHKERRQ(ierr);
      for (i=0; i<aodebug->N; i++) {
        ierr = PetscViewerASCIIPrintf(viewer,"%3D  %3D    %3D  %3D\n",i,aodebug->app[i],i,aodebug->petsc[i]);CHKERRQ(ierr);
      }
    } else {
      SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported for AOData basic",((PetscObject)viewer)->type_name);
    }
  }
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AODestroy_Basic" 
PetscErrorCode AODestroy_Basic(AO ao)
{
  AO_Basic       *aodebug = (AO_Basic*)ao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(aodebug->app);CHKERRQ(ierr);
  PetscFree(ao->data); 
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
             indicate the natural ordering)

   Output Parameter:
.  aoout - the new application ordering

   Options Database Key:
.   -ao_view - call AOView() at the conclusion of AOCreateBasic()

   Level: beginner

.keywords: AO, create

.seealso: AOCreateBasicIS(), AODestroy()
@*/
PetscErrorCode AOCreateBasic(MPI_Comm comm,PetscInt napp,const PetscInt myapp[],const PetscInt mypetsc[],AO *aoout)
{
  AO_Basic       *aobasic;
  AO             ao;
  PetscMPIInt    *lens,size,rank,nnapp,*disp;
  PetscInt       *allpetsc,*allapp,ip,ia,N,i,*petsc,start;
  PetscTruth     opt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(aoout,5);
  *aoout = 0;
#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = DMInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif

  PetscHeaderCreate(ao, _p_AO, struct _AOOps, AO_COOKIE, AO_BASIC, "AO", comm, AODestroy, AOView); 
  ierr = PetscNew(AO_Basic, &aobasic);CHKERRQ(ierr);
  PetscLogObjectMemory(ao, sizeof(struct _p_AO) + sizeof(AO_Basic));

  ierr = PetscMemcpy(ao->ops, &AOops, sizeof(AOops));CHKERRQ(ierr);
  ao->data = (void*) aobasic;

  /* transmit all lengths to all processors */
  ierr  = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  ierr  = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr  = PetscMalloc(2*size * sizeof(PetscMPIInt), &lens);CHKERRQ(ierr);
  disp  = lens + size;
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
  if (!mypetsc) {
    start = disp[rank];
    ierr  = PetscMalloc((napp+1) * sizeof(PetscInt), &petsc);CHKERRQ(ierr);
    for (i=0; i<napp; i++) {
      petsc[i] = start + i;
    }
  } else {
    petsc = (PetscInt*)mypetsc;
  }

  /* get all indices on all processors */
  ierr   = PetscMalloc(2*N * sizeof(PetscInt), &allpetsc);CHKERRQ(ierr);
  allapp = allpetsc + N;
  ierr   = MPI_Allgatherv(petsc, napp, MPIU_INT, allpetsc, lens, disp, MPIU_INT, comm);CHKERRQ(ierr);
  ierr   = MPI_Allgatherv((void*)myapp, napp, MPIU_INT, allapp, lens, disp, MPIU_INT, comm);CHKERRQ(ierr);
  ierr   = PetscFree(lens);CHKERRQ(ierr);

  /* generate a list of application and PETSc node numbers */
  ierr = PetscMalloc(2*N * sizeof(PetscInt), &aobasic->app);CHKERRQ(ierr);
  PetscLogObjectMemory(ao,2*N*sizeof(PetscInt));
  aobasic->petsc = aobasic->app + N;
  ierr = PetscMemzero(aobasic->app, 2*N*sizeof(PetscInt));CHKERRQ(ierr);
  for(i = 0; i < N; i++) {
    ip = allpetsc[i];
    ia = allapp[i];
    /* check there are no duplicates */
    if (aobasic->app[ip]) SETERRQ3(PETSC_ERR_ARG_OUTOFRANGE,"Duplicate in PETSc ordering at position %d. Already mapped to %d, not %d.", i, aobasic->app[ip]-1, ia);
    aobasic->app[ip] = ia + 1;
    if (aobasic->petsc[ia]) SETERRQ3(PETSC_ERR_ARG_OUTOFRANGE,"Duplicate in Application ordering at position %d. Already mapped to %d, not %d.", i, aobasic->petsc[ia]-1, ip);
    aobasic->petsc[ia] = ip + 1;
  }
  if (!mypetsc) {
    ierr = PetscFree(petsc);CHKERRQ(ierr);
  }
  ierr = PetscFree(allpetsc);CHKERRQ(ierr);
  /* shift indices down by one */
  for(i = 0; i < N; i++) {
    aobasic->app[i]--;
    aobasic->petsc[i]--;
  }

  ierr = PetscOptionsHasName(PETSC_NULL, "-ao_view", &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = AOView(ao, PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  }

  *aoout = ao;
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

.keywords: AO, create

.seealso: AOCreateBasic(),  AODestroy()
@*/
PetscErrorCode AOCreateBasicIS(IS isapp,IS ispetsc,AO *aoout)
{
  PetscErrorCode ierr;
  PetscInt       *mypetsc = 0,*myapp,napp,npetsc;
  MPI_Comm       comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)isapp,&comm);CHKERRQ(ierr);
  ierr = ISGetLocalSize(isapp,&napp);CHKERRQ(ierr);
  if (ispetsc) {
    ierr = ISGetLocalSize(ispetsc,&npetsc);CHKERRQ(ierr);
    if (napp != npetsc) SETERRQ(PETSC_ERR_ARG_SIZ,"Local IS lengths must match");
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

