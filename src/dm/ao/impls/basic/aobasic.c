/*$Id: aobasic.c,v 1.60 2001/03/23 23:24:52 balay Exp $*/

/*
    The most basic AO application ordering routines. These store the 
  entire orderings on each processor.
*/

#include "src/dm/ao/aoimpl.h"          /*I  "petscao.h"   I*/
#include "petscsys.h"

typedef struct {
  int N;
  int *app,*petsc;  /* app[i] is the partner for the ith PETSc slot */
                    /* petsc[j] is the partner for the jth app slot */
} AO_Basic;

#undef __FUNCT__  
#define __FUNCT__ "AOBasicGetIndices_Private" 
int AOBasicGetIndices_Private(AO ao,int **app,int **petsc)
{
  AO_Basic *basic = (AO_Basic*)ao->data;

  PetscFunctionBegin;
  if (app)   *app   = basic->app;
  if (petsc) *petsc = basic->petsc;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AODestroy_Basic" 
int AODestroy_Basic(AO ao)
{
  AO_Basic *aodebug = (AO_Basic*)ao->data;
  int      ierr;

  PetscFunctionBegin;
  ierr = PetscFree(aodebug->app);CHKERRQ(ierr);
  PetscFree(ao->data); 
  PetscLogObjectDestroy(ao);
  PetscHeaderDestroy(ao);
  PetscFunctionReturn(0);
}

/*
       All processors have the same data so processor 1 prints it
*/
#undef __FUNCT__  
#define __FUNCT__ "AOView_Basic" 
int AOView_Basic(AO ao,PetscViewer viewer)
{
  int        rank,ierr,i;
  AO_Basic   *aodebug = (AO_Basic*)ao->data;
  PetscTruth isascii;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(ao->comm,&rank);CHKERRQ(ierr);
  if (!rank){
    ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&isascii);CHKERRQ(ierr);
    if (isascii) { 
      ierr = PetscViewerASCIIPrintf(viewer,"Number of elements in ordering %d\n",aodebug->N);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"   App.   PETSc\n");CHKERRQ(ierr);
      for (i=0; i<aodebug->N; i++) {
        ierr = PetscViewerASCIIPrintf(viewer,"%d   %d    %d\n",i,aodebug->app[i],aodebug->petsc[i]);CHKERRQ(ierr);
      }
    } else {
      SETERRQ1(1,"Viewer type %s not supported for AOData basic",((PetscObject)viewer)->type_name);
    }
  }
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AOPetscToApplication_Basic"  
int AOPetscToApplication_Basic(AO ao,int n,int *ia)
{
  int      i;
  AO_Basic *aodebug = (AO_Basic*)ao->data;

  PetscFunctionBegin;
  for (i=0; i<n; i++) {
    if (ia[i] >= 0) {ia[i] = aodebug->app[ia[i]];}
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AOApplicationToPetsc_Basic" 
int AOApplicationToPetsc_Basic(AO ao,int n,int *ia)
{
  int      i;
  AO_Basic *aodebug = (AO_Basic*)ao->data;

  PetscFunctionBegin;
  for (i=0; i<n; i++) {
    if (ia[i] >= 0) {ia[i] = aodebug->petsc[ia[i]];}
  }
  PetscFunctionReturn(0);
}

static struct _AOOps myops = {AOPetscToApplication_Basic,
                              AOApplicationToPetsc_Basic};

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
int AOCreateBasic(MPI_Comm comm,int napp,int *myapp,int *mypetsc,AO *aoout)
{
  AO_Basic   *aodebug;
  AO         ao;
  int        *lens,size,rank,N,i,ierr,*petsc,start;
  int        *allpetsc,*allapp,*disp,ip,ia;
  PetscTruth flg1;

  PetscFunctionBegin;
  *aoout = 0;
  PetscHeaderCreate(ao,_p_AO,struct _AOOps,AO_COOKIE,AO_BASIC,"AO",comm,AODestroy,AOView); 
  PetscLogObjectCreate(ao);
  ierr = PetscNew(AO_Basic,&aodebug);CHKERRQ(ierr);
  PetscLogObjectMemory(ao,sizeof(struct _p_AO) + sizeof(AO_Basic));

  ierr             = PetscMemcpy(ao->ops,&myops,sizeof(myops));CHKERRQ(ierr);
  ao->ops->destroy = AODestroy_Basic;
  ao->ops->view    = AOView_Basic;
  ao->data         = (void *)aodebug;

  /* transmit all lengths to all processors */
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = PetscMalloc(2*size*sizeof(int),&lens);CHKERRQ(ierr);
  disp = lens + size;
  ierr = MPI_Allgather(&napp,1,MPI_INT,lens,1,MPI_INT,comm);CHKERRQ(ierr);
  N =  0;
  for (i=0; i<size; i++) {
    disp[i] = N;
    N += lens[i];
  }
  aodebug->N = N;

  /*
     If mypetsc is 0 then use "natural" numbering 
  */
  if (!mypetsc) {
    start = disp[rank];
    ierr  = PetscMalloc((napp+1)*sizeof(int),&petsc);CHKERRQ(ierr);
    for (i=0; i<napp; i++) {
      petsc[i] = start + i;
    }
  } else {
    petsc = mypetsc;
  }

  /* get all indices on all processors */
  ierr   = PetscMalloc(2*N*sizeof(int),&allpetsc);CHKERRQ(ierr);
  allapp = allpetsc + N;
  ierr   = MPI_Allgatherv(petsc,napp,MPI_INT,allpetsc,lens,disp,MPI_INT,comm);CHKERRQ(ierr);
  ierr   = MPI_Allgatherv(myapp,napp,MPI_INT,allapp,lens,disp,MPI_INT,comm);CHKERRQ(ierr);
  ierr   = PetscFree(lens);CHKERRQ(ierr);

  /* generate a list of application and PETSc node numbers */
  ierr = PetscMalloc(2*N*sizeof(int),&aodebug->app);CHKERRQ(ierr);
  PetscLogObjectMemory(ao,2*N*sizeof(int));
  aodebug->petsc = aodebug->app + N;
  ierr           = PetscMemzero(aodebug->app,2*N*sizeof(int));CHKERRQ(ierr);
  for (i=0; i<N; i++) {
    ip = allpetsc[i]; ia = allapp[i];
    /* check there are no duplicates */
    if (aodebug->app[ip]) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Duplicate in PETSc ordering");
    aodebug->app[ip] = ia + 1;
    if (aodebug->petsc[ia]) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Duplicate in Application ordering");
    aodebug->petsc[ia] = ip + 1;
  }
  if (!mypetsc) {ierr = PetscFree(petsc);CHKERRQ(ierr);}
  ierr = PetscFree(allpetsc);CHKERRQ(ierr);
  /* shift indices down by one */
  for (i=0; i<N; i++) {
    aodebug->app[i]--;
    aodebug->petsc[i]--;
  }

  ierr = PetscOptionsHasName(PETSC_NULL,"-ao_view",&flg1);CHKERRQ(ierr);
  if (flg1) {ierr = AOView(ao,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);}

  *aoout = ao; PetscFunctionReturn(0);
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
int AOCreateBasicIS(IS isapp,IS ispetsc,AO *aoout)
{
  int       *mypetsc = 0,*myapp,ierr,napp,npetsc;
  MPI_Comm  comm;

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

