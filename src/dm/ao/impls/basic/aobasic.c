/*$Id: aobasic.c,v 1.54 2000/04/12 04:26:13 bsmith Exp balay $*/

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

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"AOBasicGetIndices_Private" 
int AOBasicGetIndices_Private(AO ao,int **app,int **petsc)
{
  AO_Basic *basic = (AO_Basic*)ao->data;

  PetscFunctionBegin;
  if (app)   *app   = basic->app;
  if (petsc) *petsc = basic->petsc;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"AODestroy_Basic" 
int AODestroy_Basic(AO ao)
{
  AO_Basic *aodebug = (AO_Basic*)ao->data;
  int      ierr;

  PetscFunctionBegin;
  ierr = PetscFree(aodebug->app);CHKERRQ(ierr);
  PetscFree(ao->data); 
  PLogObjectDestroy(ao);
  PetscHeaderDestroy(ao);
  PetscFunctionReturn(0);
}

/*
       All processors have the same data so processor 1 prints it
*/
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"AOView_Basic" 
int AOView_Basic(AO ao,Viewer viewer)
{
  int        rank,ierr,i;
  AO_Basic   *aodebug = (AO_Basic*)ao->data;
  PetscTruth isascii;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(ao->comm,&rank);CHKERRQ(ierr);
  if (!rank){
    ierr = PetscTypeCompare((PetscObject)viewer,ASCII_VIEWER,&isascii);CHKERRQ(ierr);
    if (isascii) { 
      ierr = ViewerASCIIPrintf(viewer,"Number of elements in ordering %d\n",aodebug->N);CHKERRQ(ierr);
      ierr = ViewerASCIIPrintf(viewer,"   App.   PETSc\n");CHKERRQ(ierr);
      for (i=0; i<aodebug->N; i++) {
        ierr = ViewerASCIIPrintf(viewer,"%d   %d    %d\n",i,aodebug->app[i],aodebug->petsc[i]);CHKERRQ(ierr);
      }
    } else {
      SETERRQ1(1,1,"Viewer type %s not supported for AOData basic",((PetscObject)viewer)->type_name);
    }
  }
  ierr = ViewerFlush(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"AOPetscToApplication_Basic"  
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

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"AOApplicationToPetsc_Basic" 
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

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"AOCreateBasic" 
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
  PLogObjectCreate(ao);
  aodebug            = PetscNew(AO_Basic);
  PLogObjectMemory(ao,sizeof(struct _p_AO) + sizeof(AO_Basic));

  ierr             = PetscMemcpy(ao->ops,&myops,sizeof(myops));CHKERRQ(ierr);
  ao->ops->destroy = AODestroy_Basic;
  ao->ops->view    = AOView_Basic;
  ao->data         = (void *)aodebug;

  /* transmit all lengths to all processors */
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  lens = (int*)PetscMalloc(2*size*sizeof(int));CHKPTRQ(lens);
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
    petsc = (int*)PetscMalloc((napp+1)*sizeof(int));CHKPTRQ(petsc);
    for (i=0; i<napp; i++) {
      petsc[i] = start + i;
    }
  } else {
    petsc = mypetsc;
  }

  /* get all indices on all processors */
  allpetsc = (int*)PetscMalloc(2*N*sizeof(int));CHKPTRQ(allpetsc);
  allapp   = allpetsc + N;
  ierr = MPI_Allgatherv(petsc,napp,MPI_INT,allpetsc,lens,disp,MPI_INT,comm);CHKERRQ(ierr);
  ierr = MPI_Allgatherv(myapp,napp,MPI_INT,allapp,lens,disp,MPI_INT,comm);CHKERRQ(ierr);
  ierr = PetscFree(lens);CHKERRQ(ierr);

  /* generate a list of application and PETSc node numbers */
  aodebug->app   = (int*)PetscMalloc(2*N*sizeof(int));CHKPTRQ(aodebug->app);
  PLogObjectMemory(ao,2*N*sizeof(int));
  aodebug->petsc = aodebug->app + N;
  ierr           = PetscMemzero(aodebug->app,2*N*sizeof(int));CHKERRQ(ierr);
  for (i=0; i<N; i++) {
    ip = allpetsc[i]; ia = allapp[i];
    /* check there are no duplicates */
    if (aodebug->app[ip]) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Duplicate in PETSc ordering");
    aodebug->app[ip] = ia + 1;
    if (aodebug->petsc[ia]) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Duplicate in Application ordering");
    aodebug->petsc[ia] = ip + 1;
  }
  if (!mypetsc) {ierr = PetscFree(petsc);CHKERRQ(ierr);}
  ierr = PetscFree(allpetsc);CHKERRQ(ierr);
  /* shift indices down by one */
  for (i=0; i<N; i++) {
    aodebug->app[i]--;
    aodebug->petsc[i]--;
  }

  ierr = OptionsHasName(PETSC_NULL,"-ao_view",&flg1);CHKERRQ(ierr);
  if (flg1) {ierr = AOView(ao,VIEWER_STDOUT_SELF);CHKERRQ(ierr);}

  *aoout = ao; PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"AOCreateBasicIS" 
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
  ierr = ISGetSize(isapp,&napp);CHKERRQ(ierr);
  if (ispetsc) {
    ierr = ISGetSize(ispetsc,&npetsc);CHKERRQ(ierr);
    if (napp != npetsc) SETERRQ(PETSC_ERR_ARG_SIZ,0,"Local IS lengths must match");
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

