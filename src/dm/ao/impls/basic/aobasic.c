

#ifndef lint
static char vcid[] = "$Id: aodebug.c,v 1.6 1996/08/08 14:47:57 bsmith Exp bsmith $";
#endif

/*
    The most basic AO application ordering routines. These store the 
  entire orderings on each processor.
*/

#include "src/ao/aoimpl.h"
#include "pinclude/pviewer.h"
#include "sys.h"

typedef struct {
  int N;
  int *app,*petsc;  /* app[i] is the partner for the ith PETSc slot */
                    /* petsc[j] is the partner for the jth app slot */
} AO_Debug;

static int AODestroy_Debug(PetscObject obj)
{
  AO       ao = (AO) obj;
  AO_Debug *aodebug = (AO_Debug *) ao->data;
  PetscFree(aodebug->app);
  PetscFree(ao->data); 
  PLogObjectDestroy(ao);
  PetscHeaderDestroy(ao);
  return 0;
}

static int AOView_Debug(PetscObject obj,Viewer viewer)
{
  AO          ao = (AO) obj;
  int         rank,ierr,i;
  ViewerType  vtype;
  FILE        *fd;
  AO_Debug    *aodebug = (AO_Debug*) ao->data;

  MPI_Comm_rank(ao->comm,&rank); if (rank) return 0;

  if (!viewer) {
    viewer = VIEWER_STDOUT_SELF; 
  }

  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  if (vtype  == ASCII_FILE_VIEWER || vtype == ASCII_FILES_VIEWER) { 
    ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
    fprintf(fd,"Number of elements in ordering %d\n",aodebug->N);
    fprintf(fd,"   App.   PETSc\n");
    for ( i=0; i<aodebug->N; i++ ) {
      fprintf(fd,"%d   %d    %d\n",i,aodebug->app[i],aodebug->petsc[i]);
    }
  }
  return 0;
}

static int AOPetscToApplication_Debug(AO ao,int n,int *ia)
{
  int      i;
  AO_Debug *aodebug = (AO_Debug *) ao->data;

  for ( i=0; i<n; i++ ) {
    ia[i] = aodebug->app[ia[i]];
  }
  return 0;
}

static int AOApplicationToPetsc_Debug(AO ao,int n,int *ia)
{
  int      i;
  AO_Debug *aodebug = (AO_Debug *) ao->data;

  for ( i=0; i<n; i++ ) {
    ia[i] = aodebug->petsc[ia[i]];
  }
  return 0;
}

static struct _AOOps myops = {AOPetscToApplication_Debug,
                              AOApplicationToPetsc_Debug};

/*@
   AOCreateDebug - Creates a basic application ordering.

   Input Parameters:
.  comm - MPI communicator that is to share AO
.  isapp - index set that defines part of the ordering.
.  ispetsc - 

   Output Parameter:
.  aoout - the new application ordering

   Options Database Keys:
$   -ao_view : call AOView() at the conclusion of AOCreateDebug()

.keywords: AO, create

.seealso: AOCreateBasic(), AOCreateDebugIS(), AODestroy()
@*/
int AOCreateDebug(MPI_Comm comm,int napp,int *myapp,int *mypetsc,AO *aoout)
{
  AO_Debug  *aodebug;
  AO        ao;
  int       *lens,size,rank,N,i,flg1,ierr;
  int       *allpetsc,*allapp,*disp,ip,ia;

  *aoout = 0;
  PetscHeaderCreate(ao, _AO,AO_COOKIE,AO_DEBUG,comm); 
  PLogObjectCreate(ao);
  aodebug            = PetscNew(AO_Debug);
  PLogObjectMemory(ao,sizeof(struct _AO) + sizeof(AO_Debug));

  PetscMemcpy(&ao->ops,&myops,sizeof(myops));
  ao->destroy = AODestroy_Debug;
  ao->view    = AOView_Debug;
  ao->data    = (void *)aodebug;

  /* transmit all lengths to all processors */
  MPI_Comm_size(comm,&size);
  MPI_Comm_rank(comm,&rank);
  lens = (int *) PetscMalloc( 2*size*sizeof(int) ); CHKPTRQ(lens);
  disp = lens + size;
  MPI_Allgather(&napp,1,MPI_INT,lens,1,MPI_INT,comm);
  N =  0;
  for ( i=0; i<size; i++ ) {
    disp[i] = N;
    N += lens[i];
  }
  aodebug->N = N;

  /* get all indices on all processors */
  allpetsc = (int *) PetscMalloc( 2*N*sizeof(int) ); CHKPTRQ(allpetsc);
  allapp   = allpetsc + N;
  MPI_Allgatherv(mypetsc,napp,MPI_INT,allpetsc,lens,disp,MPI_INT,comm);
  MPI_Allgatherv(myapp,napp,MPI_INT,allapp,lens,disp,MPI_INT,comm);
  PetscFree(lens);

  /* generate a list of application and PETSc node numbers */
  aodebug->app = (int *) PetscMalloc(2*N*sizeof(int));CHKPTRQ(aodebug->app);
  aodebug->petsc = aodebug->app + N;
  PetscMemzero(aodebug->app,2*N*sizeof(int));
  for ( i=0; i<N; i++ ) {
    ip = allpetsc[i]; ia = allapp[i];
    /* check there are no duplicates */
    if (aodebug->app[ip]) SETERRQ(1,"AOCreateDebug:Duplicate in ordering");
    aodebug->app[ip] = ia + 1;
    if (aodebug->petsc[ia]) SETERRQ(1,"AOCreateDebug:Duplicate in ordering");
    aodebug->petsc[ia] = ip + 1;
  }
  PetscFree(allpetsc);
  /* shift indices down by one */
  for ( i=0; i<N; i++ ) {
    aodebug->app[i]--;
    aodebug->petsc[i]--;
  }

  ierr = OptionsHasName(PETSC_NULL,"-ao_view",&flg1); CHKERRQ(ierr);
  if (flg1) {ierr = AOView(ao,VIEWER_STDOUT_SELF); CHKERRQ(ierr);}

  *aoout = ao; return 0;
}

/*@
   AOCreateDebugIS - Creates a basic application ordering.

   Input Parameters:
.  comm - MPI communicator that is to share AO
.  isapp - index set that defines part of the ordering.
.  ispetsc - 

   Output Parameter:
.  aoout - the new application ordering

.keywords: AO, create

.seealso: AOCreateBasic(), AOCreateDebug(), AODestroy()
@*/
int AOCreateDebugIS(MPI_Comm comm,IS isapp,IS ispetsc,AO *aoout)
{
  int       *mypetsc,*myapp,ierr,napp,npetsc;

  ierr = ISGetSize(isapp,&napp); CHKERRQ(ierr);
  ierr = ISGetSize(ispetsc,&npetsc); CHKERRQ(ierr);
  if (napp != npetsc) SETERRQ(1,"AOCreateDebug:Local IS lengths must match");

  ierr = ISGetIndices(isapp,&myapp); CHKERRQ(ierr);
  ierr = ISGetIndices(ispetsc,&mypetsc); CHKERRQ(ierr);

  ierr = AOCreateDebug(comm,napp,myapp,mypetsc,aoout); CHKERRQ(ierr);

  ierr = ISRestoreIndices(isapp,&myapp); CHKERRQ(ierr);
  ierr = ISRestoreIndices(ispetsc,&mypetsc); CHKERRQ(ierr);
  return 0;
}

