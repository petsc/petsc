#ifndef lint
static char vcid[] = "$Id: iscoloring.c,v 1.4 1996/12/16 22:18:33 balay Exp balay $";
#endif

#include "sys.h"   /*I "sys.h" I*/
#include "is.h"    /*I "is.h"  I*/

#undef __FUNC__  
#define __FUNC__ "ISColoringDestroy"
/*@
     ISColoringDestroy - Destroy's a coloring context.

  Input Parameter:
.   iscoloring - the coloring context

.seealso: ISColoringView(), MatGetColoring()
@*/
int ISColoringDestroy(ISColoring iscoloring)
{
  int i,ierr,flag;

  PetscValidPointer(iscoloring);

  ierr = OptionsHasName(0,"-iscoloring_view",&flag); CHKERRQ(ierr);
  if (flag) {
    Viewer viewer;
    ierr = ViewerFileOpenASCII(iscoloring->comm,"stdout",&viewer);CHKERRQ(ierr);
    ierr = ISColoringView(iscoloring,viewer);CHKERRQ(ierr);
    ierr = ViewerDestroy(viewer); CHKERRQ(ierr);
  }

  for ( i=0; i<iscoloring->n; i++ ) {
    ierr = ISDestroy(iscoloring->is[i]); CHKERRQ(ierr);
  }
  PetscCommFree_Private(&iscoloring->comm);
  PetscFree(iscoloring->is);
  PetscFree(iscoloring);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "ISColoringView"
/*@
     ISColoringView - View's a coloring context.

  Input Parameter:
.   iscoloring - the coloring context
.   viewer- the viewer with which to view

.seealso: ISColoringDestroy(), MatGetColoring()
@*/
int ISColoringView(ISColoring iscoloring,Viewer viewer)
{
  int i,ierr;
  PetscValidPointer(iscoloring);

  for ( i=0; i<iscoloring->n; i++ ) {
    ierr = ISView(iscoloring->is[i],viewer); CHKERRQ(ierr);
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "ISColoringCreate"
/*@
    ISColoringCreate - From lists (provided by each processor) of
        colors for each node, generate a ISColoring

  Input Parameters:
.   comm - communicator for the processors creating the coloring
.   n - number of nodes on this processor
.   colors - array containing the colors for this processor, color
             numbers begin at 0.

  Output Parameter:
.   iscoloring - the resulting coloring data structure

.seealso: MatColoringCreate(), ISColoringView(),ISColoringDestroy()
@*/
int ISColoringCreate(MPI_Comm comm,int n,int *colors,ISColoring *iscoloring)
{
  int        ierr,size,rank,base,top,tag,nc,ncwork,*mcolors,**ii,i;
  MPI_Status status;
  IS         *is;

  *iscoloring = (ISColoring) PetscMalloc(sizeof(struct _ISColoring));CHKPTRQ(*iscoloring);
  PetscCommDup_Private(comm,&(*iscoloring)->comm,&tag);
  comm = (*iscoloring)->comm;

  /* compute the number of the first node on my processor */
  MPI_Comm_size(comm,&size);
  MPI_Comm_rank(comm,&rank);
  if (rank == 0) {
    base = 0;
    top  = n;
  } else {
    MPI_Recv(&base,1,MPI_INT,rank-1,tag,comm,&status);
    top = base+n;
  }
  if (rank < size-1) {
    MPI_Send(&top,1,MPI_INT,rank+1,tag,comm);
  }

  /* compute the total number of colors */
  ncwork = 0;
  for ( i=0; i<n; i++ ) {
    if (ncwork < colors[i]) ncwork = colors[i];
  }
  ncwork++;
  MPI_Allreduce(&ncwork,&nc,1,MPI_INT,MPI_MAX,comm);

  /* generate the lists of nodes for each color */
  mcolors = (int *) PetscMalloc( nc*sizeof(int) ); CHKPTRQ(colors);
  PetscMemzero(mcolors,nc*sizeof(int));
  for ( i=0; i<n; i++ ) {
    mcolors[colors[i]]++;
  }

  ii    = (int **) PetscMalloc( nc*sizeof(int*) ); CHKPTRQ(ii);
  ii[0] = (int *) PetscMalloc( n*sizeof(int) ); CHKPTRQ(ii[0]);
  for ( i=1; i<nc; i++ ) {
    ii[i] = ii[i-1] + mcolors[i-1];
  }
  PetscMemzero(mcolors,nc*sizeof(int));
  for ( i=0; i<n; i++ ) {
    ii[colors[i]][mcolors[colors[i]]++] = i + base;
  }
  is  = (IS *) PetscMalloc( nc*sizeof(IS) ); CHKPTRQ(is);
  for ( i=0; i<nc; i++ ) {
    ierr = ISCreateGeneral(MPI_COMM_SELF,mcolors[i],ii[i],is+i); CHKERRQ(ierr);
  }

  (*iscoloring)->n    = nc;
  (*iscoloring)->is   = is;

  PetscFree(ii[0]);
  PetscFree(ii);
  PetscFree(mcolors);
  return 0;
}







