#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: viewers.c,v 1.1 1999/04/19 21:29:35 bsmith Exp bsmith $";
#endif

#include "src/sys/src/viewer/viewerimpl.h"  /*I "petsc.h" I*/  

struct _p_Viewers {
   MPI_Comm comm;
   Viewer   *viewer;
   int      n,nmax;
} ;

#undef __FUNC__  
#define __FUNC__ "ViewersDestroy"
/*@C
   ViewersDestroy - Destroys a set of viewers created with ViewersCreate().

   Collective on Viewers

   Input Parameters:
.  viewers - the viewer to be destroyed.

   Level: intermediate

.seealso: ViewerSocketOpen(), ViewerASCIIOpen(), ViewerCreate(), ViewerDrawOpen(), ViewersCreate()

.keywords: Viewer, destroy
@*/
int ViewersDestroy(Viewers v)
{
  int         i,ierr;

  PetscFunctionBegin;
  for ( i=0; i<v->n; i++ ) {
    ierr = ViewerDestroy(v->viewer[i]);CHKERRQ(ierr);
  }
  PetscFree(v->viewer);
  PetscFree(v);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ViewersCreate"
/*@C
   ViewersCreate - Creates a container to hold a set of viewers.

   Collective on MPI_Comm

   Input Parameter:
.   comm - the MPI communicator

   Output Parameter:
.  viewers - the collection of viewers

   Level: intermediate

.keywords: Viewers, get, type

.seealso: ViewerCreate(), ViewersDestroy()

@*/
int ViewersCreate(MPI_Comm comm,Viewers *v)
{
  PetscFunctionBegin;
  *v = PetscNew(struct _p_Viewers);CHKPTRQ(*v);
  PetscFunctionReturn(0);
}






