/*$Id: viewers.c,v 1.7 2000/01/11 20:59:04 bsmith Exp bsmith $*/

#include "viewer.h"

struct _p_Viewers {
   MPI_Comm comm;
   Viewer   *viewer;
   int      n;
} ;

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"ViewersDestroy" 
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
  for (i=0; i<v->n; i++) {
    if (v->viewer[i]) {ierr = ViewerDestroy(v->viewer[i]);CHKERRQ(ierr);}
  }
  ierr = PetscFree(v->viewer);CHKERRQ(ierr);
  ierr = PetscFree(v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"ViewersCreate" 
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
  int ierr;

  PetscFunctionBegin;
  *v           = PetscNew(struct _p_Viewers);CHKPTRQ(*v);
  (*v)->n      = 64;
  (*v)->comm   = comm;
  (*v)->viewer = (Viewer*)PetscMalloc(64*sizeof(Viewer));CHKPTRQ(v);
  ierr = PetscMemzero((*v)->viewer,64*sizeof(Viewer));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"ViewersGetViewer" 
/*@C
   ViewersGetViewer - Gets a viewer from a viewer collection

   Not Collective, but Viewer will be collective object on Viewers

   Input Parameter:
+   viewers - object created with ViewersCreate()
-   n - number of viewer you want

   Output Parameter:
.  viewer - the viewer

   Level: intermediate

.keywords: Viewers, get, type

.seealso: ViewersCreate(), ViewersDestroy()

@*/
int ViewersGetViewer(Viewers viewers,int n,Viewer *viewer)
{
  int ierr;

  PetscFunctionBegin;
  if (n < 0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,1,"Cannot access using a negative index - %d\n",n);
  if (n >= viewers->n) {
    Viewer *v;
    int    newn = n + 64; /* add 64 new ones at a time */
     
    v    = (Viewer*)PetscMalloc(newn*sizeof(Viewer));CHKPTRQ(v);
    ierr = PetscMemzero(v,newn*sizeof(Viewer));CHKERRQ(ierr);
    ierr = PetscMemcpy(v,viewers->viewer,viewers->n*sizeof(Viewer));CHKERRQ(ierr);
    ierr = PetscFree(viewers->viewer);CHKERRQ(ierr);
    viewers->viewer = v;
  }
  if (!viewers->viewer[n]) {
    ierr = ViewerCreate(viewers->comm,&viewers->viewer[n]);CHKERRQ(ierr);
  }
  *viewer = viewers->viewer[n];
  PetscFunctionReturn(0);
}






