#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: viewreg.c,v 1.5 1999/02/01 23:19:33 bsmith Exp bsmith $";
#endif

#include "src/sys/src/viewer/viewerimpl.h"  /*I "petsc.h" I*/  

FList ViewerList              = 0;

#undef __FUNC__  
#define __FUNC__ "ViewerCreate"
/*@C
   ViewerCreate - Creates a viewing context

   Collective on MPI_Comm

   Input Parameter:
.  comm - MPI communicator

   Output Parameter:
.  inviewer - location to put the Viewer context

   Level: advanced

.keywords: Viewer, create, context

.seealso: ViewerDestroy()
@*/
int ViewerCreate(MPI_Comm comm,Viewer *inviewer)
{
  Viewer viewer;

  PetscFunctionBegin;
  *inviewer = 0;
  PetscHeaderCreate(viewer,_p_Viewer,struct _ViewerOps,VIEWER_COOKIE,-1,"Viewer",comm,ViewerDestroy,0);
  PLogObjectCreate(viewer);
  *inviewer           = viewer;
  viewer->type        = -1;
  viewer->data        = 0;
  PetscFunctionReturn(0);
}
 
#undef __FUNC__  
#define __FUNC__ "ViewerSetType"
/*@C
   ViewerSetType - Builds viewer for a particular implementation.

   Collective on Viewer

   Input Parameter:
+  viewer      - the viewer context
-  type        - for example, "ASCII"

   Options Database Command:
.  -draw_type  <type> - Sets the type; use -help for a list 
    of available methods (for instance, ascii)

   Level: advanced

   Notes:  
   See "petsc/include/viewer.h" for available methods (for instance,
   SOCKET_VIEWER)

.keywords: Viewer, set, method, type
@*/
int ViewerSetType(Viewer viewer,ViewerType type)
{
  int ierr,(*r)(Viewer);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,VIEWER_COOKIE);

  if (PetscTypeCompare(viewer->type_name,type)) PetscFunctionReturn(0);

  if (viewer->data) {
    /* destroy the old private Viewer context */
    ierr = (*viewer->ops->destroy)(viewer); CHKERRQ(ierr);
    if (viewer->type_name) PetscFree(viewer->type_name);
    viewer->data      = 0;
    viewer->type_name = 0;
  }
  /* Get the function pointers for the graphics method requested */
  if (!ViewerList) SETERRQ(1,1,"No viewer implementations registered");

  ierr =  FListFind(viewer->comm, ViewerList, type,(int (**)(void *)) &r );CHKERRQ(ierr);

  if (!r) SETERRQ1(1,1,"Unknown Viewer type given: %s",type);

  viewer->data        = 0;
  ierr = (*r)(viewer); CHKERRQ(ierr);

  if (!viewer->type_name) {
    viewer->type_name = (char *) PetscMalloc((PetscStrlen(type)+1)*sizeof(char));CHKPTRQ(viewer->type_name);
    PetscStrcpy(viewer->type_name,type);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ViewerRegisterDestroy"
/*@C
   ViewerRegisterDestroy - Frees the list of Viewer methods that were
   registered by ViewerRegister().

   Not Collective

   Level: developer

.keywords: Viewer, register, destroy

.seealso: ViewerRegister(), ViewerRegisterAll()
@*/
int ViewerRegisterDestroy(void)
{
  int ierr;

  PetscFunctionBegin;
  if (ViewerList) {
    ierr = FListDestroy( ViewerList );CHKERRQ(ierr);
    ViewerList = 0;
  }
  PetscFunctionReturn(0);
}

/*MC
   ViewerRegister - Adds a method to the Krylov subspace solver package.

   Synopsis:
   ViewerRegister(char *name_solver,char *path,char *name_create,int (*routine_create)(Viewer))

   Not Collective

   Input Parameters:
+  name_solver - name of a new user-defined solver
.  path - path (either absolute or relative) the library containing this solver
.  name_create - name of routine to create method context
-  routine_create - routine to create method context

   Level: developer

   Notes:
   ViewerRegister() may be called multiple times to add several user-defined solvers.

   If dynamic libraries are used, then the fourth input argument (routine_create)
   is ignored.

   Sample usage:
.vb
   ViewerRegister("my_viewer_type",/home/username/my_lib/lib/libO/solaris/mylib.a,
               "MyViewerCreate",MyViewerCreate);
.ve

   Then, your solver can be chosen with the procedural interface via
$     ViewerSetType(ksp,"my_viewer_type")
   or at runtime via the option
$     -viewer_type my_viewer_type

.keywords: KSP, register

.seealso: ViewerRegisterAll(), ViewerRegisterDestroy()
M*/

#undef __FUNC__  
#define __FUNC__ "ViewerRegister_Private"
int ViewerRegister_Private(char *sname,char *path,char *name,int (*function)(Viewer))
{
  int ierr;
  char fullname[256];

  PetscFunctionBegin;
  PetscStrcpy(fullname,path); PetscStrcat(fullname,":");PetscStrcat(fullname,name);
  ierr = FListAdd_Private(&ViewerList,sname,fullname,(int (*)(void*))function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ViewerSetFromOptions"
/*@C
   ViewerSetFromOptions - Sets the graphics type from the options database.
      Defaults to a PETSc X windows graphics.

   Collective on Viewer

   Input Parameter:
.     viewer - the graphics context

   Level: intermediate

   Notes: 
    Must be called after ViewerCreate() before the Viewertor is used.

.keywords: viewer

.seealso: ViewerCreate(), ViewerSetType()

@*/
int ViewerSetFromOptions(Viewer viewer)
{
  int     ierr,flg;
  char    vtype[256];

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,VIEWER_COOKIE);

  if (!ViewerList) SETERRQ(1,1,"No viewer implementations registered");
  ierr = OptionsGetString(viewer->prefix,"-viewer_type",vtype,256,&flg);
  if (flg) {
    ierr = ViewerSetType(viewer,vtype); CHKERRQ(ierr);
  }

  /* type has not been set? */
  if (!viewer->type_name) {
    ierr = ViewerSetType(viewer,ASCII_VIEWER); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}
