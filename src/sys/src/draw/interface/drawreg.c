
#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: drawreg.c,v 1.12 1999/05/12 03:26:28 bsmith Exp balay $";
#endif
/*
       Provides the registration process for PETSc Draw routines
*/
#include "src/sys/src/draw/drawimpl.h"  /*I "draw.h" I*/

/*
   Contains the list of registered Draw routines
*/
FList DrawList              = 0;

#undef __FUNC__  
#define __FUNC__ "DrawCreate"
/*@C
   DrawCreate - Creates a graphics context.

   Collective on MPI_Comm

   Input Parameter:
.  comm - MPI communicator

   Output Parameter:
.  draw - location to put the Draw context

   Level: beginner

.keywords: Draw, create, context

.seealso: DrawSetFromOptions(), DrawDestroy(), DrawSetType()
@*/
int DrawCreate(MPI_Comm comm,const char display[],const char title[],int x,int y,int w,int h,Draw *indraw)
{
  Draw draw;
  int  ierr;

  PetscFunctionBegin;
  *indraw = 0;
  PetscHeaderCreate(draw,_p_Draw,struct _DrawOps,DRAW_COOKIE,-1,"Draw",comm,DrawDestroy,0);
  PLogObjectCreate(draw);
  *indraw             = draw;
  draw->type          = -1;
  draw->data          = 0;
  if (title) {
    int len     = PetscStrlen(title);
    draw->title = (char *) PetscMalloc((len+1)*sizeof(char*));CHKPTRQ(draw->title);
    PLogObjectMemory(draw,(len+1)*sizeof(char*));
    ierr = PetscStrcpy(draw->title,title);CHKERRQ(ierr);
  } else {
    draw->title = 0;
  }
  if (display) {
    int len     = PetscStrlen(display);
    draw->display = (char *) PetscMalloc((len+1)*sizeof(char*));CHKPTRQ(draw->display);
    PLogObjectMemory(draw,(len+1)*sizeof(char*));
    ierr = PetscStrcpy(draw->display,display);CHKERRQ(ierr);
  } else {
    draw->display = 0;
  }
  draw->x = x;
  draw->y = y;
  draw->w = w;
  draw->h = h;
  PetscFunctionReturn(0);
}
 
#undef __FUNC__  
#define __FUNC__ "DrawSetType"
/*@C
   DrawSetType - Builds graphics object for a particular implementation 

   Collective on Draw

   Input Parameter:
+  draw      - the graphics context
-  type      - for example, DRAW_X

   Options Database Command:
.  -draw_type  <type> - Sets the type; use -help for a list 
    of available methods (for instance, x)

   Level: intermediate

   Notes:  
   See "petsc/include/draw.h" for available methods (for instance,
   DRAW_X)

.keywords: Draw, set, method

.seealso: DrawSetFromOptions(), DrawCreate(), DrawDestroy()
@*/
int DrawSetType(Draw draw,DrawType type)
{
  int ierr,(*r)(Draw);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);

  if (PetscTypeCompare(draw->type_name,type)) PetscFunctionReturn(0);

  if (draw->data) {
    /* destroy the old private Draw context */
    ierr = (*draw->ops->destroy)(draw);CHKERRQ(ierr);
    draw->data      = 0;
  }
  if (draw->type_name) {
    PetscFree(draw->type_name);
    draw->type_name = 0;
  }

  /* Get the function pointers for the graphics method requested */
  if (!DrawList) SETERRQ(1,1,"No draw implementations ierr");

  ierr =  FListFind(draw->comm, DrawList, type,(int (**)(void *)) &r );CHKERRQ(ierr);

  if (!r) SETERRQ1(1,1,"Unknown Draw type given: %s",type);

  draw->data        = 0;
  ierr = (*r)(draw);CHKERRQ(ierr);

  if (!draw->type_name) {
    draw->type_name = (char *) PetscMalloc((PetscStrlen(type)+1)*sizeof(char));CHKPTRQ(draw->type_name);
    ierr = PetscStrcpy(draw->type_name,type);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "DrawRegisterDestroy"
/*@C
   DrawRegisterDestroy - Frees the list of Draw methods that were
   registered by DrawRegister().

   Not Collective

   Level: developer

.keywords: Draw, register, destroy

.seealso: DrawRegister(), DrawRegisterAll()
@*/
int DrawRegisterDestroy(void)
{
  int ierr;

  PetscFunctionBegin;
  if (DrawList) {
    ierr = FListDestroy( DrawList );CHKERRQ(ierr);
    DrawList = 0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "DrawGetType"
/*@C
   DrawGetType - Gets the Draw type as a string from the Draw object.

   Not Collective

   Input Parameter:
.  draw - Krylov context 

   Output Parameters:
.  name - name of Draw method 

   Level: advanced

.keywords: Draw, get, method, name
@*/
int DrawGetType(Draw draw,DrawType *type)
{
  PetscFunctionBegin;
  *type = draw->type_name;
  PetscFunctionReturn(0);
}

/*MC
   DrawRegister - Adds a method to the Krylov subspace solver package.

   Synopsis:
   DrawRegister(char *name_solver,char *path,char *name_create,int (*routine_create)(Draw))

   Not Collective

   Input Parameters:
+  name_solver - name of a new user-defined solver
.  path - path (either absolute or relative) the library containing this solver
.  name_create - name of routine to create method context
-  routine_create - routine to create method context

   Level: developer

   Notes:
   DrawRegister() may be called multiple times to add several user-defined solvers.

   If dynamic libraries are used, then the fourth input argument (routine_create)
   is ignored.

   Sample usage:
.vb
   DrawRegister("my_draw_type",/home/username/my_lib/lib/libO/solaris/mylib.a,
               "MyDrawCreate",MyDrawCreate);
.ve

   Then, your solver can be chosen with the procedural interface via
$     DrawSetType(ksp,"my_draw_type")
   or at runtime via the option
$     -draw_type my_draw_type

.keywords: KSP, register

.seealso: DrawRegisterAll(), DrawRegisterDestroy()
M*/

#undef __FUNC__  
#define __FUNC__ "DrawRegister_Private"
int DrawRegister_Private(char *sname,char *path,char *name,int (*function)(Draw))
{
  int ierr;
  char fullname[256];

  PetscFunctionBegin;
  ierr = PetscStrcpy(fullname,path);CHKERRQ(ierr);
  ierr = PetscStrcat(fullname,":");CHKERRQ(ierr);
  ierr = PetscStrcat(fullname,name);CHKERRQ(ierr);
  ierr = FListAdd_Private(&DrawList,sname,fullname,(int (*)(void*))function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "DrawSetFromOptions"
/*@C
   DrawSetFromOptions - Sets the graphics type from the options database.
      Defaults to a PETSc X windows graphics.

   Collective on Draw

   Input Parameter:
.     draw - the graphics context

   Level: intermediate

   Notes: 
    Must be called after DrawCreate() before the Drawtor is used.

.keywords: draw

.seealso: DrawCreate(), DrawSetType()

@*/
int DrawSetFromOptions(Draw draw)
{
  int     ierr,flg;
  char    vtype[256];

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);

  if (!DrawList) SETERRQ(1,1,"No draw implementations registered");
  ierr = OptionsGetString(draw->prefix,"-draw_type",vtype,256,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = DrawSetType(draw,vtype);CHKERRQ(ierr);
  }

  /* type has not been set? */
  if (!draw->type_name) {
#if defined(PETSC_HAVE_X11)
    ierr = DrawSetType(draw,DRAW_X);CHKERRQ(ierr);
#else
    ierr = OptionsHasName(PETSC_NULL,"-nox",&flg);CHKERRQ(ierr);
    if (!flg) {
      (*PetscErrorPrintf)("PETSc installed without X windows on this machine\nproceeding without graphics\n");
    }
    ierr = DrawSetType(draw,DRAW_NULL);CHKERRQ(ierr);
#endif
  }

  PetscFunctionReturn(0);
}
