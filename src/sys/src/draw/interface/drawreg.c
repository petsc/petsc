/*$Id: drawreg.c,v 1.33 2000/08/04 15:56:10 bsmith Exp balay $*/
/*
       Provides the registration process for PETSc Draw routines
*/
#include "src/sys/src/draw/drawimpl.h"  /*I "petscdraw.h" I*/

/*
   Contains the list of registered Draw routines
*/
FList DrawList              = 0;

#undef __FUNC__  
#define __FUNC__ /*<a name="DrawCreate"></a>*/"DrawCreate" 
/*@C
   DrawCreate - Creates a graphics context.

   Collective on MPI_Comm

   Input Parameter:
+  comm - MPI communicator
.  display - X display when using X windows
.  title - optional title added to top of window
.  x,y - coordinates of lower left corner of window or PETSC_DECIDE
-  w, h - width and height of window or PETSC_DECIDE or DRAW_HALF_SIZE, DRAW_FULL_SIZE,
          or DRAW_THIRD_SIZE or DRAW_QUARTER_SIZE

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
  draw->type    = -1;
  draw->data    = 0;
  ierr          = PetscStrallocpy(title,&draw->title);CHKERRQ(ierr);
  ierr          = PetscStrallocpy(display,&draw->display);CHKERRQ(ierr);
  draw->x       = x;
  draw->y       = y;
  draw->w       = w;
  draw->h       = h;
  draw->pause   = 0;
  draw->coor_xl = 0.0;  
  draw->coor_xr = 1.0;
  draw->coor_yl = 0.0;  
  draw->coor_yr = 1.0;
  draw->port_xl = 0.0;  
  draw->port_xr = 1.0;
  draw->port_yl = 0.0;  
  draw->port_yr = 1.0;
  draw->popup   = 0;
  ierr = OptionsGetInt(PETSC_NULL,"-draw_pause",&draw->pause,PETSC_NULL);CHKERRQ(ierr);
  *indraw       = draw;
  PetscFunctionReturn(0);
}
 
#undef __FUNC__  
#define __FUNC__ /*<a name="DrawSetType"></a>*/"DrawSetType" 
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
   See "petsc/include/petscdraw.h" for available methods (for instance,
   DRAW_X)

.keywords: Draw, set, method

.seealso: DrawSetFromOptions(), DrawCreate(), DrawDestroy()
@*/
int DrawSetType(Draw draw,DrawType type)
{
  int        ierr,(*r)(Draw);
  PetscTruth match,flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  PetscValidCharPointer(type);

  ierr = PetscTypeCompare((PetscObject)draw,type,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  /*  User requests no graphics */
  ierr = OptionsHasName(PETSC_NULL,"-nox",&flg);CHKERRQ(ierr);

  /*
     This is not ideal, but it allows codes to continue to run if X graphics 
   was requested but is not installed on this machine. Mostly this is for
   testing.
   */
#if !defined(PETSC_HAVE_X11)
  {
    PetscTruth match;
    ierr = PetscStrcmp(type,DRAW_X,&match);CHKERRQ(ierr);
    if (match) flg = PETSC_TRUE;
  }
#endif
  if (flg) {
    type = DRAW_NULL;
  }

  if (draw->data) {
    /* destroy the old private Draw context */
    ierr       = (*draw->ops->destroy)(draw);CHKERRQ(ierr);
    draw->data = 0;
  }

  /* Get the function pointers for the graphics method requested */
  if (!DrawList) SETERRQ(1,1,"No draw implementations ierr");

  ierr =  FListFind(draw->comm,DrawList,type,(int (**)(void *)) &r);CHKERRQ(ierr);

  if (!r) SETERRQ1(1,1,"Unknown Draw type given: %s",type);

  ierr = PetscObjectChangeTypeName((PetscObject)draw,type);CHKERRQ(ierr);

  draw->data        = 0;
  ierr = (*r)(draw);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="DrawRegisterDestroy"></a>*/"DrawRegisterDestroy" 
/*@C
   DrawRegisterDestroy - Frees the list of Draw methods that were
   registered by DrawRegisterDynamic().

   Not Collective

   Level: developer

.keywords: Draw, register, destroy

.seealso: DrawRegisterDynamic(), DrawRegisterAll()
@*/
int DrawRegisterDestroy(void)
{
  int ierr;

  PetscFunctionBegin;
  if (DrawList) {
    ierr = FListDestroy(&DrawList);CHKERRQ(ierr);
    DrawList = 0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="DrawGetType"></a>*/"DrawGetType" 
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
   DrawRegisterDynamic - Adds a method to the Krylov subspace solver package.

   Synopsis:
   DrawRegisterDynamic(char *name_solver,char *path,char *name_create,int (*routine_create)(Draw))

   Not Collective

   Input Parameters:
+  name_solver - name of a new user-defined solver
.  path - path (either absolute or relative) the library containing this solver
.  name_create - name of routine to create method context
-  routine_create - routine to create method context

   Level: developer

   Notes:
   DrawRegisterDynamic() may be called multiple times to add several user-defined solvers.

   If dynamic libraries are used, then the fourth input argument (routine_create)
   is ignored.

   Sample usage:
.vb
   DrawRegisterDynamic("my_draw_type",/home/username/my_lib/lib/libO/solaris/mylib.a,
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
#define __FUNC__ /*<a name="DrawRegister"></a>*/"DrawRegister" 
int DrawRegister(char *sname,char *path,char *name,int (*function)(Draw))
{
  int ierr;
  char fullname[256];

  PetscFunctionBegin;
  ierr = FListConcat(path,name,fullname);CHKERRQ(ierr);
  ierr = FListAdd(&DrawList,sname,fullname,(int (*)(void*))function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="DrawSetFromOptions"></a>*/"DrawSetFromOptions" 
/*@C
   DrawSetFromOptions - Sets the graphics type from the options database.
      Defaults to a PETSc X windows graphics.

   Collective on Draw

   Input Parameter:
.     draw - the graphics context

   Options Database:
+   -nox - do not use X graphics (ignore graphics calls, but run program correctly)
-   -nox_warning - when X windows support is not installed this prevents the warning message
                   from being printed

   Level: intermediate

   Notes: 
    Must be called after DrawCreate() before the Drawtor is used.

.keywords: draw

.seealso: DrawCreate(), DrawSetType()

@*/
int DrawSetFromOptions(Draw draw)
{
  int        ierr;
  PetscTruth flg;
  char       vtype[256];

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);

  if (!DrawList) SETERRQ(1,1,"No draw implementations registered");
  ierr = OptionsGetString(draw->prefix,"-draw_type",vtype,256,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = DrawSetType(draw,vtype);CHKERRQ(ierr);
  }

  /* type has not been set? */
  if (!draw->type_name) {
#if defined(PARCH_win32)
    ierr = DrawSetType(draw,DRAW_WIN32);CHKERRQ(ierr);
#elif defined(PETSC_HAVE_X11)
    ierr = DrawSetType(draw,DRAW_X);CHKERRQ(ierr);
#else
    PetscTruth warn;
    ierr = OptionsHasName(PETSC_NULL,"-nox",&flg);CHKERRQ(ierr);
    ierr = OptionsHasName(PETSC_NULL,"-nox_warning",&warn);CHKERRQ(ierr);
    if (!flg && !warn) {
      (*PetscErrorPrintf)("PETSc installed without X windows on this machine\nproceeding without graphics\n");
    }
    ierr = DrawSetType(draw,DRAW_NULL);CHKERRQ(ierr);
#endif
  }

  PetscFunctionReturn(0);
}
