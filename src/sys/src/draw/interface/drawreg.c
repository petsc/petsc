/*$Id: drawreg.c,v 1.45 2001/06/21 21:15:18 bsmith Exp $*/
/*
       Provides the registration process for PETSc PetscDraw routines
*/
#include "src/sys/src/draw/drawimpl.h"  /*I "petscdraw.h" I*/

/*
   Contains the list of registered PetscDraw routines
*/
PetscFList PetscDrawList              = 0;

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawCreate" 
/*@C
   PetscDrawCreate - Creates a graphics context.

   Collective on MPI_Comm

   Input Parameter:
+  comm - MPI communicator
.  display - X display when using X windows
.  title - optional title added to top of window
.  x,y - coordinates of lower left corner of window or PETSC_DECIDE
-  w, h - width and height of window or PETSC_DECIDE or PETSC_DRAW_HALF_SIZE, PETSC_DRAW_FULL_SIZE,
          or PETSC_DRAW_THIRD_SIZE or PETSC_DRAW_QUARTER_SIZE

   Output Parameter:
.  draw - location to put the PetscDraw context

   Level: beginner

   Concepts: graphics^creating context
   Concepts: drawing^creating context

.seealso: PetscDrawSetFromOptions(), PetscDrawDestroy(), PetscDrawSetType()
@*/
int PetscDrawCreate(MPI_Comm comm,const char display[],const char title[],int x,int y,int w,int h,PetscDraw *indraw)
{
  PetscDraw draw;
  int  ierr;

  PetscFunctionBegin;
  *indraw = 0;
  PetscHeaderCreate(draw,_p_PetscDraw,struct _PetscDrawOps,PETSC_DRAW_COOKIE,-1,"Draw",comm,PetscDrawDestroy,0);
  PetscLogObjectCreate(draw);
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
  ierr = PetscOptionsGetInt(PETSC_NULL,"-draw_pause",&draw->pause,PETSC_NULL);CHKERRQ(ierr);
  *indraw       = draw;
  PetscFunctionReturn(0);
}
 
#undef __FUNCT__  
#define __FUNCT__ "PetscDrawSetType" 
/*@C
   PetscDrawSetType - Builds graphics object for a particular implementation 

   Collective on PetscDraw

   Input Parameter:
+  draw      - the graphics context
-  type      - for example, PETSC_DRAW_X

   Options Database Command:
.  -draw_type  <type> - Sets the type; use -help for a list 
    of available methods (for instance, x)

   Level: intermediate

   Notes:  
   See "petsc/include/petscdraw.h" for available methods (for instance,
   PETSC_DRAW_X)

   Concepts: drawing^X windows
   Concepts: X windows^graphics
   Concepts: drawing^postscript
   Concepts: postscript^graphics
   Concepts: drawing^Microsoft Windows

.seealso: PetscDrawSetFromOptions(), PetscDrawCreate(), PetscDrawDestroy()
@*/
int PetscDrawSetType(PetscDraw draw,PetscDrawType type)
{
  int        ierr,(*r)(PetscDraw);
  PetscTruth match;
  PetscTruth flg=PETSC_FALSE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_COOKIE);
  PetscValidCharPointer(type);

  ierr = PetscTypeCompare((PetscObject)draw,type,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

#if defined(PETSC_HAVE_X11)
  /*  User requests no graphics */
  ierr = PetscOptionsHasName(PETSC_NULL,"-nox",&flg);CHKERRQ(ierr);
#endif

  /*
     This is not ideal, but it allows codes to continue to run if X graphics 
   was requested but is not installed on this machine. Mostly this is for
   testing.
   */
#if !defined(PETSC_HAVE_X11)
  {
    ierr = PetscStrcmp(type,PETSC_DRAW_X,&match);CHKERRQ(ierr);
    if (match) flg = PETSC_TRUE;
  }
#endif
  if (flg) {
    type = PETSC_DRAW_NULL;
  }

  if (draw->data) {
    /* destroy the old private PetscDraw context */
    ierr       = (*draw->ops->destroy)(draw);CHKERRQ(ierr);
    draw->data = 0;
  }

  /* Get the function pointers for the graphics method requested */
  if (!PetscDrawList) SETERRQ(1,"No draw implementations ierr");

  ierr =  PetscFListFind(draw->comm,PetscDrawList,type,(void (**)(void)) &r);CHKERRQ(ierr);

  if (!r) SETERRQ1(1,"Unknown PetscDraw type given: %s",type);

  ierr = PetscObjectChangeTypeName((PetscObject)draw,type);CHKERRQ(ierr);

  draw->data        = 0;
  ierr = (*r)(draw);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawRegisterDestroy" 
/*@C
   PetscDrawRegisterDestroy - Frees the list of PetscDraw methods that were
   registered by PetscDrawRegisterDynamic().

   Not Collective

   Level: developer

.seealso: PetscDrawRegisterDynamic(), PetscDrawRegisterAll()
@*/
int PetscDrawRegisterDestroy(void)
{
  int ierr;

  PetscFunctionBegin;
  if (PetscDrawList) {
    ierr = PetscFListDestroy(&PetscDrawList);CHKERRQ(ierr);
    PetscDrawList = 0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawGetType" 
/*@C
   PetscDrawGetType - Gets the PetscDraw type as a string from the PetscDraw object.

   Not Collective

   Input Parameter:
.  draw - Krylov context 

   Output Parameters:
.  name - name of PetscDraw method 

   Level: advanced

@*/
int PetscDrawGetType(PetscDraw draw,PetscDrawType *type)
{
  PetscFunctionBegin;
  *type = draw->type_name;
  PetscFunctionReturn(0);
}

/*MC
   PetscDrawRegisterDynamic - Adds a method to the Krylov subspace solver package.

   Synopsis:
   int PetscDrawRegisterDynamic(char *name_solver,char *path,char *name_create,int (*routine_create)(PetscDraw))

   Not Collective

   Input Parameters:
+  name_solver - name of a new user-defined solver
.  path - path (either absolute or relative) the library containing this solver
.  name_create - name of routine to create method context
-  routine_create - routine to create method context

   Level: developer

   Notes:
   PetscDrawRegisterDynamic() may be called multiple times to add several user-defined solvers.

   If dynamic libraries are used, then the fourth input argument (routine_create)
   is ignored.

   Sample usage:
.vb
   PetscDrawRegisterDynamic("my_draw_type",/home/username/my_lib/lib/libO/solaris/mylib.a,
               "MyDrawCreate",MyDrawCreate);
.ve

   Then, your solver can be chosen with the procedural interface via
$     PetscDrawSetType(ksp,"my_draw_type")
   or at runtime via the option
$     -draw_type my_draw_type

   Concepts: graphics^registering new draw classes
   Concepts: PetscDraw^registering new draw classes

.seealso: PetscDrawRegisterAll(), PetscDrawRegisterDestroy()
M*/

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawRegister" 
int PetscDrawRegister(char *sname,char *path,char *name,int (*function)(PetscDraw))
{
  int  ierr;
  char fullname[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  ierr = PetscFListConcat(path,name,fullname);CHKERRQ(ierr);
  ierr = PetscFListAdd(&PetscDrawList,sname,fullname,(void (*)(void))function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawSetFromOptions" 
/*@C
   PetscDrawSetFromOptions - Sets the graphics type from the options database.
      Defaults to a PETSc X windows graphics.

   Collective on PetscDraw

   Input Parameter:
.     draw - the graphics context

   Options Database Keys:
+   -nox - do not use X graphics (ignore graphics calls, but run program correctly)
-   -nox_warning - when X windows support is not installed this prevents the warning message
                   from being printed

   Level: intermediate

   Notes: 
    Must be called after PetscDrawCreate() before the PetscDrawtor is used.

    Concepts: drawing^setting options
    Concepts: graphics^setting options

.seealso: PetscDrawCreate(), PetscDrawSetType()

@*/
int PetscDrawSetFromOptions(PetscDraw draw)
{
  int        ierr;
  PetscTruth flg,nox;
  char       vtype[256],*def;
#if !defined(PARCH_Win32) && !defined(PETSC_HAVE_X11)
  PetscTruth warn;
#endif

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_COOKIE);

  if (!PetscDrawList) SETERRQ(1,"No draw implementations registered");
  if (draw->type_name) {
    def = draw->type_name;
  } else {
    ierr = PetscOptionsHasName(PETSC_NULL,"-nox",&nox);CHKERRQ(ierr);
    def  = PETSC_DRAW_NULL;
#if defined(PARCH_win32)
    if (!nox) def = PETSC_DRAW_WIN32;
#elif defined(PETSC_HAVE_X11)
    if (!nox) def = PETSC_DRAW_X;
#else
    ierr = PetscOptionsHasName(PETSC_NULL,"-nox_warning",&warn);CHKERRQ(ierr);
    if (!nox && !warn) {
      (*PetscErrorPrintf)("PETSc installed without X windows on this machine\nproceeding without graphics\n");
    }
#endif
  }
  ierr = PetscOptionsBegin(draw->comm,draw->prefix,"Graphics (PetscDraw) Options","Draw");CHKERRQ(ierr);
    ierr = PetscOptionsList("-draw_type","Type of graphical output","PetscDrawSetType",PetscDrawList,def,vtype,256,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PetscDrawSetType(draw,vtype);CHKERRQ(ierr);
    } else if (!draw->type_name) {
      ierr = PetscDrawSetType(draw,def);CHKERRQ(ierr);
    }
    ierr = PetscOptionsName("-nox","Run without graphics","None",&nox);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
