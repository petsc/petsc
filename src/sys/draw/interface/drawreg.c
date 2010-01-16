#define PETSC_DLL
/*
       Provides the registration process for PETSc PetscDraw routines
*/
#include "../src/sys/draw/drawimpl.h"  /*I "petscdraw.h" I*/

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
PetscErrorCode PETSC_DLLEXPORT PetscDrawCreate(MPI_Comm comm,const char display[],const char title[],int x,int y,int w,int h,PetscDraw *indraw)
{
  PetscDraw      draw;
  PetscErrorCode ierr;
  PetscReal      dpause;
  PetscTruth     flag;

  PetscFunctionBegin;
#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = PetscDrawInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif
  *indraw = 0;
  ierr = PetscHeaderCreate(draw,_p_PetscDraw,struct _PetscDrawOps,PETSC_DRAW_COOKIE,-1,"Draw",comm,PetscDrawDestroy,0);CHKERRQ(ierr);
  draw->data    = 0;
  ierr          = PetscStrallocpy(title,&draw->title);CHKERRQ(ierr);
  ierr          = PetscStrallocpy(display,&draw->display);CHKERRQ(ierr);
  draw->x       = x;
  draw->y       = y;
  draw->w       = w;
  draw->h       = h;
  draw->pause   = 0.0;
  draw->coor_xl = 0.0;  
  draw->coor_xr = 1.0;
  draw->coor_yl = 0.0;  
  draw->coor_yr = 1.0;
  draw->port_xl = 0.0;  
  draw->port_xr = 1.0;
  draw->port_yl = 0.0;  
  draw->port_yr = 1.0;
  draw->popup   = 0;
  ierr = PetscOptionsGetReal(PETSC_NULL,"-draw_pause",&dpause,&flag);CHKERRQ(ierr);
  if (flag) draw->pause = dpause;
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
PetscErrorCode PETSC_DLLEXPORT PetscDrawSetType(PetscDraw draw,const PetscDrawType type)
{
  PetscErrorCode ierr,(*r)(PetscDraw);
  PetscTruth      match;
  PetscTruth      flg=PETSC_FALSE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_COOKIE,1);
  PetscValidCharPointer(type,2);

  ierr = PetscTypeCompare((PetscObject)draw,type,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  /*  User requests no graphics */
  ierr = PetscOptionsHasName(PETSC_NULL,"-nox",&flg);CHKERRQ(ierr);

  /*
     This is not ideal, but it allows codes to continue to run if X graphics 
   was requested but is not installed on this machine. Mostly this is for
   testing.
   */
#if !defined(PETSC_HAVE_X11)
  if (!flg) {
    ierr = PetscStrcmp(type,PETSC_DRAW_X,&match);CHKERRQ(ierr);
    if (match) {
      PetscTruth dontwarn = PETSC_TRUE;
      flg = PETSC_TRUE;
      ierr = PetscOptionsHasName(PETSC_NULL,"-nox_warning",&dontwarn);CHKERRQ(ierr);
      if (!dontwarn) {
        (*PetscErrorPrintf)("PETSc installed without X windows on this machine\nproceeding without graphics\n");
      }
    }
  }
#endif
  if (flg) {
    type  = PETSC_DRAW_NULL;
  }

  if (draw->data) {
    /* destroy the old private PetscDraw context */
    ierr       = (*draw->ops->destroy)(draw);CHKERRQ(ierr);
    draw->data = 0;
  }

  ierr =  PetscFListFind(PetscDrawList,((PetscObject)draw)->comm,type,(void (**)(void)) &r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown PetscDraw type given: %s",type);
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
PetscErrorCode PETSC_DLLEXPORT PetscDrawRegisterDestroy(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFListDestroy(&PetscDrawList);CHKERRQ(ierr);
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
PetscErrorCode PETSC_DLLEXPORT PetscDrawGetType(PetscDraw draw,const PetscDrawType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_COOKIE,1);
  PetscValidPointer(type,2);
  *type = ((PetscObject)draw)->type_name;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawRegister" 
PetscErrorCode PETSC_DLLEXPORT PetscDrawRegister(const char *sname,const char *path,const char *name,PetscErrorCode (*function)(PetscDraw))
{
  PetscErrorCode ierr;
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
PetscErrorCode PETSC_DLLEXPORT PetscDrawSetFromOptions(PetscDraw draw)
{
  PetscErrorCode ierr;
  PetscTruth flg,nox;
  char       vtype[256];
  const char *def;
#if !defined(PETSC_USE_WINDOWS_GRAPHICS) && !defined(PETSC_HAVE_X11)
  PetscTruth warn;
#endif

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_COOKIE,1);

  if (!PetscDrawList) {
    ierr = PetscDrawRegisterAll(PETSC_NULL);CHKERRQ(ierr);
  }

  if (((PetscObject)draw)->type_name) {
    def = ((PetscObject)draw)->type_name;
  } else {
    ierr = PetscOptionsHasName(PETSC_NULL,"-nox",&nox);CHKERRQ(ierr);
    def  = PETSC_DRAW_NULL;
#if defined(PETSC_USE_WINDOWS_GRAPHICS) && !defined(PETSC_HAVE_X11)
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
  ierr = PetscOptionsBegin(((PetscObject)draw)->comm,((PetscObject)draw)->prefix,"Graphics (PetscDraw) Options","Draw");CHKERRQ(ierr);
    ierr = PetscOptionsList("-draw_type","Type of graphical output","PetscDrawSetType",PetscDrawList,def,vtype,256,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PetscDrawSetType(draw,vtype);CHKERRQ(ierr);
    } else if (!((PetscObject)draw)->type_name) {
      ierr = PetscDrawSetType(draw,def);CHKERRQ(ierr);
    }
    ierr = PetscOptionsName("-nox","Run without graphics","None",&nox);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
