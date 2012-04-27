
/*
       Provides the registration process for PETSc PetscDraw routines
*/
#include <../src/sys/draw/drawimpl.h>  /*I "petscdraw.h" I*/

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
PetscErrorCode  PetscDrawCreate(MPI_Comm comm,const char display[],const char title[],int x,int y,int w,int h,PetscDraw *indraw)
{
  PetscDraw      draw;
  PetscErrorCode ierr;
  PetscReal      dpause;
  PetscBool      flag;

  PetscFunctionBegin;
#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = PetscDrawInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif
  *indraw = 0;
  ierr = PetscHeaderCreate(draw,_p_PetscDraw,struct _PetscDrawOps,PETSC_DRAW_CLASSID,-1,"Draw","Graphics","Draw",comm,PetscDrawDestroy,0);CHKERRQ(ierr);
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
  draw->savefilename = PETSC_NULL;
  draw->savefilemovie = PETSC_FALSE;
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
PetscErrorCode  PetscDrawSetType(PetscDraw draw,const PetscDrawType type)
{
  PetscErrorCode ierr,(*r)(PetscDraw);
  PetscBool       match;
  PetscBool       flg=PETSC_FALSE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  PetscValidCharPointer(type,2);

  ierr = PetscObjectTypeCompare((PetscObject)draw,type,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  /*  User requests no graphics */
  ierr = PetscOptionsHasName(PETSC_NULL,"-nox",&flg);CHKERRQ(ierr);

  /*
     This is not ideal, but it allows codes to continue to run if X graphics 
   was requested but is not installed on this machine. Mostly this is for
   testing.
   */
#if !defined(PETSC_HAVE_X)
  if (!flg) {
    ierr = PetscStrcmp(type,PETSC_DRAW_X,&match);CHKERRQ(ierr);
    if (match) {
      PetscBool  dontwarn = PETSC_TRUE;
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
    draw->ops->destroy = PETSC_NULL;
    draw->data = 0;
  }

  ierr =  PetscFListFind(PetscDrawList,((PetscObject)draw)->comm,type,PETSC_TRUE,(void (**)(void)) &r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown PetscDraw type given: %s",type);
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
PetscErrorCode  PetscDrawRegisterDestroy(void)
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
PetscErrorCode  PetscDrawGetType(PetscDraw draw,const PetscDrawType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  PetscValidPointer(type,2);
  *type = ((PetscObject)draw)->type_name;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawRegister" 
PetscErrorCode  PetscDrawRegister(const char *sname,const char *path,const char *name,PetscErrorCode (*function)(PetscDraw))
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
PetscErrorCode  PetscDrawSetFromOptions(PetscDraw draw)
{
  PetscErrorCode ierr;
  PetscBool      flg,nox;
  char           vtype[256];
  const char     *def;
  PetscBool      save;
#if !defined(PETSC_USE_WINDOWS_GRAPHICS) && !defined(PETSC_HAVE_X)
  PetscBool      warn;
#endif
  char           filename[PETSC_MAX_PATH_LEN];
  PetscBool      movie = PETSC_FALSE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);

  if (!PetscDrawList) {
    ierr = PetscDrawRegisterAll(PETSC_NULL);CHKERRQ(ierr);
  }

  if (((PetscObject)draw)->type_name) {
    def = ((PetscObject)draw)->type_name;
  } else {
    ierr = PetscOptionsHasName(PETSC_NULL,"-nox",&nox);CHKERRQ(ierr);
    def  = PETSC_DRAW_NULL;
#if defined(PETSC_USE_WINDOWS_GRAPHICS) && !defined(PETSC_HAVE_X)
    if (!nox) def = PETSC_DRAW_WIN32;
#elif defined(PETSC_HAVE_X)
    if (!nox) def = PETSC_DRAW_X;
#else
    ierr = PetscOptionsHasName(PETSC_NULL,"-nox_warning",&warn);CHKERRQ(ierr);
    if (!nox && !warn) {
      (*PetscErrorPrintf)("PETSc installed without X windows on this machine\nproceeding without graphics\n");
    }
#endif
  }
  ierr = PetscObjectOptionsBegin((PetscObject)draw);CHKERRQ(ierr);
    ierr = PetscOptionsList("-draw_type","Type of graphical output","PetscDrawSetType",PetscDrawList,def,vtype,256,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PetscDrawSetType(draw,vtype);CHKERRQ(ierr);
    } else if (!((PetscObject)draw)->type_name) {
      ierr = PetscDrawSetType(draw,def);CHKERRQ(ierr);
    }
    ierr = PetscOptionsName("-nox","Run without graphics","None",&nox);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-draw_save_movie","Make a movie from the images saved","PetscDrawSetSave",movie,&movie,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsString("-draw_save","Save graphics to file","PetscDrawSetSave",filename,filename,PETSC_MAX_PATH_LEN,&save);CHKERRQ(ierr);
    if (save) {
      ierr = PetscDrawSetSave(draw,filename,movie);CHKERRQ(ierr);
    }

    /* process any options handlers added with PetscObjectAddOptionsHandler() */
    ierr = PetscObjectProcessOptionsHandlers((PetscObject)draw);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawSetSave" 
/*@C
   PetscDrawSave - Saves images produced in a PetscDraw into a file as a Gif file using AfterImage

   Collective on PetscDraw

   Input Parameter:
+  draw      - the graphics context
.  filename  - name of the file, if PETSC_NULL uses name of draw object
-  movie - produce a movie of all the images 

   Options Database Command:
+  -draw_save  <filename>
-  -draw_save_movie

   Level: intermediate

   Concepts: X windows^graphics
   Concepts: drawing^postscript
   Concepts: postscript^graphics
   Concepts: drawing^Microsoft Windows

   Notes: Requires that PETSc be configured with the option --with-afterimage


.seealso: PetscDrawSetFromOptions(), PetscDrawCreate(), PetscDrawDestroy(), PetscDrawSave()
@*/
PetscErrorCode  PetscDrawSetSave(PetscDraw draw,const char *filename,PetscBool movie)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  ierr = PetscFree(draw->savefilename);CHKERRQ(ierr);
  draw->savefilecount = 0;
  draw->savefilemovie = movie;
  if (filename && filename[0]) {
    ierr = PetscStrallocpy(filename,&draw->savefilename);CHKERRQ(ierr);
  } else {
    const char *name;
    ierr = PetscObjectGetName((PetscObject)draw,&name);CHKERRQ(ierr);
    ierr = PetscStrallocpy(name,&draw->savefilename);CHKERRQ(ierr);
  }
  if (draw->ops->setsave) {
    ierr = (*draw->ops->setsave)(draw,filename);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
