#include <petsc/private/drawimpl.h>  /*I "petscdraw.h" I*/

#undef __FUNCT__
#define __FUNCT__ "PetscDrawSetSave"
/*@C
   PetscDrawSetSave - Saves images produced in a PetscDraw into a file as a Gif file using AfterImage

   Collective on PetscDraw

   Input Parameter:
+  draw      - the graphics context
.  filename  - name of the file, if .ext then uses name of draw object plus .ext using .ext to determine the image type, if NULL use .Gif image type
-  movie - produce a movie of all the images

   Options Database Command:
+  -draw_save  <filename> - filename could be name.ext or .ext (where .ext determines the type of graphics file to save, for example .Gif)
.  -draw_save_movie
.  -draw_save_final_image [optional filename] - (X windows only) saves the final image displayed in a window
.  -draw_save_on_flush - saves an image on each flush in addition to each clear
-  -draw_save_single_file - saves each new image in the same file, normally each new image is saved in a new file with filename_%d

   Level: intermediate

   Concepts: X windows^graphics

   Notes: You should call this BEFORE creating your image and calling PetscDrawFlush().

   Requires that PETSc be configured with the option --with-afterimage to save the images and ffmpeg must be in your path to make the movie

   The .ext formats that are supported depend on what formats AfterImage was configured with; on the Apple Mac both .Gif and .Jpeg are supported.

   If X windows generates an error message about X_CreateWindow() failing then Afterimage was installed without X windows. Reinstall Afterimage using the
   ./configure flags --x-includes=/pathtoXincludes --x-libraries=/pathtoXlibraries   For example under Mac OS X Mountain Lion --x-includes=/opt/X11/include -x-libraries=/opt/X11/lib


.seealso: PetscDrawSetFromOptions(), PetscDrawCreate(), PetscDrawDestroy(), PetscDrawSetSaveFinalImage()
@*/
PetscErrorCode  PetscDrawSetSave(PetscDraw draw,const char *filename,PetscBool movie)
{
  PetscErrorCode ierr;
  char           *ext;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  ierr = PetscFree(draw->savefilename);CHKERRQ(ierr);

  /* determine extension of filename */
  if (filename && filename[0]) {
    ierr = PetscStrchr(filename,'.',&ext);CHKERRQ(ierr);
    if (!ext) SETERRQ1(PetscObjectComm((PetscObject)draw),PETSC_ERR_ARG_INCOMP,"Filename %s should end with graphics extension (for example .Gif)",filename);
  } else {
    ext = (char *)".Gif";
  }
  if (ext == filename) filename = NULL;
  ierr = PetscStrallocpy(ext,&draw->savefilenameext);CHKERRQ(ierr);

  draw->savefilemovie = movie;
  if (filename && filename[0]) {
    size_t  l1,l2;
    ierr = PetscStrlen(filename,&l1);CHKERRQ(ierr);
    ierr = PetscStrlen(ext,&l2);CHKERRQ(ierr);
    ierr = PetscMalloc1(l1-l2+1,&draw->savefilename);CHKERRQ(ierr);
    ierr = PetscStrncpy(draw->savefilename,filename,l1-l2+1);CHKERRQ(ierr);
  } else {
    const char *name;

    ierr = PetscObjectGetName((PetscObject)draw,&name);CHKERRQ(ierr);
    ierr = PetscStrallocpy(name,&draw->savefilename);CHKERRQ(ierr);
  }
  ierr = PetscInfo2(NULL,"Will save images to file %s%s\n",draw->savefilename,draw->savefilenameext);CHKERRQ(ierr);
  if (draw->ops->setsave) {
    ierr = (*draw->ops->setsave)(draw,draw->savefilename);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDrawSetSaveFinalImage"
/*@C
   PetscDrawSetSaveFinalImage - Saves the finale image produced in a PetscDraw into a file as a Gif file using AfterImage

   Collective on PetscDraw

   Input Parameter:
+  draw      - the graphics context
-  filename  - name of the file, if NULL uses name of draw object

   Options Database Command:
.  -draw_save_final_image  <filename>

   Level: intermediate

   Concepts: X windows^graphics

   Notes: You should call this BEFORE creating your image and calling PetscDrawFlush().

   Requires that PETSc be configured with the option --with-afterimage to save the images and ffmpeg must be in your path to make the movie

   If X windows generates an error message about X_CreateWindow() failing then Afterimage was installed without X windows. Reinstall Afterimage using the
   ./configure flags --x-includes=/pathtoXincludes --x-libraries=/pathtoXlibraries   For example under Mac OS X Mountain Lion --x-includes=/opt/X11/include -x-libraries=/opt/X11/lib


.seealso: PetscDrawSetFromOptions(), PetscDrawCreate(), PetscDrawDestroy(), PetscDrawSetSave()
@*/
PetscErrorCode  PetscDrawSetSaveFinalImage(PetscDraw draw,const char *filename)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  ierr = PetscFree(draw->savefinalfilename);CHKERRQ(ierr);

  if (filename && filename[0]) {
    ierr = PetscStrallocpy(filename,&draw->savefinalfilename);CHKERRQ(ierr);
  } else {
    const char *name;
    ierr = PetscObjectGetName((PetscObject)draw,&name);CHKERRQ(ierr);
    ierr = PetscStrallocpy(name,&draw->savefinalfilename);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDrawSave"
/*@
   PetscDrawSave - Saves a drawn image

   Collective on PetscDraw

   Input Parameters:
.  draw - the drawing context

   Level: advanced

   Notes: this is not normally called by the user, it is called by PetscDrawFlush() to save a sequence of images.

.seealso: PetscDrawSetSave()

@*/
PetscErrorCode  PetscDrawSave(PetscDraw draw)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  if (draw->ops->save) {
    ierr = (*draw->ops->save)(draw);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
