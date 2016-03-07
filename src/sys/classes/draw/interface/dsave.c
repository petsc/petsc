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
-  -draw_save_single_file - saves each new image in the same file, normally each new image is saved in a new file with filename_%d

   Level: intermediate

   Concepts: X windows^graphics

   Notes: You should call this BEFORE creating your image and calling PetscDrawFlush().

   The ffmpeg utility must be in your path to make the movie.

   It is recommended that PETSc be configured with the option --with-afterimage to save the images. Otherwise, PETSc will write uncompressed, binary PPM files.
   The .ext formats that are supported depend on what formats AfterImage was configured with; on the Apple Mac both .Gif and .Jpeg are supported.
   If X windows generates an error message about X_CreateWindow() failing then Afterimage was installed without X windows.
   Reinstall Afterimage using the following configure flags:
   ./configure flags --x-includes=/pathtoXincludes --x-libraries=/pathtoXlibraries
   For example under Mac OS X Mountain Lion --x-includes=/opt/X11/include -x-libraries=/opt/X11/lib

.seealso: PetscDrawSetFromOptions(), PetscDrawCreate(), PetscDrawDestroy(), PetscDrawSetSaveFinalImage()
@*/
PetscErrorCode  PetscDrawSetSave(PetscDraw draw,const char *filename,PetscBool movie)
{
  const char     *name = NULL;
  const char     *ext = NULL;
  char           buf[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  if (filename) PetscValidCharPointer(filename,2);

  /* determine filename and extension */
  if (filename && filename[0]) {
    ierr = PetscStrchr(filename,'.',(char **)&ext);CHKERRQ(ierr);
    if (!ext) name = filename;
    else if (ext != filename) {
      size_t l1 = 0,l2 = 0;
      ierr = PetscStrlen(filename,&l1);CHKERRQ(ierr);
      ierr = PetscStrlen(ext,&l2);CHKERRQ(ierr);
      ierr = PetscMemzero(buf,sizeof(buf));CHKERRQ(ierr);
      ierr = PetscStrncpy(buf,filename,l1-l2+1);CHKERRQ(ierr);
      name = buf;
    }
  }
  if (!name) {ierr = PetscObjectGetName((PetscObject)draw,&name);CHKERRQ(ierr);}
#if defined(PETSC_HAVE_AFTERIMAGE)
  if (!ext) ext = ".Gif";
#else
  if (!ext) ext = ".ppm";
  else {
    PetscBool match;
    ierr = PetscStrcasecmp(ext,".ppm",&match);CHKERRQ(ierr);
    if (!match) SETERRQ1(PetscObjectComm((PetscObject)draw),PETSC_ERR_SUP,"Image extension %s not supported, use .ppm",ext);
  }
#endif

  ierr = PetscFree(draw->savefilename);CHKERRQ(ierr);
  ierr = PetscFree(draw->savefilenameext);CHKERRQ(ierr);
  ierr = PetscStrallocpy(name,&draw->savefilename);CHKERRQ(ierr);
  ierr = PetscStrallocpy(ext,&draw->savefilenameext);CHKERRQ(ierr);
  draw->savefilemovie = movie;
  if (draw->ops->setsave) {
    ierr = (*draw->ops->setsave)(draw,draw->savefilename);CHKERRQ(ierr);
  }

  if (draw->savesinglefile) {
    ierr = PetscInfo2(NULL,"Will save image to file %s%s\n",draw->savefilename,draw->savefilenameext);CHKERRQ(ierr);
  } else {
    ierr = PetscInfo3(NULL,"Will save images to file %s/%s_*.%s\n",draw->savefilename,draw->savefilename,draw->savefilenameext);CHKERRQ(ierr);
  }
  if (draw->savefilemovie) {
    ierr = PetscInfo1(NULL,"Will save movie to file %s.m4v\n",draw->savefilename);CHKERRQ(ierr);
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

   It is recommended that PETSc be configured with the option --with-afterimage to save the images.
   If X windows generates an error message about X_CreateWindow() failing then Afterimage was installed without X windows.
   Reinstall Afterimage using the following configure flags:
   ./configure flags --x-includes=/pathtoXincludes --x-libraries=/pathtoXlibraries
   For example under Mac OS X Mountain Lion: --x-includes=/opt/X11/include -x-libraries=/opt/X11/lib

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
