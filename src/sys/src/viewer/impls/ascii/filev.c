#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: filev.c,v 1.90 1999/05/12 03:26:11 bsmith Exp balay $";
#endif

#include "src/sys/src/viewer/viewerimpl.h"  /*I     "petsc.h"   I*/
#include "pinclude/petscfix.h"
#include <stdarg.h>

typedef struct {
  FILE          *fd;
  int           tab;   /* how many times text is tabbed in from left */
} Viewer_ASCII;

/* ----------------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "ViewerDestroy_ASCII"
int ViewerDestroy_ASCII(Viewer v)
{
  int          rank = 0;
  Viewer_ASCII *vascii = (Viewer_ASCII *)v->data;

  PetscFunctionBegin;
  if (!rank && vascii->fd != stderr && vascii->fd != stdout) fclose(vascii->fd);
  PetscFree(vascii);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ViewerFlush_ASCII"
int ViewerFlush_ASCII(Viewer v)
{
  int          rank,ierr;
  Viewer_ASCII *vascii = (Viewer_ASCII *)v->data;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(v->comm,&rank);CHKERRQ(ierr);
  if (rank) PetscFunctionReturn(0);
  fflush(vascii->fd);
  PetscFunctionReturn(0);  
}

#undef __FUNC__  
#define __FUNC__ "ViewerASCIIGetPointer"
/*@C
    ViewerASCIIGetPointer - Extracts the file pointer from an ASCII viewer.

    Not Collective

+   viewer - viewer context, obtained from ViewerASCIIOpen()
-   fd - file pointer

    Level: intermediate

    Fortran Note:
    This routine is not supported in Fortran.

.keywords: Viewer, file, get, pointer

.seealso: ViewerASCIIOpen()
@*/
int ViewerASCIIGetPointer(Viewer viewer, FILE **fd)
{
  Viewer_ASCII *vascii = (Viewer_ASCII *)viewer->data;

  PetscFunctionBegin;
  *fd = vascii->fd;
  PetscFunctionReturn(0);
}


/*
   If petsc_history is on, then all Petsc*Printf() results are saved
   if the appropriate (usually .petschistory) file.
*/
extern FILE *petsc_history;

#undef __FUNC__  
#define __FUNC__ "ViewerASCIIPushTab" 
/*@C
    ViewerASCIIPushTab - Adds one more tab to the amount that ViewerASCIIPrintf()
     lines are tabbed.

    Not Collective, but only first processor in set has any effect

    Input Parameters:
.    viewer - optained with ViewerASCIIOpen()

    Level: developer

    Fortran Note:
    This routine is not supported in Fortran.

.keywords: parallel, fprintf

.seealso: PetscPrintf(), PetscSynchronizedPrintf(), ViewerASCIIPrintf(),
          ViewerASCIIPopTab()
@*/
int ViewerASCIIPushTab(Viewer viewer)
{
  Viewer_ASCII *ascii = (Viewer_ASCII*) viewer->data;

  PetscFunctionBegin;
  if (!PetscTypeCompare(viewer->type_name,ASCII_VIEWER)) PetscFunctionReturn(0);
  ascii->tab++;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ViewerASCIIPopTab" 
/*@C
    ViewerASCIIPopTab - Removes one tab from the amount that ViewerASCIIPrintf()
     lines are tabbed.

    Not Collective, but only first processor in set has any effect

    Input Parameters:
.    viewer - optained with ViewerASCIIOpen()

    Level: developer

    Fortran Note:
    This routine is not supported in Fortran.

.keywords: parallel, fprintf

.seealso: PetscPrintf(), PetscSynchronizedPrintf(), ViewerASCIIPrintf(),
          ViewerASCIIPushTab()
@*/
int ViewerASCIIPopTab(Viewer viewer)
{
  Viewer_ASCII *ascii = (Viewer_ASCII*) viewer->data;

  PetscFunctionBegin;
  if (!PetscTypeCompare(viewer->type_name,ASCII_VIEWER)) PetscFunctionReturn(0);
  if (ascii->tab <= 0) SETERRQ(1,1,"More tabs popped than pushed");
  ascii->tab--;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ViewerASCIIPrintf" 
/*@C
    ViewerASCIIPrintf - Prints to a file, only from the first
    processor in the viewer

    Not Collective, but only first processor in set has any effect

    Input Parameters:
+    viewer - optained with ViewerASCIIOpen()
-    format - the usual printf() format string 

    Level: developer

    Fortran Note:
    This routine is not supported in Fortran.

.keywords: parallel, fprintf

.seealso: PetscPrintf(), PetscSynchronizedPrintf(), ViewerASCIIOpen(),
          ViewerASCIIPushTab(), ViewerASCIIPopTab()
@*/
int ViewerASCIIPrintf(Viewer viewer,const char format[],...)
{
  Viewer_ASCII *ascii = (Viewer_ASCII*) viewer->data;
  int          rank, tab, ierr;
  FILE         *fd = ascii->fd;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(viewer->comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    va_list Argp;

    tab = ascii->tab;
    while (tab--) fprintf(fd,"  ");

    va_start( Argp, format );
#if defined(PETSC_HAVE_VPRINTF_CHAR)
    vfprintf(fd,format,(char*)Argp);
#else
    vfprintf(fd,format,Argp);
#endif
    fflush(fd);
    if (petsc_history) {
      tab = ascii->tab;
      while (tab--) fprintf(fd,"  ");
#if defined(PETSC_HAVE_VPRINTF_CHAR)
      vfprintf(petsc_history,format,(char *)Argp);
#else
      vfprintf(petsc_history,format,Argp);
#endif
      fflush(petsc_history);
    }
    va_end( Argp );
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ViewerSetFilename"
/*@C
     ViewerSetFilename - Sets the name of the file the viewer uses.

    Collective on Viewer

  Input Parameters:
+  viewer - the viewer; either ASCII or binary
-  name - the name of the file it should use

    Level: advanced

.seealso: ViewerCreate(), ViewerSetType(), ViewerASCIIOpen(), ViewerBinaryOpen()

@*/
int ViewerSetFilename(Viewer viewer,const char name[])
{
  int ierr, (*f)(Viewer,const char[]);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,VIEWER_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)viewer,"ViewerSetFilename_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(viewer,name);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "ViewerSetFilename_ASCII"
int ViewerSetFilename_ASCII(Viewer viewer,const char name[])
{
  int          ierr;
  char         fname[256];
  Viewer_ASCII *vascii = (Viewer_ASCII *) viewer->data;

  PetscFunctionBegin;
  if (!name) PetscFunctionReturn(0);

  if (!PetscStrcmp(name,"stderr"))      vascii->fd = stderr;
  else if (!PetscStrcmp(name,"stdout")) vascii->fd = stdout;
  else {
    ierr         = PetscFixFilename(name,fname);CHKERRQ(ierr);
    vascii->fd   = fopen(fname,"w"); 
    if (!vascii->fd) SETERRQ1(PETSC_ERR_FILE_OPEN,0,"Cannot open viewer file: %s",fname);
  }
#if defined(PETSC_USE_LOG)
  PLogObjectState((PetscObject)viewer,"File: %s",name);
#endif

  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "ViewerCreate_ASCII"
int ViewerCreate_ASCII(Viewer v)
{
  Viewer_ASCII *vascii;
  int          ierr;

  PetscFunctionBegin;
  vascii  = PetscNew(Viewer_ASCII);CHKPTRQ(vascii);
  v->data = (void *) vascii;

  v->ops->destroy     = ViewerDestroy_ASCII;
  v->ops->flush       = ViewerFlush_ASCII;

  /* defaults to stdout unless set with ViewerSetFilename() */
  vascii->fd         = stdout;
  v->format          = VIEWER_FORMAT_ASCII_DEFAULT;
  v->iformat         = 0;
  v->outputname      = 0;
  vascii->tab        = 0;
  v->type_name    = (char *) PetscMalloc((1+PetscStrlen(ASCII_VIEWER))*sizeof(char));CHKPTRQ(v->type_name);
  ierr = PetscStrcpy(v->type_name,ASCII_VIEWER);CHKERRQ(ierr);

  ierr = PetscObjectComposeFunction((PetscObject)v,"ViewerSetFilename_C",
                                    "ViewerSetFilename_ASCII",
                                     (void*)ViewerSetFilename_ASCII);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END

