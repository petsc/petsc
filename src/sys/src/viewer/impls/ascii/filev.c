/* $Id: filev.c,v 1.102 2000/01/11 20:58:57 bsmith Exp bsmith $ */

#include "src/sys/src/viewer/viewerimpl.h"  /*I     "petsc.h"   I*/
#include "petscfix.h"
#include <stdarg.h>

typedef struct {
  FILE          *fd;
  int           tab;            /* how many times text is tabbed in from left */
  int           tab_store;      /* store tabs value while tabs are turned off */
  Viewer        bviewer;        /* if viewer is a singleton, this points to mother */
  Viewer        sviewer;        /* if viewer has a singleton, this points to singleton */
  char          *filename;
  PetscTruth    storecompressed; 
} Viewer_ASCII;

/* ----------------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "ViewerDestroy_ASCII"
int ViewerDestroy_ASCII(Viewer viewer)
{
  int          rank,ierr;
  Viewer_ASCII *vascii = (Viewer_ASCII *)viewer->data;

  PetscFunctionBegin;
  if (vascii->sviewer) {
    SETERRQ(1,1,"ASCII Viewer destroyed before restoring singleton viewer");
  }
  ierr = MPI_Comm_rank(viewer->comm,&rank);CHKERRQ(ierr);
  if (!rank && vascii->fd != stderr && vascii->fd != stdout) {
    fclose(vascii->fd);
    if (vascii->storecompressed) {
      char par[1024],buf[1024];
      FILE *fp;
      ierr = PetscStrcpy(par,"gzip ");CHKERRQ(ierr);
      ierr = PetscStrcat(par,vascii->filename);CHKERRQ(ierr);
      ierr = PetscPOpen(PETSC_COMM_SELF,PETSC_NULL,par,"r",&fp);CHKERRQ(ierr);
      if (fgets(buf,1024,fp)) {
        SETERRQ2(1,1,"Error from compression command %s %s\n%s",par,buf);
      }
    }
  }
  ierr = PetscStrfree(vascii->filename);CHKERRQ(ierr);
  ierr = PetscFree(vascii);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ViewerDestroy_ASCII_Singleton"
int ViewerDestroy_ASCII_Singleton(Viewer viewer)
{
  Viewer_ASCII *vascii = (Viewer_ASCII *)viewer->data;
  int          ierr;
  PetscFunctionBegin;
  ierr = ViewerRestoreSingleton(vascii->bviewer,&viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ViewerFlush_ASCII_Singleton_0"
int ViewerFlush_ASCII_Singleton_0(Viewer viewer)
{
  Viewer_ASCII *vascii = (Viewer_ASCII *)viewer->data;

  PetscFunctionBegin;
  fflush(vascii->fd);
  PetscFunctionReturn(0);  
}

#undef __FUNC__  
#define __FUNC__ "ViewerFlush_ASCII"
int ViewerFlush_ASCII(Viewer viewer)
{
  int          rank,ierr;
  Viewer_ASCII *vascii = (Viewer_ASCII *)viewer->data;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(viewer->comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    fflush(vascii->fd);
  }

  /*
     Also flush anything printed with ViewerASCIISynchronizedPrintf()
  */
  ierr = PetscSynchronizedFlush(viewer->comm);CHKERRQ(ierr);
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

.seealso: ViewerASCIIOpen(), ViewerDestroy(), ViewerSetType(), ViewerCreate(), ViewerASCIIPrintf(),
          ViewerASCIISynchronizedPrintf(), ViewerFlush()
@*/
int ViewerASCIIGetPointer(Viewer viewer,FILE **fd)
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
          ViewerASCIIPopTab(), ViewerASCIISynchronizedPrintf(), ViewerASCIIOpen(),
          ViewerCreate(), ViewerDestroy(), ViewerSetType(), ViewerASCIIGetPointer()
@*/
int ViewerASCIIPushTab(Viewer viewer)
{
  Viewer_ASCII *ascii = (Viewer_ASCII*)viewer->data;
  PetscTruth   isascii;
  int          ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,VIEWER_COOKIE);
  ierr = PetscTypeCompare((PetscObject)viewer,ASCII_VIEWER,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ascii->tab++;
  }
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
          ViewerASCIIPushTab(), ViewerASCIISynchronizedPrintf(), ViewerASCIIOpen(),
          ViewerCreate(), ViewerDestroy(), ViewerSetType(), ViewerASCIIGetPointer()
@*/
int ViewerASCIIPopTab(Viewer viewer)
{
  Viewer_ASCII *ascii = (Viewer_ASCII*)viewer->data;
  int          ierr;
  PetscTruth   isascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,VIEWER_COOKIE);
  ierr = PetscTypeCompare((PetscObject)viewer,ASCII_VIEWER,&isascii);CHKERRQ(ierr);
  if (isascii) {
    if (ascii->tab <= 0) SETERRQ(1,1,"More tabs popped than pushed");
    ascii->tab--;
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ViewerASCIIUseTabs" 
/*@C
    ViewerASCIIUseTabs - Turns on or off the use of tabs with the ASCII viewer

    Not Collective, but only first processor in set has any effect

    Input Parameters:
+    viewer - optained with ViewerASCIIOpen()
-    flg - PETSC_YES or PETSC_NO

    Level: developer

    Fortran Note:
    This routine is not supported in Fortran.

.keywords: parallel, fprintf

.seealso: PetscPrintf(), PetscSynchronizedPrintf(), ViewerASCIIPrintf(),
          ViewerASCIIPopTab(), ViewerASCIISynchronizedPrintf(), ViewerASCIIPushTab(), ViewerASCIIOpen(),
          ViewerCreate(), ViewerDestroy(), ViewerSetType(), ViewerASCIIGetPointer()
@*/
int ViewerASCIIUseTabs(Viewer viewer,PetscTruth flg)
{
  Viewer_ASCII *ascii = (Viewer_ASCII*)viewer->data;
  PetscTruth   isascii;
  int          ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,VIEWER_COOKIE);
  ierr = PetscTypeCompare((PetscObject)viewer,ASCII_VIEWER,&isascii);CHKERRQ(ierr);
  if (isascii) {
    if (flg) {
      ascii->tab       = ascii->tab_store;
    } else {
      ascii->tab_store = ascii->tab;
      ascii->tab       = 0;
    }
  }
  PetscFunctionReturn(0);
}


/* ----------------------------------------------------------------------- */
/*
       These are defined in the file with PetscPrintf()
*/
typedef struct _PrintfQueue *PrintfQueue;
struct _PrintfQueue {
  char        string[256];
  PrintfQueue next;
};
extern PrintfQueue queue,queuebase;
extern int         queuelength;
extern FILE        *queuefile;

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
          ViewerASCIIPushTab(), ViewerASCIIPopTab(), ViewerASCIISynchronizedPrintf(),
          ViewerCreate(), ViewerDestroy(), ViewerSetType(), ViewerASCIIGetPointer()
@*/
int ViewerASCIIPrintf(Viewer viewer,const char format[],...)
{
  Viewer_ASCII *ascii = (Viewer_ASCII*)viewer->data;
  int          rank,tab,ierr;
  FILE         *fd = ascii->fd;
  PetscTruth   isascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,VIEWER_COOKIE);
  ierr = PetscTypeCompare((PetscObject)viewer,ASCII_VIEWER,&isascii);CHKERRQ(ierr);
  if (!isascii) SETERRQ(1,1,"Not ASCII viewer");

  ierr = MPI_Comm_rank(viewer->comm,&rank);CHKERRQ(ierr);
  if (ascii->bviewer) {ierr = MPI_Comm_rank(ascii->bviewer->comm,&rank);CHKERRQ(ierr);}
  if (!rank) {
    va_list Argp;
    if (ascii->bviewer) {
      queuefile = fd;
    }

    tab = ascii->tab;
    while (tab--) fprintf(fd,"  ");

    va_start(Argp,format);
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
    va_end(Argp);
  } else if (ascii->bviewer) { /* this is a singleton viewer that is not on process 0 */
    int         len;
    va_list     Argp;

    PrintfQueue next = PetscNew(struct _PrintfQueue);CHKPTRQ(next);
    if (queue) {queue->next = next; queue = next;}
    else       {queuebase   = queue = next;}
    queuelength++;
    va_start(Argp,format);
#if defined(PETSC_HAVE_VPRINTF_CHAR)
    vsprintf(next->string,format,(char *)Argp);
#else
    vsprintf(next->string,format,Argp);
#endif
    va_end(Argp);
    ierr = PetscStrlen(next->string,&len);CHKERRQ(ierr);
    if (len > 256) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Formatted string longer then 256 bytes");
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

.seealso: ViewerCreate(), ViewerSetType(), ViewerASCIIOpen(), ViewerBinaryOpen(), ViewerDestroy(),
          ViewerASCIIGetPointer(), ViewerASCIIPrintf(), ViewerASCIISynchronizedPrintf()

@*/
int ViewerSetFilename(Viewer viewer,const char name[])
{
  int ierr,(*f)(Viewer,const char[]);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,VIEWER_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)viewer,"ViewerSetFilename_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(viewer,name);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ViewerGetFilename"
/*@C
     ViewerGetFilename - Gets the name of the file the viewer uses.

    Not Collective

  Input Parameter:
.  viewer - the viewer; either ASCII or binary

  Output Parameter:
.  name - the name of the file it is using

    Level: advanced

.seealso: ViewerCreate(), ViewerSetType(), ViewerASCIIOpen(), ViewerBinaryOpen(), ViewerSetFilename()

@*/
int ViewerGetFilename(Viewer viewer,char **name)
{
  int ierr,(*f)(Viewer,char **);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,VIEWER_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)viewer,"ViewerGetFilename_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(viewer,name);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "ViewerGetFilename_ASCII"
int ViewerGetFilename_ASCII(Viewer viewer,char **name)
{
  Viewer_ASCII *vascii = (Viewer_ASCII*)viewer->data;

  PetscFunctionBegin;
  
  *name = vascii->filename;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "ViewerSetFilename_ASCII"
int ViewerSetFilename_ASCII(Viewer viewer,const char name[])
{
  int          ierr,len;
  char         fname[256],*gz;
  Viewer_ASCII *vascii = (Viewer_ASCII*)viewer->data;
  PetscTruth   isstderr,isstdout;

  PetscFunctionBegin;
  if (!name) PetscFunctionReturn(0);

  ierr = PetscStrallocpy(name,&vascii->filename);CHKERRQ(ierr);

  /* Is this file to be compressed */
  vascii->storecompressed = PETSC_FALSE;
  ierr = PetscStrstr(vascii->filename,".gz",&gz);CHKERRQ(ierr);
  if (gz) {
    ierr = PetscStrlen(gz,&len);CHKERRQ(ierr);
    if (len == 3) {
      *gz = 0;
      vascii->storecompressed = PETSC_TRUE;
    } 
  }
  ierr = PetscStrcmp(name,"stderr",&isstderr);CHKERRQ(ierr);
  ierr = PetscStrcmp(name,"stdout",&isstdout);CHKERRQ(ierr);
  if (isstderr)      vascii->fd = stderr;
  else if (isstdout) vascii->fd = stdout;
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

#undef __FUNC__  
#define __FUNC__ "ViewerGetSingleton_ASCII"
int ViewerGetSingleton_ASCII(Viewer viewer,Viewer *outviewer)
{
  int          rank,ierr;
  Viewer_ASCII *vascii = (Viewer_ASCII *)viewer->data,*ovascii;

  PetscFunctionBegin;
  if (vascii->sviewer) {
    SETERRQ(1,1,"Singleton already obtained from viewer and not restored");
  }
  ierr         = ViewerCreate(PETSC_COMM_SELF,outviewer);CHKERRQ(ierr);
  ierr         = ViewerSetType(*outviewer,ASCII_VIEWER);CHKERRQ(ierr);
  ovascii      = (Viewer_ASCII*)(*outviewer)->data;
  ovascii->fd  = vascii->fd;
  ovascii->tab = vascii->tab;

  vascii->sviewer = *outviewer;

  (*outviewer)->format     = viewer->format;
  (*outviewer)->iformat    = viewer->iformat;
  (*outviewer)->outputname = viewer->outputname;

  ierr = MPI_Comm_rank(viewer->comm,&rank);CHKERRQ(ierr);
  ((Viewer_ASCII*)((*outviewer)->data))->bviewer = viewer;
  (*outviewer)->ops->destroy = ViewerDestroy_ASCII_Singleton;
  if (rank) {
    (*outviewer)->ops->flush = 0;
  } else {
    (*outviewer)->ops->flush = ViewerFlush_ASCII_Singleton_0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ViewerRestoreSingleton_ASCII"
int ViewerRestoreSingleton_ASCII(Viewer viewer,Viewer *outviewer)
{
  int          ierr;
  Viewer_ASCII *vascii = (Viewer_ASCII *)(*outviewer)->data;
  Viewer_ASCII *ascii  = (Viewer_ASCII *)viewer->data;

  PetscFunctionBegin;
  if (!ascii->sviewer) {
    SETERRQ(1,1,"Singleton never obtained from viewer");
  }
  if (ascii->sviewer != *outviewer) {
    SETERRQ(1,1,"This viewer did not generate singleton");
  }

  ascii->sviewer             = 0;
  vascii->fd                 = stdout;
  (*outviewer)->ops->destroy = ViewerDestroy_ASCII;
  ierr                       = ViewerDestroy(*outviewer);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "ViewerCreate_ASCII"
int ViewerCreate_ASCII(Viewer viewer)
{
  Viewer_ASCII *vascii;
  int          ierr;

  PetscFunctionBegin;
  vascii       = PetscNew(Viewer_ASCII);CHKPTRQ(vascii);
  viewer->data = (void*)vascii;

  viewer->ops->destroy          = ViewerDestroy_ASCII;
  viewer->ops->flush            = ViewerFlush_ASCII;
  viewer->ops->getsingleton     = ViewerGetSingleton_ASCII;
  viewer->ops->restoresingleton = ViewerRestoreSingleton_ASCII;

  /* defaults to stdout unless set with ViewerSetFilename() */
  vascii->fd             = stdout;
  vascii->bviewer        = 0;
  vascii->sviewer        = 0;
  viewer->format         = VIEWER_FORMAT_ASCII_DEFAULT;
  viewer->iformat        = 0;
  viewer->outputname     = 0;
  vascii->tab            = 0;
  vascii->tab_store      = 0;
  vascii->filename       = 0;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)viewer,"ViewerSetFilename_C","ViewerSetFilename_ASCII",
                                     (void*)ViewerSetFilename_ASCII);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)viewer,"ViewerGetFilename_C","ViewerGetFilename_ASCII",
                                     (void*)ViewerGetFilename_ASCII);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END


#undef __FUNC__  
#define __FUNC__ "ViewerASCIISynchronizedPrintf"
/*@C
    ViewerASCIISynchronizedFPrintf - Prints synchronized output to the specified file from
    several processors.  Output of the first processor is followed by that of the 
    second, etc.

    Not Collective, must call collective ViewerFlush() to get the results out

    Input Parameters:
+   viewer - the ASCII viewer
-   format - the usual printf() format string 

    Level: intermediate

.seealso: PetscSynchronizedPrintf(), PetscSynchronizedFlush(), PetscFPrintf(),
          PetscFOpen(), ViewerFlush(), ViewerASCIIGetPointer(), ViewerDestroy(), ViewerASCIIOpen(),
          ViewerASCIIPrintf()

@*/
int ViewerASCIISynchronizedPrintf(Viewer viewer,const char format[],...)
{
  Viewer_ASCII *vascii = (Viewer_ASCII *)viewer->data;
  int          ierr,rank;
  MPI_Comm     comm;
  FILE         *fp;
  PetscTruth   isascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,VIEWER_COOKIE);
  ierr = PetscTypeCompare((PetscObject)viewer,ASCII_VIEWER,&isascii);
  if (!isascii) SETERRQ(1,1,"Not ASCII viewer");

  comm = viewer->comm;
  fp   = vascii->fd;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  
  /* First processor prints immediately to fp */
  if (!rank) {
    va_list Argp;
    va_start(Argp,format);
#if defined(PETSC_HAVE_VPRINTF_CHAR)
    vfprintf(fp,format,(char*)Argp);
#else
    vfprintf(fp,format,Argp);
#endif
    fflush(fp);
    queuefile = fp;
    if (petsc_history) {
#if defined(PETSC_HAVE_VPRINTF_CHAR)
      vfprintf(petsc_history,format,(char *)Argp);
#else
      vfprintf(petsc_history,format,Argp);
#endif
      fflush(petsc_history);
    }
    va_end(Argp);
  } else { /* other processors add to local queue */
    int         len;
    va_list     Argp;
    PrintfQueue next = PetscNew(struct _PrintfQueue);CHKPTRQ(next);
    if (queue) {queue->next = next; queue = next;}
    else       {queuebase   = queue = next;}
    queuelength++;
    va_start(Argp,format);
#if defined(PETSC_HAVE_VPRINTF_CHAR)
    vsprintf(next->string,format,(char *)Argp);
#else
    vsprintf(next->string,format,Argp);
#endif
    va_end(Argp);
    ierr = PetscStrlen(next->string,&len);CHKERRQ(ierr);
    if (len > 256) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Formatted string longer then 256 bytes");
  }
  PetscFunctionReturn(0);
}


