#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: dl.c,v 1.4 1998/01/14 14:52:43 bsmith Exp balay $";
#endif
/*
      Routines for opening dynamic link libraries (DLLs), keeping a searchable
   path of DLLs, obtaining remote DLLs via a URL and opening them locally.
*/

#include "petsc.h"
#include "sys.h"
#include "src/sys/src/files.h"

/* ------------------------------------------------------------------------------*/
/*
      Code to maintain a list of opened dynamic libraries
*/
#if defined(USE_DYNAMIC_LIBRARIES)
#include <dlfcn.h>

struct _DLLibraryList {
  DLLibraryList next;
  void          *handle;
};

#undef __FUNC__  
#define __FUNC__ "DLObtainLibrary"
/*
   DLObtainLibrary - Obtains a library from a URL and copies into local
        disk space.

  Input Parameter:
.   libname - name of library, including entire URL

  Output Paramter:
.   llibname - name of local copy of library

*/
int DLObtainLibrary(char *libname,char *llibname)
{
#if defined(USE_PYTHON_FOR_REMOTE_DLLS)
  char *par4,buf[1024];
  FILE *fp;
  PetscFunctionBegin;

  if (PetscStrncmp(libname,"ftp://",6) && PetscStrncmp(libname,"http://",7)) {
    SETERRQ(1,1,"Only support for ftp/http DLL retrieval with \n\
      USE_PYTHON_FOR_REMOTE_DLLS installation option");
  }
    
  /* Construct the Python script run command */
  par4 = (char *) PetscMalloc(1024*sizeof(char));CHKPTRQ(par4);
  PetscStrcpy(par4,PETSC_DIR);
  PetscStrcat(par4,"/bin/urlget.py ");
  PetscStrcat(par4,libname);
  PetscStrcat(par4," ");
  PetscStrcat(par4,llibname);

  if ((fp = popen(par4,"r")) == NULL) {
    SETERRQ(1,1,"Cannot Execute $(PETSC_DIR)/bin/urlget.py\n\
      Check if python1.5 is in your path");
  }
  if (fgets(buf,1024,fp) == NULL) {
    SETERRQ(1,1,"No output from $(PETSC_DIR)/bin/urlget.py");
  }
  if (PetscStrncmp(buf,"Error",5)) { SETERRQ(1,1,buf); }
  PetscStrcpy(llibname,buf);
  PetscFree(par4);
 
#elif defined(USE_EXPECT_FOR_REMOTE_DLLS)
  char *par4;
  int  ierr;

  PetscFunctionBegin;

  if (PetscStrncmp(par2,"ftp://",6)) {
    SETERRQ(1,1,"Only support for ftp DLL retrieval with \n\
      USE_EXPECT_FOR_REMOTE_DLLS installation option");
  }

  /* create name for copy of library in /tmp */
  PetscStrcpy(llibname,"/tmp/PETScLibXXXXXX");
  mktemp(llibname);
    
  /* get library file from ftp to /tmp */
  par4 = (char *) PetscMalloc(1024*sizeof(char));CHKPTRQ(par4);
  PetscStrcpy(par4,PETSC_DIR);
  PetscStrcat(par4,"/bin/ftpget ");
  PetscStrcat(par4,libname);
  PetscStrcat(par4," ");
  PetscStrcat(par4,llibname);
  PetscStrcat(par4," ");
  if (PLogPrintInfo) PetscStrcat(par4,"1");
  else               PetscStrcat(par4,"0");

  PLogInfo(0,"About to run: %s\n",par4);
  ierr = system(par4);
  if (ierr) { /* could not get file; try again with .so.1.0 suffix */
    PetscStrcpy(par4,PETSC_DIR);
    PetscStrcat(par4,"/bin/ftpget ");
    PetscStrcat(par4,libname);
    PetscStrcat(par4,".so.1.0 ");
    PetscStrcat(par4,llibname);
    PetscStrcat(par4," ");
    if (PLogPrintInfo) PetscStrcat(par4,"1");
    else               PetscStrcat(par4,"0");

    PLogInfo(0,"About to run: %s\n",par4);
    ierr = system(par4);
    if (ierr) {
      PetscErrorPrintf("Attempting %s\n",par4);
      SETERRQ(1,1,"Unable to retreive FTP library");
    }
  }
  PetscFree(par4);
#else
  PetscFunctionBegin;
  SETERRQ(1,1,"Not compiled to obtain remote DLLs\n\
    Compile with  USE_EXPECT_FOR_REMOTE_DLLS or USE_PYTHON_FOR_REMOTE_DLLS\n\
    installation option");
#endif

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "DLOpen"
/*
     DLOpen - Opens a dynamic link library

   Input Parameter:
    libname - name of the library, can be relative or absolute

   Output Paramter:
    handle - returned from dlopen

   Notes:
     [[<http,ftp>://hostname]/directoryname/]filename[.so.1.0]

     $PETSC_ARCH and $BOPT occuring in directoryname and filename 
       will be replaced with appropriate values
*/
int DLOpen(char *libname,void **handle)
{
  char       *par2,ierr,len,*par3,arch[10];
  PetscTruth foundlibrary;
  int        flg;

  PetscFunctionBegin;

  /* 
     make copy of library name and replace $PETSC_ARCH and $BOPT and 
     so we can add to the end of it to look for something like .so.1.0 etc.
  */
  len   = PetscStrlen(libname);
  par2  = (char *) PetscMalloc((16+len+1)*sizeof(char));CHKPTRQ(par2);
  ierr  = PetscStrcpy(par2,libname);CHKERRQ(ierr);
  
  par3 = PetscStrstr(par2,"$PETSC_ARCH");
  while (par3) {
    *par3  =  0;
    par3  += 11;
    ierr   = PetscGetArchType(arch,10);
    PetscStrcat(par2,arch);
    PetscStrcat(par2,par3);
    par3 = PetscStrstr(par2,"$PETSC_ARCH");
  }

  par3 = PetscStrstr(par2,"$BOPT");
  while (par3) {
    *par3  =  0;
    par3  += 5;
    PetscStrcat(par2,PETSC_BOPT);
    PetscStrcat(par2,par3);
    par3 = PetscStrstr(par2,"$BOPT");
  }

  /* 
     Remove any file: header
  */
  flg = !PetscStrncmp(par2,"file:",5);
  if (flg) {
    PetscStrcpy(par2,par2+5);
  }

  /*
     Is it an ftp or http library?
  */
  flg = !PetscStrncmp(par2,"ftp://",6) || !PetscStrncmp(par2,"http://",7);
  if (flg) {
    par3 = (char *) PetscMalloc(64*sizeof(char));CHKPTRQ(par3);
    ierr = DLObtainLibrary(par2,par3);CHKERRQ(ierr);
    PetscStrcpy(par2,par3);
    PetscFree(par3);
  }

  /* first check original given name */
  ierr  = PetscTestFile(par2,'x',&foundlibrary);CHKERRQ(ierr);
  if (!foundlibrary) {

    /* strip out .a from it if user put it in by mistake */
    len    = PetscStrlen(par2);
    if (par2[len-1] == 'a' && par2[len-2] == '.') par2[len-2] = 0;

    /* try appending .so.1.0 */
    PetscStrcat(par2,".so.1.0");
    ierr  = PetscTestFile(par2,'x',&foundlibrary);CHKERRQ(ierr);
    if (!foundlibrary) {
      PetscErrorPrintf("Library name %s\n",par2);
      SETERRQ(1,1,"Unable to locate dynamic library");
    }
  }

  *handle = dlopen(par2,1);    
  if (!*handle) {
    PetscErrorPrintf("Library name %s\n",libname);
    SETERRQ(1,1,"Unable to locate dynamic library");
  }
  PetscFree(par2);
  PetscFunctionReturn(0);
}

/*
     DLSym - Load a symbol from the dynamic link libraries.

  Input Parameter:
.  insymbol - name of symbol

  Output Parameter:
.  value 

  Notes: Symbol can be of the form

        [/path/libname[.so.1.0]:]functionname[()] where items in [] denote optional 

*/
#undef __FUNC__  
#define __FUNC__ "DLSym"
int DLSym(DLLibraryList list,char *insymbol, void **value)
{
  char          *par1,*symbol;
  int           ierr,len;

  PetscFunctionBegin;
  *value = 0;

  /* make copy of symbol so we can edit it in place */
  len    = PetscStrlen(insymbol);
  symbol = (char *) PetscMalloc((len+1)*sizeof(char));CHKPTRQ(symbol);
  ierr   = PetscStrcpy(symbol,insymbol);CHKERRQ(ierr);

  /* 
      If symbol contains () then replace with a NULL, to support functionname() 
  */
  par1 = PetscStrchr(symbol,'(');
  if (par1) *par1 = 0;

  /* 
     check if library path is given in function name 
  */
  par1 = PetscStrrchr(symbol,':');
  if (par1 != symbol) {
    void *handle;

    par1[-1] = 0;
    ierr     = DLOpen(symbol,&handle);CHKERRQ(ierr);
    *value   = dlsym(handle,par1);
    if (!*value) {
      PetscErrorPrintf("Library path and function name %s\n",insymbol);
      SETERRQ(1,1,"Unable to locate function in dynamic library");
    }
    PLogInfo(0,"DLSym:Loading function %s from dynamic library\n",par1);
  /* 
     look for symbol in predefined path of libraries 
  */
  } else {
    while (list) {
      *value =  dlsym(list->handle,symbol);
      if (*value) {
        PLogInfo(0,"DLSym:Loading function %s from dynamic library\n",symbol);
        break;
      }
      list = list->next;
    }
  }

  PetscFree(symbol);
  PetscFunctionReturn(0);
}

/*
     DLAppend - Appends another dynamic link library to the seach list, to the end
                of the search path.

     Notes: if library is already in path will not add it.
*/
#undef __FUNC__  
#define __FUNC__ "DLAppend"
int DLAppend(DLLibraryList *outlist,char *libname)
{
  DLLibraryList list,next;
  void*         handle;
  int           ierr;

  PetscFunctionBegin;
  ierr = DLOpen(libname,&handle);CHKERRQ(ierr);

  list = (DLLibraryList) PetscMalloc(sizeof(struct _DLLibraryList));CHKPTRQ(list);
  list->next   = 0;
  list->handle = handle;

  if (!*outlist) {
    *outlist = list;
  } else {
    next = *outlist;
    if (next->handle == handle) {
      PetscFree(list);
      PetscFunctionReturn(0); /* it is already listed */
    }
    while (next->next) {
      next = next->next;
      if (next->handle == handle) {
        PetscFree(list);
        PetscFunctionReturn(0); /* it is already listed */
      }
    }
    next->next = list;
  }
  PLogInfo(0,"DLAppend:Appending %s to dynamic library search path\n",libname);
  PetscFunctionReturn(0);
}

/*
     DLPrepend - Add another dynamic library to search for symbols to the beginning of
                 the search path.

     Notes: If library is already in path will remove old reference.

*/
#undef __FUNC__  
#define __FUNC__ "DLPrepend"
int DLPrepend(DLLibraryList *outlist,char *libname)
{
  DLLibraryList list,next,prev;
  void*         handle;
  int           ierr;

  PetscFunctionBegin;
  ierr = DLOpen(libname,&handle);CHKERRQ(ierr);

  PLogInfo(0,"DLPrepend:Prepending %s to dynamic library search path\n",libname);

  list = (DLLibraryList) PetscMalloc(sizeof(struct _DLLibraryList));CHKPTRQ(list);
  list->handle = handle;

  list->next        = *outlist;
  *outlist          = list;

  /* check if library was previously open, if so remove duplicate reference */
  next = list->next;
  prev = list;
  while (next) {
    if (next->handle == handle) {
      prev->next = next->next;
      PetscFree(next);
      PetscFunctionReturn(0);
    }
    prev = next;
    next = next->next;
  }
  PetscFunctionReturn(0);
}

/*
     DLClose - Destroys the search path of dynamic libraries and closes the libraries.

*/
#undef __FUNC__  
#define __FUNC__ "DLClose"
int DLClose(DLLibraryList next)
{
  DLLibraryList prev;

  PetscFunctionBegin;

  while (next) {
    prev = next;
    next = next->next;
    dlclose(prev->handle);
    PetscFree(prev);
  }
  PetscFunctionReturn(0);
}

#endif

