
#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: dl.c,v 1.11 1998/02/02 21:57:30 bsmith Exp bsmith $";
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
      Code to maintain a list of opened dynamic libraries and load symbols
*/
#if defined(USE_DYNAMIC_LIBRARIES)
#include <dlfcn.h>

struct _DLLibraryList {
  DLLibraryList next;
  void          *handle;
  char          libname[1024];
};


#undef __FUNC__  
#define __FUNC__ "DLLibraryObtain"
/*
   DLLibraryObtain - Obtains a library from a URL and copies into local
        disk space.

  Input Parameter:
.   libname - name of library, including entire URL

  Output Paramter:
.   llibname - name of local copy of library

*/
int DLLibraryObtain(char *libname,char *llibname)
{
  char *par4,buf[1024];
  FILE *fp;
  int  i;

  PetscFunctionBegin;

  if (PetscStrncmp(libname,"ftp://",6) && PetscStrncmp(libname,"http://",7)) {
    SETERRQ(1,1,"Only support for ftp/http DLL retrieval with python");
  }
    
  /* Construct the Python script run command */
  par4 = (char *) PetscMalloc(1024*sizeof(char));CHKPTRQ(par4);
  PetscStrcpy(par4,"python1.5 ");
  PetscStrcat(par4,PETSC_DIR);
  PetscStrcat(par4,"/bin/urlget.py ");
  PetscStrcat(par4,libname);

  if ((fp = popen(par4,"r")) == NULL) {
    SETERRQ(1,1,"Cannot Execute python1.5 on $(PETSC_DIR)/bin/urlget.py\n\
      Check if python1.5 is in your path");
  }
  if (fgets(buf,1024,fp) == NULL) {
    SETERRQ(1,1,"No output from $(PETSC_DIR)/bin/urlget.py");
  }
  /* Check for \n and make it NULL */
  for ( i=0; i<1024; i++ ) {
    if ( buf[i] == '\n') {
      buf[i] = NULL;
      break;
    }
  }
  if (!PetscStrncmp(buf,"Error",5) ||!PetscStrncmp(buf,"Traceback",9) ) { SETERRQ(1,1,buf); }
  PetscStrcpy(llibname,buf);
  PetscFree(par4);

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "DLLibraryOpen"
/*
     DLLibraryOpen - Opens a dynamic link library

   Input Parameter:
    libname - name of the library, can be relative or absolute

   Output Paramter:
    handle - returned from DLLibraryOpen

   Notes:
     [[<http,ftp>://hostname]/directoryname/]filename[.so.1.0]

     $PETSC_ARCH and $BOPT occuring in directoryname and filename 
       will be replaced with appropriate values
*/
int DLLibraryOpen(char *libname,void **handle)
{
  char       *par2,ierr,len,*par3,arch[10];
  PetscTruth foundlibrary;
  int        flg;
  int        (*func)(char*);

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
    ierr = DLLibraryObtain(par2,par3);CHKERRQ(ierr);
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

  /* run the function DLRegister() if it is in the library */
  ((void *) func)   = dlsym(*handle,"DLLibraryRegister");
  if (func) {
    ierr = (*func)(libname);CHKERRQ(ierr);
    PLogInfo(0,"DLLibraryOpen:Loading registered routines from %s\n",libname);
  }
  if (PLogPrintInfo) {
    char *(*sfunc)(char *,char*),*mess;

    ((void *) sfunc)   = dlsym(*handle,"DLLibraryInfo");
    if (sfunc) {
      mess = (*sfunc)(libname,"Contents");
      if (mess) {
        PLogInfo(0,"Library %s contents \n %s",libname,mess);
      }
      mess = (*sfunc)(libname,"Authors");
      if (mess) {
        PLogInfo(0,"Library %s authors \n %s",libname,mess);
      }
      mess = (*sfunc)(libname,"Version");
      if (mess) {
        PLogInfo(0,"Library %s version \n %s",libname,mess);
      }
    }
  }

  PetscFree(par2);
  PetscFunctionReturn(0);
}

/*
     DLLibrarySym - Load a symbol from the dynamic link libraries.

  Input Parameter:
.  path     - optional complete library name
.  insymbol - name of symbol

  Output Parameter:
.  value 

  Notes: Symbol can be of the form

        [/path/libname[.so.1.0]:]functionname[()] where items in [] denote optional 

*/
#undef __FUNC__  
#define __FUNC__ "DLLibrarySym"
int DLLibrarySym(DLLibraryList *inlist,char *path,char *insymbol, void **value)
{
  char          *par1,*symbol;
  int           ierr,len;
  DLLibraryList nlist,prev,list = *inlist;

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
       Function name does include library 
       -------------------------------------
  */
  if (path) {
    void *handle;
    
    /*   
        Look if library is already opened and in path
    */
    nlist = list;
    prev  = 0;
    while (nlist) {
      if (!PetscStrcmp(nlist->libname,path)) {
        handle = nlist->handle;
        goto done;
      }
      prev = nlist;
      nlist = nlist->next;
    }
    ierr = DLLibraryOpen(path,&handle);CHKERRQ(ierr);

    nlist = (DLLibraryList) PetscMalloc(sizeof(struct _DLLibraryList));CHKPTRQ(list);
    nlist->next   = 0;
    nlist->handle = handle;
    PetscStrcpy(nlist->libname,path);

    if (prev) {
      prev->next = nlist;
    } else {
      *inlist    = list;
    }
    PLogInfo(0,"DLLibraryAppend:Appending %s to dynamic library search path\n",symbol);

    done:; 
    *value   = dlsym(handle,symbol);
    if (!*value) {
      PetscErrorPrintf("Library path %s and function name %s\n",path,insymbol);
      SETERRQ(1,1,"Unable to locate function in dynamic library");
    }
    PLogInfo(0,"DLLibrarySym:Loading function %s from dynamic library %s\n",insymbol,path);

  /*
       Function name does not include library so search path
       -----------------------------------------------------
  */
  } else {
    while (list) {
      *value =  dlsym(list->handle,symbol);
      if (*value) {
        PLogInfo(0,"DLLibrarySym:Loading function %s from dynamic library %s\n",symbol,list->libname);
        break;
      }
      list = list->next;
    }
  }

  PetscFree(symbol);
  PetscFunctionReturn(0);
}

/*
     DLLibraryAppend - Appends another dynamic link library to the seach list, to the end
                of the search path.

     Notes: if library is already in path will not add it.
*/
#undef __FUNC__  
#define __FUNC__ "DLLibraryAppend"
int DLLibraryAppend(DLLibraryList *outlist,char *libname)
{
  DLLibraryList list,prev;
  void*         handle;
  int           ierr;

  PetscFunctionBegin;

  /* see if library was already open then we are done */
  list = *outlist;
  while (list) {
    if (!PetscStrcmp(list->libname,libname)) {
      PetscFunctionReturn(0);
    }
    prev = list;
    list = list->next;
  }

  ierr = DLLibraryOpen(libname,&handle);CHKERRQ(ierr);

  list = (DLLibraryList) PetscMalloc(sizeof(struct _DLLibraryList));CHKPTRQ(list);
  list->next   = 0;
  list->handle = handle;
  PetscStrcpy(list->libname,libname);

  if (!*outlist) {
    *outlist   = list;
  } else {
    prev->next = list;
  }
  PLogInfo(0,"DLLibraryAppend:Appending %s to dynamic library search path\n",libname);
  PetscFunctionReturn(0);
}

/*
     DLLibraryPrepend - Add another dynamic library to search for symbols to the beginning of
                 the search path.

     Notes: If library is already in path will remove old reference.

*/
#undef __FUNC__  
#define __FUNC__ "DLLibraryPrepend"
int DLLibraryPrepend(DLLibraryList *outlist,char *libname)
{
  DLLibraryList list,prev;
  void*         handle;
  int           ierr;

  PetscFunctionBegin;
 
  /* see if library was already open and move it to the front */
  list = *outlist;
  prev = 0;
  while (list) {
    if (!PetscStrcmp(list->libname,libname)) {
      if (prev) prev->next = list->next;
      list->next = *outlist;
      *outlist   = list;
      PetscFunctionReturn(0);
    }
    prev = list;
    list = list->next;
  }

  /* open the library and add to front of list */
  ierr = DLLibraryOpen(libname,&handle);CHKERRQ(ierr);

  PLogInfo(0,"DLLibraryPrepend:Prepending %s to dynamic library search path\n",libname);

  list         = (DLLibraryList) PetscMalloc(sizeof(struct _DLLibraryList));CHKPTRQ(list);
  list->handle = handle;
  list->next   = *outlist;
  PetscStrcpy(list->libname,libname);
  *outlist     = list;

  PetscFunctionReturn(0);
}

/*
     DLLibraryClose - Destroys the search path of dynamic libraries and closes the libraries.

*/
#undef __FUNC__  
#define __FUNC__ "DLLibraryClose"
int DLLibraryClose(DLLibraryList next)
{
  DLLibraryList prev;

  PetscFunctionBegin;

  while (next) {
    prev = next;
    next = next->next;
    DLLibraryClose(prev->handle);
    PetscFree(prev);
  }
  PetscFunctionReturn(0);
}

#endif


