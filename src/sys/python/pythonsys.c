#define PETSC_DLL

#include "petscsys.h"       /*I "petscsys.h" I*/

/* ---------------------------------------------------------------- */

#if !defined(PETSC_PYTHON_EXE)
#define PETSC_PYTHON_EXE "python"
#endif

#undef __FUNCT__
#define __FUNCT__ "PetscPythonFindExecutable"
static PetscErrorCode PetscPythonFindExecutable(char pythonexe[PETSC_MAX_PATH_LEN])
{
  PetscTruth     flag;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  /* get the path for the Python interpreter executable */
  ierr = PetscStrncpy(pythonexe,PETSC_PYTHON_EXE,PETSC_MAX_PATH_LEN);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(PETSC_NULL,"-python",pythonexe,PETSC_MAX_PATH_LEN,&flag);CHKERRQ(ierr);
  if (!flag || pythonexe[0]==0) {
    ierr = PetscStrncpy(pythonexe,PETSC_PYTHON_EXE,PETSC_MAX_PATH_LEN);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscPythonFindLibrary"
static PetscErrorCode PetscPythonFindLibrary(char pythonexe[PETSC_MAX_PATH_LEN], 
					     char pythonlib[PETSC_MAX_PATH_LEN])
{
  const char cmdline[] = "-c 'import sys; print(sys.exec_prefix); print(sys.version[:3])'";
  char command[PETSC_MAX_PATH_LEN+1+sizeof(cmdline)+1];
  char prefix[PETSC_MAX_PATH_LEN],version[8],sep[2]={PETSC_DIR_SEPARATOR, 0},*eol;
  FILE* fp = NULL;
  char path[PETSC_MAX_PATH_LEN+1];
  PetscTruth found = PETSC_FALSE;
  PetscErrorCode ierr;
  PetscFunctionBegin;

#if defined(PETSC_PYTHON_LIB)
  ierr = PetscStrcpy(pythonlib,PETSC_PYTHON_LIB);CHKERRQ(ierr);
  PetscFunctionReturn(0);
#endif
  
  /* call Python to find out the name of the Python dynamic library */
  ierr = PetscStrncpy(command,pythonexe,PETSC_MAX_PATH_LEN);CHKERRQ(ierr);
  ierr = PetscStrcat(command," ");CHKERRQ(ierr);
  ierr = PetscStrcat(command,cmdline);CHKERRQ(ierr);
#if defined(PETSC_HAVE_POPEN)
  ierr = PetscPOpen(PETSC_COMM_SELF,PETSC_NULL,command,"r",&fp);CHKERRQ(ierr);
  if (!fgets(prefix,sizeof(prefix),fp))
    { SETERRQ1(PETSC_ERR_PLIB,"Python: bad output from executable: %s",pythonexe); }
  if (!fgets(version,sizeof(version),fp))
    { SETERRQ1(PETSC_ERR_PLIB,"Python: bad output from executable: %s",pythonexe); }
  ierr = PetscPClose(PETSC_COMM_SELF,fp);CHKERRQ(ierr);
#else
  SETERRQ(1,"Python: Aborted due to missing popen()");
#endif
  /* remove newlines */
  ierr = PetscStrchr(prefix,'\n',&eol);CHKERRQ(ierr);
  if (eol) eol[0] = 0;
  ierr = PetscStrchr(version,'\n',&eol);CHKERRQ(ierr);
  if (eol) eol[0] = 0;

  /* test for $prefix/lib64/libpythonX.X[.so]*/
  ierr = PetscStrcpy(pythonlib,prefix);CHKERRQ(ierr);
  ierr = PetscStrcat(pythonlib,sep);CHKERRQ(ierr);
  ierr = PetscStrcat(pythonlib,"lib64");CHKERRQ(ierr);
  ierr = PetscTestDirectory(pythonlib,'r',&found);CHKERRQ(ierr);
  if (found) {
    ierr = PetscStrcat(pythonlib,sep);CHKERRQ(ierr);
    ierr = PetscStrcat(pythonlib,"libpython");CHKERRQ(ierr);
    ierr = PetscStrcat(pythonlib,version);CHKERRQ(ierr);
    ierr = PetscDLLibraryRetrieve(PETSC_COMM_SELF,pythonlib,path,PETSC_MAX_PATH_LEN,&found);CHKERRQ(ierr);
    if (found) PetscFunctionReturn(0);
  }

  /* test for $prefix/lib/libpythonX.X[.so]*/
  ierr = PetscStrcpy(pythonlib,prefix);CHKERRQ(ierr);
  ierr = PetscStrcat(pythonlib,sep);CHKERRQ(ierr);
  ierr = PetscStrcat(pythonlib,"lib");CHKERRQ(ierr);
  ierr = PetscTestDirectory(pythonlib,'r',&found);CHKERRQ(ierr);
  if (found) {
    ierr = PetscStrcat(pythonlib,sep);CHKERRQ(ierr);
    ierr = PetscStrcat(pythonlib,"libpython");CHKERRQ(ierr);
    ierr = PetscStrcat(pythonlib,version);CHKERRQ(ierr);
    ierr = PetscDLLibraryRetrieve(PETSC_COMM_SELF,pythonlib,path,PETSC_MAX_PATH_LEN,&found);CHKERRQ(ierr);
    if (found) PetscFunctionReturn(0);
  }

  /* nothing good found */
  ierr = PetscMemzero(pythonlib,PETSC_MAX_PATH_LEN);CHKERRQ(ierr);
  ierr = PetscInfo(0,"Python dynamic library not found\n");CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */

typedef struct _Py_object_t PyObject; /* fake definition */

static int   	 (*Py_IsInitialized)(void);
static void  	 (*Py_InitializeEx)(int);
static void  	 (*Py_Finalize)(void);

static void      (*PySys_SetArgv)(int, char **);
static PyObject* (*PyImport_ImportModule)(const char *);

static void      (*Py_IncRef)(PyObject *);
static void      (*Py_DecRef)(PyObject *);

static void      (*PyErr_Clear)(void);
static PyObject* (*PyErr_Occurred)(void);


#define PetscDLPyLibOpen(libname) \
  PetscDLLibraryAppend(PETSC_COMM_SELF,&DLLibrariesLoaded,libname)
#define PetscDLPyLibSym(symbol, value) \
  PetscDLLibrarySym(PETSC_COMM_SELF,&DLLibrariesLoaded,PETSC_NULL,symbol,(void**)value)
#define PetscDLPyLibClose(comm) \
  do { } while(0)

#undef __FUNCT__
#define __FUNCT__ "PetscPythonLoadLibrary"
static PetscErrorCode PetscPythonLoadLibrary(const char pythonlib[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  /* open the Python dynamic library */
  ierr = PetscDLPyLibOpen(pythonlib);CHKERRQ(ierr);
  ierr = PetscInfo1(0,"Python: loaded dynamic library %s\n", pythonlib);CHKERRQ(ierr);
  /* look required symbols from the Python C-API */
  ierr = PetscDLPyLibSym("Py_IsInitialized"      , &Py_IsInitialized      );CHKERRQ(ierr);
  ierr = PetscDLPyLibSym("Py_InitializeEx"       , &Py_InitializeEx       );CHKERRQ(ierr);
  ierr = PetscDLPyLibSym("Py_Finalize"           , &Py_Finalize           );CHKERRQ(ierr);
  ierr = PetscDLPyLibSym("PySys_SetArgv"         , &PySys_SetArgv         );CHKERRQ(ierr);
  ierr = PetscDLPyLibSym("PyImport_ImportModule" , &PyImport_ImportModule );CHKERRQ(ierr);
  ierr = PetscDLPyLibSym("Py_IncRef"             , &Py_IncRef             );CHKERRQ(ierr);
  ierr = PetscDLPyLibSym("Py_DecRef"             , &Py_DecRef             );CHKERRQ(ierr);
  ierr = PetscDLPyLibSym("PyErr_Clear"           , &PyErr_Clear           );CHKERRQ(ierr);
  ierr = PetscDLPyLibSym("PyErr_Occurred"        , &PyErr_Occurred        );CHKERRQ(ierr);
  /* XXX TODO: check that ALL symbols were there !!! */
  if (!Py_IsInitialized) {SETERRQ(1,"Python: failed to load symbols from dynamic library");}
  if (!Py_InitializeEx)  {SETERRQ(1,"Python: failed to load symbols from dynamic library");}
  if (!Py_Finalize)      {SETERRQ(1,"Python: failed to load symbols from dynamic library");}
  ierr = PetscInfo(0,"Python: all required symbols loaded from Python dynamic library\n");CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */

static char       PetscPythonExe[PETSC_MAX_PATH_LEN] = { 0 };
static char       PetscPythonLib[PETSC_MAX_PATH_LEN] = { 0 };
static PetscTruth PetscBeganPython = PETSC_FALSE;

#undef __FUNCT__
#define __FUNCT__ "PetscPythonFinalize"
/*@C
  PetscPythonFinalize - Finalize Python.
  
  Level: intermediate

.keywords: Python
@*/
PetscErrorCode PETSC_DLLEXPORT PetscPythonFinalize(void)
{
  PetscFunctionBegin;
  if (PetscBeganPython) { if (Py_IsInitialized()) Py_Finalize(); }
  PetscBeganPython = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscPythonInitialize"
/*@C
  PetscPythonInitialize - Initialize Python and import petsc4py.

   Input Parameter:
+  pyexe - path to the Python interpreter executable, or PETSC_NULL.
-  pylib - full path to the Python dynamic library, or PETSC_NULL.

  Level: intermediate

.keywords: Python
  
@*/
PetscErrorCode PETSC_DLLEXPORT PetscPythonInitialize(const char pyexe[],const char pylib[])
{
  int               argc       = 0;
  char              **argv     = 0;
  PyObject          *module    = 0;
  static PetscTruth registered = PETSC_FALSE;
  PetscErrorCode    ierr;
  PetscFunctionBegin;
  if (PetscBeganPython) PetscFunctionReturn(0);
  /* Python executable */
  if (pyexe && pyexe[0] != 0) {
    ierr = PetscStrncpy(PetscPythonExe,pyexe,sizeof(PetscPythonExe));CHKERRQ(ierr);
  } else {
    ierr = PetscPythonFindExecutable(PetscPythonExe);CHKERRQ(ierr);
  }
  /* Python dynamic library */
  if (pylib && pylib[0] != 0) {
    ierr = PetscStrncpy(PetscPythonLib,pylib,sizeof(PetscPythonLib));CHKERRQ(ierr);
  } else {
    ierr = PetscPythonFindLibrary(PetscPythonExe,PetscPythonLib);CHKERRQ(ierr);
  }
  /* dynamically load Python library */
  ierr = PetscPythonLoadLibrary(PetscPythonLib);CHKERRQ(ierr);
  /* initialize Python */
  PetscBeganPython = PETSC_FALSE;
  if (!Py_IsInitialized()) {
    /* call below does not install signal handlers */
    Py_InitializeEx(0);
    /* call below required to build 'sys.argv' list */
    ierr = PetscGetArgs(&argc,&argv);CHKERRQ(ierr);
    if (argc && argv && argv[0]) PySys_SetArgv(argc,argv);
    /* register finalizer */
    if (!registered) {
      ierr = PetscRegisterFinalize(PetscPythonFinalize);CHKERRQ(ierr);
      registered = PETSC_TRUE;
    }
    PetscBeganPython = PETSC_TRUE;
  }
  /* import 'petsc4py.PETSc' module */
  module = PyImport_ImportModule("petsc4py.PETSc");
  if (module) {
    ierr = PetscInfo(0,"Python: successfully imported  module 'petsc4py.PETSc'\n");CHKERRQ(ierr);
    Py_DecRef(module); module = 0;
  } else {
    SETERRQ(PETSC_ERR_PLIB,"Python: could not import module 'petsc4py.PETSc', perhaps your PYTHONPATH does not contain it\n"); 
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
