#include <petsc/private/petscimpl.h>       /*I "petscsys.h" I*/

/* ---------------------------------------------------------------- */

#if !defined(PETSC_PYTHON_EXE)
#define PETSC_PYTHON_EXE "python"
#endif

static PetscErrorCode PetscPythonFindExecutable(char pythonexe[PETSC_MAX_PATH_LEN])
{
  PetscBool      flag;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* get the path for the Python interpreter executable */
  ierr = PetscStrncpy(pythonexe,PETSC_PYTHON_EXE,PETSC_MAX_PATH_LEN);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-python",pythonexe,PETSC_MAX_PATH_LEN,&flag);CHKERRQ(ierr);
  if (!flag || pythonexe[0]==0) {
    ierr = PetscStrncpy(pythonexe,PETSC_PYTHON_EXE,PETSC_MAX_PATH_LEN);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
    Python does not appear to have a universal way to indicate the location of Python dynamic library so try several possibilities
*/
static PetscErrorCode PetscPythonFindLibraryName(const char pythonexe[PETSC_MAX_PATH_LEN],const char attempt[PETSC_MAX_PATH_LEN],char pythonlib[PETSC_MAX_PATH_LEN],PetscBool *found)
{
  char           command[2*PETSC_MAX_PATH_LEN];
  FILE           *fp = NULL;
  char           *eol;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* call Python to find out the name of the Python dynamic library */
  ierr = PetscStrncpy(command,pythonexe,PETSC_MAX_PATH_LEN);CHKERRQ(ierr);
  ierr = PetscStrcat(command," ");CHKERRQ(ierr);
  ierr = PetscStrcat(command,attempt);CHKERRQ(ierr);
#if defined(PETSC_HAVE_POPEN)
  ierr = PetscPOpen(PETSC_COMM_SELF,NULL,command,"r",&fp);CHKERRQ(ierr);
  if (!fgets(pythonlib,PETSC_MAX_PATH_LEN,fp)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Python: bad output from executable: %s",pythonexe);
  ierr = PetscPClose(PETSC_COMM_SELF,fp);CHKERRQ(ierr);
#else
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Python: Aborted due to missing popen()");
#endif
  /* remove newlines */
  ierr = PetscStrchr(pythonlib,'\n',&eol);CHKERRQ(ierr);
  if (eol) eol[0] = 0;
  ierr = PetscTestFile(pythonlib,'r',found);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


static PetscErrorCode PetscPythonFindLibrary(const char pythonexe[PETSC_MAX_PATH_LEN],char pythonlib[PETSC_MAX_PATH_LEN])
{
  const char     cmdline1[] = "-c 'import os;from distutils import sysconfig; print(os.path.join(sysconfig.get_config_var(\"LIBDIR\"),sysconfig.get_config_var(\"LDLIBRARY\")))'";
  const char     cmdline2[] = "-c 'import os;from distutils import sysconfig; import sys;print(os.path.join(sysconfig.get_config_var(\"LIBDIR\"),\"libpython\"+sys.version[:3]+\".dylib\"))'";
  const char     cmdline3[] = "-c 'import os;from distutils import sysconfig; print(os.path.join(sysconfig.get_config_var(\"LIBPL\"),sysconfig.get_config_var(\"LDLIBRARY\")))'";
  const char     cmdline4[] = "-c 'from distutils import sysconfig; print(sysconfig.get_config_var(\"LIBPYTHON\"))'";

  PetscBool      found = PETSC_FALSE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
#if defined(PETSC_PYTHON_LIB)
  ierr = PetscStrcpy(pythonlib,PETSC_PYTHON_LIB);CHKERRQ(ierr);
  PetscFunctionReturn(0);
#endif

  ierr = PetscPythonFindLibraryName(pythonexe,cmdline1,pythonlib,&found);CHKERRQ(ierr);
  if (!found) {
    ierr = PetscPythonFindLibraryName(pythonexe,cmdline2,pythonlib,&found);CHKERRQ(ierr);
  }
  if (!found) {
    ierr = PetscPythonFindLibraryName(pythonexe,cmdline3,pythonlib,&found);CHKERRQ(ierr);
  }
  if (!found) {
    ierr = PetscPythonFindLibraryName(pythonexe,cmdline4,pythonlib,&found);CHKERRQ(ierr);
  }
  ierr = PetscInfo2(0,"Python library  %s found %d\n",pythonlib,found);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */

typedef struct _Py_object_t PyObject; /* fake definition */

static PyObject* Py_None = 0;

static const char* (*Py_GetVersion)(void);

static int       (*Py_IsInitialized)(void);
static void      (*Py_InitializeEx)(int);
static void      (*Py_Finalize)(void);

static void      (*PySys_SetArgv)(int,void*);
static PyObject* (*PySys_GetObject)(const char*);
static PyObject* (*PyObject_CallMethod)(PyObject*,const char*, const char*, ...);
static PyObject* (*PyImport_ImportModule)(const char*);

static void      (*Py_IncRef)(PyObject*);
static void      (*Py_DecRef)(PyObject*);

static void      (*PyErr_Clear)(void);
static PyObject* (*PyErr_Occurred)(void);
static void      (*PyErr_Fetch)(PyObject**,PyObject**,PyObject**);
static void      (*PyErr_NormalizeException)(PyObject**,PyObject**, PyObject**);
static void      (*PyErr_Display)(PyObject*,PyObject*,PyObject*);
static void      (*PyErr_Restore)(PyObject*,PyObject*,PyObject*);


#define PetscDLPyLibOpen(libname) \
  PetscDLLibraryAppend(PETSC_COMM_SELF,&PetscDLLibrariesLoaded,libname)
#define PetscDLPyLibSym(symbol, value) \
  PetscDLLibrarySym(PETSC_COMM_SELF,&PetscDLLibrariesLoaded,NULL,symbol,(void**)value)
#define PetscDLPyLibClose(comm) \
  do { } while (0)

static PetscErrorCode PetscPythonLoadLibrary(const char pythonlib[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* open the Python dynamic library */
  ierr = PetscDLPyLibOpen(pythonlib);CHKERRQ(ierr);
  ierr = PetscInfo1(0,"Python: loaded dynamic library %s\n", pythonlib);CHKERRQ(ierr);
  /* look required symbols from the Python C-API */
  ierr = PetscDLPyLibSym("_Py_NoneStruct"        , &Py_None               );CHKERRQ(ierr);
  ierr = PetscDLPyLibSym("Py_GetVersion"         , &Py_GetVersion         );CHKERRQ(ierr);
  ierr = PetscDLPyLibSym("Py_IsInitialized"      , &Py_IsInitialized      );CHKERRQ(ierr);
  ierr = PetscDLPyLibSym("Py_InitializeEx"       , &Py_InitializeEx       );CHKERRQ(ierr);
  ierr = PetscDLPyLibSym("Py_Finalize"           , &Py_Finalize           );CHKERRQ(ierr);
  ierr = PetscDLPyLibSym("PySys_GetObject"       , &PySys_GetObject       );CHKERRQ(ierr);
  ierr = PetscDLPyLibSym("PySys_SetArgv"         , &PySys_SetArgv         );CHKERRQ(ierr);
  ierr = PetscDLPyLibSym("PyObject_CallMethod"   , &PyObject_CallMethod   );CHKERRQ(ierr);
  ierr = PetscDLPyLibSym("PyImport_ImportModule" , &PyImport_ImportModule );CHKERRQ(ierr);
  ierr = PetscDLPyLibSym("Py_IncRef"             , &Py_IncRef             );CHKERRQ(ierr);
  ierr = PetscDLPyLibSym("Py_DecRef"             , &Py_DecRef             );CHKERRQ(ierr);
  ierr = PetscDLPyLibSym("PyErr_Clear"           , &PyErr_Clear           );CHKERRQ(ierr);
  ierr = PetscDLPyLibSym("PyErr_Occurred"        , &PyErr_Occurred        );CHKERRQ(ierr);
  ierr = PetscDLPyLibSym("PyErr_Fetch"             , &PyErr_Fetch             );CHKERRQ(ierr);
  ierr = PetscDLPyLibSym("PyErr_NormalizeException", &PyErr_NormalizeException);CHKERRQ(ierr);
  ierr = PetscDLPyLibSym("PyErr_Display",            &PyErr_Display           );CHKERRQ(ierr);
  ierr = PetscDLPyLibSym("PyErr_Restore",            &PyErr_Restore           );CHKERRQ(ierr);
  /* XXX TODO: check that ALL symbols were there !!! */
  if (!Py_None)          SETERRQ1(PETSC_COMM_SELF,1,"Python: failed to load symbols from Python dynamic library %s",pythonlib);
  if (!Py_GetVersion)    SETERRQ1(PETSC_COMM_SELF,1,"Python: failed to load symbols from Python dynamic library %s",pythonlib);
  if (!Py_IsInitialized) SETERRQ1(PETSC_COMM_SELF,1,"Python: failed to load symbols from Python dynamic library %s",pythonlib);
  if (!Py_InitializeEx)  SETERRQ1(PETSC_COMM_SELF,1,"Python: failed to load symbols from Python dynamic library %s",pythonlib);
  if (!Py_Finalize)      SETERRQ1(PETSC_COMM_SELF,1,"Python: failed to load symbols from Python dynamic library %s",pythonlib);
  ierr = PetscInfo1(0,"Python: all required symbols loaded from Python dynamic library %s\n",pythonlib);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */

static char      PetscPythonExe[PETSC_MAX_PATH_LEN] = { 0 };
static char      PetscPythonLib[PETSC_MAX_PATH_LEN] = { 0 };
static PetscBool PetscBeganPython = PETSC_FALSE;

/*@C
  PetscPythonFinalize - Finalize Python.

  Level: intermediate

@*/
PetscErrorCode  PetscPythonFinalize(void)
{
  PetscFunctionBegin;
  if (PetscBeganPython) { if (Py_IsInitialized()) Py_Finalize(); }
  PetscBeganPython = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
  PetscPythonInitialize - Initialize Python and import petsc4py.

   Input Parameter:
+  pyexe - path to the Python interpreter executable, or NULL.
-  pylib - full path to the Python dynamic library, or NULL.

  Level: intermediate

@*/
PetscErrorCode  PetscPythonInitialize(const char pyexe[],const char pylib[])
{
  PyObject       *module = 0;
  PetscErrorCode ierr;

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
    static PetscBool registered = PETSC_FALSE;
    const char       *py_version;
    PyObject         *sys_path;
    char             path[PETSC_MAX_PATH_LEN] = { 0 };

    /* initialize Python */
    Py_InitializeEx(0); /* 0: do not install signal handlers */
    /*  build 'sys.argv' list */
    py_version = Py_GetVersion();
    if (py_version[0] == '2') {
      int argc = 0; char *argv[1] = {NULL};
      PySys_SetArgv(argc,argv);
    }
    if (py_version[0] == '3') {
      int argc = 0; wchar_t *argv[1] = {NULL};
      PySys_SetArgv(argc,argv);
    }
    /* add PETSC_LIB_DIR in front of 'sys.path' */
    sys_path = PySys_GetObject("path");
    if (sys_path) {
      ierr = PetscStrreplace(PETSC_COMM_SELF,"${PETSC_LIB_DIR}",path,sizeof(path));CHKERRQ(ierr);
      Py_DecRef(PyObject_CallMethod(sys_path,"insert","is",(int)0,(char*)path));
#if defined(PETSC_PETSC4PY_INSTALL_PATH)
      {
        char *rpath;
        ierr = PetscStrallocpy(PETSC_PETSC4PY_INSTALL_PATH,&rpath);CHKERRQ(ierr);
        Py_DecRef(PyObject_CallMethod(sys_path,"insert","is",(int)0,rpath));
        ierr = PetscFree(rpath);CHKERRQ(ierr);
      }
#endif
    }
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
    PetscPythonPrintError();
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Python: could not import module 'petsc4py.PETSc', perhaps your PYTHONPATH does not contain it\n");
  }
  PetscFunctionReturn(0);
}

/*@C
  PetscPythonPrintError - Print Python errors.

  Level: developer

@*/
PetscErrorCode  PetscPythonPrintError(void)
{
  PyObject *exc=0, *val=0, *tb=0;

  PetscFunctionBegin;
  if (!PetscBeganPython) PetscFunctionReturn(0);
  if (!PyErr_Occurred()) PetscFunctionReturn(0);
  PyErr_Fetch(&exc,&val,&tb);
  PyErr_NormalizeException(&exc,&val,&tb);
  PyErr_Display(exc ? exc : Py_None,
                val ? val : Py_None,
                tb  ? tb  : Py_None);
  PyErr_Restore(exc,val,tb);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */

PETSC_EXTERN PetscErrorCode (*PetscPythonMonitorSet_C)(PetscObject,const char[]);
PetscErrorCode (*PetscPythonMonitorSet_C)(PetscObject,const char[]) = NULL;

/*@C
  PetscPythonMonitorSet - Set Python monitor

  Level: developer

@*/
PetscErrorCode PetscPythonMonitorSet(PetscObject obj, const char url[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeader(obj,1);
  PetscValidCharPointer(url,2);
  if (!PetscPythonMonitorSet_C) {
    ierr = PetscPythonInitialize(NULL,NULL);CHKERRQ(ierr);
    if (!PetscPythonMonitorSet_C) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Couldn't initialize Python support for monitors");
  }
  ierr = PetscPythonMonitorSet_C(obj,url);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
