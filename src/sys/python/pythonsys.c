#include <petsc/private/petscimpl.h> /*I "petscsys.h" I*/

/* ---------------------------------------------------------------- */

#if !defined(PETSC_PYTHON_EXE)
  #define PETSC_PYTHON_EXE "python"
#endif

static PetscErrorCode PetscPythonFindExecutable(char pythonexe[], size_t len)
{
  PetscBool flag;

  PetscFunctionBegin;
  /* get the path for the Python interpreter executable */
  PetscCall(PetscStrncpy(pythonexe, PETSC_PYTHON_EXE, len));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-python", pythonexe, len, &flag));
  if (!flag || pythonexe[0] == 0) PetscCall(PetscStrncpy(pythonexe, PETSC_PYTHON_EXE, len));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
    Python does not appear to have a universal way to indicate the location of Python dynamic library so try several possibilities
*/
static PetscErrorCode PetscPythonFindLibraryName(const char pythonexe[], const char attempt[], char pythonlib[], size_t pl, PetscBool *found)
{
  char  command[2 * PETSC_MAX_PATH_LEN];
  FILE *fp  = NULL;
  char *eol = NULL;

  PetscFunctionBegin;
  /* call Python to find out the name of the Python dynamic library */
  PetscCall(PetscStrncpy(command, pythonexe, sizeof(command)));
  PetscCall(PetscStrlcat(command, " ", sizeof(command)));
  PetscCall(PetscStrlcat(command, attempt, sizeof(command)));
#if defined(PETSC_HAVE_POPEN)
  PetscCall(PetscPOpen(PETSC_COMM_SELF, NULL, command, "r", &fp));
  PetscCheck(fgets(pythonlib, pl, fp), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Python: bad output from executable: %s\nRunning: %s", pythonexe, command);
  PetscCall(PetscPClose(PETSC_COMM_SELF, fp));
#else
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "Python: Aborted due to missing popen()");
#endif
  /* remove newlines */
  PetscCall(PetscStrchr(pythonlib, '\n', &eol));
  if (eol) eol[0] = 0;
  PetscCall(PetscTestFile(pythonlib, 'r', found));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscPythonFindLibrary(const char pythonexe[], char pythonlib[], size_t pl)
{
  const char cmdline1[] = "-c 'import os, sysconfig; print(os.path.join(sysconfig.get_config_var(\"LIBDIR\"),sysconfig.get_config_var(\"LDLIBRARY\")))'";
  const char cmdline2[] = "-c 'import os, sysconfig; print(os.path.join(sysconfig.get_path(\"stdlib\"),os.path.pardir,\"libpython\"+sysconfig.get_python_version()+\".dylib\"))'";
  const char cmdline3[] = "-c 'import os, sysconfig; print(os.path.join(sysconfig.get_config_var(\"LIBPL\"),sysconfig.get_config_var(\"LDLIBRARY\")))'";
  const char cmdline4[] = "-c 'import sysconfig; print(sysconfig.get_config_var(\"LIBPYTHON\"))'";
  const char cmdline5[] = "-c 'import os, sysconfig; import sys;print(os.path.join(sysconfig.get_config_var(\"LIBDIR\"),\"libpython\"+sys.version[:3]+\".so\"))'";

  PetscBool found = PETSC_FALSE;

  PetscFunctionBegin;
#if defined(PETSC_PYTHON_LIB)
  PetscCall(PetscStrncpy(pythonlib, PETSC_PYTHON_LIB, pl));
  PetscFunctionReturn(PETSC_SUCCESS);
#endif

  PetscCall(PetscPythonFindLibraryName(pythonexe, cmdline1, pythonlib, pl, &found));
  if (!found) PetscCall(PetscPythonFindLibraryName(pythonexe, cmdline2, pythonlib, pl, &found));
  if (!found) PetscCall(PetscPythonFindLibraryName(pythonexe, cmdline3, pythonlib, pl, &found));
  if (!found) PetscCall(PetscPythonFindLibraryName(pythonexe, cmdline4, pythonlib, pl, &found));
  if (!found) PetscCall(PetscPythonFindLibraryName(pythonexe, cmdline5, pythonlib, pl, &found));
  PetscCall(PetscInfo(NULL, "Python library  %s found %d\n", pythonlib, found));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ---------------------------------------------------------------- */

typedef struct _Py_object_t PyObject; /* fake definition */

static PyObject *Py_None = NULL;

static const char *(*Py_GetVersion)(void);

static int (*Py_IsInitialized)(void);
static void (*Py_InitializeEx)(int);
static void (*Py_Finalize)(void);

static void (*PySys_SetArgv)(int, void *);
static PyObject *(*PySys_GetObject)(const char *);
static PyObject *(*PyObject_CallMethod)(PyObject *, const char *, const char *, ...);
static PyObject *(*PyImport_ImportModule)(const char *);

static void (*Py_IncRef)(PyObject *);
static void (*Py_DecRef)(PyObject *);

static void (*PyErr_Clear)(void);
static PyObject *(*PyErr_Occurred)(void);
static void (*PyErr_Fetch)(PyObject **, PyObject **, PyObject **);
static void (*PyErr_NormalizeException)(PyObject **, PyObject **, PyObject **);
static void (*PyErr_Display)(PyObject *, PyObject *, PyObject *);
static void (*PyErr_Restore)(PyObject *, PyObject *, PyObject *);

#define PetscDLPyLibOpen(libname)      PetscDLLibraryAppend(PETSC_COMM_SELF, &PetscDLLibrariesLoaded, libname)
#define PetscDLPyLibSym(symbol, value) PetscDLLibrarySym(PETSC_COMM_SELF, &PetscDLLibrariesLoaded, NULL, symbol, (void **)value)
#define PetscDLPyLibClose(comm) \
  do { \
  } while (0)

static PetscErrorCode PetscPythonLoadLibrary(const char pythonlib[])
{
  PetscFunctionBegin;
  /* open the Python dynamic library */
  PetscCall(PetscDLPyLibOpen(pythonlib));
  PetscCall(PetscInfo(NULL, "Python: loaded dynamic library %s\n", pythonlib));
  /* look required symbols from the Python C-API */
  PetscCall(PetscDLPyLibSym("_Py_NoneStruct", &Py_None));
  PetscCall(PetscDLPyLibSym("Py_GetVersion", &Py_GetVersion));
  PetscCall(PetscDLPyLibSym("Py_IsInitialized", &Py_IsInitialized));
  PetscCall(PetscDLPyLibSym("Py_InitializeEx", &Py_InitializeEx));
  PetscCall(PetscDLPyLibSym("Py_Finalize", &Py_Finalize));
  PetscCall(PetscDLPyLibSym("PySys_GetObject", &PySys_GetObject));
  PetscCall(PetscDLPyLibSym("PySys_SetArgv", &PySys_SetArgv));
  PetscCall(PetscDLPyLibSym("PyObject_CallMethod", &PyObject_CallMethod));
  PetscCall(PetscDLPyLibSym("PyImport_ImportModule", &PyImport_ImportModule));
  PetscCall(PetscDLPyLibSym("Py_IncRef", &Py_IncRef));
  PetscCall(PetscDLPyLibSym("Py_DecRef", &Py_DecRef));
  PetscCall(PetscDLPyLibSym("PyErr_Clear", &PyErr_Clear));
  PetscCall(PetscDLPyLibSym("PyErr_Occurred", &PyErr_Occurred));
  PetscCall(PetscDLPyLibSym("PyErr_Fetch", &PyErr_Fetch));
  PetscCall(PetscDLPyLibSym("PyErr_NormalizeException", &PyErr_NormalizeException));
  PetscCall(PetscDLPyLibSym("PyErr_Display", &PyErr_Display));
  PetscCall(PetscDLPyLibSym("PyErr_Restore", &PyErr_Restore));
  /* XXX TODO: check that ALL symbols were there !!! */
  PetscCheck(Py_None, PETSC_COMM_SELF, PETSC_ERR_LIB, "Python: failed to load symbols from Python dynamic library %s", pythonlib);
  PetscCheck(Py_GetVersion, PETSC_COMM_SELF, PETSC_ERR_LIB, "Python: failed to load symbols from Python dynamic library %s", pythonlib);
  PetscCheck(Py_IsInitialized, PETSC_COMM_SELF, PETSC_ERR_LIB, "Python: failed to load symbols from Python dynamic library %s", pythonlib);
  PetscCheck(Py_InitializeEx, PETSC_COMM_SELF, PETSC_ERR_LIB, "Python: failed to load symbols from Python dynamic library %s", pythonlib);
  PetscCheck(Py_Finalize, PETSC_COMM_SELF, PETSC_ERR_LIB, "Python: failed to load symbols from Python dynamic library %s", pythonlib);
  PetscCall(PetscInfo(NULL, "Python: all required symbols loaded from Python dynamic library %s\n", pythonlib));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ---------------------------------------------------------------- */

static char      PetscPythonExe[PETSC_MAX_PATH_LEN] = {0};
static char      PetscPythonLib[PETSC_MAX_PATH_LEN] = {0};
static PetscBool PetscBeganPython                   = PETSC_FALSE;

/*@C
  PetscPythonFinalize - Finalize PETSc for use with Python.

  Level: intermediate

.seealso: `PetscPythonInitialize()`, `PetscPythonPrintError()`
@*/
PetscErrorCode PetscPythonFinalize(void)
{
  PetscFunctionBegin;
  if (PetscBeganPython) {
    if (Py_IsInitialized()) Py_Finalize();
  }
  PetscBeganPython = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscPythonInitialize - Initialize Python for use with PETSc and import petsc4py.

   Input Parameters:
+  pyexe - path to the Python interpreter executable, or NULL.
-  pylib - full path to the Python dynamic library, or NULL.

  Level: intermediate

.seealso: `PetscPythonFinalize()`, `PetscPythonPrintError()`
@*/
PetscErrorCode PetscPythonInitialize(const char pyexe[], const char pylib[])
{
  PyObject *module = NULL;

  PetscFunctionBegin;
  if (PetscBeganPython) PetscFunctionReturn(PETSC_SUCCESS);
  /* Python executable */
  if (pyexe && pyexe[0] != 0) {
    PetscCall(PetscStrncpy(PetscPythonExe, pyexe, sizeof(PetscPythonExe)));
  } else {
    PetscCall(PetscPythonFindExecutable(PetscPythonExe, sizeof(PetscPythonExe)));
  }
  /* Python dynamic library */
  if (pylib && pylib[0] != 0) {
    PetscCall(PetscStrncpy(PetscPythonLib, pylib, sizeof(PetscPythonLib)));
  } else {
    PetscCall(PetscPythonFindLibrary(PetscPythonExe, PetscPythonLib, sizeof(PetscPythonLib)));
  }
  /* dynamically load Python library */
  PetscCall(PetscPythonLoadLibrary(PetscPythonLib));
  /* initialize Python */
  PetscBeganPython = PETSC_FALSE;
  if (!Py_IsInitialized()) {
    static PetscBool registered = PETSC_FALSE;
    const char      *py_version;
    PyObject        *sys_path;
    char             path[PETSC_MAX_PATH_LEN] = {0};

    /* initialize Python. Py_InitializeEx() prints an error and EXITS the program if it is not successful! */
    PetscCall(PetscInfo(NULL, "Calling Py_InitializeEx(0);\n"));
    Py_InitializeEx(0); /* 0: do not install signal handlers */
    PetscCall(PetscInfo(NULL, "Py_InitializeEx(0) called successfully;\n"));

    /*  build 'sys.argv' list */
    py_version = Py_GetVersion();
    if (py_version[0] == '2') {
      int   argc    = 0;
      char *argv[1] = {NULL};
      PySys_SetArgv(argc, argv);
    }
    if (py_version[0] == '3') {
      int      argc    = 0;
      wchar_t *argv[1] = {NULL};
      PySys_SetArgv(argc, argv);
    }
    /* add PETSC_LIB_DIR in front of 'sys.path' */
    sys_path = PySys_GetObject("path");
    if (sys_path) {
      PetscCall(PetscStrreplace(PETSC_COMM_SELF, "${PETSC_LIB_DIR}", path, sizeof(path)));
      Py_DecRef(PyObject_CallMethod(sys_path, "insert", "is", (int)0, (char *)path));
#if defined(PETSC_PETSC4PY_INSTALL_PATH)
      {
        char *rpath;
        PetscCall(PetscStrallocpy(PETSC_PETSC4PY_INSTALL_PATH, &rpath));
        Py_DecRef(PyObject_CallMethod(sys_path, "insert", "is", (int)0, rpath));
        PetscCall(PetscFree(rpath));
      }
#endif
    }
    /* register finalizer */
    if (!registered) {
      PetscCall(PetscRegisterFinalize(PetscPythonFinalize));
      registered = PETSC_TRUE;
    }
    PetscBeganPython = PETSC_TRUE;
    PetscCall(PetscInfo(NULL, "Python initialize completed.\n"));
  }
  /* import 'petsc4py.PETSc' module */
  module = PyImport_ImportModule("petsc4py.PETSc");
  if (module) {
    PetscCall(PetscInfo(NULL, "Python: successfully imported  module 'petsc4py.PETSc'\n"));
    Py_DecRef(module);
    module = NULL;
  } else {
    PetscCall(PetscPythonPrintError());
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Python: could not import module 'petsc4py.PETSc', perhaps your PYTHONPATH does not contain it");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscPythonPrintError - Print any current Python errors.

  Level: developer

.seealso: `PetscPythonInitialize()`, `PetscPythonFinalize()`
@*/
PetscErrorCode PetscPythonPrintError(void)
{
  PyObject *exc = NULL, *val = NULL, *tb = NULL;

  PetscFunctionBegin;
  if (!PetscBeganPython) PetscFunctionReturn(PETSC_SUCCESS);
  if (!PyErr_Occurred()) PetscFunctionReturn(PETSC_SUCCESS);
  PyErr_Fetch(&exc, &val, &tb);
  PyErr_NormalizeException(&exc, &val, &tb);
  PyErr_Display(exc ? exc : Py_None, val ? val : Py_None, tb ? tb : Py_None);
  PyErr_Restore(exc, val, tb);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ---------------------------------------------------------------- */

PETSC_EXTERN PetscErrorCode (*PetscPythonMonitorSet_C)(PetscObject, const char[]);
PetscErrorCode (*PetscPythonMonitorSet_C)(PetscObject, const char[]) = NULL;

/*@C
  PetscPythonMonitorSet - Set a Python monitor for a `PetscObject`

  Level: developer

.seealso: `PetscPythonInitialize()`, `PetscPythonFinalize()`, `PetscPythonPrintError()`
@*/
PetscErrorCode PetscPythonMonitorSet(PetscObject obj, const char url[])
{
  PetscFunctionBegin;
  PetscValidHeader(obj, 1);
  PetscValidCharPointer(url, 2);
  if (!PetscPythonMonitorSet_C) {
    PetscCall(PetscPythonInitialize(NULL, NULL));
    PetscCheck(PetscPythonMonitorSet_C, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Couldn't initialize Python support for monitors");
  }
  PetscCall(PetscPythonMonitorSet_C(obj, url));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ---------------------------------------------------------------- */
