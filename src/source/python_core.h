#ifndef _PETSC_PYTHON_CORE_H
#define _PETSC_PYTHON_CORE_H

/* -------------------------------------------------------------------------- */

#include <petsc4py/petsc4py.h>

/* -------------------------------------------------------------------------- */

#if !defined(WITH_THREAD)
#undef  PyGILState_Ensure
#define PyGILState_Ensure() ((PyGILState_STATE)0)
#undef  PyGILState_Release
#define PyGILState_Release(state) (state)=((PyGILState_STATE)0)
#undef  Py_BLOCK_THREADS
#define Py_BLOCK_THREADS (_save)=(PyThreadState*)0;
#undef  Py_UNBLOCK_THREADS
#define Py_UNBLOCK_THREADS (_save)=(PyThreadState*)0;
#endif

/* -------------------------------------------------------------------------- */

#if PY_MAJOR_VERSION >= 3
#define PyInt_Check  PyLong_Check
#define PyInt_AsLong PyLong_AsLong
#endif

#if PY_MAJOR_VERSION < 3
#define PyBytes_Check    PyString_Check
#define PyBytes_AsString PyString_AsString
#endif

/* -------------------------------------------------------------------------- */

#if !defined(PETSC_VERSION_)
#define PETSC_VERSION_(MAJOR,MINOR,SUBMINOR)   \
       (PETSC_VERSION_MAJOR    == MAJOR    && \
	PETSC_VERSION_MINOR    == MINOR    && \
	PETSC_VERSION_SUBMINOR == SUBMINOR && \
	PETSC_VERSION_RELEASE  == 1)
#endif

#if (PETSC_VERSION_(3,1,0) || PETSC_VERSION_(3,0,0))
#define IS_CLASSID    IS_COOKIE
#define VEC_CLASSID   VEC_COOKIE
#define MAT_CLASSID   MAT_COOKIE
#define KSP_CLASSID   KSP_COOKIE
#define PC_CLASSID    PC_COOKIE
#define SNES_CLASSID  SNES_COOKIE
#define TS_CLASSID    TS_COOKIE
#endif

#if (PETSC_VERSION_(3,1,0) || PETSC_VERSION_(3,0,0))
typedef PetscTruth PetscBool;
#endif

#if (PETSC_VERSION_(3,1,0) || PETSC_VERSION_(3,0,0))
#define PETSCVIEWERASCII  PETSC_VIEWER_ASCII
#define PETSCVIEWERSTRING PETSC_VIEWER_STRING
#endif

#define SETERRQQ(comm,n,s) \
  return PetscError(comm,__LINE__,__FUNCT__,__FILE__,__SDIR__,\
                    n,PETSC_ERROR_INITIAL,s)
#define SETERRQQ1(comm,n,s,a1) \
  return PetscError(comm,__LINE__,__FUNCT__,__FILE__,__SDIR__,\
                    n,PETSC_ERROR_INITIAL,s,a1)
#define SETERRQQ2(comm,n,s,a1,a2) \
  return PetscError(comm,__LINE__,__FUNCT__,__FILE__,__SDIR__,\
                    n,PETSC_ERROR_INITIAL,s,a1,a2)
#define SETERRQQ3(comm,n,s,a1,a2,a3) \
  return PetscError(comm,__LINE__,__FUNCT__,__FILE__,__SDIR__,\
                    n,PETSC_ERROR_INITIAL,s,a1,a2,a3)

#if (PETSC_VERSION_(3,1,0) || PETSC_VERSION_(3,0,0))
#undef SETERRQQ
#define SETERRQQ(comm,n,s) \
  return PetscError(__LINE__,__FUNCT__,__FILE__,__SDIR__,n,1,s)
#undef SETERRQQ1
#define SETERRQQ1(comm,n,s,a1) \
  return PetscError(__LINE__,__FUNCT__,__FILE__,__SDIR__,n,1,s,a1)
#undef SETERRQQ2
#define SETERRQQ2(comm,n,s,a1,a2) \
  return PetscError(__LINE__,__FUNCT__,__FILE__,__SDIR__,n,1,s,a1,a2)
#undef SETERRQQ3
#define SETERRQQ3(comm,n,s,a1,a2,a3) \
  return PetscError(__LINE__,__FUNCT__,__FILE__,__SDIR__,n,1,s,a1,a2,a3)
#endif

#ifndef PETSC_ERR_PYTHON
#define PETSC_ERR_PYTHON (-1)
#endif

/* -------------------------------------------------------------------------- */

#if PY_VERSION_HEX < 0x02050000
#define PetscPyExceptionClassCheck(exc) PyClass_Check((exc))
#define PetscPyExceptionClassName(exc)  PyString_AsString(((PyClassObject*)(exc))->cl_name)
#else
#define PetscPyExceptionClassCheck(exc) PyExceptionClass_Check((exc))
#define PetscPyExceptionClassName(exc)  PyExceptionClass_Name((exc))
#endif

#if PY_VERSION_HEX < 0x02050000
#define PetscPyImportModule(modname) PyImport_ImportModule((char *)(modname));
#define PetscPyGetAttrStr(ob,attr) PyObject_GetAttrString((ob),(char *)(attr))
#else
#define PetscPyImportModule(modname) PyImport_ImportModule((modname));
#define PetscPyGetAttrStr(ob,attr) PyObject_GetAttrString((ob),(attr))
#endif

#if PY_VERSION_HEX < 0x02050000
static PyObject* PetscPyBuildValue(const char *format, ...)
{
  va_list va;
  PyObject* retval = NULL;
  va_start(va, format);
  retval = Py_VaBuildValue((char *)format, va);
  va_end(va);
  return retval;
}
#else
#define PetscPyBuildValue Py_BuildValue
#endif

/* -------------------------------------------------------------------------- */

static const char * PetscPythonHandleError(void)
{
  static char ExcName[256];
  PyObject*   ExcType = PyErr_Occurred();
  PyOS_snprintf(ExcName, sizeof(ExcName), "%s", "<unknown>");
  if (ExcType){
    if (PetscPyExceptionClassCheck(ExcType)) {
      const char *name = PetscPyExceptionClassName(ExcType);
      if (name != NULL) {
        const char *dot = strrchr(name, '.');
        if (dot != NULL) name = dot+1;
        PyOS_snprintf(ExcName, sizeof(ExcName), "%s", name);
      }
    }
    {
      PyObject *exc,*val,*tb;
      PyErr_Fetch(&exc,&val,&tb);
      PyErr_NormalizeException(&exc, &val, &tb);
      PyErr_Display(exc ? exc : Py_None,
                    val ? val : Py_None,
                    tb  ? tb  : Py_None);
      PyErr_Restore(exc,val,tb);
    }
  }
  return ExcName;
}

/* -------------------------------------------------------------------------- */

#define PETSC_PYTHON_CALL_HEAD(PySelf, PyMethod)                        \
do {                                                                    \
  PyObject   *_self = PySelf;                                           \
  const char *_meth = PyMethod;                                         \
  PyObject   *_call = NULL;                                             \
  PyObject   *_args = NULL;                                             \
  PyObject   *_retv = NULL;                                             \
  if (!Py_IsInitialized()) {                                            \
    SETERRQQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Python is not initialized");\
    PetscFunctionReturn(PETSC_ERR_PYTHON);                              \
  }                                                                     \
  do {                                                                  \
    if (_self != NULL && _self != Py_None) {                            \
      _call = PetscPyGetAttrStr(_self, _meth);                          \
      if      (_call == NULL)    { PyErr_Clear(); }                     \
      else if (_call == Py_None) { Py_DecRef(_call); _call = NULL; }    \
    }                                                                   \
  } while(0)                                                            \
/**/

#define PETSC_PYTHON_CALL_JUMP(LABEL)                                   \
  do { if (_call == NULL) goto LABEL; } while(0)                        \
/**/

#define PETSC_PYTHON_CALL_BODY(Py_BV_ARG_LIST)                          \
  if (_call != NULL) {                                                  \
    do {                                                                \
      _args = PetscPyBuildValue Py_BV_ARG_LIST;                         \
      if (_args != NULL) {                                              \
        if (_args == Py_None)                                           \
          _retv = PyObject_CallObject(_call, NULL);                     \
        else if (PyTuple_CheckExact(_args))                             \
          _retv = PyObject_CallObject(_call, _args);                    \
        else                                                            \
          _retv = PyObject_CallFunctionObjArgs(_call, _args, NULL);     \
        Py_DecRef(_args); _args = NULL;                                 \
      }                                                                 \
      Py_DecRef(_call); _call = NULL;                                   \
    } while(0)                                                          \
/**/

#define PETSC_PYTHON_CALL_TAIL()                                        \
    if (_retv == NULL) {                                                \
      const char *_exc = PetscPythonHandleError();                      \
      SETERRQQ2(PETSC_COMM_SELF,PETSC_ERR_PYTHON,"calling Python, "     \
                "method %s(), exception '%s'", _meth, _exc);            \
      PetscFunctionReturn(PETSC_ERR_PYTHON);                            \
    } else {                                                            \
      Py_DecRef(_retv);                                                 \
    }                                                                   \
  }                                                                     \
} while(0)                                                              \
/**/

#define PETSC_PYTHON_NOTIMPLEMENTED(obj,PyMethod)                       \
  SETERRQQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,                              \
            "method %s() not implemented",PyMethod);                    \
/**/

#define PETSC_PYTHON_INCREF(h) \
  {PetscObject _1_obj = (PetscObject)(h); \
  if (_1_obj && _1_obj->refct != 0) _1_obj = 0; \
  if (_1_obj) _1_obj->refct++; \
/**/

#define PETSC_PYTHON_DECREF(h) \
  if (_1_obj) _1_obj->refct--;} \
/**/

/* -------------------------------------------------------------------------- */

#undef __FUNCT__
#define __FUNCT__ "PetscPythonImportPetsc4Py"
static PetscErrorCode PetscPythonImportPetsc4Py(void)
{
  static PetscBool initialized = PETSC_FALSE;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (initialized) PetscFunctionReturn(0);
  if (import_petsc4py() < 0) goto fail;
  initialized = PETSC_TRUE;
  PetscFunctionReturn(0);
 fail:
  PetscPythonHandleError();
  SETERRQQ(PETSC_COMM_SELF,PETSC_ERR_PYTHON,
           "could not import Python package 'petsc4py.PETSc'");
  PetscFunctionReturn(ierr);
}


#undef __FUNCT__
#define __FUNCT__ "PetscCreatePythonObject"
static PetscErrorCode PetscCreatePythonObject(const char fullname[],
                                              PyObject **outself)
{
  char modname[2*PETSC_MAX_PATH_LEN],*clsname=0,*dot;
  PyObject *mod, *cls, *inst, *self;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidCharPointer(fullname,1);
  PetscValidCharPointer(outself,2);

  ierr = PetscStrncpy(modname,fullname,sizeof(modname));CHKERRQ(ierr);
  ierr = PetscStrrchr(modname,'.',&dot);CHKERRQ(ierr);
  if (dot != modname) { dot[-1] = 0; clsname = dot; }

  /* import the Python package/module */
  self = mod = PetscPyImportModule(modname);
  if (mod == NULL) {
    const char *excname = PetscPythonHandleError();
    SETERRQQ2(PETSC_COMM_SELF,PETSC_ERR_PYTHON,"Python: error importing "
              "module '%s', exception '%s'",modname,excname);
    PetscFunctionReturn(PETSC_ERR_PYTHON);
  }
  if (!clsname) goto done;
  /* get the Python module/class/callable */
  self = cls = PetscPyGetAttrStr(mod,clsname);
  Py_DecRef(mod);
  if (cls == NULL) {
    const char *excname = PetscPythonHandleError();
    SETERRQQ3(PETSC_COMM_SELF,PETSC_ERR_PYTHON,"Python: error getting "
              "function/class '%s' from module '%s', exception '%s'",
              clsname,modname,excname);
    PetscFunctionReturn(PETSC_ERR_PYTHON);
  }
  if (!PyCallable_Check(cls)) goto done;
  /* create the Python instance */
  self = inst = PyObject_CallFunction(cls, NULL);
  Py_DecRef(cls);
  if (inst == NULL) {
    const char *excname = PetscPythonHandleError();
    SETERRQQ3(PETSC_COMM_SELF,PETSC_ERR_PYTHON,"Python: error calling "
              "function/class '%s' from module '%s', exception '%s'",
              clsname,modname,excname);
    PetscFunctionReturn(PETSC_ERR_PYTHON);
  }
 done:
  *outself = self;
  PetscFunctionReturn(0);
}

static PyObject *PetscPythonAsString(PyObject *ob, const char* buf[])
{
  if (ob && PyUnicode_Check(ob)) {
    PyObject *tmp = PyUnicode_AsASCIIString(ob);
    Py_DecRef(ob);
    ob = tmp;
  }
  if (ob && PyBytes_Check(ob))
    *buf = PyBytes_AsString(ob);
  return ob;
}

#undef __FUNCT__
#define __FUNCT__ "PetscPythonGetFullName"
static PetscErrorCode PetscPythonGetFullName(PyObject *self, char *pyname[])
{
  PyObject *cls=NULL, *omodname=NULL, *oclsname=NULL;
  const char *ModName = 0;
  const char *ClsName = 0;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (self) PetscValidPointer(self,1);
  PetscValidPointer(pyname,2);
  *pyname = PETSC_NULL;
  if (self == NULL) PetscFunctionReturn(0);
  /* --- */
  if (PyModule_Check(self)) {
    omodname = PetscPyGetAttrStr(self,"__name__");
    omodname = PetscPythonAsString(omodname, &ModName);
    if (!omodname) PyErr_Clear();
  } else {
    cls = PetscPyGetAttrStr(self,"__class__");
    if (!cls) PyErr_Clear();
    else {
      omodname = PetscPyGetAttrStr(cls,"__module__");
      omodname = PetscPythonAsString(omodname, &ModName);
      if (!omodname) PyErr_Clear();
      oclsname = PetscPyGetAttrStr(cls,"__name__");
      oclsname = PetscPythonAsString(oclsname, &ClsName);
      if (!oclsname) PyErr_Clear();
    }
  }
  /* --- */
  if (ModName) {
    if (ClsName) {
      size_t len1, len2;
      ierr = PetscStrlen(ModName,&len1);CHKERRQ(ierr);
      ierr = PetscStrlen(ClsName,&len2);CHKERRQ(ierr);
      ierr = PetscMalloc((len1+1+len2+1)*sizeof(char),pyname);CHKERRQ(ierr);
      ierr = PetscMemzero(*pyname,(len1+1+len2+1)*sizeof(char));CHKERRQ(ierr);
      ierr = PetscStrncpy(*pyname,ModName,len1);CHKERRQ(ierr);
      ierr = PetscStrncat(*pyname,".",1);CHKERRQ(ierr);
      ierr = PetscStrncat(*pyname,ClsName,len2);CHKERRQ(ierr);
    } else {
      ierr = PetscStrallocpy(ModName,pyname);CHKERRQ(ierr);
    }
  } else if (ClsName) {
    ierr = PetscStrallocpy(ClsName,pyname);CHKERRQ(ierr);
  }
  /* --- */
  if (cls)      Py_DecRef(cls);
  if (omodname) Py_DecRef(omodname);
  if (oclsname) Py_DecRef(oclsname);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscCreatePythonObject_GIL(const char fullname[],
                                                  PyObject **outself)
{
  PetscErrorCode ierr;
  PyGILState_STATE _save = PyGILState_Ensure();
  ierr = PetscCreatePythonObject(fullname, outself);
  PyGILState_Release(_save);
  return ierr;
}
#define PetscCreatePythonObject PetscCreatePythonObject_GIL
static PetscErrorCode PetscPythonGetFullName_GIL(PyObject *self, char *pyname[])
{
  PetscErrorCode ierr;
  PyGILState_STATE _save = PyGILState_Ensure();
  ierr = PetscPythonGetFullName(self, pyname);
  PyGILState_Release(_save);
  return ierr;
}
#define PetscPythonGetFullName PetscPythonGetFullName_GIL

/* -------------------------------------------------------------------------- */

#endif /* !_PETSC_PYTHON_H */
