#ifndef _PETSC_PYTHON_H
#define _PETSC_PYTHON_H

/* -------------------------------------------------------------------------- */

#include <petsc.h>

#if !defined(PETSC_USE_ERRORCHECKING)
#undef  SETERRQ
#define SETERRQ(n,s)                     return(n)
#undef  SETERRQ1
#define SETERRQ1(n,s,a1)                 return(n)
#undef  SETERRQ2
#define SETERRQ2(n,s,a1,a2)              return(n)
#undef  SETERRQ3
#define SETERRQ3(n,s,a1,a2,a3)           return(n)
#undef  SETERRQ4
#define SETERRQ4(n,s,a1,a2,a3,a4)        return(n)
#undef  SETERRQ5
#define SETERRQ5(n,s,a1,a2,a3,a4,a5)     return(n)
#undef  SETERRQ6
#define SETERRQ6(n,s,a1,a2,a3,a4,a5,a6)  return(n)
#undef  CHKERRQ
#define CHKERRQ(n)                       if(n)return(n)
#endif

#include <petsc4py/petsc4py.h>

#ifdef import_petsc
#define import_petsc4py() \
  (import_petsc(),(PyErr_Occurred()?-1:0))
#define PyPetscViewer_New  PyPetscViewer->Arg
#define PyPetscVec_New     PyPetscVec->Arg
#define PyPetscMat_New     PyPetscMat->Arg
#define PyPetscPC_New      PyPetscPC->Arg
#define PyPetscKSP_New     PyPetscKSP->Arg
#define PyPetscSNES_New    PyPetscSNES->Arg
#define PyPetscTS_New      PyPetscTS->Arg
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
#define PetscPyObjectGetAttrStr(ob,attr) PyObject_GetAttrString((ob),(char *)(attr))
#else
#define PetscPyImportModule(modname) PyImport_ImportModule((modname));
#define PetscPyObjectGetAttrStr(ob,attr) PyObject_GetAttrString((ob),(attr))
#endif

/* -------------------------------------------------------------------------- */

static const char * PetscHandlePythonError(void)
{
  PyObject *excType = PyErr_Occurred();
  if (excType && PetscPyExceptionClassCheck(excType)) {
    const char *className = PetscPyExceptionClassName(excType);
    if (className != NULL) {
      char *dot = strrchr(className, '.');
      if (dot != NULL) className = dot+1;
      return className;
    }
  }
  return "<unknown>";
}

/* -------------------------------------------------------------------------- */

#undef __FUNCT__
#define __FUNCT__ "PetscInitializePython"
static PetscErrorCode PetscInitializePython(void)
{
  static PetscTruth initialized = PETSC_FALSE;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (initialized) PetscFunctionReturn(0);
  if (!Py_IsInitialized()) Py_InitializeEx(0);
  if (import_petsc4py() < 0) goto fail;
  initialized = PETSC_TRUE;
  PetscFunctionReturn(0);
 fail:
  PetscHandlePythonError();
  SETERRQ(PETSC_ERR_PYTHON,"could not import Python package 'petsc4py.PETSc'");
  PetscFunctionReturn(ierr);
}

/* -------------------------------------------------------------------------- */

#define PETSC_PYTHON_CALL_HEAD(PySelf, PyMethod)			\
do {									\
  PyObject   *_self = PySelf;						\
  const char *_meth = PyMethod;						\
  PyObject   *_call = NULL;						\
  PyObject   *_args = NULL;						\
  PyObject   *_retv = NULL;						\
  if (!Py_IsInitialized()) {						\
    SETERRQ(PETSC_ERR_LIB,"Python is not initialized");			\
    PetscFunctionReturn(PETSC_ERR_PYTHON);				\
  }									\
  do {									\
    if (_self != NULL && _self != Py_None) {				\
      _call = PetscPyObjectGetAttrStr(_self, _meth);			\
      if      (_call == NULL)    { PyErr_Clear(); }			\
      else if (_call == Py_None) { Py_DecRef(_call); _call = NULL; }	\
    }									\
  } while(0)								\
/**/

#define PETSC_PYTHON_CALL_JUMP(LABEL)			\
  do { if (_call == NULL) goto LABEL; } while(0)	\
/**/
 
#define PETSC_PYTHON_CALL_BODY(Py_BV_ARGS)				\
  if (_call != NULL) {							\
    do {								\
      _args = Py_BuildValue Py_BV_ARGS;					\
      if (_args != NULL) {						\
	if (_args == Py_None)						\
	  _retv = PyObject_CallObject(_call, NULL);			\
	else if (PyTuple_CheckExact(_args))				\
	  _retv = PyObject_CallObject(_call, _args);			\
	else								\
	  _retv = PyObject_CallFunctionObjArgs(_call, _args, NULL);	\
	Py_DecRef(_args); _args = NULL;					\
      }									\
      Py_DecRef(_call); _call = NULL;					\
    } while(0)								\
/**/

#define PETSC_PYTHON_CALL_TAIL()					\
    if (_retv == NULL) {						\
      const char *_exc = PetscHandlePythonError();			\
      SETERRQ2(PETSC_ERR_PYTHON,"calling Python, "			\
	       "method %s(), exception '%s'", _meth, _exc);		\
      PetscFunctionReturn(PETSC_ERR_PYTHON);				\
    } else {								\
      Py_DecRef(_retv);							\
    }									\
  }									\
} while(0)								\
/**/

/* -------------------------------------------------------------------------- */

#undef __FUNCT__
#define __FUNCT__ "PetscCreatePythonObject"
static PetscErrorCode PetscCreatePythonObject(const char* modname,
					      const char* clsname,
					      PyObject **outself)
{
  PyObject *mod, *cls, *self;
  PetscFunctionBegin;
  if (outself) *outself = NULL;
  PetscValidCharPointer(modname,1);
  PetscValidCharPointer(clsname,2);
  PetscValidCharPointer(outself,3);
  /* import the Python module */
  mod = PetscPyImportModule(modname);
  if (mod == NULL) {
    const char *excname = PetscHandlePythonError();
    SETERRQ2(PETSC_ERR_PYTHON,"Python: error importing "
	     "module '%s', exception '%s'",modname,excname);
    PetscFunctionReturn(PETSC_ERR_PYTHON);
  }
  /* get the Python class (or callable) */
  cls = PetscPyObjectGetAttrStr(mod,clsname);
  Py_DecRef(mod);
  if (cls == NULL) {
    const char *excname = PetscHandlePythonError();
    SETERRQ3(PETSC_ERR_PYTHON,"Python: error getting "
	     "function/class '%s' from module '%s', exception '%s'",
	     clsname,modname,excname);
    PetscFunctionReturn(PETSC_ERR_PYTHON);
  }
  /* create the Python instance */
  self = PyObject_CallFunction(cls, NULL);
  Py_DecRef(cls);
  if (self == NULL) {
    const char *excname = PetscHandlePythonError();
    SETERRQ3(PETSC_ERR_PYTHON,"Python: error calling "
	     "function/class '%s' from module '%s', exception '%s'",
	     clsname,modname,excname);
    PetscFunctionReturn(PETSC_ERR_PYTHON);
  }
  *outself = self;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscPythonGetModuleAndClass"
static PetscErrorCode PetscPythonGetModuleAndClass(PyObject *self,
						   char *modname[],
						   char *clsname[])
{
  PyObject *cls=NULL, *omodname=NULL, *oclsname=NULL;
  const char *ModName = "<unknown>";
  const char *ClsName = "<unknown>";
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (self) PetscValidPointer(self,1);
  PetscValidPointer(modname,2);
  PetscValidPointer(clsname,3);
  if (self == NULL) {
    *modname = *clsname = PETSC_NULL;
    PetscFunctionReturn(0);
  }
  cls = PetscPyObjectGetAttrStr(self,"__class__");
  if (cls != NULL) {
    omodname = PetscPyObjectGetAttrStr(cls,"__module__");
    if (omodname != NULL) {
      if (PyString_Check(omodname))
	ModName = PyString_AsString(omodname);
    } else PyErr_Clear();
    oclsname = PetscPyObjectGetAttrStr(cls,"__name__"); 
    if (oclsname != NULL) {
      if (PyString_Check(oclsname))
	ClsName = PyString_AsString(oclsname);
    } else PyErr_Clear();
  } else PyErr_Clear();
  ierr = PetscStrallocpy(ModName,modname);CHKERRQ(ierr);
  ierr = PetscStrallocpy(ClsName,clsname);CHKERRQ(ierr);
  if (cls)      Py_DecRef(cls);
  if (omodname) Py_DecRef(omodname);
  if (oclsname) Py_DecRef(oclsname);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */

#endif /* !_PETSC_PYTHON_H */
