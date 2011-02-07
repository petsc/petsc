#undef  __FUNCT__
#define __FUNCT__ "PetscPyObjDestroy"
static PetscErrorCode PetscPyObjDestroy(void* ptr)
{
  PyObject *pyobj = 0;
  PetscFunctionBegin;
  pyobj = (PyObject*) ptr;
  if (pyobj && Py_IsInitialized()) {
    PyGILState_STATE _save = PyGILState_Ensure();
    Py_DecRef(pyobj);
    PyGILState_Release(_save);
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "PetscObjectGetPyDict"
static PetscErrorCode PetscObjectGetPyDict(PetscObject obj, PetscBool create, void **dict)
{
  PyObject      *pydict = NULL;
  PetscFunctionBegin;
  if (dict) *dict = NULL;
  PetscValidHeader(obj, 1);
  if (dict) PetscValidPointer(dict, 2);
  if (obj->python_context != NULL) {
    pydict = (PyObject *) obj->python_context;
    if (!PyDict_CheckExact(pydict)) {
      SETERRQQ(PETSC_COMM_SELF,
               PETSC_ERR_LIB,
               "internal Python object is not a Python dictionary");
    }
  } else if (create) {
    pydict = PyDict_New();
    if (pydict == NULL) {
      SETERRQQ(PETSC_COMM_SELF,
               PETSC_ERR_LIB,
              "failed to create internal Python dictionary");
    }
    obj->python_context = (void *) pydict;
    obj->python_destroy = PetscPyObjDestroy;
  } else {
    pydict = Py_None;
  }
  if (dict) *dict = (void *) pydict;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "PetscObjectSetPyObj"
static PetscErrorCode PetscObjectSetPyObj(PetscObject obj, const char name[], void *op)
{
  PyObject       *pydct = NULL;
  PyObject       *pykey = NULL;
  PyObject       *pyobj = NULL;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeader(obj, 1);
  PetscValidCharPointer(name, 2);
  if (op) PetscValidPointer(op, 3);
#if PY_MAJOR_VERSION < 3
  pykey = PyString_InternFromString(name);
#else
  pykey = PyUnicode_InternFromString(name);
#endif
  if (pykey == NULL) {
    SETERRQQ(PETSC_COMM_SELF,
             PETSC_ERR_LIB,
             "failed to create Python string");
  }
  pyobj = (PyObject *) op;
  if (pyobj == Py_None) pyobj = NULL;
  if (pyobj == NULL) {
    ierr = PetscObjectGetPyDict(obj, PETSC_FALSE, (void**)&pydct);CHKERRQ(ierr);
    if (pydct != NULL && pydct != Py_None && PyDict_CheckExact(pydct)) {
      int ret = PyDict_DelItem(pydct, pykey);
      Py_DecRef(pykey); pykey = NULL;
      if (ret < 0) {
        SETERRQQ(PETSC_COMM_SELF,
                 PETSC_ERR_LIB,
                 "failed to remove object from internal Python dictionary");
      }
    }
  } else {
    ierr = PetscObjectGetPyDict(obj, PETSC_TRUE, (void**)&pydct);CHKERRQ(ierr);
    if (pydct != NULL && pydct != Py_None && PyDict_CheckExact(pydct)) {
      int ret = PyDict_SetItem(pydct, pykey, pyobj);
      Py_DecRef(pykey); pykey = NULL;
      if (ret < 0) {
        SETERRQQ(PETSC_COMM_SELF,
                 PETSC_ERR_LIB,
                 "failed to set object in internal Python dictionary");
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "PetscObjectGetPyObj"
static PetscErrorCode PetscObjectGetPyObj(PetscObject obj, const char name[], void **op)
{
  PyObject       *pydct = NULL;
  PyObject       *pykey = NULL;
  PyObject       *pyobj = NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeader(obj, 1);
  PetscValidCharPointer(name, 2);
  PetscValidPointer(op, 3);
  *op = (void*) Py_None;
  ierr = PetscObjectGetPyDict(obj, PETSC_FALSE, (void**)&pydct);CHKERRQ(ierr);
  if (pydct != NULL && pydct != Py_None && PyDict_CheckExact(pydct)) {
#if PY_MAJOR_VERSION < 3
    pykey = PyString_InternFromString(name);
#else
    pykey = PyUnicode_InternFromString(name);
#endif
    if (pykey == NULL) {
      SETERRQQ(PETSC_COMM_SELF,
               PETSC_ERR_LIB,
               "failed to create Python string");
    }
    pyobj = PyDict_GetItem(pydct, pykey);
    Py_DecRef(pykey); pykey = NULL;
    if (pyobj == NULL) pyobj = Py_None;
    *op = (void*) pyobj;
  }
  PetscFunctionReturn(0);
}

/*
  Local variables:
  c-basic-offset: 2
  indent-tabs-mode: nil
  End:
*/
