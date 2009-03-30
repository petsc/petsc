
#undef  __FUNCT__
#define __FUNCT__ "PetscPyObjDestroy"
static PetscErrorCode
PetscPyObjDestroy(void* ptr) {
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

#if (PETSC_VERSION_MAJOR    == 2  && \
     PETSC_VERSION_MINOR    == 3  && \
     PETSC_VERSION_SUBMINOR == 3  && \
     PETSC_VERSION_RELEASE  == 1) || \
    (PETSC_VERSION_MAJOR    == 2  && \
     PETSC_VERSION_MINOR    == 3  && \
     PETSC_VERSION_SUBMINOR == 2  && \
     PETSC_VERSION_RELEASE  == 1)

/* Implementation for PETSc-2.3.3 and PETSc-2.3.2 */

#warning "using former implementation of Python context management"

#undef  __FUNCT__
#define __FUNCT__ "PetscObjectGetPyDict"
PETSC_STATIC_INLINE PetscErrorCode
PetscObjectGetPyDict(PetscObject obj, PetscTruth create, void **dict)
{
  PyObject*      pydict = NULL;
  PetscContainer container = PETSC_NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeader(obj, 1);
  if (dict) PetscValidPointer(dict, 2);
  if (dict) *dict = NULL;
  ierr = PetscOListFind(obj->olist,"__python__",(PetscObject*)&container);CHKERRQ(ierr);
  if (container != PETSC_NULL) {
    if (((PetscObject)container)->cookie != PETSC_CONTAINER_COOKIE)
      SETERRQ(1, "composed object is not a PETSc container");
    ierr = PetscContainerGetPointer(container,(void**)&pydict); CHKERRQ(ierr);
    if (pydict == NULL)
      SETERRQ(1, "object in container is NULL");
    if (!PyDict_CheckExact(pydict))
      SETERRQ(1, "object in container is not a Python dictionary");
  } else if (create) {
    pydict = PyDict_New();
    if (pydict == NULL)
      SETERRQ(1, "failed to create internal Python dictionary");
    ierr = PetscContainerCreate(obj->comm,&container);CHKERRQ(ierr);
    ierr = PetscContainerSetUserDestroy(container,PetscPyObjDestroy);CHKERRQ(ierr);
    ierr = PetscContainerSetPointer(container,(void*)pydict);CHKERRQ(ierr);
    ierr = PetscOListAdd(&obj->olist,"__python__",(PetscObject)container);CHKERRQ(ierr);
    ierr = PetscObjectDestroy((PetscObject)container);CHKERRQ(ierr);
  } else {
    pydict = Py_None;
  }
  if (dict) *dict = pydict;
  PetscFunctionReturn(0);
}

#else

/* Implementation for PETSc-3.0.0 and above */

#undef  __FUNCT__
#define __FUNCT__ "PetscObjectGetPyDict"
PETSC_STATIC_INLINE PetscErrorCode
PetscObjectGetPyDict(PetscObject obj, PetscTruth create, void **dict)
{
  PyObject      *pydict = NULL;
  PetscFunctionBegin;
  PetscValidHeader(obj, 1);
  if (dict) PetscValidPointer(dict, 2);
  if (dict) *dict = NULL;
  if (obj->python_context != NULL) {
    pydict = (PyObject *) obj->python_context;
    if (!PyDict_CheckExact(pydict)) {
      SETERRQ(PETSC_ERR_LIB, "internal Python object is not a Python dictionary");
      PetscFunctionReturn(PETSC_ERR_LIB);
    }
  } else if (create) {
    pydict = PyDict_New();
    if (pydict == NULL) {
      SETERRQ(PETSC_ERR_LIB, "failed to create internal Python dictionary");
      PetscFunctionReturn(PETSC_ERR_LIB);
    }
    obj->python_context = (void *) pydict;
    obj->python_destroy = PetscPyObjDestroy;
  } else {
    pydict = Py_None;
  }
  if (dict) *dict = (void *) pydict;
  PetscFunctionReturn(0);
}

#endif

#undef  __FUNCT__
#define __FUNCT__ "PetscObjectSetPyObj"
PETSC_STATIC_INLINE PetscErrorCode
PetscObjectSetPyObj(PetscObject obj, const char name[], void *op)
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
  if (pykey == NULL) { SETERRQ(1, "failed to create Python string"); }
  pyobj = (PyObject *) op;
  if (pyobj == NULL || pyobj == Py_None) {
    ierr = PetscObjectGetPyDict(obj, PETSC_FALSE, (void**)&pydct);CHKERRQ(ierr);
    if (pydct != NULL && pydct != Py_None && PyDict_CheckExact(pydct)) {
      int ret = PyDict_DelItem(pydct, pykey);
      Py_DecRef(pykey); pykey = NULL;
      if (ret < 0) SETERRQ(1, "failed to remove object from internal Python dictionary");
    }
  } else {
    ierr = PetscObjectGetPyDict(obj, PETSC_TRUE, (void**)&pydct);CHKERRQ(ierr);
    if (pydct != NULL && pydct != Py_None && PyDict_CheckExact(pydct)) {
      int ret = PyDict_SetItem(pydct, pykey, pyobj);
      Py_DecRef(pykey); pykey = NULL;
      if (ret < 0) SETERRQ(1, "failed to set object in internal Python dictionary");
    }
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "PetscObjectGetPyObj"
PETSC_STATIC_INLINE PetscErrorCode
PetscObjectGetPyObj(PetscObject obj, const char name[], void **op)
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
    if (pykey == NULL) { SETERRQ(1, "failed to create Python string"); }
    pyobj = PyDict_GetItem(pydct, pykey);
    Py_DecRef(pykey); pykey = NULL;
    if (pyobj == NULL) pyobj = Py_None;
    *op = (void*) pyobj;
  }
  PetscFunctionReturn(0);
}
