#ifndef PETSC4PY_PEP3118_H
#define PETSC4PY_PEP3118_H

#if defined(PETSC_USE_64BIT_INDICES)
# define _FMT_PETSC_INT     "q"
#else
# define _FMT_PETSC_INT     "i"
#endif

#if   defined(PETSC_USE_SCALAR_SINGLE) || defined(PETSC_USE_SINGLE)
# define _FMT_PETSC_REAL    "f"
# define _FMT_PETSC_COMPLEX "Zf"
#elif defined(PETSC_USE_SCALAR_LONG_DOUBLE) ||defined(PETSC_USE_LONG_DOUBLE)
# define _FMT_PETSC_REAL    "g"
# define _FMT_PETSC_COMPLEX "Zg"
#elif defined(PETSC_USE_SCALAR_INT) || defined(PETSC_USE_INT)
# define _FMT_PETSC_REAL    "i"
# define _FMT_PETSC_COMPLEX "Zi"
#else    /*  (PETSC_USE_SCALAR_DOUBLE) || (PETSC_USE_DOUBLE)  */
# define _FMT_PETSC_REAL    "d"
# define _FMT_PETSC_COMPLEX "Zd"
#endif

#if defined(PETSC_USE_COMPLEX)
# define _FMT_PETSC_SCALAR  _FMT_PETSC_COMPLEX
#else    /* PETSC_USE_REAL */
# define _FMT_PETSC_SCALAR  _FMT_PETSC_REAL
#endif

static 
int 
PyPetscBuffer_FillInfo(Py_buffer *view, PyObject* ob, 
                       char typechar, int flags)
{
  view->obj = NULL;
  switch (typechar) {
  case 'i': view->itemsize = sizeof(PetscInt);    break;
  case 'r': view->itemsize = sizeof(PetscReal);   break;
  case 's': view->itemsize = sizeof(PetscScalar); break;
  case 'c': view->itemsize = 2*sizeof(PetscReal); break;
  default:  view->itemsize = 1;
  }
  view->format = NULL;
  if ((flags & PyBUF_FORMAT) == PyBUF_FORMAT) {
    switch (typechar) {
    case 'i': view->format = (char *) _FMT_PETSC_INT;     break;
    case 'r': view->format = (char *) _FMT_PETSC_REAL;    break;
    case 's': view->format = (char *) _FMT_PETSC_SCALAR;  break;
    case 'c': view->format = (char *) _FMT_PETSC_COMPLEX; break;
    default:  view->format = (char *) "B";
    }
  }
  view->ndim = 0;
  view->shape = NULL;
  view->strides = NULL;
  view->suboffsets = NULL;
  view->internal = NULL;
  if ((flags & PyBUF_ND) == PyBUF_ND) {
    view->ndim = 1;
    #if PY_VERSION_HEX >= 0x02070000
    view->shape = &view->smalltable[0];
    #else
    view->internal = PyMem_Malloc(2*sizeof(Py_ssize_t));
    if (!view->internal) {
      PyErr_NoMemory();
      return -1;
    }
    view->shape = (Py_ssize_t *) view->internal;
    #endif
    view->shape[0] = view->len/view->itemsize;
    if ((flags & PyBUF_STRIDES) == PyBUF_STRIDES) {
      view->strides = &view->shape[1];
      view->strides[0] = view->itemsize;
    }
  }
  Py_XINCREF(ob);
  view->obj = ob;
  return 0;
}

static
void
PyPetscBuffer_Release(Py_buffer *view)
{
  #if PY_VERSION_HEX < 0x02070000
  if (view->internal) PyMem_Free(view->internal);
  view->internal = NULL;
  #endif
}

#undef _FMT_PETSC_INT
#undef _FMT_PETSC_REAL
#undef _FMT_PETSC_SCALAR

#endif /* !PETSC4PY_PEP3118_H */
