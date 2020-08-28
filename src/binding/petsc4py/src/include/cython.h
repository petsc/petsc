static
void *Cython_ImportFunction(PyObject   *module,
                            const char *funcname,
                            const char *signature)
{
  PyObject *capi = NULL, *capsule = NULL; void *p = NULL;
  capi = PyObject_GetAttrString(module, (char *)"__pyx_capi__");
  if (!capi)
    goto bad;
  capsule = PyDict_GetItemString(capi, (char *)funcname);
  if (!capsule) {
    PyErr_Format(PyExc_ImportError,
                 "%s does not export expected C function %s",
                 PyModule_GetName(module), funcname);
    goto bad;
  }
#if PY_VERSION_HEX < 0x03020000
  if (PyCObject_Check(capsule)) {
    const char *desc, *s1, *s2;
    desc = (const char *)PyCObject_GetDesc(capsule);
    if (!desc)
      goto bad;
    s1 = desc; s2 = signature;
    while (*s1 != '\0' && *s1 == *s2) { s1++; s2++; }
    if (*s1 != *s2) {
      PyErr_Format(PyExc_TypeError,
                   "C function %s.%s has wrong signature "
                   "(expected %s, got %s)",
                   PyModule_GetName(module), funcname, signature, desc);
      goto bad;
    }
    p = PyCObject_AsVoidPtr(capsule);
  }
#endif
#if PY_VERSION_HEX >= 0x02070000
  if (PyCapsule_CheckExact(capsule)) {
    if (!PyCapsule_IsValid(capsule, signature)) {
      const char *desc = PyCapsule_GetName(capsule);
      PyErr_Format(PyExc_TypeError,
                   "C function %s.%s has wrong signature "
                   "(expected %s, got %s)",
                   PyModule_GetName(module), funcname, signature, desc);
      goto bad;
    }
    p = PyCapsule_GetPointer(capsule, signature);
  }
#endif
  Py_DECREF(capi);
  return p;
 bad:
  Py_XDECREF(capi);
  return NULL;
}
