#include <Python.h>

#ifdef HAVE_CYGWIN

#include <sys/cygwin.h>
#include <sys/param.h>
static PyObject *cygwinpath_convertToFullWin32Path(PyObject *self, PyObject *args) {
  char *cygpath;
  char  winpath[MAXPATHLEN];

  if (!PyArg_ParseTuple(args, (char *) "s", &cygpath)) {
    return NULL;
  }
  cygwin_conv_to_full_win32_path(cygpath, winpath);
  return(Py_BuildValue((char *) "s", winpath));
}

#else

static PyObject *cygwinpath_convertToFullWin32Path(PyObject *self, PyObject *args) {
  char *cygpath;

  if (!PyArg_ParseTuple(args, (char *) "s", &cygpath)) {
    return NULL;
  }
  return(Py_BuildValue((char *) "s", cygpath));
}

#endif /* HAVE_CYGWIN */

static PyMethodDef cygwinpath_methods[] = {
  {(char *) "convertToFullWin32Path", cygwinpath_convertToFullWin32Path, METH_VARARGS, (char *) "Convert a cygwin path to its WIN32 path."},
  {NULL, NULL, 0, NULL}
};

void initcygwinpath(void);
void initcygwinpath(void) {
  (void) Py_InitModule((char *) "cygwinpath", cygwinpath_methods);
}
