#include <Python.h>
#include "cygwinpath_Module.h"

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

static PyObject *
cygwinpath_convertToFullWin32Path(
  PyObject *_self,
  PyObject *_args,
  PyObject *_kwdict
)
{
  char *cygpath;

  if (!PyArg_ParseTuple(_args, (char *) "s", &cygpath)) {
    return NULL;
  }
  return(Py_BuildValue((char *) "s", cygpath));
}

#endif /* HAVE_CYGWIN */

static PyMethodDef _cygwinpath_methods[] = {
  {(char *) "convertToFullWin32Path", (PyCFunction) cygwinpath_convertToFullWin32Path, (METH_VARARGS | METH_KEYWORDS), (char *) "Convert a cygwin path to its WIN32 path.\n"},
  {NULL, NULL}
};

void initcygwinpath(void);
void initcygwinpath(void) {
  (void) Py_InitModule3((char *) "cygwinpath", _cygwinpath_methods, (char *) "Path conversion for Cygwin\n");
}
