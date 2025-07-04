/* Author:  Lisandro Dalcin   */
/* Contact: dalcinl@gmail.com */

/* -------------------------------------------------------------------------- */

#if defined(Py_LIMITED_API) && Py_LIMITED_API+0 < 0x030B0000

#define Py_bf_getbuffer 1
#define Py_bf_releasebuffer 2

typedef struct {
  void *buf;
  PyObject *obj;
  Py_ssize_t len;
  Py_ssize_t itemsize;
  int readonly;
  int ndim;
  char *format;
  Py_ssize_t *shape;
  Py_ssize_t *strides;
  Py_ssize_t *suboffsets;
  void *internal;
} Py_buffer;

#define PyBUF_SIMPLE 0
#define PyBUF_WRITABLE 0x0001

#define PyBUF_FORMAT 0x0004
#define PyBUF_ND 0x0008
#define PyBUF_STRIDES (0x0010 | PyBUF_ND)
#define PyBUF_C_CONTIGUOUS (0x0020 | PyBUF_STRIDES)
#define PyBUF_F_CONTIGUOUS (0x0040 | PyBUF_STRIDES)
#define PyBUF_ANY_CONTIGUOUS (0x0080 | PyBUF_STRIDES)
#define PyBUF_INDIRECT (0x0100 | PyBUF_STRIDES)

#define PyBUF_CONTIG (PyBUF_ND | PyBUF_WRITABLE)
#define PyBUF_CONTIG_RO (PyBUF_ND)

#define PyBUF_STRIDED (PyBUF_STRIDES | PyBUF_WRITABLE)
#define PyBUF_STRIDED_RO (PyBUF_STRIDES)

#define PyBUF_RECORDS (PyBUF_STRIDES | PyBUF_WRITABLE | PyBUF_FORMAT)
#define PyBUF_RECORDS_RO (PyBUF_STRIDES | PyBUF_FORMAT)

#define PyBUF_FULL (PyBUF_INDIRECT | PyBUF_WRITABLE | PyBUF_FORMAT)
#define PyBUF_FULL_RO (PyBUF_INDIRECT | PyBUF_FORMAT)

#define PyBUF_READ  0x100
#define PyBUF_WRITE 0x200

PyAPI_FUNC(int)  PyObject_CheckBuffer(PyObject *);
PyAPI_FUNC(int)  PyObject_GetBuffer(PyObject *, Py_buffer *, int);
PyAPI_FUNC(void) PyBuffer_Release(Py_buffer *);
PyAPI_FUNC(int)  PyBuffer_FillInfo(Py_buffer *, PyObject *,
                                   void *, Py_ssize_t, int, int);

#endif

/* -------------------------------------------------------------------------- */

#if (defined(Py_LIMITED_API) && Py_LIMITED_API+0 < 0x030C0000) || PY_VERSION_HEX < 0X30C0000

#define PyErr_GetRaisedException PyErr_GetRaisedException_312
static PyObject *PyErr_GetRaisedException(void)
{
    PyObject *t, *v, *tb;
    PyErr_Fetch(&t, &v, &tb);
    PyErr_NormalizeException(&t, &v, &tb);
    if (tb != NULL) PyException_SetTraceback(v, tb);
    Py_XDECREF(t);
    Py_XDECREF(tb);
    return v;
}

#define PyErr_SetRaisedException PyErr_SetRaisedException_312
static void PyErr_SetRaisedException_312(PyObject *v)
{
    PyObject *t = (PyObject *)Py_TYPE(v);
    PyObject *tb = PyException_GetTraceback(v);
    Py_XINCREF(t);
    Py_XINCREF(tb);
    PyErr_Restore(t, v, tb);
}

#endif

/* -------------------------------------------------------------------------- */
