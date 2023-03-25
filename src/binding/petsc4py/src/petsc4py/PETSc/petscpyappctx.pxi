cdef set appctx_registry = set()

cdef inline object registerAppCtx(void *appctx):
    cdef object key = <Py_uintptr_t> appctx
    appctx_registry.add(key)

cdef inline object toAppCtx(void *appctx):
    cdef object key = <Py_uintptr_t> appctx
    if key in appctx_registry:
        return <object> appctx
    else:
        if appctx != NULL:
            return PyLong_FromVoidPtr(appctx)
        else:
            return None
