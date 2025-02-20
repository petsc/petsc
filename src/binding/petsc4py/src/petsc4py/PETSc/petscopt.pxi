cdef extern from * nogil:

    ctypedef struct _n_PetscOptions
    ctypedef _n_PetscOptions* PetscOptions

    PetscErrorCode PetscOptionsCreate(PetscOptions*)
    PetscErrorCode PetscOptionsDestroy(PetscOptions*)
    PetscErrorCode PetscOptionsView(PetscOptions, PetscViewer)
    PetscErrorCode PetscOptionsClear(PetscOptions)

    PetscErrorCode PetscOptionsPrefixPush(PetscOptions, char[])
    PetscErrorCode PetscOptionsPrefixPop(PetscOptions)

    PetscErrorCode PetscOptionsHasName(PetscOptions, char[], char[], PetscBool*)
    PetscErrorCode PetscOptionsUsed(PetscOptions, char[], PetscBool*)
    PetscErrorCode PetscOptionsSetAlias(PetscOptions, char[], char[])
    PetscErrorCode PetscOptionsSetValue(PetscOptions, char[], char[])
    PetscErrorCode PetscOptionsClearValue(PetscOptions, char[])

    PetscErrorCode PetscOptionsInsertString(PetscOptions, char[])
    PetscErrorCode PetscOptionsInsertFile(PetscOptions, char[])
    PetscErrorCode PetscOptionsGetAll(PetscOptions, char*[])

    PetscErrorCode PetscOptionsGetBool(PetscOptions, char[], char[], PetscBool*, PetscBool*)
    PetscErrorCode PetscOptionsGetBoolArray(PetscOptions, char[], char[], PetscBool[], PetscInt*, PetscBool*)
    PetscErrorCode PetscOptionsGetInt(PetscOptions, char[], char[], PetscInt*, PetscBool*)
    PetscErrorCode PetscOptionsGetIntArray(PetscOptions, char[], char[], PetscInt[], PetscInt*, PetscBool*)
    PetscErrorCode PetscOptionsGetReal(PetscOptions, char[], char[], PetscReal*, PetscBool*)
    PetscErrorCode PetscOptionsGetRealArray(PetscOptions, char[], char[], PetscReal[], PetscInt*, PetscBool*)
    PetscErrorCode PetscOptionsGetScalar(PetscOptions, char[], char[], PetscScalar*, PetscBool*)
    PetscErrorCode PetscOptionsGetScalarArray(PetscOptions, char[], char[], PetscScalar[], PetscInt*, PetscBool*)
    PetscErrorCode PetscOptionsGetString(PetscOptions, char[], char[], char[], size_t, PetscBool*)

    ctypedef struct _p_PetscToken
    ctypedef _p_PetscToken* PetscToken
    PetscErrorCode PetscTokenCreate(char[], char, PetscToken*)
    PetscErrorCode PetscTokenDestroy(PetscToken*)
    PetscErrorCode PetscTokenFind(PetscToken, const char*[])
    PetscErrorCode PetscOptionsValidKey(char[], PetscBool*)

#

cdef getprefix(prefix, deft=None):
    if prefix is None:
        prefix = deft
    elif isinstance(prefix, Options):
        prefix = prefix.prefix
    elif isinstance(prefix, Object):
        prefix = prefix.getOptionsPrefix()
    elif not isinstance(prefix, str):
        raise TypeError('option prefix must be string')
    if not prefix:
        return None
    if prefix.count(' '):
        raise ValueError('option prefix should not have spaces')
    if prefix.startswith('-'):
        raise ValueError('option prefix should not start with a hyphen')
    return prefix

#

cdef opt2str(const char *pre, const char *name):
    p = bytes2str(pre)  if pre!=NULL else None
    n = bytes2str(name) if name[0]!=c'-' else bytes2str(&name[1])
    return '(prefix:%s, name:%s)' % (p, n)

cdef getopt_Bool(PetscOptions opt, const char *pre, const char *name, object deft):
    cdef PetscBool value = PETSC_FALSE
    cdef PetscBool flag  = PETSC_FALSE
    CHKERR(PetscOptionsGetBool(opt, pre, name, &value, &flag))
    if flag==PETSC_TRUE: return toBool(value)
    if deft is not None: return toBool(asBool(deft))
    raise KeyError(opt2str(pre, name))

cdef getopt_BoolArray(PetscOptions opt, const char *pre, const char *name, object deft):
    cdef PetscBool value[1024], *ivalue = value, *ivaluedeft = NULL
    cdef PetscInt nmax = 1024, ndeft = 0
    cdef PetscBool flag = PETSC_FALSE
    cdef object unused
    if deft is not None:
        deft = [toBool(asBool(d)) for d in deft]
        deft = iarray_b(deft, &ndeft, &ivaluedeft)
        if ndeft > nmax:
            unused = oarray_b(empty_b(ndeft), &nmax, &ivalue)
        memcpy(ivalue, ivaluedeft, <size_t>ndeft*sizeof(PetscBool))
    CHKERR(PetscOptionsGetBoolArray(opt, pre, name, ivalue, &nmax, &flag))
    if flag==PETSC_TRUE: return array_b(nmax, ivalue).astype('bool')
    if deft is not None: return deft.astype('bool')
    raise KeyError(opt2str(pre, name))

cdef getopt_Int(PetscOptions opt, const char *pre, const char *name, object deft):
    cdef PetscInt value = 0
    cdef PetscBool flag = PETSC_FALSE
    CHKERR(PetscOptionsGetInt(opt, pre, name, &value, &flag))
    if flag==PETSC_TRUE: return toInt(value)
    if deft is not None: return toInt(asInt(deft))
    raise KeyError(opt2str(pre, name))

cdef getopt_IntArray(PetscOptions opt, const char *pre, const char *name, object deft):
    cdef PetscInt value[1024], *ivalue = value, *ivaluedeft = NULL
    cdef PetscInt nmax = 1024, ndeft = 0
    cdef PetscBool flag = PETSC_FALSE
    cdef object unused
    if deft is not None:
        deft = iarray_i(deft, &ndeft, &ivaluedeft)
        if ndeft > nmax:
            unused = oarray_i(empty_i(ndeft), &nmax, &ivalue)
        memcpy(ivalue, ivaluedeft, <size_t>ndeft*sizeof(PetscInt))
    CHKERR(PetscOptionsGetIntArray(opt, pre, name, ivalue, &nmax, &flag))
    if flag==PETSC_TRUE: return array_i(nmax, ivalue)
    if deft is not None: return deft
    raise KeyError(opt2str(pre, name))

cdef getopt_Real(PetscOptions opt, const char *pre, const char *name, object deft):
    cdef PetscReal value = 0
    cdef PetscBool flag = PETSC_FALSE
    CHKERR(PetscOptionsGetReal(opt, pre, name, &value, &flag))
    if flag==PETSC_TRUE: return toReal(value)
    if deft is not None: return toReal(asReal(deft))
    raise KeyError(opt2str(pre, name))

cdef getopt_RealArray(PetscOptions opt, const char *pre, const char *name, object deft):
    cdef PetscReal value[1024], *ivalue = value, *ivaluedeft = NULL
    cdef PetscInt nmax = 1024, ndeft = 0
    cdef PetscBool flag = PETSC_FALSE
    cdef object unused
    if deft is not None:
        deft = iarray_r(deft, &ndeft, &ivaluedeft)
        if ndeft > nmax:
            unused = oarray_r(empty_r(ndeft), &nmax, &ivalue)
        memcpy(ivalue, ivaluedeft, <size_t>ndeft*sizeof(PetscReal))
    CHKERR(PetscOptionsGetRealArray(opt, pre, name, ivalue, &nmax, &flag))
    if flag==PETSC_TRUE: return array_r(nmax, ivalue)
    if deft is not None: return deft
    raise KeyError(opt2str(pre, name))

cdef getopt_Scalar(PetscOptions opt, const char *pre, const char *name, object deft):
    cdef PetscScalar value = 0
    cdef PetscBool flag = PETSC_FALSE
    CHKERR(PetscOptionsGetScalar(opt, pre, name, &value, &flag))
    if flag==PETSC_TRUE: return toScalar(value)
    if deft is not None: return toScalar(asScalar(deft))
    raise KeyError(opt2str(pre, name))

cdef getopt_ScalarArray(PetscOptions opt, const char *pre, const char *name, object deft):
    cdef PetscScalar value[1024], *ivalue = value, *ivaluedeft = NULL
    cdef PetscInt nmax = 1024, ndeft = 0
    cdef PetscBool flag = PETSC_FALSE
    cdef object unused
    if deft is not None:
        deft = iarray_s(deft, &ndeft, &ivaluedeft)
        if ndeft > nmax:
            unused = oarray_s(empty_s(ndeft), &nmax, &ivalue)
        memcpy(ivalue, ivaluedeft, <size_t>ndeft*sizeof(PetscScalar))
    CHKERR(PetscOptionsGetScalarArray(opt, pre, name, ivalue, &nmax, &flag))
    if flag==PETSC_TRUE: return array_s(nmax, ivalue)
    if deft is not None: return deft
    raise KeyError(opt2str(pre, name))

cdef getopt_String(PetscOptions opt, const char *pre, const char *name, object deft):
    cdef char value[1024+1]
    cdef PetscBool flag = PETSC_FALSE
    CHKERR(PetscOptionsGetString(opt, pre, name, value, 1024, &flag))
    if flag==PETSC_TRUE: return bytes2str(value)
    if deft is not None: return str(deft)
    raise KeyError(opt2str(pre, name))

cdef enum PetscOptType:
    OPT_BOOL
    OPT_BOOLARRAY
    OPT_INT
    OPT_INTARRAY
    OPT_REAL
    OPT_REALARRAY
    OPT_SCALAR
    OPT_SCALARARRAY
    OPT_STRING

cdef getpair(prefix, name, const char **pr, const char **nm):
    # --
    cdef const char *p = NULL
    prefix = str2bytes(prefix, &p)
    if p != NULL and p[0] == c'-':
        p = &p[1]
    # --
    cdef const char *n = NULL
    name = str2bytes(name, &n)
    if n != NULL and n[0] != c'-':
        name = b'-' + name
        name = str2bytes(name, &n)
    # --
    pr[0] = p
    nm[0] = n
    return (prefix, name)

cdef getopt(PetscOptions opt, PetscOptType otype, prefix, name, deft):
    cdef const char *pr = NULL
    cdef const char *nm = NULL
    cdef object unused = getpair(prefix, name, &pr, &nm)
    if otype == OPT_BOOL        : return getopt_Bool        (opt, pr, nm, deft)
    if otype == OPT_BOOLARRAY   : return getopt_BoolArray   (opt, pr, nm, deft)
    if otype == OPT_INT         : return getopt_Int         (opt, pr, nm, deft)
    if otype == OPT_INTARRAY    : return getopt_IntArray    (opt, pr, nm, deft)
    if otype == OPT_REAL        : return getopt_Real        (opt, pr, nm, deft)
    if otype == OPT_REALARRAY   : return getopt_RealArray   (opt, pr, nm, deft)
    if otype == OPT_SCALAR      : return getopt_Scalar      (opt, pr, nm, deft)
    if otype == OPT_SCALARARRAY : return getopt_ScalarArray (opt, pr, nm, deft)
    if otype == OPT_STRING      : return getopt_String      (opt, pr, nm, deft)


# simple minded options parser

cdef tokenize(options):
    cdef PetscToken t = NULL
    cdef const char *s = NULL
    cdef const char *p = NULL
    options = str2bytes(options, &s)
    cdef list tokens = []
    CHKERR(PetscTokenCreate(s, c' ', &t))
    try:
        CHKERR(PetscTokenFind(t, <const char**>&p))
        while p != NULL:
            tokens.append(bytes2str(p))
            CHKERR(PetscTokenFind(t, <const char**>&p))
    finally:
        CHKERR(PetscTokenDestroy(&t))
    return tokens

cdef bint iskey(key):
    cdef const char *k = NULL
    cdef PetscBool b = PETSC_FALSE
    if key:
        key = str2bytes(key, &k)
        CHKERR(PetscOptionsValidKey(k, &b))
        if b == PETSC_TRUE:
            return True
    return False

cdef gettok(tokens):
    if tokens:
        return tokens.pop(0)
    else:
        return None

cdef getkey(key, prefix):
    if not iskey(key):
        return None
    key = key[1:]
    if key[0] == '-':
        key = key[1:]
    if not key.startswith(prefix):
        return None
    return key.replace(prefix, '', 1)

cdef parseopt(options, prefix):
    if isinstance(options, str):
        tokens = tokenize(options)
    else:
        tokens = list(options)
    prefix = prefix or ''
    # parser loop
    opts = {}
    first = gettok(tokens)
    while first:
        key = getkey(first, prefix)
        if not key:
            first = gettok(tokens)
        else:
            second = gettok(tokens)
            if getkey(second, prefix):
                value = None
                first = second
            else:
                value = second
                first = gettok(tokens)
            opts[key] = value
    # we are done
    return opts
