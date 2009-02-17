cdef extern from "petsc.h" nogil:

    int PetscOptionsCreate()
    int PetscOptionsDestroy()
    int PetscOptionsSetFromOptions()

    int PetscOptionsHasName(char[],char[],PetscTruth*)
    int PetscOptionsSetAlias(char[],char[])
    int PetscOptionsSetValue(char[],char[])
    int PetscOptionsClearValue(char[])
    int PetscOptionsClear()

    int PetscOptionsInsertString(char[])
    int PetscOptionsInsertFile(char[])
    int PetscOptionsGetAll(char*[])

    int PetscOptionsGetTruth(char[],char[],PetscTruth*,PetscTruth*)
    int PetscOptionsGetInt(char[],char[],PetscInt*,PetscTruth*)
    int PetscOptionsGetReal(char[],char[],PetscReal*,PetscTruth*)
    int PetscOptionsGetScalar(char[],char[],PetscScalar*,PetscTruth*)
    int PetscOptionsGetString(char[],char[],char[],size_t,PetscTruth*)

    int PetscStrfree(char*)

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
        raise ValueError('option prefix should not start with a hypen')
    return prefix

#

cdef opt2str(char *pre, char *name):
    p = cp2str(pre)  if pre!=NULL else None
    n = cp2str(name) if name[0]!=c'-' else cp2str(&name[1])
    return '(prefix:%s, name:%s)' % (p, n)

cdef getopt_Truth(char *pre, char *name, object deft):
    cdef PetscTruth value = PETSC_FALSE
    cdef PetscTruth flag = PETSC_FALSE
    CHKERR( PetscOptionsGetTruth(pre, name, &value, &flag) )
    if flag==PETSC_TRUE: return value
    if deft is not None: return deft
    raise KeyError(opt2str(pre, name))

cdef getopt_Int(char *pre, char *name, object deft):
    cdef PetscInt value = 0
    cdef PetscTruth flag = PETSC_FALSE
    CHKERR( PetscOptionsGetInt(pre, name, &value, &flag) )
    if flag==PETSC_TRUE: return value
    if deft is not None: return deft
    raise KeyError(opt2str(pre, name))

cdef getopt_Real(char *pre, char *name, object deft):
    cdef PetscReal value = 0
    cdef PetscTruth flag = PETSC_FALSE
    CHKERR( PetscOptionsGetReal(pre, name, &value, &flag) )
    if flag==PETSC_TRUE: return value
    if deft is not None: return deft
    raise KeyError(opt2str(pre, name))

cdef getopt_Scalar(char *pre, char *name, object deft):
    cdef PetscScalar value = 0
    cdef PetscTruth flag = PETSC_FALSE
    CHKERR( PetscOptionsGetScalar(pre, name, &value, &flag) )
    if flag==PETSC_TRUE: return toScalar(value)
    if deft is not None: return deft
    raise KeyError(opt2str(pre, name))

cdef getopt_String(char *pre, char *name, object deft):
    cdef char value[1024+1]
    cdef PetscTruth flag = PETSC_FALSE
    CHKERR( PetscOptionsGetString(pre, name, value, 1024, &flag) )
    if flag==PETSC_TRUE: return cp2str(value)
    if deft is not None: return deft
    raise KeyError(opt2str(pre, name))


cdef enum PetscOptType:
    OPT_TRUTH
    OPT_INT
    OPT_REAL
    OPT_SCALAR
    OPT_STRING

cdef getpair(prefix, name, char **pr, char **nm):
    # --
    cdef char *p = str2cp(prefix)
    if p and p[0] == c'-':
        p = &p[1]
    # --
    cdef char *n = str2cp(name)
    if n and n[0] != c'-':
        name = '-' + name
        n = str2cp(name)
    # --
    pr[0] = p
    nm[0] = n
    return (prefix, name)

cdef getopt(PetscOptType otype, prefix, name, deft):
    cdef char *pre = NULL, *nm = NULL
    tmp = getpair(prefix, name, &pre, &nm)
    if otype == OPT_TRUTH  : return getopt_Truth  (pre, nm, deft)
    if otype == OPT_INT    : return getopt_Int    (pre, nm, deft)
    if otype == OPT_REAL   : return getopt_Real   (pre, nm, deft)
    if otype == OPT_SCALAR : return getopt_Scalar (pre, nm, deft)
    if otype == OPT_STRING : return getopt_String (pre, nm, deft)


# simple minded options parser

cdef gettok(tokens):
    if not tokens: return None
    else: return tokens.pop(0)

cdef getkey(key, prefix):
    if not key or key[0] != '-' :
        return None
    key = key[1:]
    if not key.startswith(prefix):
        return None
    return key.replace(prefix, '', 1)

cdef parseopt(options, prefix):
    if isinstance(options, str):
        tokens = options.split()
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
