cdef extern from "petsc.h" nogil:

    int PetscOptionsCreate()
    int PetscOptionsDestroy()
    int PetscOptionsSetFromOptions()

    int PetscOptionsHasName(char[],char[],PetscBool*)
    int PetscOptionsSetAlias(char[],char[])
    int PetscOptionsSetValue(char[],char[])
    int PetscOptionsClearValue(char[])
    int PetscOptionsClear()

    int PetscOptionsInsertString(char[])
    int PetscOptionsInsertFile(char[])
    int PetscOptionsGetAll(char*[])

    int PetscOptionsGetBool(char[],char[],PetscBool*,PetscBool*)
    int PetscOptionsGetInt(char[],char[],PetscInt*,PetscBool*)
    int PetscOptionsGetReal(char[],char[],PetscReal*,PetscBool*)
    int PetscOptionsGetScalar(char[],char[],PetscScalar*,PetscBool*)
    int PetscOptionsGetString(char[],char[],char[],size_t,PetscBool*)

    ctypedef struct _p_PetscToken
    ctypedef _p_PetscToken* PetscToken
    int PetscTokenCreate(char[],char,PetscToken*)
    int PetscTokenDestroy(PetscToken)
    int PetscTokenFind(PetscToken,char*[])

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

cdef opt2str(const_char *pre, const_char *name):
    p = bytes2str(pre)  if pre!=NULL else None
    n = bytes2str(name) if name[0]!=c'-' else bytes2str(&name[1])
    return '(prefix:%s, name:%s)' % (p, n)

cdef getopt_Bool(const_char *pre, const_char *name, object deft):
    cdef PetscBool value = PETSC_FALSE
    cdef PetscBool flag  = PETSC_FALSE
    CHKERR( PetscOptionsGetBool(pre, name, &value, &flag) )
    if flag==PETSC_TRUE: return <bint>value
    if deft is not None: return deft
    raise KeyError(opt2str(pre, name))

cdef getopt_Int(const_char *pre, const_char *name, object deft):
    cdef PetscInt value = 0
    cdef PetscBool flag = PETSC_FALSE
    CHKERR( PetscOptionsGetInt(pre, name, &value, &flag) )
    if flag==PETSC_TRUE: return toInt(value)
    if deft is not None: return deft
    raise KeyError(opt2str(pre, name))

cdef getopt_Real(const_char *pre, const_char *name, object deft):
    cdef PetscReal value = 0
    cdef PetscBool flag = PETSC_FALSE
    CHKERR( PetscOptionsGetReal(pre, name, &value, &flag) )
    if flag==PETSC_TRUE: return toReal(value)
    if deft is not None: return deft
    raise KeyError(opt2str(pre, name))

cdef getopt_Scalar(const_char *pre, const_char *name, object deft):
    cdef PetscScalar value = 0
    cdef PetscBool flag = PETSC_FALSE
    CHKERR( PetscOptionsGetScalar(pre, name, &value, &flag) )
    if flag==PETSC_TRUE: return toScalar(value)
    if deft is not None: return deft
    raise KeyError(opt2str(pre, name))

cdef getopt_String(const_char *pre, const_char *name, object deft):
    cdef char value[1024+1]
    cdef PetscBool flag = PETSC_FALSE
    CHKERR( PetscOptionsGetString(pre, name, value, 1024, &flag) )
    if flag==PETSC_TRUE: return bytes2str(value)
    if deft is not None: return deft
    raise KeyError(opt2str(pre, name))


cdef enum PetscOptType:
    OPT_BOOL
    OPT_INT
    OPT_REAL
    OPT_SCALAR
    OPT_STRING

cdef getpair(prefix, name, const_char **pr, const_char **nm):
    # --
    cdef const_char *p = NULL
    prefix = str2bytes(prefix, &p)
    if p != NULL and p[0] == c'-':
        p = &p[1]
    # --
    cdef const_char *n = NULL
    name = str2bytes(name, &n)
    if n != NULL and n[0] != c'-':
        name = b'-' + name
        name = str2bytes(name, &n)
    # --
    pr[0] = p
    nm[0] = n
    return (prefix, name)

cdef getopt(PetscOptType otype, prefix, name, deft):
    cdef const_char *pr = NULL
    cdef const_char *nm = NULL
    tmp = getpair(prefix, name, &pr, &nm)
    if otype == OPT_BOOL   : return getopt_Bool   (pr, nm, deft)
    if otype == OPT_INT    : return getopt_Int    (pr, nm, deft)
    if otype == OPT_REAL   : return getopt_Real   (pr, nm, deft)
    if otype == OPT_SCALAR : return getopt_Scalar (pr, nm, deft)
    if otype == OPT_STRING : return getopt_String (pr, nm, deft)


# simple minded options parser

cdef tokenize(options):
  cdef PetscToken t
  cdef const_char *s = NULL
  cdef const_char *p = NULL
  options = str2bytes(options, &s)
  cdef list tokens = []
  CHKERR( PetscTokenCreate(s, c' ', &t) )
  try:
      CHKERR( PetscTokenFind(t, <char**>&p) )
      while p != NULL:
          tokens.append(bytes2str(p))
          CHKERR( PetscTokenFind(t, <char**>&p) )
  finally:
      CHKERR( PetscTokenDestroy(t) )
  return tokens

cdef gettok(tokens):
    if tokens: 
        return tokens.pop(0)
    else: 
        return None

cdef getkey(key, prefix):
    if not key or key[0] != '-' :
        return None
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

#
