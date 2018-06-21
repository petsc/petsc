/* Author:  Lisandro Dalcin   */
/* Contact: dalcinl@gmail.com */

/* ---------------------------------------------------------------- */

%header %{#include "petsc4py/petsc4py.h"%}
%init   %{import_petsc4py();%}

/* ---------------------------------------------------------------- */

%runtime %{
SWIGINTERNINLINE PyObject*
SWIG_getattr_this(PyObject* obj) {
  if (!obj) return NULL;
  obj = PyObject_GetAttr(obj, SWIG_This());
  if (!obj) PyErr_Clear();
  return obj;
}
SWIGINTERNINLINE int
SWIG_convert_ptr(PyObject *obj, void **ptr, swig_type_info *ty, int flags) {
  int res = SWIG_ConvertPtr(obj, ptr, ty, flags);
  if (!SWIG_IsOK(res)) {
    PyObject* _this = SWIG_getattr_this(obj);
    res = SWIG_ConvertPtr(_this, ptr, ty, flags);
    Py_XDECREF(_this);
  }
  return res;
}
#undef  SWIG_ConvertPtr
#define SWIG_ConvertPtr(obj, pptr, type, flags) \
        SWIG_convert_ptr(obj, pptr, type, flags)
%}

/* ---------------------------------------------------------------- */


/* ---------------------------------------------------------------- */
/* PETSc Error Codes                                                */
/* ---------------------------------------------------------------- */

%define %petsc4py_errt(Pkg, PyType, Type)

%wrapper %{
#ifndef Py##Pkg##_ChkErrQ
#define Py##Pkg##_ChkErrQ(ierr)                         \
    do {                                                \
      if (ierr != 0) {                                  \
        Py##Pkg##PyType##_Set((ierr)); SWIG_fail;       \
      }                                                 \
    } while (0)
#endif /* defined Py##Pkg##_ChkErrQ */
%}

%typemap(out,noblock=1) Type {
Py##Pkg##_ChkErrQ($1); %set_output(VOID_Object);
}

%enddef

/* ---------------------------------------------------------------- */



/* ---------------------------------------------------------------- */
/* Numeric Types                                                    */
/* ---------------------------------------------------------------- */

%define SWIG_TYPECHECK_PETSC_INT      SWIG_TYPECHECK_INT32   %enddef
%define SWIG_TYPECHECK_PETSC_REAL     SWIG_TYPECHECK_DOUBLE  %enddef
%define SWIG_TYPECHECK_PETSC_COMPLEX  SWIG_TYPECHECK_CPLXDBL %enddef
%define SWIG_TYPECHECK_PETSC_SCALAR   SWIG_TYPECHECK_CPLXDBL %enddef

/* PetscInt */
/* -------- */
%fragment(SWIG_From_frag(PetscInt64),"header",
          fragment=SWIG_From_frag(long long),
          fragment=SWIG_From_frag(long))
{
SWIGINTERN SWIG_Object
SWIG_From_dec(PetscInt64)(PetscInt64 val) {
%#if PETSC_SIZEOF_LONG == 8
  return SWIG_From(long)(%numeric_cast(val,long));
%#else
  return SWIG_From(long long)(%numeric_cast(val,long long));
%#endif
}
}
%fragment(SWIG_AsVal_frag(PetscInt64),"header",
          fragment=SWIG_AsVal_frag(long long),
          fragment=SWIG_AsVal_frag(long))
{
SWIGINTERN int
SWIG_AsVal_dec(PetscInt64)(SWIG_Object obj, PetscInt64 *val) {
%#if PETSC_SIZEOF_LONG == 8
  long v; int res = SWIG_AsVal(long)(obj, &v);
%#else
  long long v; int res = SWIG_AsVal(long long)(obj, &v);
%#endif
  if (SWIG_IsOK(res) && val) if (val) *val = %numeric_cast(v,PetscInt64);
  return res;
}
}
%fragment(SWIG_From_frag(PetscInt),"header",
          fragment=SWIG_From_frag(PetscInt64),
          fragment=SWIG_From_frag(int))
{
%#if defined(PETSC_USE_64BIT_INDICES)
%define_as(SWIG_From(PetscInt), SWIG_From(PetscInt64))
%#else
%define_as(SWIG_From(PetscInt), SWIG_From(int))
%#endif
}
%fragment(SWIG_AsVal_frag(PetscInt),"header",
          fragment=SWIG_AsVal_frag(PetscInt64),
          fragment=SWIG_AsVal_frag(int))
{
%#if defined(PETSC_USE_64BIT_INDICES)
%define_as(SWIG_AsVal(PetscInt), SWIG_AsVal(PetscInt64))
%#else
%define_as(SWIG_AsVal(PetscInt), SWIG_AsVal(int))
%#endif
}

/* PetscReal */
/* --------- */
%fragment(SWIG_From_frag(long double),"header",
          fragment=SWIG_From_frag(double))
{
SWIGINTERN SWIG_Object
SWIG_From_dec(long double)(long double val) {
  return SWIG_From(double)((double)val);
}
}
%fragment(SWIG_AsVal_frag(long double),"header",
          fragment=SWIG_AsVal_frag(double))
{
SWIGINTERN int
SWIG_AsVal_dec(long double)(SWIG_Object obj, long double *val) {
  double v; int res = SWIG_AsVal(double)(obj, &v);
  if (SWIG_IsOK(res) && val) if (val) *val = %numeric_cast(v,long double);
  return res;
}
}
%fragment(SWIG_From_frag(PetscReal),"header",
          fragment=SWIG_From_frag(long double),
          fragment=SWIG_From_frag(double),
          fragment=SWIG_From_frag(float))
{
%#if   defined(PETSC_USE_REAL_SINGLE)
%define_as(SWIG_From(PetscReal), SWIG_From(float))
%#elif defined(PETSC_USE_REAL_DOUBLE)
%define_as(SWIG_From(PetscReal), SWIG_From(double))
%#elif defined(PETSC_USE_REAL_LONG_DOUBLE)
%define_as(SWIG_From(PetscReal), SWIG_From(long double))
%#endif
}
%fragment(SWIG_AsVal_frag(PetscReal),"header",
          fragment=SWIG_AsVal_frag(long double),
          fragment=SWIG_AsVal_frag(double),
          fragment=SWIG_AsVal_frag(float))
{
%#if   defined(PETSC_USE_REAL_SINGLE)
%define_as(SWIG_AsVal(PetscReal),  SWIG_AsVal(float))
%#elif defined(PETSC_USE_REAL_DOUBLE)
%define_as(SWIG_AsVal(PetscReal),  SWIG_AsVal(double))
%#elif defined(PETSC_USE_REAL_LONG_DOUBLE)
%define_as(SWIG_AsVal(PetscReal),  SWIG_AsVal(long double))
%#endif
}

/* PetscComplex */
/* ------------ */
%include complex.i
%fragment(SWIG_From_frag(PetscComplex),"header",
#ifdef __cplusplus
          fragment=SWIG_From_frag(std::complex<long double>),
          fragment=SWIG_From_frag(std::complex<double>),
          fragment=SWIG_From_frag(std::complex<float>))
#else
          fragment=SWIG_From_frag(long double complex),
          fragment=SWIG_From_frag(double complex),
          fragment=SWIG_From_frag(float complex))
#endif
{
%#if defined(PETSC_CLANGUAGE_CXX)
%define_as(SWIG_From(PetscComplex), SWIG_From(std::complex<double>))
%#else
%define_as(SWIG_From(PetscComplex), SWIG_From(double complex))
%#endif
}

%fragment(SWIG_AsVal_frag(PetscComplex),"header",
#ifdef __cplusplus
          fragment=SWIG_AsVal_frag(std::complex<long double>),
          fragment=SWIG_AsVal_frag(std::complex<double>),
          fragment=SWIG_AsVal_frag(std::complex<float>))
#else
          fragment=SWIG_AsVal_frag(long double complex),
          fragment=SWIG_AsVal_frag(double complex),
          fragment=SWIG_AsVal_frag(float complex))
#endif
{
%#if defined(PETSC_CLANGUAGE_CXX)
%define_as(SWIG_AsVal(PetscComplex), SWIG_AsVal(std::complex<double>))
%#else
%define_as(SWIG_AsVal(PetscComplex), SWIG_AsVal(double complex))
%#endif
}

/* PetscScalar */
/* ----------- */
%fragment(SWIG_From_frag(PetscScalar), "header",
          fragment=SWIG_From_frag(PetscReal),
          fragment=SWIG_From_frag(PetscComplex))
{
%#if   defined(PETSC_USE_SCALAR_COMPLEX)
%define_as(SWIG_From(PetscScalar), SWIG_From(PetscComplex))
%#elif defined(PETSC_USE_SCALAR_REAL)
%define_as(SWIG_From(PetscScalar), SWIG_From(PetscReal))
%#endif
}
%fragment(SWIG_AsVal_frag(PetscScalar), "header",
          fragment=SWIG_AsVal_frag(PetscReal),
          fragment=SWIG_AsVal_frag(PetscComplex))
{
%#if   defined(PETSC_USE_SCALAR_COMPLEX)
%define_as(SWIG_AsVal(PetscScalar), SWIG_AsVal(PetscComplex))
%#elif defined(PETSC_USE_SCALAR_REAL)
%define_as(SWIG_AsVal(PetscScalar), SWIG_AsVal(PetscReal))
%#endif
}

%define %petsc4py_numt(Pkg, PyType, Type, CheckCode, UNUSED)
%types(Type,Type*);
%typemaps_primitive(%checkcode(CheckCode), Type);
/* INPUT value typemap*/
%typemap(typecheck,
         precedence=%checkcode(CheckCode),
         fragment=SWIG_AsVal_frag(Type))
Type, const Type & {
  int res = SWIG_AsVal(Type)($input, 0);
  $1 = SWIG_CheckState(res);
}
%typemap(in,noblock=1,fragment=SWIG_AsVal_frag(Type))
Type (Type val, int ecode = 0) {
  ecode = SWIG_AsVal(Type)($input, &val);
  if (!SWIG_IsOK(ecode)) %argument_fail(ecode, "$ltype", $symname, $argnum);
  $1 = %static_cast(val,$ltype);
}
%typemap(in,noblock=1,fragment=SWIG_AsVal_frag(Type))
const Type & ($*ltype temp, Type val, int ecode = 0) {
  ecode = SWIG_AsVal(Type)($input, &val);
  if (!SWIG_IsOK(ecode)) %argument_fail(ecode, "$*ltype", $symname, $argnum);
  temp = %static_cast(val,$*ltype);
  $1 = &temp;
}
%typemap(freearg) Type, const Type & "";
/* INPUT pointer/reference typemap */
%typemap(typecheck,
         precedence=%checkcode(CheckCode),
         fragment=SWIG_AsVal_frag(Type))
Type *INPUT, Type &INPUT {
  int res = SWIG_AsVal(Type)($input, 0);
  $1 = SWIG_CheckState(res);
}
%typemap(in,noblock=1,fragment=SWIG_AsVal_frag(Type))
Type *INPUT ($*ltype temp, int res = 0) {
  res = SWIG_AsVal(Type)($input, &temp);
  if (!SWIG_IsOK(res)) %argument_fail(res, "$*ltype",$symname, $argnum);
  $1 = &temp;
}
%typemap(in,noblock=1,fragment=SWIG_AsVal_frag(Type))
Type &INPUT ($*ltype temp, int res = 0) {
  res = SWIG_AsVal(Type)($input, &temp);
  if (!SWIG_IsOK(res)) %argument_fail(res, "$*ltype",$symname, $argnum);
  $1 = &temp;
}
%typemap(freearg) Type *INPUT, Type &INPUT "";
/* OUTPUT pointer/reference typemap */
%typemap(in,numinputs=0,noblock=1)
 Type *OUTPUT ($*ltype temp=0) "$1 = &temp;";
%typemap(in,numinputs=0,noblock=1)
Type &OUTPUT ($*ltype temp=0)  "$1 = &temp;";
%typemap(argout,noblock=1,fragment=SWIG_From_frag(Type))
Type* OUTPUT, Type &OUTPUT { %append_output(SWIG_From(Type)((*$1))); }
%typemap(freearg) Type *OUTPUT "";
/* INOUT pointer/reference typemap */
%typemap(typecheck) Type *INOUT = Type *INPUT;
%typemap(in)        Type *INOUT = Type *INPUT;
%typemap(argout)    Type *INOUT = Type *OUTPUT;
%typemap(freearg)   Type *INOUT "";
%typemap(typecheck) Type &INOUT = Type &INPUT;
%typemap(in)        Type &INOUT = Type &INPUT;
%typemap(argout)    Type &INOUT = Type &OUTPUT;
%typemap(freearg)   Type &INOUT "";
/* default typemap for pointer argument */
%apply Type *OUTPUT { Type * }
%enddef /* %petsc4py_numt */

/* ---------------------------------------------------------------- */



/* ---------------------------------------------------------------- */
/* Enumerations                                                     */
/* ---------------------------------------------------------------- */

%define SWIG_TYPECHECK_PETSC_ENUM SWIG_TYPECHECK_INT32 %enddef

%fragment(SWIG_From_frag(PetscEnum),"header",
          fragment=SWIG_From_frag(int))
{
SWIGINTERN SWIG_Object
SWIG_From_dec(PetscEnum)(PetscEnum val) {
  return SWIG_From(int)((PetscEnum)val);
}
}
%fragment(SWIG_AsVal_frag(PetscEnum),"header",
          fragment=SWIG_AsVal_frag(int))
{
SWIGINTERN int
SWIG_AsVal_dec(PetscEnum)(SWIG_Object obj, PetscEnum *val) {
  int v; int res = SWIG_AsVal(int)(obj, &v);
  if (SWIG_IsOK(res) && val) *val = %static_cast(v,PetscEnum);
  return res;
}
}

%typemaps_primitive(%checkcode(PETSC_ENUM), PetscEnum);

%typemap(in,numinputs=0)
PetscEnum *OUTPUT ($*ltype temp) "$1 = &temp;"
%typemap(argout,noblock=1,fragment=SWIG_From_frag(PetscEnum))
PetscEnum *OUTPUT { %append_output(SWIG_From(PetscEnum)(%static_cast(*$1,PetscEnum))); }
%apply PetscEnum *INPUT { PetscEnum const * }
%typemap(argout) PetscEnum const* "";
%apply PetscEnum *OUTPUT { PetscEnum * }

%define %petsc4py_enum(EnumType)
%apply PetscEnum         { EnumType         }
%apply PetscEnum *       { EnumType *       }
%apply PetscEnum *INPUT  { EnumType *INPUT  }
%apply PetscEnum *OUTPUT { EnumType *OUTPUT }
%apply PetscEnum *INOUT  { EnumType *INOUT  }
%enddef

/* ---------------------------------------------------------------- */


%define %petsc4py_fragments(Pkg, PyType, Type, OBJECT_DEFAULT)
/* AsVal */
/* ----- */
%fragment(SWIG_AsVal_frag(Type),"header")
{
SWIGINTERN int
SWIG_AsVal_dec(Type)(SWIG_Object input, Type *v) {
  if (input == Py_None) {
    if (v) *v = OBJECT_DEFAULT;
    return SWIG_OK;
  } else if (PyObject_TypeCheck(input,&Py##Pkg##PyType##_Type)) {
    if (v) *v = Py##Pkg##PyType##_Get(input);
    return SWIG_OK;
  } else {
    void *argp = 0;
    int res = SWIG_ConvertPtr(input,&argp,%descriptor(p_##Type), 0);
    if (!SWIG_IsOK(res)) return res;
    if (!argp) return SWIG_ValueError;
    if (v) *v = *(%static_cast(argp,Type*));
    return SWIG_OK;
  }
}
}
/* AsPtr */
/* ----- */
%fragment(SWIG_AsPtr_frag(Type),"header",
          fragment=%fragment_name(GetPtr,Type))
{
SWIGINTERN int
SWIG_AsPtr_dec(Type)(SWIG_Object input, Type **p) {
  if (input == Py_None) {
    if (p) *p = 0;
    return SWIG_OK;
  } else if (PyObject_TypeCheck(input,&Py##Pkg##PyType##_Type)) {
    if (p) *p = Py##Pkg##PyType##_GetPtr(input);
    return SWIG_OK;
  } else {
    void *argp = 0;
    int res = SWIG_ConvertPtr(input,&argp,%descriptor(p_##Type), 0);
    if (!SWIG_IsOK(res)) return res;
    if (!argp) return SWIG_ValueError;
    if (p) *p = %static_cast(argp,Type*);
    return SWIG_OK;
  }
}
}
/* From */
/* ---- */
%fragment(SWIG_From_frag(Type),"header")
{
SWIGINTERN SWIG_Object
SWIG_From_dec(Type)(Type v) {
  return Py##Pkg##PyType##_New(v);
}
}
%enddef /*petsc4py_fragments*/



/* ---------------------------------------------------------------- */
/* MPI Communicator                                                 */
/* ---------------------------------------------------------------- */

%define SWIG_TYPECHECK_MPI_COMM 600 %enddef

%define %petsc4py_comm(Pkg, PyType, Type, CODE, OBJECT_NULL)

/* pointer type */
%types(Type*);  /* XXX find better way */
/* fragments */
%fragment(%fragment_name(GetPtr,MPI_Comm),"header") { }
%petsc4py_fragments(Pkg, PyType, Type, PETSC_COMM_WORLD)
/* base typemaps */
%typemaps_asvalfromn(%checkcode(MPI_COMM), Type);
/* custom typemaps */
%typemap(check,noblock=1) Type {
  if ($1 == OBJECT_NULL)
    %argument_nullref("$ltype",$symname,$argnum);
}

%enddef  /* %petsc4py_comm */


/* ---------------------------------------------------------------- */
/* PETSc Objects                                                    */
/* ---------------------------------------------------------------- */

%define SWIG_TYPECHECK_PETSC_OBJECT  500 %enddef
%define SWIG_TYPECHECK_PETSC_VIEWER  501 %enddef
%define SWIG_TYPECHECK_PETSC_RANDOM  502 %enddef

%define SWIG_TYPECHECK_PETSC_IS            510 %enddef
%define SWIG_TYPECHECK_PETSC_IS_LTOGM      511 %enddef
%define SWIG_TYPECHECK_PETSC_SF            512 %enddef
%define SWIG_TYPECHECK_PETSC_VEC           513 %enddef
%define SWIG_TYPECHECK_PETSC_VEC_SCATTER   514 %enddef
%define SWIG_TYPECHECK_PETSC_SECTION       515 %enddef

%define SWIG_TYPECHECK_PETSC_MAT           520 %enddef
%define SWIG_TYPECHECK_PETSC_MAT_NULLSPACE 521 %enddef

%define SWIG_TYPECHECK_PETSC_KSP           530 %enddef
%define SWIG_TYPECHECK_PETSC_PC            531 %enddef
%define SWIG_TYPECHECK_PETSC_SNES          532 %enddef
%define SWIG_TYPECHECK_PETSC_TS            533 %enddef
%define SWIG_TYPECHECK_PETSC_TAO           534 %enddef

%define SWIG_TYPECHECK_PETSC_AO            540 %enddef
%define SWIG_TYPECHECK_PETSC_DM            541 %enddef
%define SWIG_TYPECHECK_PETSC_DS            542 %enddef
%define SWIG_TYPECHECK_PETSC_PARTITIONER   543 %enddef


%define %petsc4py_objt(Pkg, PyType, Type, CODE)

/* pointer type */
%types(Type*); /* XXX find better way */
/* fragments */
%fragment(%fragment_name(GetPtr,Type),"header")
{ /* XXX implement this better*/
%define_as(Py##Pkg##PyType##_GetPtr(ob), (Type *)PyPetscObject_GetPtr(ob))
}
%petsc4py_fragments(Pkg, PyType, Type, NULL)
/* base typemaps */
%typemaps_asptrfromn(%checkcode(CODE), Type);


/* Custom Typemaps */
/* --------------- */

/* freearg */
%typemap(freearg) Type, Type*, Type& "";

/* check */
%typemap(check,noblock=1) Type INPUT {
  if ($1 == NULL)
    %argument_nullref("$type", $symname, $argnum);
}

/* input pointer */
%typemap(in,fragment=SWIG_AsPtr_frag(Type)) Type *INPUT (int res = SWIG_OLDOBJ) {
  Type *ptr = (Type *)0;
  res = SWIG_AsPtr(Type)($input, &ptr);
  if (!SWIG_IsOK(res)) { %argument_fail(res,"$type", $symname, $argnum); }
  $1 = ptr;
}
%typemap(check,noblock=1) Type *INPUT {
  if ($1 == NULL || (*$1) == NULL)
    %argument_nullref("$type", $symname, $argnum);
}

/* input reference */
%apply Type *INPUT { Type& }

/* optional value */
%typemap(arginit) Type OPTIONAL "$1 = NULL;"
%typemap(in,fragment=SWIG_AsPtr_frag(Type)) Type OPTIONAL (int res = 0) {
  Type *ptr = (Type *)0;
  res = SWIG_AsPtr(Type)($input, &ptr);
  if (!SWIG_IsOK(res)) { %argument_fail(res, "$type", $symname, $argnum); }
  if (ptr) $1 = *ptr;
}

/* optional reference */
%typemap(in,fragment=SWIG_AsPtr_frag(Type)) Type& OPTIONAL (int res = 0) {
  Type *ptr = (Type *)0;
  res = SWIG_AsPtr(Type)($input, &ptr);
  if (!SWIG_IsOK(res)) { %argument_fail(res, "$type", $symname, $argnum); }
  if (!ptr) %argument_nullref("$type", $symname, $argnum);
  $1 = ptr;
  if (SWIG_IsNewObj(res)) %delete(ptr);
}

%typemap(in,numinputs=0) Type* OUTREF, Type* OUTNEW
($*ltype temp = NULL) "$1 = &temp;";
%typemap(freearg) Type* OUTREF, Type* OUTNEW  "";
%typemap(check) Type* OUTREF, Type* OUTNEW  "";
%typemap(argout) Type* OUTREF {
  SWIG_Object o = Py##Pkg##PyType##_New(*$1);
  %append_output(o);
}
%typemap(argout) Type* OUTNEW {
  SWIG_Object o = Py##Pkg##PyType##_New(*$1);
  if (o!=NULL) PetscObjectDereference((PetscObject)(*$1));
  %append_output(o);
}


%apply Type  OPTIONAL { Type  MAYBE  }
%apply Type& OPTIONAL { Type& MAYBE  }
%apply Type* OUTNEW   { Type* NEWOBJ }
%apply Type* OUTREF   { Type* NEWREF }

%enddef /* %petsc4py_objt */


/* ---------------------------------------------------------------- */
/*                                                                  */
/* ---------------------------------------------------------------- */

%petsc4py_errt( Petsc, Error , PetscErrorCode )

%petsc4py_numt( Petsc , Int     , PetscInt     , PETSC_INT     , 0 )
%petsc4py_numt( Petsc , Real    , PetscReal    , PETSC_REAL    , 0 )
%petsc4py_numt( Petsc , Complex , PetscComplex , PETSC_COMPLEX , 0 )
%petsc4py_numt( Petsc , Scalar  , PetscScalar  , PETSC_SCALAR  , 0 )

%petsc4py_comm( Petsc, Comm , MPI_Comm , MPI_COMM , MPI_COMM_NULL )

%petsc4py_objt( Petsc , Object      , PetscObject            , PETSC_OBJECT        )
%petsc4py_objt( Petsc , Viewer      , PetscViewer            , PETSC_VIEWER        )
%petsc4py_objt( Petsc , Random      , PetscRandom            , PETSC_RANDOM        )
%petsc4py_objt( Petsc , IS          , IS                     , PETSC_IS            )
%petsc4py_objt( Petsc , LGMap       , ISLocalToGlobalMapping , PETSC_IS_LTOGM      )
%petsc4py_objt( Petsc , SF          , PetscSF                , PETSC_SF            )
%petsc4py_objt( Petsc , Vec         , Vec                    , PETSC_VEC           )
%petsc4py_objt( Petsc , Scatter     , VecScatter             , PETSC_VEC_SCATTER   )
%petsc4py_objt( Petsc , Section     , PetscSection           , PETSC_SECTION       )
%petsc4py_objt( Petsc , Mat         , Mat                    , PETSC_MAT           )
%petsc4py_objt( Petsc , NullSpace   , MatNullSpace           , PETSC_MAT_NULLSPACE )
%petsc4py_objt( Petsc , KSP         , KSP                    , PETSC_KSP           )
%petsc4py_objt( Petsc , PC          , PC                     , PETSC_PC            )
%petsc4py_objt( Petsc , SNES        , SNES                   , PETSC_SNES          )
%petsc4py_objt( Petsc , TS          , TS                     , PETSC_TS            )
%petsc4py_objt( Petsc , TAO         , Tao                    , PETSC_TAO           )
%petsc4py_objt( Petsc , AO          , AO                     , PETSC_AO            )
%petsc4py_objt( Petsc , DM          , DM                     , PETSC_DM            )
%petsc4py_objt( Petsc , DS          , PetscDS                , PETSC_DS            )
%petsc4py_objt( Petsc , Partitioner , PetscPartitioner       , PETSC_PARTITIONER   )

/* ---------------------------------------------------------------- */

/*
 * Local Variables:
 * mode: C
 * End:
 */
