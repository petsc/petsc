/*
    Defines the basic header of all PETSc objects.
*/
#pragma once
#include <petscsys.h>

/* SUBMANSEC = Sys */

#if defined(PETSC_CLANG_STATIC_ANALYZER)
  #define PetscDisableStaticAnalyzerForExpressionUnderstandingThatThisIsDangerousAndBugprone(...)
#else
  #define PetscDisableStaticAnalyzerForExpressionUnderstandingThatThisIsDangerousAndBugprone(...) __VA_ARGS__
#endif

#if PetscDefined(USE_DEBUG) && !PetscDefined(HAVE_THREADSAFETY)
PETSC_INTERN PetscErrorCode PetscStackSetCheck(PetscBool);
PETSC_INTERN PetscErrorCode PetscStackReset(void);
PETSC_INTERN PetscErrorCode PetscStackCopy(PetscStack *, PetscStack *);
PETSC_INTERN PetscErrorCode PetscStackPrint(PetscStack *, FILE *);
#else
  #define PetscStackSetCheck(check)         PETSC_SUCCESS
  #define PetscStackReset()                 PETSC_SUCCESS
  #define PetscStackCopy(stackin, stackout) PETSC_SUCCESS
  #define PetscStackPrint(stack, file)      PETSC_SUCCESS
#endif

/* These are used internally by PETSc ASCII IO routines*/
#include <stdarg.h>
PETSC_EXTERN PetscErrorCode PetscVFPrintfDefault(FILE *, const char[], va_list);

/*
   All major PETSc data structures have a common core; this is defined
   below by PETSCHEADER.

   PetscHeaderCreate() should be used whenever creating a PETSc structure.
*/

/*
   PetscOps: structure of core operations that all PETSc objects support.

      view()            - Is the routine for viewing the entire PETSc object; for
                          example, MatView() is the general matrix viewing routine.
                          This is used by PetscObjectView((PetscObject)obj) to allow
                          viewing any PETSc object.
      destroy()         - Is the routine for destroying the entire PETSc object;
                          for example,MatDestroy() is the general matrix
                          destruction routine.
                          This is used by PetscObjectDestroy((PetscObject*)&obj) to allow
                          destroying any PETSc object.
*/

typedef struct {
  PetscErrorCode (*view)(PetscObject, PetscViewer);
  PetscErrorCode (*destroy)(PetscObject *);
} PetscOps;

/*E
   PetscFortranCallbackType  - Indicates if a Fortran callback stored in a `PetscObject` is associated with the class or the current particular type of the object

  Values:
+ `PETSC_FORTRAN_CALLBACK_CLASS`   - the callback is associated with the class
- `PETSC_FORTRAN_CALLBACK_SUBTYPE` - the callback is associated with the current particular subtype

  Level: developer

  Developer Note:
  The two sets of callbacks are stored in different arrays in the `PetscObject` because the `PETSC_FORTRAN_CALLBACK_SUBTYPE` callbacks must
  be removed whenever the type of the object is changed (because they are not appropriate for other types). The removal is done in
  `PetscObjectChangeTypeName()`.

.seealso: `PetscFortranCallbackFn`, `PetscObjectSetFortranCallback()`, `PetscObjectGetFortranCallback()`, `PetscObjectChangeTypeName()`
E*/
typedef enum {
  PETSC_FORTRAN_CALLBACK_CLASS,
  PETSC_FORTRAN_CALLBACK_SUBTYPE,
  PETSC_FORTRAN_CALLBACK_MAXTYPE
} PetscFortranCallbackType;

typedef size_t PetscFortranCallbackId;
#define PETSC_SMALLEST_FORTRAN_CALLBACK ((PetscFortranCallbackId)1000)
PETSC_EXTERN PetscErrorCode PetscFortranCallbackRegister(PetscClassId, const char *, PetscFortranCallbackId *);
PETSC_EXTERN PetscErrorCode PetscFortranCallbackGetSizes(PetscClassId, PetscFortranCallbackId *, PetscFortranCallbackId *);

/*S
  PetscFortranCallbackFn - A prototype of a Fortran function provided as a callback

  Level: advanced

  Notes:
  `PetscFortranCallbackFn *` plays the role of `void *` for function pointers in the PETSc Fortran API.

.seealso: `PetscVoidFn`, `PetscErrorCodeFn`
S*/
PETSC_EXTERN_TYPEDEF typedef void(PetscFortranCallbackFn)(void);

typedef struct {
  PetscFortranCallbackFn *func;
  void                   *ctx;
} PetscFortranCallback;

/*
   All PETSc objects begin with the fields defined in PETSCHEADER.
   The PetscObject is a way of examining these fields regardless of
   the specific object. In C++ this could be a base abstract class
   from which all objects are derived.
*/
#define PETSC_MAX_OPTIONS_HANDLER 5
typedef struct _p_PetscObject {
  PetscOps      bops[1];
  PetscClassId  classid;
  MPI_Comm      comm;
  PetscObjectId id; /* this is used to compare object for identity that may no longer exist since memory addresses get recycled for new objects */
  PetscInt      refct;
  PetscErrorCode (*non_cyclic_references)(PetscObject, PetscInt *);
  PetscInt64        cidx;
  PetscMPIInt       tag;
  PetscFunctionList qlist;
  PetscObjectList   olist;
  char             *class_name; /*  for example, "Vec" */
  char             *description;
  char             *mansec;
  char             *type_name; /*  this is the subclass, for example VECSEQ which equals "seq" */
  char             *name;
  char             *prefix;
  PetscInt          tablevel;
  void             *cpp;
  PetscObjectState  state;
  PetscInt          int_idmax, intstar_idmax;
  PetscObjectState *intcomposedstate, *intstarcomposedstate;
  PetscInt         *intcomposeddata, **intstarcomposeddata;
  PetscInt          real_idmax, realstar_idmax;
  PetscObjectState *realcomposedstate, *realstarcomposedstate;
  PetscReal        *realcomposeddata, **realstarcomposeddata;
#if PetscDefined(USE_COMPLEX)
  PetscInt          scalar_idmax, scalarstar_idmax;
  PetscObjectState *scalarcomposedstate, *scalarstarcomposedstate;
  PetscScalar      *scalarcomposeddata, **scalarstarcomposeddata;
#endif
  PetscFortranCallbackFn **fortran_func_pointers;     /* used by Fortran interface functions to stash user provided Fortran functions */
  PetscFortranCallbackId   num_fortran_func_pointers; /* number of Fortran function pointers allocated */
  PetscFortranCallback    *fortrancallback[PETSC_FORTRAN_CALLBACK_MAXTYPE];
  PetscFortranCallbackId   num_fortrancallback[PETSC_FORTRAN_CALLBACK_MAXTYPE];
  void                    *python_context;
  PetscErrorCode (*python_destroy)(void *);

  PetscInt noptionhandler;
  PetscErrorCode (*optionhandler[PETSC_MAX_OPTIONS_HANDLER])(PetscObject, PetscOptionItems, void *);
  PetscErrorCode (*optiondestroy[PETSC_MAX_OPTIONS_HANDLER])(PetscObject, void *);
  void *optionctx[PETSC_MAX_OPTIONS_HANDLER];
#if defined(PETSC_HAVE_SAWS)
  PetscBool amsmem;          /* if PETSC_TRUE then this object is registered with SAWs and visible to clients */
  PetscBool amspublishblock; /* if PETSC_TRUE and publishing objects then will block at PetscObjectSAWsBlock() */
#endif
  PetscOptions options; /* options database used, NULL means default */
  PetscBool    optionsprinted;
  PetscBool    donotPetscObjectPrintClassNamePrefixType;
} _p_PetscObject;

#define PETSCHEADER(ObjectOps) \
  _p_PetscObject hdr; \
  ObjectOps      ops[1]

#define PETSCFREEDHEADER -1

/*S
  PetscObjectDestroyFn - A prototype of a function that can destroy a `PetscObject`

  Calling Sequence:
. obj - the `PetscObject` to destroy

  Level: beginner

  Note:
  The deprecated `PetscObjectDestroyFunction` works as a replacement for `PetscObjectDestroyFn` *.

.seealso: `PetscObject`, `PetscObjectDestroy()`
S*/
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode PetscObjectDestroyFn(PetscObject *obj);

PETSC_EXTERN_TYPEDEF typedef PetscObjectDestroyFn *PetscObjectDestroyFunction;

/*S
  PetscObjectViewFn - A prototype of a function that can view a `PetscObject`

  Calling Sequence:
+ obj - the `PetscObject` to view
- v - the viewer

  Level: beginner

  Note:
  The deprecated `PetscObjectViewFunction` works as a replacement for `PetscObjectViewFn` *.

.seealso: `PetscObject`, `PetscObjectDestroy()`, `PetscViewer`, `PetscObjectView()`
S*/
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode PetscObjectViewFn(PetscObject obj, PetscViewer v);

PETSC_EXTERN_TYPEDEF typedef PetscObjectViewFn *PetscObjectViewFunction;

/*MC
    PetscHeaderCreate - Creates a raw PETSc object of a particular class

  Synopsis:
  #include <petsc/private/petscimpl.h>
  PetscErrorCode PetscHeaderCreate(PetscObject h, PetscClassId classid, const char class_name[], const char descr[], const char mansec[], MPI_Comm comm, PetscObjectDestroyFn * destroy, PetscObjectViewFn * view)

  Collective

  Input Parameters:
+ classid    - The classid associated with this object (for example `VEC_CLASSID`)
. class_name - String name of class; should be static (for example "Vec"), may be `PETSC_NULLPTR`
. descr      - String containing short description; should be static (for example "Vector"), may be `PETSC_NULLPTR`
. mansec     - String indicating section in manual pages; should be static (for example "Vec"), may be `PETSC_NULLPTR`
. comm       - The MPI Communicator
. destroy    - The destroy routine for this object (for example `VecDestroy()`)
- view       - The view routine for this object (for example `VecView()`), may be `PETSC_NULLPTR`

  Output Parameter:
. h - The newly created `PetscObject`

  Level: developer

  Notes:
  Can only be used to create a `PetscObject`. A `PetscObject` is defined as a pointer to a
  C/C++ structure which satisfies all of the following\:

  1. The first member of the structure must be a `_p_PetscObject`.
  2. C++ structures must be "Standard Layout". Generally speaking a standard layout class\:
     - Has no virtual functions or base classes.
     - Has only standard layout non-static members (if any).
     - Has only standard layout base classes (if any).

     See https://en.cppreference.com/w/cpp/language/classes#Standard-layout_class for further
     information.

  Example Usage:
  Existing `PetscObject`s may be easily created as shown. Unless otherwise stated, a particular
  objects `destroy` and `view` functions are exactly `<OBJECT_TYPE>Destroy()` and
  `<OBJECT_TYPE>View()`.
.vb
  Vec v;

  PetscHeaderCreate(v, VEC_CLASSID, "Vec", "A distributed vector class", "Vec", PETSC_COMM_WORLD, VecDestroy, VecView);
.ve

  It is possible to create custom `PetscObject`s, note however that they must abide by the
  restrictions set forth above.
.vb
  // OK, first member of C structure is _p_PetscObject
  struct MyCPetscObject_s
  {
    _p_PetscObject header;
    int            some_data;
  };
  typedef struct *MyCPetscObject_s MyCPetscObject;

  PetscErrorCode MyObjectDestroy(MyObject *);
  PetscErrorCode MyObjectView(MyObject);

  MyCPetscObject obj;

  // assume MY_PETSC_CLASSID is already registered
  PetscHeaderCreate(obj, MY_PETSC_CLASSID, "MyObject", "A custom PetscObject", PETSC_NULLPTR, PETSC_COMM_SELF, MyObjectDestroy, MyObjectView);

  // OK, only destroy function must be given, all others may be NULL
  PetscHeaderCreate(obj, MY_PETSC_CLASSID, PETSC_NULLPTR, PETSC_NULLPTR, PETSC_NULLPTR, PETSC_COMM_SELF, MyObjectDestroy, PETSC_NULLPTR);

  // ERROR must be a single-level pointer
  PetscHeaderCreate(&obj, ...);
.ve

  Illustrating proper construction from C++\:
.vb
  // ERROR, class is not standard layout, first member must be publicly accessible
  class BadCppPetscObject
  {
    _p_PetscObject header;
  };

  // ERROR, class is not standard layout, has a virtual function and virtual inheritance
  class BadCppPetscObject2 : virtual BaseClass
  {
  public:
    _p_PetscObject header;

    virtual void foo();
  };

  // ERROR, class is not standard layout! Has non-standard layout member
  class BadCppPetscObject2
  {
  public:
    _p_PetscObject    header;
    BadCppPetscObject non_standard_layout;
  };

  // OK, class is standard layout!
  class GoodCppPetscObject;
  using MyCppObject = GoodCppPetscObject *;

  // OK, non-virtual inheritance of other standard layout class does not affect layout
  class GoodCppPetscObject : StandardLayoutClass
  {
  public:
    // OK, non standard layout member is static, does not affect layout
    static BadCppPetscObject non_standard_layout;

    // OK, first non-static member is _p_PetscObject
    _p_PetscObject header;

     // OK, non-virtual member functions do not affect class layout
    void foo();

    // OK, may use "member" functions for destroy and view so long as they are static
    static PetscErrorCode destroy(MyCppObject *);
    static PetscErrorCode view(MyCppObject);
  };

  // OK, usage via pointer
  MyObject obj;

  PetscHeaderCreate(obj, MY_PETSC_CLASSID, "MyObject", "A custom C++ PetscObject", nullptr, PETSC_COMM_SELF, GoodCppPetscObject::destroy, GoodCppPetscObject::view);
.ve

.seealso: `PetscObject`, `PetscHeaderDestroy()`, `PetscClassIdRegister()`
M*/
#define PetscHeaderCreate(h, classid, class_name, descr, mansec, comm, destroy, view) \
  PetscHeaderCreate_Function(PetscNew(&(h)), (PetscObject *)&(h), (classid), (class_name), (descr), (mansec), (comm), (PetscObjectDestroyFn *)(destroy), (PetscObjectViewFn *)(view))

PETSC_EXTERN PetscErrorCode PetscHeaderCreate_Function(PetscErrorCode, PetscObject *, PetscClassId, const char[], const char[], const char[], MPI_Comm, PetscObjectDestroyFn *, PetscObjectViewFn *);
PETSC_EXTERN PetscErrorCode PetscHeaderCreate_Private(PetscObject, PetscClassId, const char[], const char[], const char[], MPI_Comm, PetscObjectDestroyFn *, PetscObjectViewFn *);
PETSC_EXTERN PetscErrorCode PetscHeaderDestroy_Function(PetscObject *);
PETSC_EXTERN PetscErrorCode PetscComposedQuantitiesDestroy(PetscObject obj);
PETSC_INTERN PetscObjectId  PetscObjectNewId_Internal(void);

/*MC
  PetscHeaderDestroy - Final step in destroying a `PetscObject`

  Synopsis:
  #include <petsc/private/petscimpl.h>
  PetscErrorCode PetscHeaderDestroy(PetscObject *obj)

  Collective

  Input Parameter:
. h - A pointer to the header created with `PetscHeaderCreate()`

  Level: developer

  Notes:
  `h` is freed and set to `PETSC_NULLPTR` when this routine returns.

  Example Usage:
.vb
  PetscObject obj;

  PetscHeaderCreate(obj, ...);
  // use obj...

  // note pointer to obj is used
  PetscHeaderDestroy(&obj);
.ve

  Note that this routine is the _last_ step when destroying higher-level `PetscObject`s as it
  deallocates the memory for the structure itself\:
.vb
  typedef struct MyPetscObject_s *MyPetscObject;
  struct MyPetscObject_s
  {
    _p_PetscObject  hdr;
    PetscInt       *foo;
    PetscScalar    *bar;
  };

  // assume obj is created/initialized elsewhere...
  MyPetscObject obj;

  // OK, should dispose of all dynamically allocated resources before calling
  // PetscHeaderDestroy()
  PetscFree(obj->foo);

  // OK, dispose of obj
  PetscHeaderDestroy(&obj);

  // ERROR, obj points to NULL here, accessing obj->bar may result in segmentation violation!
  // obj->bar is potentially leaked!
  PetscFree(obj->bar);
.ve

.seealso: `PetscObject`, `PetscHeaderCreate()`
M*/
#define PetscHeaderDestroy(h) PetscHeaderDestroy_Function((PetscObject *)h)

PETSC_EXTERN PetscErrorCode                PetscHeaderDestroy_Private(PetscObject, PetscBool);
PETSC_INTERN PetscErrorCode                PetscHeaderDestroy_Private_Unlogged(PetscObject, PetscBool);
PETSC_SINGLE_LIBRARY_INTERN PetscErrorCode PetscHeaderReset_Internal(PetscObject);
PETSC_EXTERN PetscErrorCode                PetscObjectCopyFortranFunctionPointers(PetscObject, PetscObject);
PETSC_EXTERN PetscErrorCode                PetscObjectSetFortranCallback(PetscObject, PetscFortranCallbackType, PetscFortranCallbackId *, PetscFortranCallbackFn *, void *ctx);
PETSC_EXTERN PetscErrorCode                PetscObjectGetFortranCallback(PetscObject, PetscFortranCallbackType, PetscFortranCallbackId, PetscFortranCallbackFn **, void **ctx);

PETSC_INTERN PetscErrorCode PetscCitationsInitialize(void);
PETSC_INTERN PetscErrorCode PetscFreeMPIResources(void);
PETSC_INTERN PetscErrorCode PetscOptionsHasHelpIntro_Internal(PetscOptions, PetscBool *);

/* Code shared between C and Fortran */
PETSC_INTERN PetscErrorCode PetscInitialize_Common(const char *, const char *, const char *, PetscBool, PetscInt);

#if PetscDefined(HAVE_SETJMP_H)
PETSC_EXTERN PetscBool PetscCheckPointer(const void *, PetscDataType);
#else
  #define PetscCheckPointer(ptr, data_type) (ptr ? PETSC_TRUE : PETSC_FALSE)
#endif

#if defined(PETSC_CLANG_STATIC_ANALYZER)
template <typename T>
extern void PetscValidHeaderSpecificType(T, PetscClassId, int, const char[]);
template <typename T>
extern void PetscValidHeaderSpecific(T, PetscClassId, int);
template <typename T>
extern void PetscValidHeader(T, int);
template <typename T>
extern void PetscAssertPointer(T, int)
{
}
template <typename T>
extern void PetscValidFunction(T, int);
#else
  // Macros to test if a PETSc object is valid and if pointers are valid
  #if PetscDefined(USE_DEBUG)
    /*  This check is for subtype methods such as DMDAGetCorners() that do not use the PetscTryMethod() or PetscUseMethod() paradigm */
    #define PetscValidHeaderSpecificType(h, ck, arg, t) \
      do { \
        PetscBool _7_same; \
        PetscValidHeaderSpecific(h, ck, arg); \
        PetscCall(PetscObjectTypeCompare((PetscObject)(h), t, &_7_same)); \
        PetscCheck(_7_same, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Wrong subtype object:Parameter # %d must have implementation %s it is %s", arg, t, ((PetscObject)(h))->type_name); \
      } while (0)

    #define PetscAssertPointer_Internal(ptr, arg, ptype, ptrtype) \
      do { \
        PetscCheck(ptr, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Null Pointer: Parameter # %d", arg); \
        PetscCheck(PetscCheckPointer(ptr, ptype), PETSC_COMM_SELF, PETSC_ERR_ARG_BADPTR, "Invalid Pointer to %s: Argument '" PetscStringize(ptr) "' (parameter # %d)", ptrtype, arg); \
      } while (0)

    #define PetscValidHeaderSpecific(h, ck, arg) \
      do { \
        PetscAssertPointer_Internal(h, arg, PETSC_OBJECT, "PetscObject"); \
        if (((PetscObject)(h))->classid != ck) { \
          PetscCheck(((PetscObject)(h))->classid != PETSCFREEDHEADER, PETSC_COMM_SELF, PETSC_ERR_ARG_CORRUPT, "Object already free: Parameter # %d", arg); \
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Wrong type of object: Parameter # %d", arg); \
        } \
      } while (0)

    #define PetscValidHeader(h, arg) \
      do { \
        PetscAssertPointer_Internal(h, arg, PETSC_OBJECT, "PetscObject"); \
        PetscCheck(((PetscObject)(h))->classid != PETSCFREEDHEADER, PETSC_COMM_SELF, PETSC_ERR_ARG_CORRUPT, "Object already free: Parameter # %d", arg); \
        PetscCheck(((PetscObject)(h))->classid >= PETSC_SMALLEST_CLASSID && ((PetscObject)(h))->classid <= PETSC_LARGEST_CLASSID, PETSC_COMM_SELF, PETSC_ERR_ARG_CORRUPT, "Invalid type of object: Parameter # %d", arg); \
      } while (0)

    #if defined(__cplusplus)
      #include <type_traits> // std::decay

namespace Petsc
{

namespace util
{

template <typename T>
struct PetscAssertPointerImpl {
  PETSC_NODISCARD static constexpr PetscDataType type() noexcept { return PETSC_CHAR; }
  PETSC_NODISCARD static constexpr const char   *string() noexcept { return "memory"; }
};

      #define PETSC_ASSERT_POINTER_IMPL_SPECIALIZATION(T, PETSC_TYPE) \
        template <> \
        struct PetscAssertPointerImpl<T *> { \
          PETSC_NODISCARD static constexpr PetscDataType type() noexcept { return PETSC_TYPE; } \
          PETSC_NODISCARD static constexpr const char   *string() noexcept { return PetscStringize(T); } \
        }; \
        template <> \
        struct PetscAssertPointerImpl<const T *> : PetscAssertPointerImpl<T *> { }; \
        template <> \
        struct PetscAssertPointerImpl<volatile T *> : PetscAssertPointerImpl<T *> { }; \
        template <> \
        struct PetscAssertPointerImpl<const volatile T *> : PetscAssertPointerImpl<T *> { }

PETSC_ASSERT_POINTER_IMPL_SPECIALIZATION(char, PETSC_CHAR);
PETSC_ASSERT_POINTER_IMPL_SPECIALIZATION(signed char, PETSC_CHAR);
PETSC_ASSERT_POINTER_IMPL_SPECIALIZATION(unsigned char, PETSC_CHAR);
PETSC_ASSERT_POINTER_IMPL_SPECIALIZATION(short, PETSC_SHORT);
PETSC_ASSERT_POINTER_IMPL_SPECIALIZATION(unsigned short, PETSC_SHORT);
PETSC_ASSERT_POINTER_IMPL_SPECIALIZATION(PetscBool, PETSC_BOOL);
PETSC_ASSERT_POINTER_IMPL_SPECIALIZATION(float, PETSC_FLOAT);
PETSC_ASSERT_POINTER_IMPL_SPECIALIZATION(double, PETSC_DOUBLE);
PETSC_ASSERT_POINTER_IMPL_SPECIALIZATION(int32_t, PETSC_INT32);
PETSC_ASSERT_POINTER_IMPL_SPECIALIZATION(uint32_t, PETSC_INT32);
PETSC_ASSERT_POINTER_IMPL_SPECIALIZATION(int64_t, PETSC_INT64);
PETSC_ASSERT_POINTER_IMPL_SPECIALIZATION(uint64_t, PETSC_INT64);
      #if defined(PETSC_HAVE_COMPLEX)
PETSC_ASSERT_POINTER_IMPL_SPECIALIZATION(PetscComplex, PETSC_COMPLEX);
      #endif

      #undef PETSC_ASSERT_POINTER_IMPL_SPECIALIZATION

} // namespace util

} // namespace Petsc

      #define PetscAssertPointer_PetscDataType(h) ::Petsc::util::PetscAssertPointerImpl<typename std::decay<decltype(h)>::type>::type()
      #define PetscAssertPointer_String(h)        ::Petsc::util::PetscAssertPointerImpl<typename std::decay<decltype(h)>::type>::string()

    #elif PETSC_C_VERSION >= 11
      #define PETSC_GENERIC_CV(type, result) type * : result, const type * : result, volatile type * : result, const volatile type * : result

      #if PetscDefined(HAVE_COMPLEX)
        #define PETSC_GENERIC_CV_COMPLEX(result) , PETSC_GENERIC_CV(PetscComplex, result)
      #else
        #define PETSC_GENERIC_CV_COMPLEX(result)
      #endif

      #define PetscAssertPointer_PetscDataType(h) \
        _Generic((h), \
          default: PETSC_CHAR, \
          PETSC_GENERIC_CV(          char, PETSC_CHAR), \
          PETSC_GENERIC_CV(   signed char, PETSC_CHAR), \
          PETSC_GENERIC_CV( unsigned char, PETSC_CHAR), \
          PETSC_GENERIC_CV(         short, PETSC_SHORT), \
          PETSC_GENERIC_CV(unsigned short, PETSC_SHORT), \
          PETSC_GENERIC_CV(         float, PETSC_FLOAT), \
          PETSC_GENERIC_CV(        double, PETSC_DOUBLE), \
          PETSC_GENERIC_CV(       int32_t, PETSC_INT32), \
          PETSC_GENERIC_CV(      uint32_t, PETSC_INT32), \
          PETSC_GENERIC_CV(       int64_t, PETSC_INT64), \
          PETSC_GENERIC_CV(      uint64_t, PETSC_INT64) \
          PETSC_GENERIC_CV_COMPLEX(PETSC_COMPLEX))

      #define PETSC_GENERIC_CV_STRINGIZE(type) PETSC_GENERIC_CV(type, PetscStringize(type))

      #if PetscDefined(HAVE_COMPLEX)
        #define PETSC_GENERIC_CV_STRINGIZE_COMPLEX , PETSC_GENERIC_CV_STRINGIZE(PetscComplex)
      #else
        #define PETSC_GENERIC_CV_STRINGIZE_COMPLEX
      #endif

      #define PetscAssertPointer_String(h) \
        _Generic((h), \
          default: "memory", \
          PETSC_GENERIC_CV_STRINGIZE(char), \
          PETSC_GENERIC_CV_STRINGIZE(signed char), \
          PETSC_GENERIC_CV_STRINGIZE(unsigned char), \
          PETSC_GENERIC_CV_STRINGIZE(short), \
          PETSC_GENERIC_CV_STRINGIZE(unsigned short), \
          PETSC_GENERIC_CV_STRINGIZE(float), \
          PETSC_GENERIC_CV_STRINGIZE(double), \
          PETSC_GENERIC_CV_STRINGIZE(int32_t), \
          PETSC_GENERIC_CV_STRINGIZE(uint32_t), \
          PETSC_GENERIC_CV_STRINGIZE(int64_t), \
          PETSC_GENERIC_CV_STRINGIZE(uint64_t) \
          PETSC_GENERIC_CV_STRINGIZE_COMPLEX)
    #else // PETSC_C_VERSION >= 11 || defined(__cplusplus)
      #define PetscAssertPointer_PetscDataType(h) PETSC_CHAR
      #define PetscAssertPointer_String(h)        "memory"
    #endif // PETSC_C_VERSION >= 11 || defined(__cplusplus)
    #define PetscAssertPointer(h, arg) PetscAssertPointer_Internal(h, arg, PetscAssertPointer_PetscDataType(h), PetscAssertPointer_String(h))
    #define PetscValidFunction(f, arg) PetscCheck((f), PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Null Function Pointer: Parameter # %d", arg)
  #else // PetscDefined(USE_DEBUG)
    #define PetscValidHeaderSpecific(h, ck, arg) \
      do { \
        (void)(h); \
      } while (0)
    #define PetscValidHeaderSpecificType(h, ck, arg, t) \
      do { \
        (void)(h); \
      } while (0)
    #define PetscValidHeader(h, arg) \
      do { \
        (void)(h); \
      } while (0)
    #define PetscAssertPointer(h, arg) \
      do { \
        (void)(h); \
      } while (0)
    #define PetscValidFunction(h, arg) \
      do { \
        (void)(h); \
      } while (0)
  #endif // PetscDefined(USE_DEBUG)
#endif   // PETSC_CLANG_STATIC_ANALYZER

#define PetscValidPointer(h, arg)       PETSC_DEPRECATED_MACRO(3, 20, 0, "PetscAssertPointer()", ) PetscAssertPointer(h, arg)
#define PetscValidCharPointer(h, arg)   PETSC_DEPRECATED_MACRO(3, 20, 0, "PetscAssertPointer()", ) PetscAssertPointer(h, arg)
#define PetscValidIntPointer(h, arg)    PETSC_DEPRECATED_MACRO(3, 20, 0, "PetscAssertPointer()", ) PetscAssertPointer(h, arg)
#define PetscValidInt64Pointer(h, arg)  PETSC_DEPRECATED_MACRO(3, 20, 0, "PetscAssertPointer()", ) PetscAssertPointer(h, arg)
#define PetscValidCountPointer(h, arg)  PETSC_DEPRECATED_MACRO(3, 20, 0, "PetscAssertPointer()", ) PetscAssertPointer(h, arg)
#define PetscValidBoolPointer(h, arg)   PETSC_DEPRECATED_MACRO(3, 20, 0, "PetscAssertPointer()", ) PetscAssertPointer(h, arg)
#define PetscValidScalarPointer(h, arg) PETSC_DEPRECATED_MACRO(3, 20, 0, "PetscAssertPointer()", ) PetscAssertPointer(h, arg)
#define PetscValidRealPointer(h, arg)   PETSC_DEPRECATED_MACRO(3, 20, 0, "PetscAssertPointer()", ) PetscAssertPointer(h, arg)

#define PetscSorted(n, idx, sorted) \
  do { \
    (sorted) = PETSC_TRUE; \
    for (PetscCount _i_ = 1; _i_ < (n); ++_i_) { \
      if ((idx)[_i_] < (idx)[_i_ - 1]) { \
        (sorted) = PETSC_FALSE; \
        break; \
      } \
    } \
  } while (0)

#if !defined(PETSC_CLANG_STATIC_ANALYZER)
  #if !defined(PETSC_USE_DEBUG)

    #define PetscCheckSameType(a, arga, b, argb) \
      do { \
        (void)(a); \
        (void)(b); \
      } while (0)
    #define PetscCheckTypeName(a, type) \
      do { \
        (void)(a); \
      } while (0)
    #define PetscCheckTypeNames(a, type1, type2) \
      do { \
        (void)(a); \
      } while (0)
    #define PetscValidType(a, arg) \
      do { \
        (void)(a); \
      } while (0)
    #define PetscCheckSameComm(a, arga, b, argb) \
      do { \
        (void)(a); \
        (void)(b); \
      } while (0)
    #define PetscCheckSameTypeAndComm(a, arga, b, argb) \
      do { \
        (void)(a); \
        (void)(b); \
      } while (0)
    #define PetscValidLogicalCollectiveScalar(a, b, arg) \
      do { \
        (void)(a); \
        (void)(b); \
      } while (0)
    #define PetscValidLogicalCollectiveReal(a, b, arg) \
      do { \
        (void)(a); \
        (void)(b); \
      } while (0)
    #define PetscValidLogicalCollectiveInt(a, b, arg) \
      do { \
        (void)(a); \
        (void)(b); \
      } while (0)
    #define PetscValidLogicalCollectiveIntComm(a, b, arg) \
      do { \
        (void)(a); \
        (void)(b); \
      } while (0)
    #define PetscValidLogicalCollectiveCount(a, b, arg) \
      do { \
        (void)(a); \
        (void)(b); \
      } while (0)
    #define PetscValidLogicalCollectiveMPIInt(a, b, arg) \
      do { \
        (void)(a); \
        (void)(b); \
      } while (0)
    #define PetscValidLogicalCollectiveBool(a, b, arg) \
      do { \
        (void)(a); \
        (void)(b); \
      } while (0)
    #define PetscValidLogicalCollectiveEnum(a, b, arg) \
      do { \
        (void)(a); \
        (void)(b); \
      } while (0)
    #define PetscCheckSorted(n, idx) \
      do { \
        (void)(n); \
        (void)(idx); \
      } while (0)

  #else

    /*
  This macro currently does nothing, the plan is for each PetscObject to have a PetscInt "type"
  member associated with the string type_name that can be quickly compared.

  **Do not swap this macro to compare string type_name!**

  This macro is used incorrectly in the code. Many places that do not need identity of the
  types incorrectly call this check and would need to be fixed if this macro is enabled.
*/
    #if 0
      #define PetscCheckSameType(a, arga, b, argb) \
        do { \
          PetscBool pcst_type_eq_ = PETSC_TRUE; \
          PetscCall(PetscStrcmp(((PetscObject)(a))->type_name, ((PetscObject)(b))->type_name, &pcst_type_eq_)); \
          PetscCheck(pcst_type_eq_, PETSC_COMM_SELF, PETSC_ERR_ARG_NOTSAMETYPE, "Objects not of same type : Argument # % d and % d, % s != % s ", arga, argb, ((PetscObject)(a))->type_name, ((PetscObject)(b))->type_name); \
        } while (0)
    #else
      #define PetscCheckSameType(a, arga, b, argb) \
        do { \
          (void)(a); \
          (void)(b); \
        } while (0)
    #endif

    /*
    Check type_name
*/
    #define PetscCheckTypeName(a, type) \
      do { \
        PetscBool _7_match; \
        PetscCall(PetscObjectTypeCompare(((PetscObject)(a)), (type), &_7_match)); \
        PetscCheck(_7_match, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Object (%s) is not %s", ((PetscObject)(a))->type_name, type); \
      } while (0)

    #define PetscCheckTypeNames(a, type1, type2) \
      do { \
        PetscBool _7_match; \
        PetscCall(PetscObjectTypeCompareAny(((PetscObject)(a)), &_7_match, (type1), (type2), "")); \
        PetscCheck(_7_match, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Object (%s) is not %s or %s", ((PetscObject)(a))->type_name, type1, type2); \
      } while (0)

    /*
   Use this macro to check if the type is set
*/
    #define PetscValidType(a, arg) PetscCheck(((PetscObject)(a))->type_name, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "%s object's type is not set: Argument # %d", ((PetscObject)(a))->class_name, arg)

    /*
   Sometimes object must live on same communicator to inter-operate
*/
    #define PetscCheckSameComm(a, arga, b, argb) \
      do { \
        PetscMPIInt _7_flag; \
        PetscCallMPI(MPI_Comm_compare(PetscObjectComm((PetscObject)(a)), PetscObjectComm((PetscObject)(b)), &_7_flag)); \
        PetscCheck(_7_flag == MPI_CONGRUENT || _7_flag == MPI_IDENT, PETSC_COMM_SELF, PETSC_ERR_ARG_NOTSAMECOMM, "Different communicators in the two objects: Argument # %d and %d flag %d", arga, argb, _7_flag); \
      } while (0)

    #define PetscCheckSameTypeAndComm(a, arga, b, argb) \
      do { \
        PetscCheckSameType(a, arga, b, argb); \
        PetscCheckSameComm(a, arga, b, argb); \
      } while (0)

    #define PetscValidLogicalCollectiveScalar(a, b, arg) \
      do { \
        PetscScalar b0 = (b); \
        PetscReal   b1[5]; \
        if (PetscIsNanScalar(b0)) { \
          b1[4] = 1; \
        } else { \
          b1[4] = 0; \
        }; \
        b1[0] = -PetscRealPart(b0); \
        b1[1] = PetscRealPart(b0); \
        b1[2] = -PetscImaginaryPart(b0); \
        b1[3] = PetscImaginaryPart(b0); \
        PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, b1, 5, MPIU_REAL, MPIU_MAX, PetscObjectComm((PetscObject)(a)))); \
        PetscCheck(b1[4] > 0 || (PetscEqualReal(-b1[0], b1[1]) && PetscEqualReal(-b1[2], b1[3])), PetscObjectComm((PetscObject)(a)), PETSC_ERR_ARG_WRONG, "Scalar value must be same on all processes, argument # %d", arg); \
      } while (0)

    #define PetscValidLogicalCollectiveReal(a, b, arg) \
      do { \
        PetscReal b0 = (b), b1[3]; \
        if (PetscIsNanReal(b0)) { \
          b1[2] = 1; \
        } else { \
          b1[2] = 0; \
        }; \
        b1[0] = -b0; \
        b1[1] = b0; \
        PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, b1, 3, MPIU_REAL, MPIU_MAX, PetscObjectComm((PetscObject)(a)))); \
        PetscCheck(b1[2] > 0 || PetscEqualReal(-b1[0], b1[1]), PetscObjectComm((PetscObject)(a)), PETSC_ERR_ARG_WRONG, "Real value must be same on all processes, argument # %d", arg); \
      } while (0)

    #define PetscValidLogicalCollectiveInt(a, b, arg) \
      do { \
        PetscInt b0 = (b), b1[2]; \
        b1[0]       = -b0; \
        b1[1]       = b0; \
        PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, b1, 2, MPIU_INT, MPI_MAX, PetscObjectComm((PetscObject)(a)))); \
        PetscCheck(-b1[0] == b1[1], PetscObjectComm((PetscObject)(a)), PETSC_ERR_ARG_WRONG, "Int value must be same on all processes, argument # %d", arg); \
      } while (0)

    #define PetscValidLogicalCollectiveIntComm(a, b, arg) \
      do { \
        PetscInt b1[2]; \
        b1[0] = -b; \
        b1[1] = b; \
        PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, b1, 2, MPIU_INT, MPI_MAX, a)); \
        PetscCheck(-b1[0] == b1[1], a, PETSC_ERR_ARG_WRONG, "Int value must be same on all processes, argument # %d", arg); \
      } while (0)

    #define PetscValidLogicalCollectiveCount(a, b, arg) \
      do { \
        PetscCount b0 = (b), b1[2]; \
        b1[0]         = -b0; \
        b1[1]         = b0; \
        PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, b1, 2, MPIU_COUNT, MPI_MAX, PetscObjectComm((PetscObject)(a)))); \
        PetscCheck(-b1[0] == b1[1], PetscObjectComm((PetscObject)(a)), PETSC_ERR_ARG_WRONG, "Int value must be same on all processes, argument # %d", arg); \
      } while (0)

    #define PetscValidLogicalCollectiveMPIInt(a, b, arg) \
      do { \
        PetscMPIInt b0 = (b), b1[2]; \
        b1[0]          = -b0; \
        b1[1]          = b0; \
        PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, b1, 2, MPI_INT, MPI_MAX, PetscObjectComm((PetscObject)(a)))); \
        PetscCheck(-b1[0] == b1[1], PetscObjectComm((PetscObject)(a)), PETSC_ERR_ARG_WRONG, "PetscMPIInt value must be same on all processes, argument # %d", arg); \
      } while (0)

    #define PetscValidLogicalCollectiveBool(a, b, arg) \
      do { \
        PetscBool b0 = (PetscBool)(b), b1[2]; \
        b1[0]        = !b0; \
        b1[1]        = b0; \
        PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, b1, 2, MPI_C_BOOL, MPI_LAND, PetscObjectComm((PetscObject)(a)))); \
        PetscCheck(!b1[0] == b1[1], PetscObjectComm((PetscObject)(a)), PETSC_ERR_ARG_WRONG, "Bool value must be same on all processes, argument # %d", arg); \
      } while (0)

    #define PetscValidLogicalCollectiveEnum(a, b, arg) \
      do { \
        PetscMPIInt b0 = (PetscMPIInt)(b), b1[2]; \
        b1[0]          = -b0; \
        b1[1]          = b0; \
        PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, b1, 2, MPI_INT, MPI_MAX, PetscObjectComm((PetscObject)(a)))); \
        PetscCheck(-b1[0] == b1[1], PetscObjectComm((PetscObject)(a)), PETSC_ERR_ARG_WRONG, "Enum value must be same on all processes, argument # %d", arg); \
      } while (0)

    #define PetscCheckSorted(n, idx) \
      do { \
        PetscBool _1_flg; \
        PetscSorted(n, idx, _1_flg); \
        PetscCheck(_1_flg, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Input array needs to be sorted"); \
      } while (0)

  #endif
#else  /* PETSC_CLANG_STATIC_ANALYZER */
template <typename Ta, typename Tb>
extern void PetscCheckSameType(Ta, int, Tb, int);
template <typename Ta, typename Tb>
extern void PetscCheckTypeName(Ta, Tb);
template <typename Ta, typename Tb, typename Tc>
extern void PetscCheckTypeNames(Ta, Tb, Tc);
template <typename T>
extern void PetscValidType(T, int);
template <typename Ta, typename Tb>
extern void PetscCheckSameComm(Ta, int, Tb, int);
template <typename Ta, typename Tb>
extern void PetscCheckSameTypeAndComm(Ta, int, Tb, int);
template <typename Ta, typename Tb>
extern void PetscValidLogicalCollectiveScalar(Ta, Tb, int);
template <typename Ta, typename Tb>
extern void PetscValidLogicalCollectiveReal(Ta, Tb, int);
template <typename Ta, typename Tb>
extern void PetscValidLogicalCollectiveInt(Ta, Tb, int);
template <typename Ta, typename Tb>
extern void PetscValidLogicalCollectiveCount(Ta, Tb, int);
template <typename Ta, typename Tb>
extern void PetscValidLogicalCollectiveMPIInt(Ta, Tb, int);
template <typename Ta, typename Tb>
extern void PetscValidLogicalCollectiveBool(Ta, Tb, int);
template <typename Ta, typename Tb>
extern void PetscValidLogicalCollectiveEnum(Ta, Tb, int);
template <typename T>
extern void PetscCheckSorted(PetscInt, T);
#endif /* PETSC_CLANG_STATIC_ANALYZER */

/*MC
   PetscTryMethod - Queries a `PetscObject` for a method added with `PetscObjectComposeFunction()`, if it exists then calls it.

  Synopsis:
   #include "petsc/private/petscimpl.h"
   PetscTryMethod(PetscObject obj, const char *name, (arg_types), (arg_value))

   Input Parameters:
+   obj - the object, for example a `Mat`, that does not need to be cast to `PetscObject`
.   name - the name of the method, for example, "KSPGMRESSetRestart_C" for the function `KSPGMRESSetRestart()`
.   arg_types - the argument types for the method, for example, (KSP,PetscInt)
-   args - the arguments for the method, for example, (ksp,restart))

   Level: developer

   Notes:
   This does not return an error code, it is a macro that returns from the subroutine with an error code on error.

   Use `PetscUseTypeMethod()` or `PetscTryTypeMethod()` to call functions that are included in the object's function table, the `ops` array
   in the object.

.seealso: `PetscUseMethod()`, `PetscCall()`, `PetscUseTypeMethod()`, `PetscTryTypeMethod()`, `PetscCheck()`, `PetscObject`
M*/
#define PetscTryMethod(obj, A, B, C) \
  do { \
    PetscErrorCode(*_7_f) B; \
    PetscCall(PetscObjectQueryFunction((PetscObject)(obj), A, &_7_f)); \
    if (_7_f) PetscCall((*_7_f)C); \
  } while (0)

/*MC
   PetscUseMethod - Queries a `PetscObject` for a method added with `PetscObjectComposeFunction()`, if it exists then calls it, otherwise generates an error.

  Synopsis:
   #include "petsc/private/petscimpl.h"
   PetscUseMethod(PetscObject obj, const char *name, (arg_types), (arg_value))

   Input Parameters:
+   obj - the object, for example a `Mat`, that does not need to be cast to `PetscObject`
.   name - the name of the method, for example, "KSPGMRESSetRestart_C" for the function `KSPGMRESSetRestart()`
.   arg_types - the argument types for the method, for example, (KSP,PetscInt)
-   args - the arguments for the method, for example, (ksp,restart))

   Level: developer

   Notes:
   This does not return an error code, it is a macro that returns from the subroutine with an error code on error.

   Use `PetscUseTypeMethod()` or `PetscTryTypeMethod()` to call functions that are included in the object's function table, the `ops` array
   in the object.

.seealso: `PetscTryMethod()`, `PetscCall()`, `PetscUseTypeMethod()`, `PetscTryTypeMethod()`, `PetscCheck()`, `PetscObject`
M*/
#define PetscUseMethod(obj, A, B, C) \
  do { \
    PetscErrorCode(*_7_f) B; \
    PetscCall(PetscObjectQueryFunction((PetscObject)(obj), A, &_7_f)); \
    PetscCheck(_7_f, PetscObjectComm((PetscObject)(obj)), PETSC_ERR_SUP, "Cannot locate function %s in object", A); \
    PetscCall((*_7_f)C); \
  } while (0)

/*
  Use Microsoft traditional preprocessor.

  The Microsoft compiler option -Zc:preprocessor available in recent versions of the compiler
  sets  _MSVC_TRADITIONAL to zero so this code path is not used.

  It appears the Intel Microsoft Windows compiler icl does not have an equivalent of -Zc:preprocessor

  These macros use the trick that Windows compilers remove the , before the __VA_ARGS__ if __VA_ARGS__ does not exist

  PetscCall() cannot be used in the macros because the remove the , trick does not work in a macro in a macro
*/
#if (defined(_MSC_VER) && (!defined(_MSVC_TRADITIONAL) || _MSVC_TRADITIONAL)) || defined(__ICL)

  #define PetscUseTypeMethod(obj, OP, ...) \
    do { \
      PetscErrorCode ierr_p_; \
      PetscStackUpdateLine; \
      PetscCheck((obj)->ops->OP, PetscObjectComm((PetscObject)obj), PETSC_ERR_SUP, "No method %s for %s of type %s", PetscStringize(OP), ((PetscObject)obj)->class_name, ((PetscObject)obj)->type_name); \
      ierr_p_ = (*(obj)->ops->OP)(obj, __VA_ARGS__); \
      PetscCall(ierr_p_); \
    } while (0)

  #define PetscTryTypeMethod(obj, OP, ...) \
    do { \
      if ((obj)->ops->OP) { \
        PetscErrorCode ierr_p_; \
        PetscStackUpdateLine; \
        ierr_p_ = (*(obj)->ops->OP)(obj, __VA_ARGS__); \
        PetscCall(ierr_p_); \
      } \
    } while (0)

#else

  /*MC
   PetscUseTypeMethod - Call a method on a `PetscObject`, that is a function in the objects function table `obj->ops`, error if the method does not exist

  Synopsis:
   #include "petsc/private/petscimpl.h"
   PetscUseTypeMethod(obj, method, other_args)

   Input Parameters:
+   obj - the object, for example a `Mat`, that does not need to be cast to `PetscObject`
.   method - the name of the method, for example, mult for the PETSc routine `MatMult()`
-   other_args - the other arguments for the method, `obj` is the first argument

   Level: developer

   Note:
   This does not return an error code, it is a macro that returns from the subroutine with an error code on error.

   Use `PetscUseMethod()` or `PetscTryMethod()` to call functions that have been composed to an object with `PetscObjectComposeFunction()`

.seealso: `PetscTryMethod()`, `PetscUseMethod()`, `PetscCall()`, `PetscCheck()`, `PetscTryTypeMethod()`, `PetscCallBack()`
M*/
  #define PetscUseTypeMethod(obj, ...) \
    do { \
      PetscCheck((obj)->ops->PETSC_FIRST_ARG((__VA_ARGS__, unused)), PetscObjectComm((PetscObject)obj), PETSC_ERR_SUP, "No method %s for %s of type %s", \
                 PetscStringize(PETSC_FIRST_ARG((__VA_ARGS__,unused))), ((PetscObject)obj)->class_name, ((PetscObject)obj)->type_name); \
      PetscCall((*(obj)->ops->PETSC_FIRST_ARG((__VA_ARGS__, unused)))(obj PETSC_REST_ARG(__VA_ARGS__))); \
    } while (0)

  /*MC
   PetscTryTypeMethod - Call a method on a `PetscObject`, that is a function in the objects function table `obj->ops`, skip if the method does not exist

  Synopsis:
   #include "petsc/private/petscimpl.h"
   PetscTryTypeMethod(obj, method, other_args)

   Input Parameters:
+   obj - the object, for example a `Mat`, that does not need to be cast to `PetscObject`
.   method - the name of the method, for example, mult for the PETSc routine `MatMult()`
-   other_args - the other arguments for the method, `obj` is the first argument

   Level: developer

   Note:
   This does not return an error code, it is a macro that returns from the subroutine with an error code on error.

   Use `PetscUseMethod()` or `PetscTryMethod()` to call functions that have been composed to an object with `PetscObjectComposeFunction()`

.seealso: `PetscTryMethod()`, `PetscUseMethod()`, `PetscCall()`, `PetscCheck()`, `PetscUseTypeMethod()`
M*/
  #define PetscTryTypeMethod(obj, ...) \
    do { \
      if ((obj)->ops->PETSC_FIRST_ARG((__VA_ARGS__, unused))) PetscCall((*(obj)->ops->PETSC_FIRST_ARG((__VA_ARGS__, unused)))(obj PETSC_REST_ARG(__VA_ARGS__))); \
    } while (0)

#endif

/*MC
   PetscObjectStateIncrease - Increases the state of any `PetscObject`

   Synopsis:
   #include "petsc/private/petscimpl.h"
   PetscErrorCode PetscObjectStateIncrease(PetscObject obj)

   Logically Collective

   Input Parameter:
.  obj - any PETSc object, for example a `Vec`, `Mat` or `KSP`. This must be
         cast with a (PetscObject), for example,
         `PetscObjectStateIncrease`((`PetscObject`)mat);

   Level: developer

   Notes:
   Object state is a 64-bit integer which gets increased every time
   the object is changed internally. By saving and later querying the object state
   one can determine whether information about the object is still current.
   Currently, state is maintained for `Vec` and `Mat` objects.

   This routine is mostly for internal use by PETSc; a developer need only
   call it after explicit access to an object's internals. Routines such
   as `VecSet()` or `MatScale()` already call this routine. It is also called, as a
   precaution, in `VecRestoreArray()`, `MatRestoreRow()`, `MatDenseRestoreArray()`.

   Routines such as `VecNorm()` can bypass the computation if the norm has already been computed and the vector's state has not changed.

   This routine is logically collective because state equality comparison needs to be possible without communication.

   `Mat` also has `MatGetNonzeroState()` for tracking changes to the nonzero structure.

.seealso: `PetscObjectStateGet()`, `PetscObject`
M*/
#define PetscObjectStateIncrease(obj) ((PetscErrorCode)((obj)->state++, PETSC_SUCCESS))

PETSC_EXTERN PetscErrorCode PetscObjectStateGet(PetscObject, PetscObjectState *);
PETSC_EXTERN PetscErrorCode PetscObjectStateSet(PetscObject, PetscObjectState);
PETSC_EXTERN PetscErrorCode PetscObjectComposedDataRegister(PetscInt *);
PETSC_EXTERN PetscErrorCode PetscObjectComposedDataIncreaseInt(PetscObject);
PETSC_EXTERN PetscErrorCode PetscObjectComposedDataIncreaseIntstar(PetscObject);
PETSC_EXTERN PetscErrorCode PetscObjectComposedDataIncreaseReal(PetscObject);
PETSC_EXTERN PetscErrorCode PetscObjectComposedDataIncreaseRealstar(PetscObject);
PETSC_EXTERN PetscErrorCode PetscObjectComposedDataIncreaseScalar(PetscObject);
PETSC_EXTERN PetscErrorCode PetscObjectComposedDataIncreaseScalarstar(PetscObject);
PETSC_EXTERN PetscInt       PetscObjectComposedDataMax;

/*MC
   PetscObjectComposedDataSetInt - attach `PetscInt` data to a `PetscObject` that may be later accessed with `PetscObjectComposedDataGetInt()`

   Synopsis:
   #include "petsc/private/petscimpl.h"
   PetscErrorCode PetscObjectComposedDataSetInt(PetscObject obj, PetscInt id, PetscInt data)

   Not Collective

   Input Parameters:
+  obj - the object to which data is to be attached
.  id - the identifier for the data
-  data - the data to  be attached, a `PetscInt`

   Level: developer

   Notes:
   The `data` identifier can be created through a call to `PetscObjectComposedDataRegister()`

   This allows the efficient composition of a single integer value with a `PetscObject`. Complex data may be
   attached with `PetscObjectCompose()`

.seealso: `PetscObjectComposedDataGetInt()`, `PetscObjectComposedDataGetReal()`, `PetscObjectComposedDataSetReal()`,
          `PetscObjectComposedDataGetIntstar()`, `PetscObjectComposedDataSetIntstar()`, `PetscObject`,
          `PetscObjectCompose()`, `PetscObjectQuery()`
M*/
#define PetscObjectComposedDataSetInt(obj, id, data) \
  ((PetscErrorCode)((((obj)->int_idmax < PetscObjectComposedDataMax) && PetscObjectComposedDataIncreaseInt(obj)) || ((obj)->intcomposeddata[id] = data, (obj)->intcomposedstate[id] = (obj)->state, PETSC_SUCCESS)))

/*MC
   PetscObjectComposedDataGetInt - retrieve `PetscInt` data attached to a `PetscObject` `PetscObjectComposedDataSetInt()`

   Synopsis:
   #include "petsc/private/petscimpl.h"
   PetscErrorCode PetscObjectComposedDataGetInt(PetscObject obj, PetscInt id, PetscInt data, PetscBool flag)

   Not Collective

   Input Parameters:
+  obj - the object from which data is to be retrieved
-  id - the identifier for the data

   Output Parameters:
+  data - the data to be retrieved, a `PetscInt`
-  flag - `PETSC_TRUE` if the data item exists and is valid, `PETSC_FALSE` otherwise

   Level: developer

   Notes:
   The `data` and `flag` variables are inlined, so they are not pointers.

.seealso: `PetscObjectComposedDataSetInt()`, `PetscObjectComposedDataGetReal()`, `PetscObjectComposedDataSetReal()`,
          `PetscObjectComposedDataGetIntstar()`, `PetscObjectComposedDataSetIntstar()`, `PetscObject`,
          `PetscObjectCompose()`, `PetscObjectQuery()`
M*/
#define PetscObjectComposedDataGetInt(obj, id, data, flag) ((PetscErrorCode)(((obj)->intcomposedstate ? (data = (obj)->intcomposeddata[id], flag = (PetscBool)((obj)->intcomposedstate[id] == (obj)->state)) : (flag = PETSC_FALSE)), PETSC_SUCCESS))

/*MC
   PetscObjectComposedDataSetIntstar - attach `PetscInt` array data to a `PetscObject` that may be accessed later with `PetscObjectComposedDataGetIntstar()`

   Synopsis:
   #include "petsc/private/petscimpl.h"
   PetscErrorCode PetscObjectComposedDataSetIntstar(PetscObject obj, PetscInt id, PetscInt *data)

   Not Collective

   Input Parameters:
+  obj - the object to which data is to be attached
.  id - the identifier for the data
-  data - the data to  be attached, a `PetscInt` array

   Level: developer

   Notes:
   The `data` identifier can be determined through a call to `PetscObjectComposedDataRegister()`

   The length of the array accessed must be known, it is not available through this API.

.seealso: `PetscObjectComposedDataSetInt()`, `PetscObjectComposedDataGetReal()`, `PetscObjectComposedDataSetReal()`,
          `PetscObjectComposedDataGetIntstar()`, `PetscObjectComposedDataGetInt()`, `PetscObject`,
          `PetscObjectCompose()`, `PetscObjectQuery()`
M*/
#define PetscObjectComposedDataSetIntstar(obj, id, data) \
  ((PetscErrorCode)((((obj)->intstar_idmax < PetscObjectComposedDataMax) && PetscObjectComposedDataIncreaseIntstar(obj)) || ((obj)->intstarcomposeddata[id] = data, (obj)->intstarcomposedstate[id] = (obj)->state, PETSC_SUCCESS)))

/*MC
   PetscObjectComposedDataGetIntstar - retrieve `PetscInt` array data attached to a `PetscObject` with `PetscObjectComposedDataSetIntstar()`

   Synopsis:
   #include "petsc/private/petscimpl.h"
   PetscErrorCode PetscObjectComposedDataGetIntstar(PetscObject obj, PetscInt id, PetscInt *data, PetscBool flag)

   Not Collective

   Input Parameters:
+  obj - the object from which data is to be retrieved
-  id - the identifier for the data

   Output Parameters:
+  data - the data to be retrieved, a `PetscInt` array
-  flag - `PETSC_TRUE` if the data item exists and is valid, `PETSC_FALSE` otherwise

   Level: developer

   Notes:
   The `data` and `flag` variables are inlined, so they are not pointers.

   The length of the array accessed must be known, it is not available through this API.

.seealso: `PetscObjectComposedDataSetInt()`, `PetscObjectComposedDataGetReal()`, `PetscObjectComposedDataSetReal()`,
          `PetscObjectComposedDataSetIntstar()`, `PetscObjectComposedDataGetInt()`, `PetscObject`,
          `PetscObjectCompose()`, `PetscObjectQuery()`
M*/
#define PetscObjectComposedDataGetIntstar(obj, id, data, flag) \
  ((PetscErrorCode)(((obj)->intstarcomposedstate ? (data = (obj)->intstarcomposeddata[id], flag = (PetscBool)((obj)->intstarcomposedstate[id] == (obj)->state)) : (flag = PETSC_FALSE)), PETSC_SUCCESS))

/*MC
   PetscObjectComposedDataSetReal - attach `PetscReal` data to a `PetscObject` that may be later accessed with `PetscObjectComposedDataGetReal()`

   Synopsis:
   #include "petsc/private/petscimpl.h"
   PetscErrorCode PetscObjectComposedDataSetReal(PetscObject obj, PetscInt id, PetscReal data)

   Not Collective

   Input Parameters:
+  obj - the object to which data is to be attached
.  id - the identifier for the data
-  data - the data to  be attached, a `PetscReal`

   Level: developer

   Note:
   The `data` identifier can be determined through a call to  `PetscObjectComposedDataRegister()`

.seealso: `PetscObjectComposedDataSetInt()`, `PetscObjectComposedDataGetReal()`, `PetscObjectComposedDataSetIntstar()`,
          `PetscObjectComposedDataSetIntstar()`, `PetscObjectComposedDataGetInt()`, `PetscObject`,
          `PetscObjectCompose()`, `PetscObjectQuery()`
M*/
#define PetscObjectComposedDataSetReal(obj, id, data) \
  ((PetscErrorCode)((((obj)->real_idmax < PetscObjectComposedDataMax) && PetscObjectComposedDataIncreaseReal(obj)) || ((obj)->realcomposeddata[id] = data, (obj)->realcomposedstate[id] = (obj)->state, PETSC_SUCCESS)))

/*MC
   PetscObjectComposedDataGetReal - retrieve `PetscReal` data attached to a `PetscObject` set with `PetscObjectComposedDataSetReal()`

   Synopsis:
   #include "petsc/private/petscimpl.h"
   PetscErrorCode PetscObjectComposedDataGetReal(PetscObject obj, PetscInt id, PetscReal data, PetscBool flag)

   Not Collective

   Input Parameters:
+  obj - the object from which data is to be retrieved
-  id - the identifier for the data

   Output Parameters:
+  data - the data to be retrieved, a `PetscReal`
-  flag - `PETSC_TRUE` if the data item exists and is valid, `PETSC_FALSE` otherwise

   Level: developer

   Note:
   The `data` and `flag` variables are inlined, so they are not pointers.

.seealso: `PetscObjectComposedDataSetInt()`, `PetscObjectComposedDataSetReal()`, `PetscObjectComposedDataSetIntstar()`,
          `PetscObjectComposedDataSetIntstar()`, `PetscObjectComposedDataGetInt()`, `PetscObject`,
          `PetscObjectCompose()`, `PetscObjectQuery()`
M*/
#define PetscObjectComposedDataGetReal(obj, id, data, flag) ((PetscErrorCode)(((obj)->realcomposedstate ? (data = (obj)->realcomposeddata[id], flag = (PetscBool)((obj)->realcomposedstate[id] == (obj)->state)) : (flag = PETSC_FALSE)), PETSC_SUCCESS))

/*MC
   PetscObjectComposedDataSetRealstar - attach `PetscReal` array data to a `PetscObject` that may be retrieved with `PetscObjectComposedDataGetRealstar()`

   Synopsis:
   #include "petsc/private/petscimpl.h"
   PetscErrorCode PetscObjectComposedDataSetRealstar(PetscObject obj, PetscInt id, PetscReal *data)

   Not Collective

   Input Parameters:
+  obj - the object to which data is to be attached
.  id - the identifier for the data
-  data - the data to  be attached

   Level: developer

   Notes:
   The `data` identifier can be determined through a call to `PetscObjectComposedDataRegister()`

   The length of the array accessed must be known, it is not available through this API.

.seealso: `PetscObjectComposedDataSetInt()`, `PetscObjectComposedDataSetReal()`, `PetscObjectComposedDataGetReal()`, `PetscObjectComposedDataSetIntstar()`,
          `PetscObjectComposedDataSetIntstar()`, `PetscObjectComposedDataGetInt()`, `PetscObject`,
          `PetscObjectCompose()`, `PetscObjectQuery()`, `PetscObjectComposedDataGetRealstar()`
M*/
#define PetscObjectComposedDataSetRealstar(obj, id, data) \
  ((PetscErrorCode)((((obj)->realstar_idmax < PetscObjectComposedDataMax) && PetscObjectComposedDataIncreaseRealstar(obj)) || ((obj)->realstarcomposeddata[id] = data, (obj)->realstarcomposedstate[id] = (obj)->state, PETSC_SUCCESS)))

/*MC
   PetscObjectComposedDataGetRealstar - retrieve `PetscReal` array data attached to a `PetscObject` with `PetscObjectComposedDataSetRealstar()`

   Synopsis:
   #include "petsc/private/petscimpl.h"
   PetscErrorCode PetscObjectComposedDataGetRealstar(PetscObject obj, PetscInt id, PetscReal *data, PetscBool flag)

   Not Collective

   Input Parameters:
+  obj - the object from which data is to be retrieved
-  id - the identifier for the data

   Output Parameters:
+  data - the data to be retrieved, a `PetscReal` array
-  flag - `PETSC_TRUE` if the data item exists and is valid, `PETSC_FALSE` otherwise

   Level: developer

   Notes:
   The `data` and `flag` variables are inlined, so they are not pointers.

   The length of the array accessed must be known, it is not available through this API.

.seealso: `PetscObjectComposedDataSetInt()`, `PetscObjectComposedDataSetReal()`, `PetscObjectComposedDataGetReal()`, `PetscObjectComposedDataSetIntstar()`,
          `PetscObjectComposedDataSetIntstar()`, `PetscObjectComposedDataGetInt()`, `PetscObject`,
          `PetscObjectCompose()`, `PetscObjectQuery()`, `PetscObjectComposedDataSetRealstar()`
M*/
#define PetscObjectComposedDataGetRealstar(obj, id, data, flag) \
  ((PetscErrorCode)(((obj)->realstarcomposedstate ? (data = (obj)->realstarcomposeddata[id], flag = (PetscBool)((obj)->realstarcomposedstate[id] == (obj)->state)) : (flag = PETSC_FALSE)), PETSC_SUCCESS))

/*MC
   PetscObjectComposedDataSetScalar - attach `PetscScalar` data to a `PetscObject` that may be later retrieved with `PetscObjectComposedDataGetScalar()`

   Synopsis:
   #include "petsc/private/petscimpl.h"
   PetscErrorCode PetscObjectComposedDataSetScalar(PetscObject obj, PetscInt id, PetscScalar data)

   Not Collective

   Input Parameters:
+  obj - the object to which data is to be attached
.  id - the identifier for the data
-  data - the data to  be attached, a `PetscScalar`

   Level: developer

   Note:
   The `data` identifier can be determined through a call to `PetscObjectComposedDataRegister()`

.seealso: `PetscObjectComposedDataSetInt()`, `PetscObjectComposedDataSetReal()`, `PetscObjectComposedDataGetReal()`, `PetscObjectComposedDataSetIntstar()`,
          `PetscObjectComposedDataSetIntstar()`, `PetscObjectComposedDataGetInt()`, `PetscObject`,
          `PetscObjectCompose()`, `PetscObjectQuery()`, `PetscObjectComposedDataSetRealstar()`, `PetscObjectComposedDataGetScalar()`
M*/
#if defined(PETSC_USE_COMPLEX)
  #define PetscObjectComposedDataSetScalar(obj, id, data) \
    ((PetscErrorCode)((((obj)->scalar_idmax < PetscObjectComposedDataMax) && PetscObjectComposedDataIncreaseScalar(obj)) || ((obj)->scalarcomposeddata[id] = data, (obj)->scalarcomposedstate[id] = (obj)->state, PETSC_SUCCESS)))
#else
  #define PetscObjectComposedDataSetScalar(obj, id, data) PetscObjectComposedDataSetReal(obj, id, data)
#endif
/*MC
   PetscObjectComposedDataGetScalar - retrieve `PetscScalar` data attached to a `PetscObject` that was set with `PetscObjectComposedDataSetScalar()`

   Synopsis:
   #include "petsc/private/petscimpl.h"
   PetscErrorCode PetscObjectComposedDataGetScalar(PetscObject obj, PetscInt id, PetscScalar data, PetscBool flag)

   Not Collective

   Input Parameters:
+  obj - the object from which data is to be retrieved
-  id - the identifier for the data

   Output Parameters:
+  data - the data to be retrieved, a `PetscScalar`
-  flag - `PETSC_TRUE` if the data item exists and is valid, `PETSC_FALSE` otherwise

   Level: developer

   Note:
   The `data` and `flag` variables are inlined, so they are not pointers.

.seealso: `PetscObjectComposedDataSetInt()`, `PetscObjectComposedDataSetReal()`, `PetscObjectComposedDataGetReal()`, `PetscObjectComposedDataSetIntstar()`,
          `PetscObjectComposedDataSetIntstar()`, `PetscObjectComposedDataGetInt()`, `PetscObject`,
          `PetscObjectCompose()`, `PetscObjectQuery()`, `PetscObjectComposedDataSetRealstar()`, `PetscObjectComposedDataSetScalar()`
M*/
#if defined(PETSC_USE_COMPLEX)
  #define PetscObjectComposedDataGetScalar(obj, id, data, flag) \
    ((PetscErrorCode)(((obj)->scalarcomposedstate ? (data = (obj)->scalarcomposeddata[id], flag = (PetscBool)((obj)->scalarcomposedstate[id] == (obj)->state)) : (flag = PETSC_FALSE)), PETSC_SUCCESS))
#else
  #define PetscObjectComposedDataGetScalar(obj, id, data, flag) PetscObjectComposedDataGetReal(obj, id, data, flag)
#endif

/*MC
   PetscObjectComposedDataSetScalarstar - attach `PetscScalar` array data to a `PetscObject` that may be later retrieved with `PetscObjectComposedDataSetScalarstar()`

   Synopsis:
   #include "petsc/private/petscimpl.h"
   PetscErrorCode PetscObjectComposedDataSetScalarstar(PetscObject obj, PetscInt id, PetscScalar *data)

   Not Collective

   Input Parameters:
+  obj - the object to which data is to be attached
.  id - the identifier for the data
-  data - the data to  be attached, a `PetscScalar` array

   Level: developer

   Notes:
   The `data` identifier can be determined through a call to `PetscObjectComposedDataRegister()`

   The length of the array accessed must be known, it is not available through this API.

.seealso: `PetscObjectComposedDataSetInt()`, `PetscObjectComposedDataSetReal()`, `PetscObjectComposedDataGetReal()`, `PetscObjectComposedDataSetIntstar()`,
          `PetscObjectComposedDataSetIntstar()`, `PetscObjectComposedDataGetInt()`, `PetscObject`,
          `PetscObjectCompose()`, `PetscObjectQuery()`, `PetscObjectComposedDataSetRealstar()`, `PetscObjectComposedDataGetScalarstar()`
M*/
#if defined(PETSC_USE_COMPLEX)
  #define PetscObjectComposedDataSetScalarstar(obj, id, data) \
    ((PetscErrorCode)((((obj)->scalarstar_idmax < PetscObjectComposedDataMax) && PetscObjectComposedDataIncreaseScalarstar(obj)) || ((obj)->scalarstarcomposeddata[id] = data, (obj)->scalarstarcomposedstate[id] = (obj)->state, PETSC_SUCCESS)))
#else
  #define PetscObjectComposedDataSetScalarstar(obj, id, data) PetscObjectComposedDataSetRealstar(obj, id, data)
#endif
/*MC
   PetscObjectComposedDataGetScalarstar - retrieve `PetscScalar` array data attached to a `PetscObject` that was set with `PetscObjectComposedDataSetScalarstar()`
   attached to an object

   Synopsis:
   #include "petsc/private/petscimpl.h"
   PetscErrorCode PetscObjectComposedDataGetScalarstar(PetscObject obj, PetscInt id, PetscScalar *data, PetscBool flag)

   Not Collective

   Input Parameters:
+  obj - the object from which data is to be retrieved
-  id - the identifier for the data

   Output Parameters:
+  data - the data to be retrieved, a `PetscScalar` array
-  flag - `PETSC_TRUE` if the data item exists and is valid, `PETSC_FALSE` otherwise

   Level: developer

   Notes:
   The `data` and `flag` variables are inlined, so they are not pointers.

   The length of the array accessed must be known, it is not available through this API.

.seealso: `PetscObjectComposedDataSetInt()`, `PetscObjectComposedDataSetReal()`, `PetscObjectComposedDataGetReal()`, `PetscObjectComposedDataSetIntstar()`,
          `PetscObjectComposedDataSetIntstar()`, `PetscObjectComposedDataGetInt()`, `PetscObject`,
          `PetscObjectCompose()`, `PetscObjectQuery()`, `PetscObjectComposedDataSetRealstar()`, `PetscObjectComposedDataSetScalarstar()`
M*/
#if defined(PETSC_USE_COMPLEX)
  #define PetscObjectComposedDataGetScalarstar(obj, id, data, flag) \
    ((PetscErrorCode)(((obj)->scalarstarcomposedstate ? (data = (obj)->scalarstarcomposeddata[id], flag = (PetscBool)((obj)->scalarstarcomposedstate[id] == (obj)->state)) : (flag = PETSC_FALSE)), PETSC_SUCCESS))
#else
  #define PetscObjectComposedDataGetScalarstar(obj, id, data, flag) PetscObjectComposedDataGetRealstar(obj, id, data, flag)
#endif

PETSC_INTERN PetscMPIInt Petsc_Counter_keyval;
PETSC_INTERN PetscMPIInt Petsc_InnerComm_keyval;
PETSC_INTERN PetscMPIInt Petsc_OuterComm_keyval;
PETSC_INTERN PetscMPIInt Petsc_Seq_keyval;
PETSC_INTERN PetscMPIInt Petsc_ShmComm_keyval;
PETSC_EXTERN PetscMPIInt Petsc_CreationIdx_keyval;
PETSC_INTERN PetscMPIInt Petsc_Garbage_HMap_keyval;

PETSC_INTERN PetscMPIInt Petsc_SharedWD_keyval;
PETSC_INTERN PetscMPIInt Petsc_SharedTmp_keyval;

struct PetscCommStash {
  struct PetscCommStash *next;
  MPI_Comm               comm;
};

/*
  PETSc communicators have this attribute, see
  PetscCommDuplicate(), PetscCommDestroy(), PetscCommGetNewTag(), PetscObjectGetName()
*/
typedef struct {
  PetscMPIInt            tag;       /* next free tag value */
  PetscInt               refcount;  /* number of references, communicator can be freed when this reaches 0 */
  PetscInt               namecount; /* used to generate the next name, as in Vec_0, Mat_1, ... */
  PetscMPIInt           *iflags;    /* length of comm size, shared by all calls to PetscCommBuildTwoSided_Allreduce/RedScatter on this comm */
  struct PetscCommStash *comms;     /* communicators available for PETSc to pass off to other packages */
} PetscCommCounter;

typedef enum {
  STATE_BEGIN,
  STATE_PENDING,
  STATE_END
} SRState;

typedef enum {
  PETSC_SR_REDUCE_SUM = 0,
  PETSC_SR_REDUCE_MAX = 1,
  PETSC_SR_REDUCE_MIN = 2
} PetscSRReductionType;

typedef struct {
  MPI_Comm              comm;
  MPI_Request           request;
  PetscBool             mix;
  PetscBool             async;
  PetscScalar          *lvalues;    /* this are the reduced values before call to MPI_Allreduce() */
  PetscScalar          *gvalues;    /* values after call to MPI_Allreduce() */
  void                **invecs;     /* for debugging only, vector/memory used with each op */
  PetscSRReductionType *reducetype; /* is particular value to be summed or maxed? */
  struct {
    PetscScalar v;
    PetscInt    i;
  }          *lvalues_mix, *gvalues_mix; /* used when mixing reduce operations */
  SRState     state;                     /* are we calling xxxBegin() or xxxEnd()? */
  PetscMPIInt maxops;                    /* total amount of space we have for requests */
  PetscMPIInt numopsbegin;               /* number of requests that have been queued in */
  PetscMPIInt numopsend;                 /* number of requests that have been gotten by user */
} PetscSplitReduction;

PETSC_EXTERN PetscErrorCode PetscSplitReductionGet(MPI_Comm, PetscSplitReduction **);
PETSC_EXTERN PetscErrorCode PetscSplitReductionEnd(PetscSplitReduction *);
PETSC_EXTERN PetscErrorCode PetscSplitReductionExtend(PetscSplitReduction *);

#if defined(PETSC_HAVE_THREADSAFETY)
  #if defined(PETSC_HAVE_CONCURRENCYKIT)
    #if defined(__cplusplus)
/*  CK does not have extern "C" protection in their include files */
extern "C" {
    #endif
    #include <ck_spinlock.h>
    #if defined(__cplusplus)
}
    #endif
typedef ck_spinlock_t PetscSpinlock;

static inline PetscErrorCode PetscSpinlockCreate(PetscSpinlock *ck_spinlock)
{
  ck_spinlock_init(ck_spinlock);
  return PETSC_SUCCESS;
}
static inline PetscErrorCode PetscSpinlockLock(PetscSpinlock *ck_spinlock)
{
  ck_spinlock_lock(ck_spinlock);
  return PETSC_SUCCESS;
}
static inline PetscErrorCode PetscSpinlockUnlock(PetscSpinlock *ck_spinlock)
{
  ck_spinlock_unlock(ck_spinlock);
  return PETSC_SUCCESS;
}
static inline PetscErrorCode PetscSpinlockDestroy(PetscSpinlock *ck_spinlock)
{
  return PETSC_SUCCESS;
}
  #elif (defined(__cplusplus) && defined(PETSC_HAVE_CXX_ATOMIC)) || (!defined(__cplusplus) && defined(PETSC_HAVE_STDATOMIC_H))
    #if defined(__cplusplus)
      // See the example at https://en.cppreference.com/w/cpp/atomic/atomic_flag
      #include <atomic>
      #define petsc_atomic_flag                 std::atomic_flag
      #define petsc_atomic_flag_test_and_set(p) std::atomic_flag_test_and_set_explicit(p, std::memory_order_acquire)
      #define petsc_atomic_flag_clear(p)        std::atomic_flag_clear_explicit(p, std::memory_order_release)
    #else
      #include <stdatomic.h>
      #define petsc_atomic_flag                 atomic_flag
      #define petsc_atomic_flag_test_and_set(p) atomic_flag_test_and_set_explicit(p, memory_order_acquire)
      #define petsc_atomic_flag_clear(p)        atomic_flag_clear_explicit(p, memory_order_release)
    #endif

typedef petsc_atomic_flag PetscSpinlock;

static inline PetscErrorCode PetscSpinlockCreate(PetscSpinlock *spinlock)
{
  petsc_atomic_flag_clear(spinlock);
  return PETSC_SUCCESS;
}
static inline PetscErrorCode PetscSpinlockLock(PetscSpinlock *spinlock)
{
  do {
  } while (petsc_atomic_flag_test_and_set(spinlock));
  return PETSC_SUCCESS;
}
static inline PetscErrorCode PetscSpinlockUnlock(PetscSpinlock *spinlock)
{
  petsc_atomic_flag_clear(spinlock);
  return PETSC_SUCCESS;
}
static inline PetscErrorCode PetscSpinlockDestroy(PETSC_UNUSED PetscSpinlock *spinlock)
{
  return PETSC_SUCCESS;
}
    #undef petsc_atomic_flag_test_and_set
    #undef petsc_atomic_flag_clear
    #undef petsc_atomic_flag

  #elif defined(PETSC_HAVE_OPENMP)

    #include <omp.h>
typedef omp_lock_t PetscSpinlock;

static inline PetscErrorCode PetscSpinlockCreate(PetscSpinlock *omp_lock)
{
  omp_init_lock(omp_lock);
  return PETSC_SUCCESS;
}
static inline PetscErrorCode PetscSpinlockLock(PetscSpinlock *omp_lock)
{
  omp_set_lock(omp_lock);
  return PETSC_SUCCESS;
}
static inline PetscErrorCode PetscSpinlockUnlock(PetscSpinlock *omp_lock)
{
  omp_unset_lock(omp_lock);
  return PETSC_SUCCESS;
}
static inline PetscErrorCode PetscSpinlockDestroy(PetscSpinlock *omp_lock)
{
  omp_destroy_lock(omp_lock);
  return PETSC_SUCCESS;
}
  #else
    #if defined(__cplusplus)
      #error "Thread safety requires either --download-concurrencykit, std::atomic, or --with-openmp"
    #else
      #error "Thread safety requires either --download-concurrencykit, stdatomic.h, or --with-openmp"
    #endif
  #endif

#else
typedef int PetscSpinlock;
  #define PetscSpinlockCreate(a)  PETSC_SUCCESS
  #define PetscSpinlockLock(a)    PETSC_SUCCESS
  #define PetscSpinlockUnlock(a)  PETSC_SUCCESS
  #define PetscSpinlockDestroy(a) PETSC_SUCCESS
#endif

#if defined(PETSC_HAVE_THREADSAFETY)
PETSC_INTERN PetscSpinlock PetscViewerASCIISpinLockOpen;
PETSC_INTERN PetscSpinlock PetscViewerASCIISpinLockStdout;
PETSC_INTERN PetscSpinlock PetscViewerASCIISpinLockStderr;
PETSC_INTERN PetscSpinlock PetscCommSpinLock;
#endif

PETSC_EXTERN PetscLogEvent PETSC_Barrier;
PETSC_EXTERN PetscLogEvent PETSC_BuildTwoSided;
PETSC_EXTERN PetscLogEvent PETSC_BuildTwoSidedF;
PETSC_EXTERN PetscBool     use_gpu_aware_mpi;
PETSC_EXTERN PetscBool     PetscPrintFunctionList;

#if defined(PETSC_HAVE_ADIOS)
PETSC_EXTERN int64_t Petsc_adios_group;
#endif

#if defined(PETSC_HAVE_KOKKOS)
PETSC_INTERN PetscBool      PetscBeganKokkos;
PETSC_EXTERN PetscBool      PetscKokkosInitialized;
PETSC_INTERN PetscErrorCode PetscKokkosIsInitialized_Private(PetscBool *);
PETSC_INTERN PetscErrorCode PetscKokkosFinalize_Private(void);
#endif

#if defined(PETSC_HAVE_OPENMP)
PETSC_EXTERN PetscInt PetscNumOMPThreads;
#endif

struct _n_PetscObjectList {
  char            name[256];
  PetscBool       skipdereference; /* when the PetscObjectList is destroyed do not call PetscObjectDereference() on this object */
  PetscObject     obj;
  PetscObjectList next;
};
