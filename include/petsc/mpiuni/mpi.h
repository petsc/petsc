/*
   This is a special set of bindings for uni-processor use of MPI by the PETSc library.

   NOT ALL THE MPI CALLS ARE IMPLEMENTED CORRECTLY! Only those needed in PETSc.

   For example,
   * Does not implement send to self.
   * Does not implement attributes correctly.
*/

/*
  The following info is a response to one of the petsc-maint questions
  regarding MPIUNI.

  MPIUNI was developed with the aim of getting PETSc compiled, and
  usable in the absence of a full MPI implementation. With this, we
  were able to provide PETSc on Windows, Windows64 even before any MPI
  implementation was available on these platforms. [Or with certain
  compilers - like borland, that do not have a usable MPI
  implementation]

  However - providing a seqential, standards compliant MPI
  implementation is *not* the goal of MPIUNI. The development strategy
  was - to make enough changes to it so that PETSc sources, examples
  compile without errors, and runs in the uni-processor mode. This is
  the reason each function is not documented.

  PETSc usage of MPIUNI is primarily from C. However a minimal fortran
  interface is also provided - to get PETSc fortran examples with a
  few MPI calls working.

  One of the optimzation with MPIUNI, is to avoid the function call
  overhead, when possible. Hence most of the C functions are
  implemented as macros. However the function calls cannot be avoided
  with fortran usage.

  Most PETSc objects have both sequential and parallel
  implementations, which are separate. For eg: We have two types of
  sparse matrix storage formats - SeqAIJ, and MPIAIJ. Some MPI
  routines are used in the Seq part, but most of them are used in the
  MPI part. The send/receive calls can be found mostly in the MPI
  part.

  When MPIUNI is used, only the Seq version of the PETSc objects are
  used, even though the MPI variant of the objects are compiled. Since
  there are no send/receive calls in the Seq variant, PETSc works fine
  with MPIUNI in seq mode.

  The reason some send/receive functions are defined to abort(), is to
  detect sections of code that use send/receive functions, and gets
  executed in the sequential mode. (which shouldn't happen in case of
  PETSc).

  Proper implementation of send/receive would involve writing a
  function for each of them. Inside each of these functions, we have
  to check if the send is to self or receive is from self, and then
  doing the buffering accordingly (until the receive is called) - or
  what if a nonblocking receive is called, do a copy etc.. Handling
  the buffering aspects might be complicated enough, that in this
  case, a proper implementation of MPI might as well be used. This is
  the reason the send to self is not implemented in MPIUNI, and never
  will be.

  Proper implementations of MPI [for eg: MPICH & OpenMPI] are
  available for most machines. When these packages are available, Its
  generally preferable to use one of them instead of MPIUNI - even if
  the user is using PETSc sequentially.

    - MPIUNI does not support all MPI functions [or functionality].
    Hence it might not work with external packages or user code that
    might have MPI calls in it.

    - MPIUNI is not a standards compliant implementation for np=1.
    For eg: if the user code has send/recv to self, then it will
    abort. [Similar issues with a number of other MPI functionality]
    However MPICH & OpenMPI are the correct implementations of MPI
    standard for np=1.

    - When user code uses multiple MPI based packages that have their
    own *internal* stubs equivalent to MPIUNI - in sequential mode,
    invariably these multiple implementations of MPI for np=1 conflict
    with each other. The correct thing to do is: make all such
    packages use the *same* MPI implementation for np=1. MPICH/OpenMPI
    satisfy this requirement correctly [and hence the correct choice].

    - Using MPICH/OpenMPI sequentially should have minimal
    disadvantages. [for eg: these binaries can be run without
    mpirun/mpiexec as ./executable, without requiring any extra
    configurations for ssh/rsh/daemons etc..]. This should not be a
    reason to avoid these packages for sequential use.

*/

#if !defined(MPIUNI_H)
#define MPIUNI_H

/* Required by abort() in mpi.c & for win64 */
#include <petscconf.h>
#include <stddef.h>

/*  This is reproduced from petscsys.h so that mpi.h can be used standalone without first including petscsys.h */
#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
#  define MPIUni_PETSC_DLLEXPORT __declspec(dllexport)
#  define MPIUni_PETSC_DLLIMPORT __declspec(dllimport)
#elif defined(PETSC_USE_VISIBILITY_CXX) && defined(__cplusplus)
#  define MPIUni_PETSC_DLLEXPORT __attribute__((visibility ("default")))
#  define MPIUni_PETSC_DLLIMPORT __attribute__((visibility ("default")))
#elif defined(PETSC_USE_VISIBILITY_C) && !defined(__cplusplus)
#  define MPIUni_PETSC_DLLEXPORT __attribute__((visibility ("default")))
#  define MPIUni_PETSC_DLLIMPORT __attribute__((visibility ("default")))
#else
#  define MPIUni_PETSC_DLLEXPORT
#  define MPIUni_PETSC_DLLIMPORT
#endif

#if defined(petsc_EXPORTS)
#  define MPIUni_PETSC_VISIBILITY_PUBLIC MPIUni_PETSC_DLLEXPORT
#else  /* Win32 users need this to import symbols from petsc.dll */
#  define MPIUni_PETSC_VISIBILITY_PUBLIC MPIUni_PETSC_DLLIMPORT
#endif

#if defined(__cplusplus)
#define MPIUni_PETSC_EXTERN extern "C" MPIUni_PETSC_VISIBILITY_PUBLIC
#else
#define MPIUni_PETSC_EXTERN extern MPIUni_PETSC_VISIBILITY_PUBLIC
#endif

#if defined(__cplusplus)
extern "C" {
#endif

/* MPI_Aint has to be a signed integral type large enough to hold a pointer */
typedef ptrdiff_t MPI_Aint;

/* old 32bit MS compiler does not support long long */
#if defined(PETSC_SIZEOF_LONG_LONG)
typedef long long MPIUNI_INT64;
typedef unsigned long long MPIUNI_UINT64;
#elif defined(PETSC_HAVE___INT64)
typedef _int64 MPIUNI_INT64;
typedef unsigned _int64 MPIUNI_UINT64;
#else
#error "cannot determine MPIUNI_INT64, MPIUNI_UINT64 types"
#endif

/*

 MPIUNI_ARG is used in the macros below only to stop various C/C++ compilers
 from generating warning messages about unused variables while compiling PETSc.
*/
MPIUni_PETSC_EXTERN void *MPIUNI_TMP;
#define MPIUNI_ARG(arg) (MPIUNI_TMP = (void *)(MPI_Aint) (arg))

#define MPI_IDENT            0
#define MPI_CONGRUENT        1
#define MPI_SIMILAR          2
#define MPI_UNEQUAL          3

#define MPI_BOTTOM   ((void *) 0)
#define MPI_IN_PLACE ((void *)-1)

#define MPI_PROC_NULL      (-1)
#define MPI_ANY_SOURCE     (-2)
#define MPI_ANY_TAG        (-1)
#define MPI_UNDEFINED  (-32766)

#define MPI_SUCCESS          0
#define MPI_ERR_OTHER       17
#define MPI_ERR_UNKNOWN     18
#define MPI_ERR_INTERN      21

#define MPI_KEYVAL_INVALID   0
#define MPI_TAG_UB           0

#define MPI_MAX_PROCESSOR_NAME 1024
#define MPI_MAX_ERROR_STRING   2056

typedef int MPI_Comm;
#define MPI_COMM_NULL  0
#define MPI_COMM_SELF  1
#define MPI_COMM_WORLD 2
#define MPI_COMM_TYPE_SHARED 1

typedef int MPI_Info;
#define MPI_INFO_NULL 0

typedef struct {int MPI_SOURCE,MPI_TAG,MPI_ERROR;} MPI_Status;
#define MPI_STATUS_IGNORE   (MPI_Status *)0
#define MPI_STATUSES_IGNORE (MPI_Status *)0

/* 32-bit packing scheme: [combiner:4 | type-index:8 | count:12 | base-bytes:8] */
/* Any changes here must also be reflected in mpif.h */
typedef int MPI_Datatype;
#define MPI_DATATYPE_NULL      0
#define MPI_PACKED             0

#define MPI_FLOAT              (1 << 20 | 1 << 8 | (int)sizeof(float))
#define MPI_DOUBLE             (1 << 20 | 1 << 8 | (int)sizeof(double))
#define MPI_LONG_DOUBLE        (1 << 20 | 1 << 8 | (int)sizeof(long double))

#define MPI_COMPLEX            (2 << 20 | 1 << 8 | 2*(int)sizeof(float))
#define MPI_C_COMPLEX          (2 << 20 | 1 << 8 | 2*(int)sizeof(float))
#define MPI_C_FLOAT_COMPLEX    (2 << 20 | 1 << 8 | 2*(int)sizeof(float))
#define MPI_DOUBLE_COMPLEX     (2 << 20 | 1 << 8 | 2*(int)sizeof(double))
#define MPI_C_DOUBLE_COMPLEX   (2 << 20 | 1 << 8 | 2*(int)sizeof(double))

#define MPI_CHAR               (3 << 20 | 1 << 8 | (int)sizeof(char))
#define MPI_BYTE               (3 << 20 | 1 << 8 | (int)sizeof(char))
#define MPI_SIGNED_CHAR        (3 << 20 | 1 << 8 | (int)sizeof(signed char))
#define MPI_UNSIGNED_CHAR      (3 << 20 | 1 << 8 | (int)sizeof(unsigned char))

#define MPI_SHORT              (4 << 20 | 1 << 8 | (int)sizeof(short))
#define MPI_INT                (4 << 20 | 1 << 8 | (int)sizeof(int))
#define MPI_LONG               (4 << 20 | 1 << 8 | (int)sizeof(long))
#define MPI_LONG_LONG          (4 << 20 | 1 << 8 | (int)sizeof(MPIUNI_INT64))
#define MPI_LONG_LONG_INT      MPI_LONG_LONG
#define MPI_INTEGER8           MPI_LONG_LONG

#define MPI_UNSIGNED_SHORT     (5 << 20 | 1 << 8 | (int)sizeof(unsigned short))
#define MPI_UNSIGNED           (5 << 20 | 1 << 8 | (int)sizeof(unsigned))
#define MPI_UNSIGNED_LONG      (5 << 20 | 1 << 8 | (int)sizeof(unsigned long))
#define MPI_UNSIGNED_LONG_LONG (5 << 20 | 1 << 8 | (int)sizeof(MPIUNI_UINT64))

#define MPI_FLOAT_INT          (10 << 20 | 1 << 8 | (int)(sizeof(float) + sizeof(int)))
#define MPI_DOUBLE_INT         (11 << 20 | 1 << 8 | (int)(sizeof(double) + sizeof(int)))
#define MPI_LONG_INT           (12 << 20 | 1 << 8 | (int)(sizeof(long) + sizeof(int)))
#define MPI_SHORT_INT          (13 << 20 | 1 << 8 | (int)(sizeof(short) + sizeof(int)))
#define MPI_2INT               (14 << 20 | 1 << 8 | (int)(2*sizeof(int)))
#define MPI_2DOUBLE            (15 << 20 | 1 << 8 | (int)(2*sizeof(double)))

/* Fortran datatypes; Jed Brown says they should be defined here */
#define MPI_INTEGER MPI_INT
#define MPI_DOUBLE_PRECISION MPI_DOUBLE
#define MPI_COMPLEX16 MPI_C_DOUBLE_COMPLEX
#define MPI_2DOUBLE_PRECISION MPI_2DOUBLE

#define MPI_ORDER_C            0
#define MPI_ORDER_FORTRAN      1

#define MPI_sizeof_default(datatype) ((((datatype) >> 8) & 0xfff) * ((datatype) & 0xff))
#if defined(PETSC_USE_REAL___FP16)
MPIUni_PETSC_EXTERN MPI_Datatype MPIU___FP16;
#define MPI_sizeof(datatype) ((datatype == MPIU___FP16) ? (int)(2*sizeof(char)) : MPI_sizeof_default(datatype))
#elif defined(PETSC_USE_REAL___FLOAT128)
MPIUni_PETSC_EXTERN MPI_Datatype MPIU___FLOAT128;
#define MPI_sizeof(datatype) ((datatype == MPIU___FLOAT128) ? (int)(2*sizeof(double)) : MPI_sizeof_default(datatype))
#else
#define MPI_sizeof(datatype) (MPI_sizeof_default(datatype))
#endif

MPIUni_PETSC_EXTERN int MPIUNI_Memcpy(void*,const void*,int);

typedef int MPI_Request;
#define MPI_REQUEST_NULL 0

typedef int MPI_Group;
#define MPI_GROUP_NULL  0
#define MPI_GROUP_EMPTY 0

typedef int MPI_Op;
#define MPI_OP_NULL    0
#define MPI_SUM        1
#define MPI_MAX        2
#define MPI_MIN        3
#define MPI_REPLACE    4
#define MPI_PROD       5
#define MPI_LAND       6
#define MPI_BAND       7
#define MPI_LOR        8
#define MPI_BOR        9
#define MPI_LXOR       10
#define MPI_BXOR       11
#define MPI_MAXLOC     12
#define MPI_MINLOC     13

typedef void (MPI_User_function)(void*, void *, int *, MPI_Datatype *);

typedef int MPI_Errhandler;
#define MPI_ERRHANDLER_NULL  0
#define MPI_ERRORS_RETURN    0
#define MPI_ERRORS_ARE_FATAL 0
typedef void (MPI_Handler_function)(MPI_Comm *, int *, ...);

/*
  Prototypes of some functions which are implemented in mpi.c
*/
typedef int (MPI_Copy_function)(MPI_Comm,int,void *,void *,void *,int *);
typedef int (MPI_Delete_function)(MPI_Comm,int,void *,void *);
#define MPI_NULL_COPY_FN   (MPI_Copy_function*)0
#define MPI_NULL_DELETE_FN (MPI_Delete_function*)0

/*
  To enable linking PETSc+MPIUNI with any other package that might have its
  own MPIUNI (equivalent implementation) we need to avoid using 'MPI'
  namespace for MPIUNI functions that go into the petsc library.

  For C functions below (that get compiled into petsc library) - we map
  the 'MPI' functions to use 'Petsc_MPI' namespace.

  With fortran we use similar mapping - thus requiring the use of
  c-preprocessor with mpif.h
*/
#define MPI_Abort         Petsc_MPI_Abort
#define MPIUni_Abort      Petsc_MPIUni_Abort
#define MPI_Attr_get      Petsc_MPI_Attr_get
#define MPI_Keyval_free   Petsc_MPI_Keyval_free
#define MPI_Attr_put      Petsc_MPI_Attr_put
#define MPI_Attr_delete   Petsc_MPI_Attr_delete
#define MPI_Keyval_create Petsc_MPI_Keyval_create
#define MPI_Comm_free     Petsc_MPI_Comm_free
#define MPI_Comm_dup      Petsc_MPI_Comm_dup
#define MPI_Comm_create   Petsc_MPI_Comm_create
#define MPI_Init          Petsc_MPI_Init
#define MPI_Finalize      Petsc_MPI_Finalize
#define MPI_Initialized   Petsc_MPI_Initialized
#define MPI_Finalized     Petsc_MPI_Finalized
#define MPI_Comm_size     Petsc_MPI_Comm_size
#define MPI_Comm_rank     Petsc_MPI_Comm_rank
#define MPI_Wtime         Petsc_MPI_Wtime
#define MPI_Type_get_envelope Petsc_MPI_Type_get_envelope
#define MPI_Type_get_contents Petsc_MPI_Type_get_contents
#define MPI_Add_error_class   Petsc_MPI_Add_error_class
#define MPI_Add_error_code    Petsc_MPI_Add_error_code

/* identical C bindings */
#define MPI_Comm_copy_attr_function   MPI_Copy_function
#define MPI_Comm_delete_attr_function MPI_Delete_function
#define MPI_COMM_NULL_COPY_FN         MPI_NULL_COPY_FN
#define MPI_COMM_NULL_DELETE_FN       MPI_NULL_DELETE_FN
#define MPI_Comm_create_keyval        Petsc_MPI_Keyval_create
#define MPI_Comm_free_keyval          Petsc_MPI_Keyval_free
#define MPI_Comm_get_attr             Petsc_MPI_Attr_get
#define MPI_Comm_set_attr             Petsc_MPI_Attr_put
#define MPI_Comm_delete_attr          Petsc_MPI_Attr_delete

MPIUni_PETSC_EXTERN int    MPIUni_Abort(MPI_Comm,int);
MPIUni_PETSC_EXTERN int    MPI_Abort(MPI_Comm,int);
MPIUni_PETSC_EXTERN int    MPI_Attr_get(MPI_Comm comm,int keyval,void *attribute_val,int *flag);
MPIUni_PETSC_EXTERN int    MPI_Keyval_free(int*);
MPIUni_PETSC_EXTERN int    MPI_Attr_put(MPI_Comm,int,void *);
MPIUni_PETSC_EXTERN int    MPI_Attr_delete(MPI_Comm,int);
MPIUni_PETSC_EXTERN int    MPI_Keyval_create(MPI_Copy_function *,MPI_Delete_function *,int *,void *);
MPIUni_PETSC_EXTERN int    MPI_Comm_free(MPI_Comm*);
MPIUni_PETSC_EXTERN int    MPI_Comm_dup(MPI_Comm,MPI_Comm *);
MPIUni_PETSC_EXTERN int    MPI_Comm_create(MPI_Comm,MPI_Group,MPI_Comm *);
MPIUni_PETSC_EXTERN int    MPI_Init(int *, char ***);
MPIUni_PETSC_EXTERN int    MPI_Finalize(void);
MPIUni_PETSC_EXTERN int    MPI_Initialized(int*);
MPIUni_PETSC_EXTERN int    MPI_Finalized(int*);
MPIUni_PETSC_EXTERN int    MPI_Comm_size(MPI_Comm,int*);
MPIUni_PETSC_EXTERN int    MPI_Comm_rank(MPI_Comm,int*);
MPIUni_PETSC_EXTERN double MPI_Wtime(void);

MPIUni_PETSC_EXTERN int MPI_Type_get_envelope(MPI_Datatype,int*,int*,int*,int*);
MPIUni_PETSC_EXTERN int MPI_Type_get_contents(MPI_Datatype,int,int,int,int*,MPI_Aint*,MPI_Datatype*);
MPIUni_PETSC_EXTERN int MPI_Add_error_class(int*);
MPIUni_PETSC_EXTERN int MPI_Add_error_code(int,int*);

/*
    Routines we have replace with macros that do nothing
    Some return error codes others return success
*/

typedef int MPI_Fint;
#define MPI_Comm_f2c(comm) (MPI_Comm)(comm)
#define MPI_Comm_c2f(comm) (MPI_Fint)(comm)
#define MPI_Type_f2c(type) (MPI_Datatype)(type)
#define MPI_Type_c2f(type) (MPI_Fint)(type)
#define MPI_Op_f2c(op)     (MPI_Op)(op)
#define MPI_Op_c2f(op)     (MPI_Fint)(op)

#define MPI_Send(buf,count,datatype,dest,tag,comm)  \
     (MPIUNI_ARG(buf),\
      MPIUNI_ARG(count),\
      MPIUNI_ARG(datatype),\
      MPIUNI_ARG(dest),\
      MPIUNI_ARG(tag),\
      MPIUNI_ARG(comm),\
      MPIUni_Abort(MPI_COMM_WORLD,0))
#define MPI_Recv(buf,count,datatype,source,tag,comm,status) \
     (MPIUNI_ARG(buf),\
      MPIUNI_ARG(count),\
      MPIUNI_ARG(datatype),\
      MPIUNI_ARG(source),\
      MPIUNI_ARG(tag),\
      MPIUNI_ARG(comm),\
      MPIUNI_ARG(status),\
      MPIUni_Abort(MPI_COMM_WORLD,0))
#define MPI_Get_count(status,datatype,count) \
     (MPIUNI_ARG(status),\
      MPIUNI_ARG(datatype),\
      MPIUNI_ARG(count),\
      MPIUni_Abort(MPI_COMM_WORLD,0))
#define MPI_Bsend(buf,count,datatype,dest,tag,comm)  \
     (MPIUNI_ARG(buf),\
      MPIUNI_ARG(count),\
      MPIUNI_ARG(datatype),\
      MPIUNI_ARG(dest),\
      MPIUNI_ARG(tag),\
      MPIUNI_ARG(comm),\
      MPIUni_Abort(MPI_COMM_WORLD,0))
#define MPI_Ssend(buf,count,datatype,dest,tag,comm) \
     (MPIUNI_ARG(buf),\
      MPIUNI_ARG(count),\
      MPIUNI_ARG(datatype),\
      MPIUNI_ARG(dest),\
      MPIUNI_ARG(tag),\
      MPIUNI_ARG(comm),\
      MPIUni_Abort(MPI_COMM_WORLD,0))
#define MPI_Rsend(buf,count,datatype,dest,tag,comm) \
     (MPIUNI_ARG(buf),\
      MPIUNI_ARG(count),\
      MPIUNI_ARG(datatype),\
      MPIUNI_ARG(dest),\
      MPIUNI_ARG(tag),\
      MPIUNI_ARG(comm),\
      MPIUni_Abort(MPI_COMM_WORLD,0))
#define MPI_Buffer_attach(buffer,size) \
     (MPIUNI_ARG(buffer),\
      MPIUNI_ARG(size),\
      MPI_SUCCESS)
#define MPI_Buffer_detach(buffer,size)\
     (MPIUNI_ARG(buffer),\
      MPIUNI_ARG(size),\
      MPI_SUCCESS)
#define MPI_Ibsend(buf,count,datatype,dest,tag,comm,request) \
     (MPIUNI_ARG(buf),\
      MPIUNI_ARG(count),\
      MPIUNI_ARG(datatype),\
      MPIUNI_ARG(dest),\
      MPIUNI_ARG(tag),\
      MPIUNI_ARG(comm),\
      MPIUNI_ARG(request),\
      MPIUni_Abort(MPI_COMM_WORLD,0))
#define MPI_Issend(buf,count,datatype,dest,tag,comm,request) \
     (MPIUNI_ARG(buf),\
      MPIUNI_ARG(count),\
      MPIUNI_ARG(datatype),\
      MPIUNI_ARG(dest),\
      MPIUNI_ARG(tag),\
      MPIUNI_ARG(comm),\
      MPIUNI_ARG(request),\
      MPIUni_Abort(MPI_COMM_WORLD,0))
#define MPI_Irsend(buf,count,datatype,dest,tag,comm,request) \
     (MPIUNI_ARG(buf),\
      MPIUNI_ARG(count),\
      MPIUNI_ARG(datatype),\
      MPIUNI_ARG(dest),\
      MPIUNI_ARG(tag),\
      MPIUNI_ARG(comm),\
      MPIUNI_ARG(request),\
      MPIUni_Abort(MPI_COMM_WORLD,0))
#define MPI_Irecv(buf,count,datatype,source,tag,comm,request) \
     (MPIUNI_ARG(buf),\
      MPIUNI_ARG(count),\
      MPIUNI_ARG(datatype),\
      MPIUNI_ARG(source),\
      MPIUNI_ARG(tag),\
      MPIUNI_ARG(comm),\
      MPIUNI_ARG(request),\
      MPIUni_Abort(MPI_COMM_WORLD,0))
#define MPI_Isend(buf,count,datatype,dest,tag,comm,request) \
     (MPIUNI_ARG(buf),\
      MPIUNI_ARG(count),\
      MPIUNI_ARG(datatype),\
      MPIUNI_ARG(dest),\
      MPIUNI_ARG(tag),\
      MPIUNI_ARG(comm),\
      MPIUNI_ARG(request),\
      MPIUni_Abort(MPI_COMM_WORLD,0))
#define MPI_Wait(request,status) \
     (MPIUNI_ARG(request),\
      MPIUNI_ARG(status),\
      MPI_SUCCESS)
#define MPI_Test(request,flag,status) \
     (MPIUNI_ARG(request),\
      MPIUNI_ARG(status),\
      *(flag) = 0,\
      MPI_SUCCESS)
#define MPI_Request_free(request) \
     (MPIUNI_ARG(request),\
      MPI_SUCCESS)
#define MPI_Waitany(count,array_of_requests,index,status) \
     (MPIUNI_ARG(count),\
      MPIUNI_ARG(array_of_requests),\
      MPIUNI_ARG(status),\
      (*(status)).MPI_SOURCE = 0,               \
      *(index) = 0,\
      MPI_SUCCESS)
#define MPI_Testany(a,b,c,d,e) \
     (MPIUNI_ARG(a),\
      MPIUNI_ARG(b),\
      MPIUNI_ARG(c),\
      MPIUNI_ARG(d),\
      MPIUNI_ARG(e),\
      MPI_SUCCESS)
#define MPI_Waitall(count,array_of_requests,array_of_statuses) \
     (MPIUNI_ARG(count),\
      MPIUNI_ARG(array_of_requests),\
      MPIUNI_ARG(array_of_statuses),\
      MPI_SUCCESS)
#define MPI_Testall(count,array_of_requests,flag,array_of_statuses) \
     (MPIUNI_ARG(count),\
      MPIUNI_ARG(array_of_requests),\
      MPIUNI_ARG(flag),\
      MPIUNI_ARG(array_of_statuses),\
      MPI_SUCCESS)
#define MPI_Waitsome(incount,array_of_requests,outcount,\
                     array_of_indices,array_of_statuses)        \
     (MPIUNI_ARG(incount),\
      MPIUNI_ARG(array_of_requests),\
      MPIUNI_ARG(outcount),\
      MPIUNI_ARG(array_of_indices),\
      MPIUNI_ARG(array_of_statuses),\
      MPI_SUCCESS)
#define MPI_Comm_group(comm,group) \
     (MPIUNI_ARG(comm),\
      *group = 1,\
      MPI_SUCCESS)
#define MPI_Group_incl(group,n,ranks,newgroup) \
     (MPIUNI_ARG(group),\
      MPIUNI_ARG(n),\
      MPIUNI_ARG(ranks),\
      MPIUNI_ARG(newgroup),\
      MPI_SUCCESS)
#define MPI_Testsome(incount,array_of_requests,outcount,\
                     array_of_indices,array_of_statuses) \
     (MPIUNI_ARG(incount),\
      MPIUNI_ARG(array_of_requests),\
      MPIUNI_ARG(outcount),\
      MPIUNI_ARG(array_of_indices),\
      MPIUNI_ARG(array_of_statuses),\
      MPI_SUCCESS)
#define MPI_Iprobe(source,tag,comm,flag,status) \
     (MPIUNI_ARG(source),\
      MPIUNI_ARG(tag),\
      MPIUNI_ARG(comm),\
      *(flag)=0,\
      MPIUNI_ARG(status),\
      MPI_SUCCESS)
#define MPI_Probe(source,tag,comm,status) \
     (MPIUNI_ARG(source),\
      MPIUNI_ARG(tag),\
      MPIUNI_ARG(comm),\
      MPIUNI_ARG(status),\
      MPI_SUCCESS)
#define MPI_Cancel(request) \
     (MPIUNI_ARG(request),\
      MPI_SUCCESS)
#define MPI_Test_cancelled(status,flag) \
     (MPIUNI_ARG(status),\
      *(flag)=0,\
      MPI_SUCCESS)
#define MPI_Send_init(buf,count,datatype,dest,tag,comm,request) \
     (MPIUNI_ARG(buf),\
      MPIUNI_ARG(count),\
      MPIUNI_ARG(datatype),\
      MPIUNI_ARG(dest),\
      MPIUNI_ARG(tag),\
      MPIUNI_ARG(comm),\
      MPIUNI_ARG(request),\
      MPI_SUCCESS)
#define MPI_Bsend_init(buf,count,datatype,dest,tag,comm,request) \
     (MPIUNI_ARG(buf),\
      MPIUNI_ARG(count),\
      MPIUNI_ARG(datatype),\
      MPIUNI_ARG(dest),\
      MPIUNI_ARG(tag),\
      MPIUNI_ARG(comm),\
      MPIUNI_ARG(request),\
      MPI_SUCCESS)
#define MPI_Ssend_init(buf,count,datatype,dest,tag,comm,request) \
     (MPIUNI_ARG(buf),\
      MPIUNI_ARG(count),\
      MPIUNI_ARG(datatype),\
      MPIUNI_ARG(dest),\
      MPIUNI_ARG(tag),\
      MPIUNI_ARG(comm),\
      MPIUNI_ARG(request),\
      MPI_SUCCESS)
#define MPI_Bsend_init(buf,count,datatype,dest,tag,comm,request) \
     (MPIUNI_ARG(buf),\
      MPIUNI_ARG(count),\
      MPIUNI_ARG(datatype),\
      MPIUNI_ARG(dest),\
      MPIUNI_ARG(tag),\
      MPIUNI_ARG(comm),\
      MPIUNI_ARG(request),\
      MPI_SUCCESS)
#define MPI_Rsend_init(buf,count,datatype,dest,tag,comm,request) \
     (MPIUNI_ARG(buf),\
      MPIUNI_ARG(count),\
      MPIUNI_ARG(datatype),\
      MPIUNI_ARG(dest),\
      MPIUNI_ARG(tag),\
      MPIUNI_ARG(comm),\
      MPIUNI_ARG(request),\
      MPI_SUCCESS)
#define MPI_Recv_init(buf,count,datatype,source,tag,comm,request) \
     (MPIUNI_ARG(buf),\
      MPIUNI_ARG(count),\
      MPIUNI_ARG(datatype),\
      MPIUNI_ARG(source),\
      MPIUNI_ARG(tag),\
      MPIUNI_ARG(comm),\
      MPIUNI_ARG(request),\
      MPI_SUCCESS)
#define MPI_Start(request) \
     (MPIUNI_ARG(request),\
      MPI_SUCCESS)
#define MPI_Startall(count,array_of_requests) \
     (MPIUNI_ARG(count),\
      MPIUNI_ARG(array_of_requests),\
      MPI_SUCCESS)
#define MPI_Sendrecv(sendbuf,sendcount,sendtype,\
                     dest,sendtag,recvbuf,recvcount,\
                     recvtype,source,recvtag,\
                     comm,status) \
     (MPIUNI_ARG(dest),\
      MPIUNI_ARG(sendtag),\
      MPIUNI_ARG(recvcount),\
      MPIUNI_ARG(recvtype),\
      MPIUNI_ARG(source),\
      MPIUNI_ARG(recvtag),\
      MPIUNI_ARG(comm),\
      MPIUNI_ARG(status),\
      MPIUNI_Memcpy(recvbuf,sendbuf,(sendcount)*MPI_sizeof(sendtype)))
#define MPI_Sendrecv_replace(buf,count,datatype,dest,sendtag,\
                             source,recvtag,comm,status) \
     (MPIUNI_ARG(buf),\
      MPIUNI_ARG(count),\
      MPIUNI_ARG(datatype),\
      MPIUNI_ARG(dest),\
      MPIUNI_ARG(sendtag),\
      MPIUNI_ARG(source),\
      MPIUNI_ARG(recvtag),\
      MPIUNI_ARG(comm),\
      MPIUNI_ARG(status),\
      MPI_SUCCESS)

#define MPI_COMBINER_NAMED      0
#define MPI_COMBINER_DUP        1
#define MPI_COMBINER_CONTIGUOUS 2
  /* 32-bit packing scheme: [combiner:4 | type-index:8 | count:12 | base-bytes:8] */
#define MPI_Type_dup(oldtype,newtype) \
     (*(newtype) = oldtype, MPI_SUCCESS)
#define MPI_Type_contiguous(count,oldtype,newtype) \
     (*(newtype) = (MPI_COMBINER_CONTIGUOUS<<28)|((oldtype)&0x0ff00000)|(((oldtype)>>8&0xfff)*(count))<<8|((oldtype)&0xff), MPI_SUCCESS)
#define MPI_Type_vector(count,blocklength,stride,oldtype,newtype) \
     (MPIUNI_ARG(count),\
      MPIUNI_ARG(blocklength),\
      MPIUNI_ARG(stride),\
      MPIUNI_ARG(oldtype),\
      MPIUNI_ARG(newtype),\
      MPIUni_Abort(MPI_COMM_WORLD,0))
#define MPI_Type_hvector(count,blocklength,stride,oldtype,newtype) \
     (MPIUNI_ARG(count),\
      MPIUNI_ARG(blocklength),\
      MPIUNI_ARG(stride),\
      MPIUNI_ARG(oldtype),\
      MPIUNI_ARG(newtype),\
      MPIUni_Abort(MPI_COMM_WORLD,0))
#define MPI_Type_indexed(count,array_of_blocklengths,array_of_displacements,oldtype,newtype) \
     (MPIUNI_ARG(count),\
      MPIUNI_ARG(array_of_blocklengths),\
      MPIUNI_ARG(array_of_displacements),\
      MPIUNI_ARG(oldtype),\
      MPIUNI_ARG(newtype),\
      MPIUni_Abort(MPI_COMM_WORLD,0))
#define MPI_Type_hindexed(count,array_of_blocklengths,array_of_displacements,oldtype,newtype) \
     (MPIUNI_ARG(count),\
      MPIUNI_ARG(array_of_blocklengths),\
      MPIUNI_ARG(array_of_displacements),\
      MPIUNI_ARG(oldtype),\
      MPIUNI_ARG(newtype),\
      MPIUni_Abort(MPI_COMM_WORLD,0))
#define MPI_Type_struct(count,array_of_blocklengths,array_of_displacements,array_of_types,newtype) \
     (MPIUNI_ARG(count),\
      MPIUNI_ARG(array_of_blocklengths),\
      MPIUNI_ARG(array_of_displacements),\
      MPIUNI_ARG(array_of_types),\
      MPIUNI_ARG(newtype),\
      MPIUni_Abort(MPI_COMM_WORLD,0))
#define MPI_Address(location,address) \
     (*(address) = (MPI_Aint)((char *)(location)), MPI_SUCCESS)
#define MPI_Type_size(datatype,size) (*(size) = MPI_sizeof((datatype)), MPI_SUCCESS)
#define MPI_Type_lb(datatype,lb) (MPIUNI_ARG(datatype), *(lb) = 0, MPI_SUCCESS)
#define MPI_Type_ub(datatype,ub) (*(ub) = MPI_sizeof((datatype)), MPI_SUCCESS)
#define MPI_Type_extent(datatype,extent) \
     (*(extent) = MPI_sizeof((datatype)), MPI_SUCCESS)
#define MPI_Type_get_extent(datatype,lb,extent) \
     (*(lb) = 0, *(extent) = MPI_sizeof((datatype)), MPI_SUCCESS)
#define MPI_Type_commit(datatype) (MPIUNI_ARG(datatype), MPI_SUCCESS)
#define MPI_Type_free(datatype) (*(datatype) = MPI_DATATYPE_NULL, MPI_SUCCESS)
#define MPI_Get_elements(status,datatype,count) \
     (MPIUNI_ARG(status),\
      MPIUNI_ARG(datatype),\
      MPIUNI_ARG(count),\
      MPIUni_Abort(MPI_COMM_WORLD,0))
#define MPI_Pack(inbuf,incount,datatype,outbuf,outsize,position,comm) \
     (MPIUNI_ARG(inbuf),\
      MPIUNI_ARG(incount),\
      MPIUNI_ARG(datatype),\
      MPIUNI_ARG(outbuf),\
      MPIUNI_ARG(outsize),\
      MPIUNI_ARG(position),\
      MPIUNI_ARG(comm),\
      MPIUni_Abort(MPI_COMM_WORLD,0))
#define MPI_Unpack(inbuf,insize,position,outbuf,outcount,datatype,comm) \
     (MPIUNI_ARG(inbuf),\
      MPIUNI_ARG(insize),\
      MPIUNI_ARG(position),\
      MPIUNI_ARG(outbuf),\
      MPIUNI_ARG(outcount),\
      MPIUNI_ARG(datatype),\
      MPIUNI_ARG(comm),\
      MPIUni_Abort(MPI_COMM_WORLD,0))
#define MPI_Pack_size(incount,datatype,comm,size) \
     (MPIUNI_ARG(incount),\
      MPIUNI_ARG(datatype),\
      MPIUNI_ARG(comm),\
      MPIUNI_ARG(size),\
      MPIUni_Abort(MPI_COMM_WORLD,0))
#define MPI_Barrier(comm) \
     (MPIUNI_ARG(comm),\
      MPI_SUCCESS)
#define MPI_Bcast(buffer,count,datatype,root,comm) \
     (MPIUNI_ARG(buffer),\
      MPIUNI_ARG(count),\
      MPIUNI_ARG(datatype),\
      MPIUNI_ARG(root),\
      MPIUNI_ARG(comm),\
      MPI_SUCCESS)
#define MPI_Gather(sendbuf,sendcount,sendtype,\
                   recvbuf,recvcount, recvtype,\
                   root,comm) \
     (MPIUNI_ARG(recvcount),\
      MPIUNI_ARG(root),\
      MPIUNI_ARG(recvtype),\
      MPIUNI_ARG(comm),\
      MPIUNI_Memcpy(recvbuf,sendbuf,(sendcount)*MPI_sizeof(sendtype)))
#define MPI_Gatherv(sendbuf,sendcount,sendtype,\
                    recvbuf,recvcounts,displs,\
                    recvtype,root,comm) \
     (MPIUNI_ARG(recvcounts),\
      MPIUNI_ARG(displs),\
      MPIUNI_ARG(recvtype),\
      MPIUNI_ARG(root),\
      MPIUNI_ARG(comm),\
      MPIUNI_Memcpy(recvbuf,sendbuf,(sendcount)*MPI_sizeof(sendtype)))
#define MPI_Scatter(sendbuf,sendcount,sendtype,\
                    recvbuf,recvcount,recvtype,\
                    root,comm) \
     (MPIUNI_ARG(sendcount),\
      MPIUNI_ARG(sendtype),\
      MPIUNI_ARG(recvbuf),\
      MPIUNI_ARG(recvtype),\
      MPIUNI_ARG(root),\
      MPIUNI_ARG(comm),\
      MPIUNI_Memcpy(recvbuf,sendbuf,(recvcount)*MPI_sizeof(recvtype)))
#define MPI_Scatterv(sendbuf,sendcounts,displs,\
                     sendtype,recvbuf,recvcount,\
                     recvtype,root,comm) \
     (MPIUNI_ARG(displs),\
      MPIUNI_ARG(sendtype),\
      MPIUNI_ARG(sendcounts),\
      MPIUNI_ARG(root),\
      MPIUNI_ARG(comm),\
      MPIUNI_Memcpy(recvbuf,sendbuf,(recvcount)*MPI_sizeof(recvtype)))
#define MPI_Allgather(sendbuf,sendcount,sendtype,\
                     recvbuf,recvcount,recvtype,comm) \
     (MPIUNI_ARG(recvcount),\
      MPIUNI_ARG(recvtype),\
      MPIUNI_ARG(comm),\
      MPIUNI_Memcpy(recvbuf,sendbuf,(sendcount)*MPI_sizeof(sendtype)))
#define MPI_Allgatherv(sendbuf,sendcount,sendtype,\
     recvbuf,recvcounts,displs,recvtype,comm) \
     (MPIUNI_ARG(recvcounts),\
      MPIUNI_ARG(displs),\
      MPIUNI_ARG(recvtype),\
      MPIUNI_ARG(comm),\
      MPIUNI_Memcpy(recvbuf,sendbuf,(sendcount)*MPI_sizeof(sendtype)))
#define MPI_Alltoall(sendbuf,sendcount,sendtype,\
                     recvbuf,recvcount,recvtype,comm) \
     (MPIUNI_ARG(recvcount),\
      MPIUNI_ARG(recvtype),\
      MPIUNI_ARG(comm),\
      MPIUNI_Memcpy(recvbuf,sendbuf,(sendcount)*MPI_sizeof(sendtype)))
#define MPI_Alltoallv(sendbuf,sendcounts,sdispls,sendtype,\
                      recvbuf,recvcounts,rdispls,recvtype,comm) \
     (MPIUNI_ARG(sendbuf),\
      MPIUNI_ARG(sendcounts),\
      MPIUNI_ARG(sdispls),\
      MPIUNI_ARG(sendtype),\
      MPIUNI_ARG(recvbuf),\
      MPIUNI_ARG(recvcounts),\
      MPIUNI_ARG(rdispls),\
      MPIUNI_ARG(recvtype),\
      MPIUNI_ARG(comm),\
      MPIUni_Abort(MPI_COMM_WORLD,0))
#define MPI_Alltoallw(sendbuf,sendcounts,sdispls,sendtypes,\
                      recvbuf,recvcounts,rdispls,recvtypes,comm) \
     (MPIUNI_ARG(sendbuf),\
      MPIUNI_ARG(sendcounts),\
      MPIUNI_ARG(sdispls),\
      MPIUNI_ARG(sendtypes),\
      MPIUNI_ARG(recvbuf),\
      MPIUNI_ARG(recvcount),\
      MPIUNI_ARG(rdispls),\
      MPIUNI_ARG(recvtypes),\
      MPIUNI_ARG(comm),\
      MPIUni_Abort(MPI_COMM_WORLD,0))
#define MPI_Reduce(sendbuf,recvbuf,count,datatype,op,root,comm) \
     (MPIUNI_ARG(op),\
      MPIUNI_ARG(root),\
      MPIUNI_ARG(comm),\
      MPIUNI_Memcpy(recvbuf,sendbuf,(count)*MPI_sizeof(datatype)))
#define MPI_Allreduce(sendbuf, recvbuf,count,datatype,op,comm) \
     (MPIUNI_ARG(op),\
      MPIUNI_ARG(comm),\
      MPIUNI_Memcpy(recvbuf,sendbuf,(count)*MPI_sizeof(datatype)))
#define MPI_Scan(sendbuf, recvbuf,count,datatype,op,comm) \
     (MPIUNI_ARG(op),\
      MPIUNI_ARG(comm),\
      MPIUNI_Memcpy(recvbuf,sendbuf,(count)*MPI_sizeof(datatype)))
#define MPI_Exscan(sendbuf,recvbuf,count,datatype,op,comm) \
     (MPIUNI_ARG(sendbuf),\
      MPIUNI_ARG(recvbuf),\
      MPIUNI_ARG(count),\
      MPIUNI_ARG(datatype),\
      MPIUNI_ARG(op),\
      MPIUNI_ARG(comm),\
      MPI_SUCCESS)
#define MPI_Reduce_scatter(sendbuf,recvbuf,recvcounts,datatype,op,comm) \
     (MPIUNI_ARG(op),\
      MPIUNI_ARG(comm),\
      MPIUNI_Memcpy(recvbuf,sendbuf,(*recvcounts)*MPI_sizeof(datatype)))
#define MPI_Op_create(function,commute,op) \
     (MPIUNI_ARG(function),\
      MPIUNI_ARG(commute),\
      MPIUNI_ARG(op),\
      MPI_SUCCESS)
#define MPI_Op_free(op) \
     (*(op) = MPI_OP_NULL, MPI_SUCCESS)

#define MPI_Group_size(group,size) \
  (MPIUNI_ARG(group),\
   *(size)=1,\
   MPI_SUCCESS)
#define MPI_Group_rank(group,rank) \
  (MPIUNI_ARG(group),\
   *(rank)=0,\
   MPI_SUCCESS)
#define MPI_Group_translate_ranks(group1,n,ranks1,group2,ranks2) \
     (MPIUNI_ARG(group1),\
      MPIUNI_ARG(group2),\
      MPIUNI_Memcpy((ranks2),(ranks1),(n)*sizeof(int)))
#define MPI_Group_compare(group1,group2,result) \
    (MPIUNI_ARG(group1),\
     MPIUNI_ARG(group2),\
     *(result)=1,\
     MPI_SUCCESS)
#define MPI_Group_union(group1,group2,newgroup) MPI_SUCCESS
#define MPI_Group_intersection(group1,group2,newgroup) MPI_SUCCESS
#define MPI_Group_difference(group1,group2,newgroup) MPI_SUCCESS
#define MPI_Group_excl(group,n,ranks,newgroup) MPI_SUCCESS
#define MPI_Group_range_incl(group,n,ranges,newgroup) MPI_SUCCESS
#define MPI_Group_range_excl(group,n,ranges,newgroup) MPI_SUCCESS
#define MPI_Group_free(group) \
     (*(group) = MPI_GROUP_NULL, MPI_SUCCESS)

#define MPI_Comm_compare(comm1,comm2,result) \
     (MPIUNI_ARG(comm1),\
      MPIUNI_ARG(comm2),\
      *(result)=MPI_IDENT,\
      MPI_SUCCESS)
#define MPI_Comm_split(comm,color,key,newcomm) \
     (MPIUNI_ARG(color),\
      MPIUNI_ARG(key),\
      MPI_Comm_dup(comm,newcomm))
#define MPI_Comm_split_type(comm,color,key,info,newcomm) \
     (MPIUNI_ARG(color),\
      MPIUNI_ARG(key),\
      MPIUNI_ARG(info),\
      MPI_Comm_dup(comm,newcomm))
#define MPI_Comm_test_inter(comm,flag) (*(flag)=1, MPI_SUCCESS)
#define MPI_Comm_remote_size(comm,size) (*(size)=1 ,MPI_SUCCESS)
#define MPI_Comm_remote_group(comm,group) MPI_SUCCESS
#define MPI_Intercomm_create(local_comm,local_leader,peer_comm,\
     remote_leader,tag,newintercomm) MPI_SUCCESS
#define MPI_Intercomm_merge(intercomm,high,newintracomm) MPI_SUCCESS
#define MPI_Topo_test(comm,flag) MPI_SUCCESS
#define MPI_Cart_create(comm_old,ndims,dims,periods,\
     reorder,comm_cart) MPIUni_Abort(MPI_COMM_WORLD,0)
#define MPI_Dims_create(nnodes,ndims,dims) MPIUni_Abort(MPI_COMM_WORLD,0)
#define MPI_Graph_create(comm,a,b,c,d,e) MPIUni_Abort(MPI_COMM_WORLD,0)
#define MPI_Graphdims_Get(comm,nnodes,nedges) MPIUni_Abort(MPI_COMM_WORLD,0)
#define MPI_Graph_get(comm,a,b,c,d) MPIUni_Abort(MPI_COMM_WORLD,0)
#define MPI_Cartdim_get(comm,ndims) MPIUni_Abort(MPI_COMM_WORLD,0)
#define MPI_Cart_get(comm,maxdims,dims,periods,coords) \
     MPIUni_Abort(MPI_COMM_WORLD,0)
#define MPI_Cart_rank(comm,coords,rank) MPIUni_Abort(MPI_COMM_WORLD,0)
#define MPI_Cart_coords(comm,rank,maxdims,coords) \
     MPIUni_Abort(MPI_COMM_WORLD,0)
#define MPI_Graph_neighbors_count(comm,rank,nneighbors) \
     MPIUni_Abort(MPI_COMM_WORLD,0)
#define MPI_Graph_neighbors(comm,rank,maxneighbors,neighbors) \
     MPIUni_Abort(MPI_COMM_WORLD,0)
#define MPI_Cart_shift(comm,direction,disp,rank_source,rank_dest) \
     MPIUni_Abort(MPI_COMM_WORLD,0)
#define MPI_Cart_sub(comm,remain_dims,newcomm) MPIUni_Abort(MPI_COMM_WORLD,0)
#define MPI_Cart_map(comm,ndims,dims,periods,newrank) MPIUni_Abort(MPI_COMM_WORLD,0)
#define MPI_Graph_map(comm,a,b,c,d) MPIUni_Abort(MPI_COMM_WORLD,0)

#define MPI_Get_processor_name(name,result_len)                         \
     (*(result_len) = 9,MPIUNI_Memcpy(name,"localhost",10*sizeof(char)))
#define MPI_Errhandler_create(function,errhandler) \
     (MPIUNI_ARG(function),\
      *(errhandler) = MPI_ERRORS_RETURN,\
      MPI_SUCCESS)
#define MPI_Errhandler_set(comm,errhandler) \
     (MPIUNI_ARG(comm),\
      MPIUNI_ARG(errhandler),\
      MPI_SUCCESS)
#define MPI_Errhandler_get(comm,errhandler) \
     (MPIUNI_ARG(comm),\
      (*errhandler) = MPI_ERRORS_RETURN,\
      MPI_SUCCESS)
#define MPI_Errhandler_free(errhandler) \
     (*(errhandler) = MPI_ERRHANDLER_NULL,\
      MPI_SUCCESS)
#define MPI_Error_string(errorcode,string,result_len) \
     (MPIUNI_ARG(errorcode),\
      *(result_len) = 9,\
      MPIUNI_Memcpy(string,"MPI error",10*sizeof(char)))
#define MPI_Error_class(errorcode,errorclass) \
     (*(errorclass) = errorcode, MPI_SUCCESS)
#define MPI_Wtick() 1.0
#define MPI_Pcontrol(level) MPI_SUCCESS

/* MPI-IO additions */

typedef int MPI_File;
#define MPI_FILE_NULL 0

typedef int MPI_Offset;

#define MPI_MODE_RDONLY  0
#define MPI_MODE_WRONLY  0
#define MPI_MODE_CREATE  0

#define MPI_File_open(comm,filename,amode,info,mpi_fh) \
  (MPIUNI_ARG(comm),\
   MPIUNI_ARG(filename),\
   MPIUNI_ARG(amode),\
   MPIUNI_ARG(info),\
   MPIUNI_ARG(mpi_fh),\
   MPIUni_Abort(MPI_COMM_WORLD,0))

#define MPI_File_close(mpi_fh) \
  (MPIUNI_ARG(mpi_fh),\
   MPIUni_Abort(MPI_COMM_WORLD,0))

#define MPI_File_set_view(mpi_fh,disp,etype,filetype,datarep,info) \
  (MPIUNI_ARG(mpi_fh),\
   MPIUNI_ARG(disp),\
   MPIUNI_ARG(etype),\
   MPIUNI_ARG(filetype),\
   MPIUNI_ARG(datarep),\
   MPIUNI_ARG(info),\
   MPIUni_Abort(MPI_COMM_WORLD,0))

#define MPI_File_write_all(mpi_fh,buf,count,datatype,status) \
  (MPIUNI_ARG(mpi_fh),\
   MPIUNI_ARG(buf),\
   MPIUNI_ARG(count),\
   MPIUNI_ARG(datatype),\
   MPIUNI_ARG(status),\
   MPIUni_Abort(MPI_COMM_WORLD,0))

#define MPI_File_read_all(mpi_fh,buf,count,datatype,status) \
  (MPIUNI_ARG(mpi_fh),\
   MPIUNI_ARG(buf),\
   MPIUNI_ARG(count),\
   MPIUNI_ARG(datatype),\
   MPIUNI_ARG(status),\
   MPIUni_Abort(MPI_COMM_WORLD,0))

  /* called from PetscInitialize() - so return success */
#define MPI_Register_datarep(name,read_conv_fn,write_conv_fn,extent_fn,state) \
  (MPIUNI_ARG(name),\
   MPIUNI_ARG(read_conv_fn),\
   MPIUNI_ARG(write_conv_fn),\
   MPIUNI_ARG(extent_fn),\
   MPIUNI_ARG(state),\
   MPI_SUCCESS)

#define MPI_Type_create_subarray(ndims,array_of_sizes,array_of_subsizes,array_of_starts,order,oldtype,newtype) \
  (MPIUNI_ARG(ndims),\
   MPIUNI_ARG(array_of_sizes),\
   MPIUNI_ARG(array_of_subsizes),\
   MPIUNI_ARG(array_of_starts),\
   MPIUNI_ARG(order),\
   MPIUNI_ARG(oldtype),\
   MPIUNI_ARG(newtype),\
   MPIUni_Abort(MPI_COMM_WORLD,0))

#define MPI_Type_create_resized(oldtype,lb,extent,newtype) \
  (MPIUNI_ARG(oldtype),\
   MPIUNI_ARG(lb),\
   MPIUNI_ARG(extent),\
   MPIUNI_ARG(newtype),\
   MPIUni_Abort(MPI_COMM_WORLD,0))

#define MPI_Type_create_indexed_block(count,blocklength,array_of_displacements,oldtype,newtype) \
  (MPIUNI_ARG(count),\
   MPIUNI_ARG(blocklength),\
   MPIUNI_ARG(array_of_displacements),\
   MPIUNI_ARG(oldtype),\
   MPIUNI_ARG(newtype),\
   MPIUni_Abort(MPI_COMM_WORLD,0))

#if defined(__cplusplus)
}
#endif
#endif
