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

    Instructions for building standalone MPIUNI [for eg: linux/gcc+gfortran]:
    - extract include/mpiuni/mpi.h,mpif.f, src/sys/mpiuni/mpi.c from PETSc
    - remove reference to petscconf.h from mpi.h
    - gcc -c mpi.c -DPETSC_HAVE_STDLIB_H -DPETSC_HAVE_FORTRAN_UNDERSCORE
    - ar cr libmpiuni.a mpi.o

*/

#if !defined(__MPIUNI_H)
#define __MPIUNI_H

/* Requred by abort() in mpi.c & for win64 */
#include "petscconf.h"

#if defined(__cplusplus)
extern "C" {
#endif

/* require an int variable large enough to hold a pointer */
#if !defined(MPIUNI_INTPTR)
#define MPIUNI_INTPTR long
#endif

/*

    MPIUNI_TMP is used in the macros below only to stop various C/C++ compilers
from generating warning messages about unused variables while compiling PETSc.
*/
extern void *MPIUNI_TMP;

#define MPI_COMM_WORLD       1
#define MPI_COMM_SELF        MPI_COMM_WORLD
#define MPI_COMM_NULL        0
#define MPI_SUCCESS          0
#define MPI_IDENT            0
#define MPI_CONGRUENT        1
#define MPI_SIMILAR          2
#define MPI_UNEQUAL          3
#define MPI_ANY_SOURCE     (-2)
#define MPI_KEYVAL_INVALID   0
#define MPI_ERR_UNKNOWN     18
#define MPI_ERR_INTERN      21
#define MPI_ERR_OTHER        1
#define MPI_TAG_UB           0
#define MPI_ERRORS_RETURN    0
#define MPI_UNDEFINED      (-32766)
#define MPI_ERRORS_ARE_FATAL (-32765)
#define MPI_MAXLOC           5


/* External types */
typedef int    MPI_Comm;
typedef void   *MPI_Request;
typedef void   *MPI_Group;
typedef struct {int MPI_TAG,MPI_SOURCE,MPI_ERROR;} MPI_Status;
typedef char   *MPI_Errhandler;
typedef int    MPI_Fint;
typedef int    MPI_File;
typedef int    MPI_Info;
typedef int    MPI_Offset;


/* In order to handle datatypes, we make them into "sizeof(raw-type)";
    this allows us to do the MPIUNI_Memcpy's easily */
#define MPI_Datatype         int
#define MPI_FLOAT            sizeof(float)
#define MPI_DOUBLE           sizeof(double)
#define MPI_LONG_DOUBLE      sizeof(long double)
#define MPI_CHAR             sizeof(char)
#define MPI_BYTE             sizeof(char)
#define MPI_INT              sizeof(int)
#define MPI_LONG             sizeof(long)
#define MPI_LONG_LONG_INT    sizeof(long long)
#define MPI_SHORT            sizeof(short)
#define MPI_UNSIGNED_SHORT   sizeof(unsigned short)
#define MPI_UNSIGNED         sizeof(unsigned)
#define MPI_UNSIGNED_CHAR    sizeof(unsigned char)
#define MPI_UNSIGNED_LONG    sizeof(unsigned long)
#define MPI_COMPLEX          2*sizeof(float)
#define MPI_C_COMPLEX        2*sizeof(float)
#define MPI_C_DOUBLE_COMPLEX 2*sizeof(double)
#define MPI_FLOAT_INT        (sizeof(float) + sizeof(int))
#define MPI_DOUBLE_INT       (sizeof(double) + sizeof(int))
#define MPI_LONG_INT         (sizeof(long) + sizeof(int))
#define MPI_SHORT_INT        (sizeof(short) + sizeof(int))
#define MPI_2INT             (2* sizeof(int))

#if defined(PETSC_USE_REAL___FLOAT128)
extern MPI_Datatype MPIU___FLOAT128;
#define MPI_sizeof(datatype) ((datatype == MPIU___FLOAT128) ? 2*sizeof(double) : datatype)
#else
#define MPI_sizeof(datatype) (datatype)
#endif
extern int MPIUNI_Memcpy(void*,const void*,int);


#define MPI_REQUEST_NULL     ((MPI_Request)0)
#define MPI_GROUP_NULL       ((MPI_Group)0)
#define MPI_INFO_NULL        ((MPI_Info)0)
#define MPI_BOTTOM           (void *)0
typedef int MPI_Op;

#define MPI_MODE_RDONLY   0
#define MPI_MODE_WRONLY   0
#define MPI_MODE_CREATE   0

#define MPI_SUM           0
#define MPI_MAX           0
#define MPI_MIN           0
#define MPI_REPLACE       0
#define MPI_ANY_TAG     (-1)
#define MPI_DATATYPE_NULL 0
#define MPI_PACKED        0
#define MPI_MAX_ERROR_STRING 2056
#define MPI_STATUS_IGNORE (MPI_Status *)1
#define MPI_ORDER_FORTRAN        57
#define MPI_IN_PLACE      (void *) -1

/*
  Prototypes of some functions which are implemented in mpi.c
*/
typedef int   (MPI_Copy_function)(MPI_Comm,int,void *,void *,void *,int *);
typedef int   (MPI_Delete_function)(MPI_Comm,int,void *,void *);
typedef void  (MPI_User_function)(void*, void *, int *, MPI_Datatype *); 

/*
  In order that the PETSc MPIUNI can be used with another package that has its
  own MPIUni we map the following function names to a unique PETSc name. Those functions
  are defined in mpi.c and put into the libpetscsys.a or libpetsc.a library.

  Note that this does not work for the MPIUni Fortran symbols which are explicitly in the 
  PETSc libraries unless the flag MPIUNI_AVOID_MPI_NAMESPACE is set.
*/
#define MPI_Abort         Petsc_MPI_Abort
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

/* identical C bindings */
#define MPI_Comm_create_keyval Petsc_MPI_Keyval_create
#define MPI_Comm_free_keyval   Petsc_MPI_Keyval_free
#define MPI_Comm_get_attr      Petsc_MPI_Attr_get
#define MPI_Comm_set_attr      Petsc_MPI_Attr_put

extern int    MPI_Abort(MPI_Comm,int);
extern int    MPI_Attr_get(MPI_Comm comm,int keyval,void *attribute_val,int *flag);
extern int    MPI_Keyval_free(int*);
extern int    MPI_Attr_put(MPI_Comm,int,void *);
extern int    MPI_Attr_delete(MPI_Comm,int);
extern int    MPI_Keyval_create(MPI_Copy_function *,MPI_Delete_function *,int *,void *);
extern int    MPI_Comm_free(MPI_Comm*);
extern int    MPI_Comm_dup(MPI_Comm,MPI_Comm *);
extern int    MPI_Comm_create(MPI_Comm,MPI_Group,MPI_Comm *);
extern int    MPI_Init(int *, char ***);
extern int    MPI_Finalize(void);
extern int    MPI_Initialized(int*);
extern int    MPI_Finalized(int*);
extern int    MPI_Comm_size(MPI_Comm,int*);
extern int    MPI_Comm_rank(MPI_Comm,int*);

#define MPI_Aint MPIUNI_INTPTR
/* 
    Routines we have replace with macros that do nothing 
    Some return error codes others return success
*/

#define MPI_Comm_f2c(comm) (MPI_Comm)(comm)
#define MPI_Comm_c2f(comm) (MPI_Fint)(comm)

#define MPI_Send(buf,count,datatype,dest,tag,comm)  \
     (MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (buf),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (count),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (datatype),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (dest),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (tag),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (comm),\
      MPI_Abort(MPI_COMM_WORLD,0))
#define MPI_Recv(buf,count,datatype,source,tag,comm,status) \
     (MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (buf),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (count),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (datatype),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (source),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (tag),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (comm),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (status),\
      MPI_Abort(MPI_COMM_WORLD,0))
#define MPI_Get_count(status, datatype,count) \
     (MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (status),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (datatype),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (count),\
      MPI_Abort(MPI_COMM_WORLD,0))
#define MPI_Bsend(buf,count,datatype,dest,tag,comm)  \
     (MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (buf),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (count),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (datatype),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (dest),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (tag),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (comm),\
      MPI_Abort(MPI_COMM_WORLD,0))
#define MPI_Ssend(buf,count, datatype,dest,tag,comm) \
     (MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (buf),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (count),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (datatype),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (dest),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (tag),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (comm),\
      MPI_Abort(MPI_COMM_WORLD,0))
#define MPI_Rsend(buf,count, datatype,dest,tag,comm) \
     (MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (buf),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (count),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (datatype),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (dest),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (tag),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (comm),\
      MPI_Abort(MPI_COMM_WORLD,0))
#define MPI_Buffer_attach(buffer,size) \
     (MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (buffer),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (size),\
      MPI_SUCCESS)
#define MPI_Buffer_detach(buffer,size)\
     (MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (buffer),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (size),\
      MPI_SUCCESS)
#define MPI_Ibsend(buf,count, datatype,dest,tag,comm,request) \
     (MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (buf),\
       MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (count),\
       MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (datatype),\
       MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (dest),\
       MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (tag),\
       MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (comm),\
       MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (request),\
       MPI_Abort(MPI_COMM_WORLD,0))
#define MPI_Issend(buf,count, datatype,dest,tag,comm,request) \
     (MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (buf),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (count),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (datatype),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (dest),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (tag),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (comm),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (request),\
      MPI_Abort(MPI_COMM_WORLD,0))
#define MPI_Irsend(buf,count, datatype,dest,tag,comm,request) \
     (MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (buf),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (count),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (datatype),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (dest),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (tag),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (comm),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (request),\
      MPI_Abort(MPI_COMM_WORLD,0))
#define MPI_Irecv(buf,count, datatype,source,tag,comm,request) \
     (MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (buf),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (count),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (datatype),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (source),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (tag),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (comm),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (request),\
      MPI_Abort(MPI_COMM_WORLD,0))
#define MPI_Isend(buf,count, datatype,dest,tag,comm,request) \
     (MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (buf),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (count),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (datatype),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (dest),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (tag),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (comm),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (request),\
      MPI_Abort(MPI_COMM_WORLD,0))
#define MPI_Wait(request,status) \
     (MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (request),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (status),\
      MPI_SUCCESS)
#define MPI_Test(request,flag,status) \
     (MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (request),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (status),\
      *(flag) = 0, \
      MPI_SUCCESS)
#define MPI_Request_free(request) \
     (MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (request),\
      MPI_SUCCESS)
#define MPI_Waitany(a,b,c,d) \
     (MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (a),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (b),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (c),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (d),(*c = 0), \
      MPI_SUCCESS)
#define MPI_Testany(a,b,c,d,e) \
     (MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (a),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (b),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (c),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (d),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (e),\
      MPI_SUCCESS)
#define MPI_Waitall(count,array_of_requests,array_of_statuses) \
     (MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (count),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (array_of_requests),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (array_of_statuses),\
      MPI_SUCCESS)
#define MPI_Testall(count,array_of_requests,flag,array_of_statuses) \
     (MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (count),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (array_of_requests),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (flag),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (array_of_statuses),\
      MPI_SUCCESS)
#define MPI_Waitsome(incount,array_of_requests,outcount,\
                     array_of_indices,array_of_statuses) \
     (MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (incount),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (array_of_requests),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (outcount),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (array_of_indices),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (array_of_statuses),\
      MPI_SUCCESS)
#define MPI_Comm_group(comm,group) \
     (MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (comm),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (group),\
      MPI_SUCCESS)
#define MPI_Group_incl(group,n,ranks,newgroup) \
     (MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (group),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (n),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (ranks),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (newgroup),\
      MPI_SUCCESS)
#define MPI_Testsome(incount,array_of_requests,outcount,\
                     array_of_indices,array_of_statuses) MPI_SUCCESS
#define MPI_Iprobe(source,tag,comm,flag,status) (*(flag)=0, MPI_SUCCESS)
#define MPI_Probe(source,tag,comm,status) MPI_SUCCESS
#define MPI_Cancel(request) (MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (request),MPI_SUCCESS)
#define MPI_Test_cancelled(status,flag) (*(flag)=0,MPI_SUCCESS)
#define MPI_Send_init(buf,count, datatype,dest,tag,comm,request) \
     (MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (buf),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (count),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (datatype),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (dest),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (tag),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (comm),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (request),\
     MPI_SUCCESS)
#define MPI_Bsend_init(buf,count, datatype,dest,tag,comm,request) \
     (MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (buf),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (count),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (datatype),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (dest),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (tag),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (comm),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (request),\
     MPI_SUCCESS)
#define MPI_Ssend_init(buf,count, datatype,dest,tag,comm,request) \
     (MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (buf),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (count),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (datatype),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (dest),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (tag),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (comm),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (request),\
     MPI_SUCCESS)
#define MPI_Bsend_init(buf,count, datatype,dest,tag,comm,request) \
     (MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (buf),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (count),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (datatype),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (dest),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (tag),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (comm),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (request),\
     MPI_SUCCESS)
#define MPI_Rsend_init(buf,count, datatype,dest,tag,comm,request) \
     (MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (buf),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (count),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (datatype),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (dest),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (tag),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (comm),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (request),\
     MPI_SUCCESS)
#define MPI_Recv_init(buf,count, datatype,source,tag,comm,request) \
     (MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (buf),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (count),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (datatype),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (source),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (tag),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (comm),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (request),\
     MPI_SUCCESS)
#define MPI_Start(request) (MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (request),MPI_SUCCESS)
#define MPI_Startall(count,array_of_requests) \
     (MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (count),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (array_of_requests),\
     MPI_SUCCESS)
#define MPI_Op_create(function,commute,op) \
     (MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (function),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (commute),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (op),\
     MPI_SUCCESS)
#define MPI_Op_free(op) \
     (MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (op),\
     MPI_SUCCESS)
     /* Need to determine sizeof "sendtype" */
#define MPI_Sendrecv(sendbuf,sendcount, sendtype,\
     dest,sendtag,recvbuf,recvcount,\
     recvtype,source,recvtag,\
     comm,status) \
  MPIUNI_Memcpy(recvbuf,sendbuf,(sendcount) * MPI_sizeof(sendtype))
#define MPI_Sendrecv_replace(buf,count, datatype,dest,sendtag,\
     source,recvtag,comm,status) MPI_SUCCESS
#define MPI_Type_contiguous(count, oldtype,newtype) \
     (*(newtype) = (count)*(oldtype),MPI_SUCCESS)
#define MPI_Type_vector(count,blocklength,stride,oldtype, newtype) MPI_SUCCESS
#define MPI_Type_hvector(count,blocklength,stride,oldtype, newtype) MPI_SUCCESS
#define MPI_Type_indexed(count,array_of_blocklengths,\
     array_of_displacements, oldtype,\
     newtype) MPI_SUCCESS
#define MPI_Type_hindexed(count,array_of_blocklengths,\
     array_of_displacements, oldtype,\
     newtype) MPI_SUCCESS
#define MPI_Type_struct(count,array_of_blocklengths,\
     array_of_displacements,\
     array_of_types, newtype) \
     (MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (count),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (array_of_blocklengths),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (array_of_displacements),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (array_of_types),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (newtype),\
      MPI_SUCCESS)
#define MPI_Address(location,address) \
     (*(address) = (MPIUNI_INTPTR)(char *)(location),MPI_SUCCESS)
#define MPI_Type_extent(datatype,extent) *(extent) = datatype
#define MPI_Type_size(datatype,size) *(size) = datatype
#define MPI_Type_lb(datatype,displacement) \
     MPI_Abort(MPI_COMM_WORLD,0)
#define MPI_Type_ub(datatype,displacement) \
     MPI_Abort(MPI_COMM_WORLD,0)
#define MPI_Type_commit(datatype) (MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (datatype),\
     MPI_SUCCESS)
#define MPI_Type_free(datatype) MPI_SUCCESS
#define MPI_Get_elements(status, datatype,count) \
     MPI_Abort(MPI_COMM_WORLD,0)
#define MPI_Pack(inbuf,incount, datatype,outbuf,\
     outsize,position, comm) \
     MPI_Abort(MPI_COMM_WORLD,0)
#define MPI_Unpack(inbuf,insize,position,outbuf,\
     outcount, datatype,comm) \
     MPI_Abort(MPI_COMM_WORLD,0)
#define MPI_Pack_size(incount, datatype,comm,size) \
     MPI_Abort(MPI_COMM_WORLD,0)
#define MPI_Barrier(comm) \
     (MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (comm),\
     MPI_SUCCESS)
#define MPI_Bcast(buffer,count,datatype,root,comm) \
     (MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (buffer),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (count),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (datatype),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (comm),\
     MPI_SUCCESS)
#define MPI_Gather(sendbuf,sendcount, sendtype,\
     recvbuf,recvcount, recvtype,\
     root,comm) \
     (MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (recvcount),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (root),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (recvtype),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (comm),\
     MPIUNI_Memcpy(recvbuf,sendbuf,(sendcount)*MPI_sizeof(sendtype)),\
     MPI_SUCCESS)
#define MPI_Gatherv(sendbuf,sendcount, sendtype,\
     recvbuf,recvcounts,displs,\
     recvtype,root,comm) \
     (MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (recvcounts),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (displs),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (recvtype),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (root),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (comm),\
     MPIUNI_Memcpy(recvbuf,sendbuf,(sendcount)*MPI_sizeof(sendtype)),\
     MPI_SUCCESS)
#define MPI_Scatter(sendbuf,sendcount, sendtype,\
     recvbuf,recvcount, recvtype,\
     root,comm) \
     (MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (sendbuf),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (sendcount),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (sendtype),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (recvbuf),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (recvcount),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (recvtype),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (root),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (comm),MPI_Abort(MPI_COMM_WORLD,0))
#define MPI_Scatterv(sendbuf,sendcounts,displs,\
     sendtype, recvbuf,recvcount,\
     recvtype,root,comm) \
     (MPIUNI_Memcpy(recvbuf,sendbuf,(recvcount)*MPI_sizeof(recvtype)),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (displs),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (sendtype),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (sendcounts),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (root),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (comm),\
     MPI_SUCCESS)
#define MPI_Allgather(sendbuf,sendcount, sendtype,\
     recvbuf,recvcount, recvtype,comm) \
     (MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (recvcount),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (recvtype),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (comm),\
     MPIUNI_Memcpy(recvbuf,sendbuf,(sendcount)*MPI_sizeof(sendtype)),\
     MPI_SUCCESS)
#define MPI_Allgatherv(sendbuf,sendcount, sendtype,\
     recvbuf,recvcounts,displs,recvtype,comm) \
     (MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (recvcounts),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (displs),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (recvtype),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (comm),\
     MPIUNI_Memcpy((recvbuf),(sendbuf),(sendcount)*MPI_sizeof(sendtype)), \
     MPI_SUCCESS)
#define MPI_Alltoall(sendbuf,sendcount, sendtype,\
     recvbuf,recvcount, recvtype,comm) \
     (MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (recvcount),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (recvtype),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (comm),\
      MPIUNI_Memcpy(recvbuf,sendbuf,(sendcount)*MPI_sizeof(sendtype)),\
      MPI_SUCCESS)
#define MPI_Alltoallv(sendbuf,sendcounts,sdispls,\
     sendtype, recvbuf,recvcounts,\
     rdispls, recvtype,comm) MPI_Abort(MPI_COMM_WORLD,0)
#define MPI_Alltoallw(sendbuf,sendcounts,sdispls,\
     sendtypes, recvbuf,recvcounts,\
     rdispls, recvtypes,comm) MPI_Abort(MPI_COMM_WORLD,0)
#define MPI_Reduce(sendbuf, recvbuf,count,\
     datatype,op,root,comm) \
     (MPIUNI_Memcpy(recvbuf,sendbuf,(count)*MPI_sizeof(datatype)),\
      MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (comm),MPI_SUCCESS)
#define MPI_Allreduce(sendbuf, recvbuf,count,datatype,op,comm) \
    (MPIUNI_Memcpy(recvbuf,sendbuf,(count)*MPI_sizeof(datatype)), \
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (comm),MPI_SUCCESS)
#define MPI_Scan(sendbuf, recvbuf,count,datatype,op,comm) \
     (MPIUNI_Memcpy(recvbuf,sendbuf,(count)*MPI_sizeof(datatype)),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (comm),MPI_SUCCESS)
#define MPI_Exscan(sendbuf, recvbuf,count,datatype,op,comm) MPI_SUCCESS
#define MPI_Reduce_scatter(sendbuf, recvbuf,recvcounts,\
     datatype,op,comm) \
     MPI_Abort(MPI_COMM_WORLD,0)
#define MPI_Group_size(group,size) (*(size)=1,MPI_SUCCESS)
#define MPI_Group_rank(group,rank) (*(rank)=0,MPI_SUCCESS)
#define MPI_Group_translate_ranks (group1,n,ranks1,\
     group2,ranks2) MPI_Abort(MPI_COMM_WORLD,0)
#define MPI_Group_compare(group1,group2,result) \
     (*(result)=1,MPI_SUCCESS)
#define MPI_Group_union(group1,group2,newgroup) MPI_SUCCESS
#define MPI_Group_intersection(group1,group2,newgroup) MPI_SUCCESS
#define MPI_Group_difference(group1,group2,newgroup) MPI_SUCCESS
#define MPI_Group_excl(group,n,ranks,newgroup) MPI_SUCCESS
#define MPI_Group_range_incl(group,n,ranges,newgroup) MPI_SUCCESS
#define MPI_Group_range_excl(group,n,ranges,newgroup) MPI_SUCCESS
#define MPI_Group_free(group) \
     (MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (group),\
     MPI_SUCCESS)
#define MPI_Comm_compare(comm1,comm2,result) \
     (MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (comm1),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (comm2),\
     *(result)=MPI_IDENT,\
     MPI_SUCCESS)
#define MPI_Comm_split(comm,color,key,newcomm) \
  (MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (color),\
  MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (key),\
   MPI_Comm_dup(comm,newcomm))
#define MPI_Comm_test_inter(comm,flag) (*(flag)=1,MPI_SUCCESS)
#define MPI_Comm_remote_size(comm,size) (*(size)=1,MPI_SUCCESS)
#define MPI_Comm_remote_group(comm,group) MPI_SUCCESS
#define MPI_Intercomm_create(local_comm,local_leader,peer_comm,\
     remote_leader,tag,newintercomm) MPI_SUCCESS
#define MPI_Intercomm_merge(intercomm,high,newintracomm) MPI_SUCCESS

#define MPI_Topo_test(comm,status) MPI_SUCCESS
#define MPI_Cart_create(comm_old,ndims,dims,periods,\
     reorder,comm_cart) MPI_SUCCESS
#define MPI_Dims_create(nnodes,ndims,dims) MPI_Abort(MPI_COMM_WORLD,0)
#define MPI_Graph_create(comm,a,b,c,d,e) MPI_SUCCESS
#define MPI_Graphdims_Get(comm,nnodes,nedges) MPI_Abort(MPI_COMM_WORLD,0)
#define MPI_Graph_get(comm,a,b,c,d) MPI_Abort(MPI_COMM_WORLD,0)
#define MPI_Cartdim_get(comm,ndims) MPI_Abort(MPI_COMM_WORLD,0)
#define MPI_Cart_get(comm,maxdims,dims,periods,coords) \
     MPI_Abort(MPI_COMM_WORLD,0)
#define MPI_Cart_rank(comm,coords,rank) MPI_Abort(MPI_COMM_WORLD,0)
#define MPI_Cart_coords(comm,rank,maxdims,coords) \
     MPI_Abort(MPI_COMM_WORLD,0)
#define MPI_Graph_neighbors_count(comm,rank,nneighbors) \
     MPI_Abort(MPI_COMM_WORLD,0)
#define MPI_Graph_neighbors(comm,rank,maxneighbors,neighbors) \
     MPI_Abort(MPI_COMM_WORLD,0)
#define MPI_Cart_shift(comm,direction,disp,rank_source,rank_dest) \
     MPI_Abort(MPI_COMM_WORLD,0)
#define MPI_Cart_sub(comm,remain_dims,newcomm) MPI_Abort(MPI_COMM_WORLD,0)
#define MPI_Cart_map(comm,ndims,dims,periods,newrank) MPI_Abort(MPI_COMM_WORLD,0)
#define MPI_Graph_map(comm,a,b,c,d) MPI_Abort(MPI_COMM_WORLD,0)
#define MPI_Get_processor_name(name,result_len) \
     (MPIUNI_Memcpy(name,"localhost",9*sizeof(char)),name[10] = 0,*(result_len) = 10)
#define MPI_Errhandler_create(function,errhandler) (*(errhandler) = (MPI_Errhandler) 0, MPI_SUCCESS)    
#define MPI_Errhandler_set(comm,errhandler) \
     (MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (comm),\
     MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (errhandler),\
     MPI_SUCCESS)
#define MPI_Errhandler_get(comm,errhandler) MPI_SUCCESS
#define MPI_Errhandler_free(errhandler) MPI_SUCCESS
#define MPI_Error_string(errorcode,string,result_len) MPI_SUCCESS
#define MPI_Error_class(errorcode,errorclass) MPI_SUCCESS
#define MPI_Wtick() 1.0
#define MPI_Wtime() 0.0
#define MPI_Pcontrol(level) MPI_SUCCESS

#define MPI_NULL_COPY_FN   0
#define MPI_NULL_DELETE_FN 0

  /* MPI-IO additions */

#define MPI_File_open(comm,filename,amode,info,mpi_fh) \
  (MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (comm),  \
   MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (filename), \
   MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (amode), \
   MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (info), \
   MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (mpi_fh), \
   MPI_Abort(MPI_COMM_WORLD,0))

#define MPI_File_close(mpi_fh) \
  (MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (mpi_fh),  \
   MPI_Abort(MPI_COMM_WORLD,0))

#define MPI_File_set_view(mpi_fh,disp,etype,filetype,datarep,info) \
  (MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (mpi_fh),  \
   MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (disp), \
   MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (etype), \
   MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (filetype), \
   MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (datarep), \
   MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (info), \
   MPI_Abort(MPI_COMM_WORLD,0))

#define MPI_Type_get_extent(datatype,lb,extent) \
  (MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (datatype),      \
   *(lb) = 0, *(extent) = datatype,0)

#define MPI_File_write_all(mpi_fh,buf,count,datatype,status) \
  (MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (mpi_fh),             \
   MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (buf), \
   MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (count), \
   MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (datatype), \
   MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (status), \
   MPI_Abort(MPI_COMM_WORLD,0))

#define MPI_File_read_all(mpi_fh,buf,count,datatype,status) \
  (MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (mpi_fh),            \
   MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (buf), \
   MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (count), \
   MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (datatype), \
   MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (status), \
   MPI_Abort(MPI_COMM_WORLD,0))

  /* called from PetscInitialize() - so return success */
#define MPI_Register_datarep(name,read_conv_fn,write_conv_fn,extent_fn,state) \
  (MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (name),                          \
   MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (read_conv_fn), \
   MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (write_conv_fn), \
   MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (extent_fn), \
   MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (state), \
   MPI_SUCCESS)

#define MPI_Type_create_subarray(ndims,array_of_sizes,array_of_subsizes,array_of_starts,order,oldtype,newtype) \
  (MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (ndims),                         \
   MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (array_of_sizes), \
   MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (array_of_subsizes), \
   MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (array_of_starts), \
   MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (order), \
   MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (oldtype), \
   MPIUNI_TMP = (void*)(MPIUNI_INTPTR) (newtype), \
   MPI_Abort(MPI_COMM_WORLD,0))

#if defined(__cplusplus)
}
#endif
#endif

