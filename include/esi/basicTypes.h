#ifndef __ESI_basicTypes_h
#define __ESI_basicTypes_h

/** This header is compatible with ANSI C or C++ compilers.
    This defines some minimum infrastructure to get an
    Equation Solver Interface prototype going.
*/

#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#define ESI_MAJOR_VERSION    0
#define ESI_MINOR_VERSION    9
#define ESI_PATCH_LEVEL      6

namespace esi {

/** ESI return codes

    \verbatim
    All ESI functions returning an int error code follow this
    convention:
    
       N == 0: all ok.
       N < 0: some error condition logically fatal has occured.
       N > 0: some informational condition.

    The non-zero values of N are (for ESI compliance purposes)
    significant only in the sign bit. Individual implementations may 
    assign significance to bits other than the sign bit, for instance
    to carry an enum.
    \endverbatim
*/

/** ESI method return codes and message string */
#define ESI_ERROR_CHARS 2030
typedef char ErrorMsg[ESI_ERROR_CHARS];
typedef int ErrorCode;

/** ESI preconditioner side terms */
typedef enum { PRECONDITIONER_LEFT, 
               PRECONDITIONER_RIGHT,
               PRECONDITIONER_TWO_SIDED } PreconditionerSide;

/** ESI precision handling.

    We deal with variations in int, float, and complex precision
    by passing data through interfaces as (void *, char *) pairs.
    The format of the char * is fixed; it contains one or more of 
    I<m>, R<n>, C<p> where m, n, and p are the number of bytes 
    used to represent integer, real, and 'a' of the complex a+bi 
    pair. This string is fixed 10 characters long, for easy f77 
    compatibility.
*/
#define ESI_PRECISION_CHARS 10
typedef char Precision[ESI_PRECISION_CHARS];

/* Since const is in disfavor, (see http://z.ca.sandia.gov/~esi-dev)
 * we have the following define to preserve visual hints regarding
 * what is in vs out/inout in some sIDLized future.
 * This is easier than trying to go back and rediscover what we think
 * should be "in" later.
 * C/C++ compilers will not see const any more this way.
 * Anywhere in a prototype that you have the urge to put const
 * or hint to developers that 'mathematically' an object is invariant
 * in its use, stick a CONST in front of it (or after in the case of
 * member functions in C++).
 */
#define CONST

#ifdef __cplusplus
#define VD(cppclass) virtual ~cppclass(){}
#endif /* __cplusplus */

/* esi long is 64 bits. changes this define to whatever you must
   to make that true.
*/
#define ESI_long long long

/* ESI_int is the machine definition of int (which may vary in size*/
#define ESI_int int

/* Include the file of macro definitions used to configure esi */
/* This file will define, in the autoconf way,
   SIZEOF_DOUBLE,
   SIZEOF_FLOAT, 
   SIZEOF_LONG_LONG,
   SIZEOF_LONG,
   SIZEOF_INT,
   SIZEOF_VOID_P
   and other configure time parameters that must be the same
   for all libraries/apps to adhere to ESI on a given platform.
*/
#include "./config.h"

/** Archaic typedefs, but left in for backwards compatibility. */
typedef long esi_long;
typedef int esi_int;

/** Basic integeger, 'real', and 'complex' type definitions */

#if (SIZEOF_DOUBLE==8)
typedef double real8;
typedef struct {double a; double b;} complex8;
#define ESI_HAVE_REAL8 1
#endif

#if (SIZEOF_FLOAT == 4)
typedef float real4;
typedef struct {float a; float b;} complex4;
#define ESI_HAVE_REAL4
#endif

/* the following case is not reported to exist for
   current c/c++ compilers on any architecture. It
   may occur for FORTRAN compilers on CRAY architectures.
*/
#if (SIZEOF_FLOAT==8)
#ifndef ESI_HAVE_REAL8
typedef float real8;
typedef struct {float a; float b;} complex8;
#define ESI_HAVE_REAL8 1
#endif
#define real4 exit(1)
#endif

#if (SIZEOF_LONG_LONG == 8)
typedef long long int8;
#define ESI_HAVE_INT8 1
#define ESI_INT8_MAX         (9223372036854775807LL)
#endif

#ifndef ESI_HAVE_INT8
#if (SIZEOF_LONG == 8)
typedef long int8;
#define ESI_HAVE_INT8 1
#define ESI_INT8_MAX         (9223372036854775807L)
#endif
#endif

#if (SIZEOF_INT == 4)
typedef int int4;
#define ESI_HAVE_INT4 1
#endif

# define ESI_INT1_MAX         (127)
# define ESI_INT2_MAX         (32767)
# define ESI_INT4_MAX         (2147483647)

/** If you intend to use ESI with MPI code, your app/library 
    needs to include esi/ESI-MPI.h after esi/ESI.h, both of 
    which are found in esi/cxx/include (the ESI header 
    include path directory).  ESI-MPI.h is a generated file
    which remaps the following macros so that ESI named 
    types will pass where an MPI Data_type is needed.
*/
#define ESI_MPI_INT4 MPI_DATATYPE_NULL
#define ESI_MPI_INT8 MPI_DATATYPE_NULL
#define ESI_MPI_REAL4 MPI_DATATYPE_NULL
#define ESI_MPI_REAL8 MPI_DATATYPE_NULL

};     /* esi namespace */

#endif /* __ESI_basicTypes_h */
