
/**********************************const.h*************************************

Author: Henry M. Tufo III

e-mail: hmt@cs.brown.edu

snail-mail:
Division of Applied Mathematics
Brown University
Providence, RI 02912

Last Modification: 
6.21.97
***********************************const.h************************************/

/**********************************const.h*************************************
File Description:
-----------------

***********************************const.h************************************/
#include "petsc.h"
#include <limits.h>
#include "petscblaslapack.h"
#if defined(PETSC_HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#if defined(PETSC_HAVE_MALLOC_H) && !defined(__cplusplus)
#include <malloc.h>
#endif

#define X          0
#define Y          1
#define Z          2
#define XY         3
#define XZ         4
#define YZ         5


#define THRESH          0.2
#define N_HALF          4096
#define PRIV_BUF_SZ     45

/*4096 8192 32768 65536 1048576 */
#define MAX_MSG_BUF     32768

/* fortran gs limit */
#define MAX_GS_IDS      100

#define FULL          2
#define PARTIAL       1
#define NONE          0

#define BYTE		8
#define BIT_0		0x1
#define BIT_1		0x2
#define BIT_2		0x4
#define BIT_3		0x8
#define BIT_4		0x10
#define BIT_5		0x20
#define BIT_6		0x40
#define BIT_7		0x80
#define TOP_BIT         INT_MIN
#define ALL_ONES        -1

#define FALSE		0
#define TRUE		1

#define C		0
#define FORTRAN 	1


#define MAX_VEC		1674
#define FORMAT		30
#define MAX_COL_LEN    	100
#define MAX_LINE	FORMAT*MAX_COL_LEN
#define   DELIM         " \n \t"
#define LINE		12
#define C_LINE		80

#define REAL_MAX	DBL_MAX
#define REAL_MIN	DBL_MIN

#define   UT            5               /* dump upper 1/2 */
#define   LT            6               /* dump lower 1/2 */
#define   SYMM          8               /* we assume symm and dump upper 1/2 */
#define   NON_SYMM      9

#define   ROW          10
#define   COL          11

#define EPS   1.0e-14
#define EPS2  1.0e-07


#define MPI   1
#define NX    2


#define LOG2(x)		(PetscScalar)log((double)x)/log(2)
#define SWAP(a,b)       temp=(a); (a)=(b); (b)=temp;
#define P_SWAP(a,b)     ptr=(a); (a)=(b); (b)=ptr;

#define MAX_FABS(x,y)   ((double)fabs(x)>(double)fabs(y)) ? ((PetscScalar)x) : ((PetscScalar)y)
#define MIN_FABS(x,y)   ((double)fabs(x)<(double)fabs(y)) ? ((PetscScalar)x) : ((PetscScalar)y)

/* specer's existence ... can be done w/MAX_ABS */
#define EXISTS(x,y)     ((x)==0.0) ? (y) : (x)

#define MULT_NEG_ONE(a) (a) *= -1;
#define NEG(a)          (a) |= BIT_31;
#define POS(a)          (a) &= INT_MAX;



