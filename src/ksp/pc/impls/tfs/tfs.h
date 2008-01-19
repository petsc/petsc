
#if !defined(__TFS_H)
#define __TFS_H

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
#include "petscsys.h"
#if defined(PETSC_HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#if defined(PETSC_HAVE_MALLOC_H)
#include <malloc.h>
#endif
#include "petscblaslapack.h"
#include <limits.h>
#include <float.h>

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




/**********************************types.h*************************************

Author: Henry M. Tufo III

e-mail: hmt@cs.brown.edu

snail-mail:
Division of Applied Mathematics
Brown University
Providence, RI 02912

Last Modification: 
6.21.97
***********************************types.h************************************/

/**********************************types.h*************************************
File Description:
-----------------

***********************************types.h************************************/
typedef PetscErrorCode (*vfp)(void*,void*,int,...);
typedef PetscErrorCode (*rbfp)(PetscScalar *, PetscScalar *, int len);
#define vbfp MPI_User_function *
typedef int (*bfp)(void*, void *, int *len, MPI_Datatype *dt); 

/***********************************comm.h*************************************

Author: Henry M. Tufo III

e-mail: hmt@cs.brown.edu

snail-mail:
Division of Applied Mathematics
Brown University
Providence, RI 02912

Last Modification: 
6.21.97
***********************************comm.h*************************************/

/***********************************comm.h*************************************
File Description:
-----------------

***********************************comm.h*************************************/

/***********************************comm.h*************************************
Function:

Input : 
Output: 
Return: 
Description: 
Usage: 
***********************************comm.h*************************************/
extern PetscMPIInt my_id;
extern PetscMPIInt num_nodes;
extern PetscMPIInt floor_num_nodes;
extern PetscMPIInt i_log2_num_nodes;

extern PetscErrorCode giop(int *vals, int *work, int n, int *oprs);
extern PetscErrorCode grop(PetscScalar *vals, PetscScalar *work, int n, int *oprs);
extern PetscErrorCode gfop(void *vals, void *wk, int n, vbfp fp, MPI_Datatype dt, int comm_type);
extern PetscErrorCode comm_init(void);
extern PetscErrorCode giop_hc(int *vals, int *work, int n, int *oprs, int dim);
extern PetscErrorCode grop_hc(PetscScalar *vals, PetscScalar *work, int n, int *oprs, int dim);
extern PetscErrorCode grop_hc_vvl(PetscScalar *vals, PetscScalar *work, int *n, int *oprs, int dim);
extern PetscErrorCode ssgl_radd(PetscScalar *vals, PetscScalar *work, int level, int *segs);

#define MSGTAG0 101
#define MSGTAG1 1001
#define MSGTAG2 76207
#define MSGTAG3 100001
#define MSGTAG4 163841
#define MSGTAG5 249439
#define MSGTAG6 10000001


/**********************************error.h*************************************

Author: Henry M. Tufo III

e-mail: hmt@cs.brown.edu

snail-mail:
Division of Applied Mathematics
Brown University
Providence, RI 02912

Last Modification: 
6.21.97
**********************************error.h*************************************/

/**********************************error.h*************************************
File Description:
-----------------

**********************************error.h*************************************/

/**********************************error.h*************************************
Function: error_msg_fatal()

Input : formatted string and arguments.
Output: conversion printed to stdout.
Return: na.
Description: prints error message and terminates program.
Usage: error_msg_fatal("this is my %d'st test",test_num)
**********************************error.h*************************************/
extern PetscErrorCode error_msg_fatal(const char msg[], ...);



/**********************************error.h*************************************
Function: error_msg_warning()

Input : formatted string and arguments.
Output: conversion printed to stdout.
Return: na.
Description: prints error message.
Usage: error_msg_warning("this is my %d'st test",test_num)
**********************************error.h*************************************/
extern PetscErrorCode error_msg_warning(const char msg[], ...);

/*$Id: vector.c,v 1.228 2001/03/23 23:21:22 balay Exp $*/
/**********************************ivec.h**************************************

Author: Henry M. Tufo III

e-mail: hmt@cs.brown.edu

snail-mail:
Division of Applied Mathematics
Brown University
Providence, RI 02912

Last Modification: 
6.21.97
***********************************ivec.h*************************************/

/**********************************ivec.h**************************************
File Description:
-----------------

***********************************ivec.h*************************************/

#define SORT_REAL		1
#define SORT_INTEGER	        0
#define SORT_INT_PTR	        2
           

#define NON_UNIFORM     0
#define GL_MAX          1
#define GL_MIN          2
#define GL_MULT         3
#define GL_ADD          4
#define GL_B_XOR        5
#define GL_B_OR         6
#define GL_B_AND        7
#define GL_L_XOR        8
#define GL_L_OR         9
#define GL_L_AND        10
#define GL_MAX_ABS      11
#define GL_MIN_ABS      12
#define GL_EXISTS       13



/**********************************ivec.h**************************************
Function:

Input : 
Output: 
Return: 
Description: 
Usage: 
***********************************ivec.h*************************************/
extern PetscErrorCode ivec_dump(int *v, int n, int tag, int tag2, char * s);
extern PetscErrorCode ivec_lb_ub(int *arg1, int n, int *lb, int *ub);
extern int *ivec_copy(int *arg1, int *arg2, int n);
/*void ivec_copy(int *arg1, int *arg2, int n); */

extern PetscErrorCode ivec_comp(int *arg1, int n);

extern int ivec_reduce_and(int *arg1, int n);
extern int ivec_reduce_or(int *arg1, int n);

extern PetscErrorCode ivec_zero(int *arg1, int n);
extern PetscErrorCode ivec_pos_one(int *arg1, int n);
extern PetscErrorCode ivec_neg_one(int *arg1, int n);
extern PetscErrorCode ivec_set(int *arg1, int arg2, int n);
extern int ivec_cmp(int *arg1, int *arg2, int n);

extern int ivec_lb(int *work, int n);
extern int ivec_ub(int *work, int n);
extern int ivec_sum(int *arg1, int n);
extern int ivec_u_sum(unsigned *arg1, int n);
extern int ivec_prod(int *arg1, int n);

extern vfp ivec_fct_addr(int type);

extern PetscErrorCode ivec_non_uniform(int *arg1, int *arg2, int n, int *arg3);
extern PetscErrorCode ivec_max(int *arg1, int *arg2, int n);
extern PetscErrorCode ivec_min(int *arg1, int *arg2, int n);
extern PetscErrorCode ivec_mult(int *arg1, int *arg2, int n);
extern PetscErrorCode ivec_add(int *arg1, int *arg2, int n);
extern PetscErrorCode ivec_xor(int *arg1, int *arg2, int n);
extern PetscErrorCode ivec_or(int *arg1, int *arg2, int len);
extern PetscErrorCode ivec_and(int *arg1, int *arg2, int len);
extern PetscErrorCode ivec_lxor(int *arg1, int *arg2, int n);
extern PetscErrorCode ivec_lor(int *arg1, int *arg2, int len);
extern PetscErrorCode ivec_land(int *arg1, int *arg2, int len);

extern PetscErrorCode ivec_or3 (int *arg1, int *arg2, int *arg3, int len);
extern PetscErrorCode ivec_and3(int *arg1, int *arg2, int *arg3, int n);

extern int ivec_split_buf(int *buf1, int **buf2, int size);


extern PetscErrorCode ivec_sort_companion(int *ar, int *ar2, int size);
extern PetscErrorCode ivec_sort(int *ar, int size);
extern PetscErrorCode SMI_sort(void *ar1, void *ar2, int size, int type);
extern int ivec_binary_search(int item, int *list, int n);
extern int ivec_linear_search(int item, int *list, int n);

extern PetscErrorCode ivec_c_index(int *arg1, int n);
extern PetscErrorCode ivec_fortran_index(int *arg1, int n);
extern PetscErrorCode ivec_sort_companion_hack(int *ar, int **ar2, int size);


extern PetscErrorCode rvec_dump(PetscScalar *v, int n, int tag, int tag2, char * s);
extern PetscErrorCode rvec_zero(PetscScalar *arg1, int n);
extern PetscErrorCode rvec_one(PetscScalar *arg1, int n);
extern PetscErrorCode rvec_neg_one(PetscScalar *arg1, int n);
extern PetscErrorCode rvec_set(PetscScalar *arg1, PetscScalar arg2, int n);
extern PetscErrorCode rvec_copy(PetscScalar *arg1, PetscScalar *arg2, int n);
extern PetscErrorCode rvec_lb_ub(PetscScalar *arg1, int n, PetscScalar *lb, PetscScalar *ub);
extern PetscErrorCode rvec_scale(PetscScalar *arg1, PetscScalar arg2, int n);

extern vfp rvec_fct_addr(int type);
extern PetscErrorCode rvec_add(PetscScalar *arg1, PetscScalar *arg2, int n);
extern PetscErrorCode rvec_mult(PetscScalar *arg1, PetscScalar *arg2, int n);
extern PetscErrorCode rvec_max(PetscScalar *arg1, PetscScalar *arg2, int n);
extern PetscErrorCode rvec_max_abs(PetscScalar *arg1, PetscScalar *arg2, int n);
extern PetscErrorCode rvec_min(PetscScalar *arg1, PetscScalar *arg2, int n);
extern PetscErrorCode rvec_min_abs(PetscScalar *arg1, PetscScalar *arg2, int n);
extern PetscErrorCode vec_exists(PetscScalar *arg1, PetscScalar *arg2, int n);


extern PetscErrorCode rvec_sort(PetscScalar *ar, int size);
extern PetscErrorCode rvec_sort_companion(PetscScalar *ar, int *ar2, int size);

extern PetscScalar rvec_dot(PetscScalar *arg1, PetscScalar *arg2, int n);

extern PetscErrorCode rvec_axpy(PetscScalar *arg1, PetscScalar *arg2, PetscScalar scale, int n);

extern int  rvec_binary_search(PetscScalar item, PetscScalar *list, int rh);


/**********************************queue.h*************************************

Author: Henry M. Tufo III

e-mail: hmt@cs.brown.edu

snail-mail:
Division of Applied Mathematics
Brown University
Providence, RI 02912

Last Modification: 
6.21.97
**********************************queue.h*************************************/

/**********************************queue.h*************************************
File Description:
-----------------
  This file provides an interface to a simple queue abstraction.
**********************************queue.h*************************************/

/**********************************queue.h*************************************
Type: queue_ADT
---------------
  This line defines the abstract queue type as a pointer to
  its concrete counterpart.  Clients have no access to the
  underlying representation.
**********************************queue.h*************************************/
typedef struct queue_CDT *queue_ADT;



/**********************************queue.h*************************************
Function: new_queue()

Input : na
Output: na
Return: pointer to ADT.
Description: This function allocates and returns an empty queue.
Usage: queue = new_queue();
**********************************queue.h*************************************/
extern queue_ADT new_queue(void);



/**********************************queue.h*************************************
Function: free_queue()

Input : pointer to ADT.
Output: na
Return: na
Description: This function frees the storage associated with queue but not any
pointer contained w/in.
Usage: free_queue(queue);
**********************************queue.h*************************************/
extern PetscErrorCode free_queue(queue_ADT queue);



/**********************************queue.h*************************************
Function: enqueue()

Input : pointer to ADT and pointer to object
Output: na
Return: na
Description: This function adds obj to the end of the queue.
Usage: enqueue(queue, obj);
**********************************queue.h*************************************/
extern PetscErrorCode enqueue(queue_ADT queue, void *obj);



/**********************************queue.h*************************************
Function: dequeue()  

Input : pointer to ADT
Output: na 
Return: void * to element
Description: This function removes the data value at the head of the queue
and returns it to the client.  dequeueing an empty queue is an error
Usage: obj = dequeue(queue);
**********************************queue.h*************************************/
extern PetscErrorCode *dequeue(queue_ADT queue);



/**********************************queue.h*************************************
Function: len_queue()

Input : pointer to ADT
Output: na
Return: integer number of elements
Description: This function returns the number of elements in the queue.
Usage: n = len_queue(queue);
**********************************queue.h*************************************/
EXTERN int len_queue(queue_ADT queue);



/*$Id: vector.c,v 1.228 2001/03/23 23:21:22 balay Exp $*/
/***********************************gs.h***************************************

Author: Henry M. Tufo III

e-mail: hmt@cs.brown.edu

snail-mail:
Division of Applied Mathematics
Brown University
Providence, RI 02912

Last Modification: 
6.21.97
************************************gs.h**************************************/

/***********************************gs.h***************************************
File Description:
-----------------

************************************gs.h**************************************/

/***********************************gs.h***************************************
Type: gs_ADT
------------

************************************gs.h**************************************/

typedef struct gather_scatter_id *gs_ADT;
typedef PetscErrorCode (*Rbfp)(PetscScalar *, PetscScalar *, int len);

/***********************************gs.h***************************************
Function:

Input : 
Output: 
Return: 
Description: 
Usage: 
************************************gs.h**************************************/
extern gs_ADT gs_init(int *elms, int nel, int level);
extern PetscErrorCode   gs_gop(gs_ADT gs_handle, PetscScalar *vals, const char *op);
extern PetscErrorCode   gs_gop_vec(gs_ADT gs_handle, PetscScalar *vals, const char *op, int step);
extern PetscErrorCode   gs_gop_binary(gs_ADT gs, PetscScalar *vals, Rbfp fct);
extern PetscErrorCode   gs_gop_hc(gs_ADT gs_handle, PetscScalar *vals, const char *op, int dim);
extern PetscErrorCode   gs_free(gs_ADT gs_handle);
extern PetscErrorCode   gs_init_msg_buf_sz(int buf_size);
extern PetscErrorCode   gs_init_vec_sz(int size);



/*************************************xxt.h************************************
Module Name: xxt
Module Info: need xxt.{c,h} gs.{c,h} comm.{c,h} ivec.{c,h} error.{c,h} 

author:  Henry M. Tufo III
e-mail:  hmt@asci.uchicago.edu
contact:
+--------------------------------+--------------------------------+
|MCS Division - Building 221     |Department of Computer Science  |
|Argonne National Laboratory     |Ryerson 152                     |
|9700 S. Cass Avenue             |The University of Chicago       |
|Argonne, IL  60439              |Chicago, IL  60637              |
|(630) 252-5354/5986 ph/fx       |(773) 702-6019/8487 ph/fx       |
+--------------------------------+--------------------------------+

Last Modification: 3.20.01
**************************************xxt.h***********************************/

/*************************************xxt.h************************************
File Description:
**************************************xxt.h***********************************/

/*************************************xxt.h************************************
Notes on Usage: 
**************************************xxt.h***********************************/


typedef struct xxt_CDT *xxt_ADT;


/*************************************xxt.h************************************
Function: XXT_new()

Input :
Output:
Return: ADT ptr or NULL upon failure.
Description: This function allocates and returns an xxt handle
Usage: xxt_handle = xxt_new();
**************************************xxt.h***********************************/
extern xxt_ADT XXT_new(void);


/*************************************xxt.h************************************
Function: XXT_free()

Input : pointer to ADT.
Output:
Return:
Description: This function frees the storage associated with an xxt handle
Usage: XXT_free(xxt_handle);
**************************************xxt.h***********************************/
EXTERN int XXT_free(xxt_ADT xxt_handle);


/*************************************xxt.h************************************
Function: XXT_factor

Input : ADT ptr,  and pointer to object
Output:
Return: 0 on failure, 1 on success
Description: This function sets the xxt solver 

xxt assumptions: given n rows of global coarse matrix (E_loc) where
   o global dofs N = sum_p(n), p=0,P-1 
   (i.e. row dist. with no dof replication)
   (5.21.00 will handle dif replication case)
   o m is the number of columns in E_loc (m>=n)
   o local2global holds global number of column i (i=0,...,m-1)
   o local2global holds global number of row    i (i=0,...,n-1)
   o mylocmatvec performs E_loc . x_loc where x_loc is an vector of
   length m in 1-1 correspondence with local2global
   (note that gs package takes care of communication).
   (note do not zero out upper m-n entries!)
   o mylocmatvec(void *grid_data, double *in, double *out)

ML beliefs/usage: move this to to ML_XXT_factor routine
   o my_ml holds address of ML struct associated w/E_loc, grid_data, grid_tag
   o grid_tag, grid_data, my_ml used in
      ML_Set_CSolve(my_ml, grid_tag, grid_data, ML_Do_CoarseDirect);
   o grid_data used in 
      A_matvec(grid_data,v,u);

Usage: 
**************************************xxt.h***********************************/
extern int XXT_factor(xxt_ADT xxt_handle,   /* prev. allocated xxt  handle */
                      int *local2global,    /* global column mapping       */
		      int n,                /* local num rows              */
		      int m,                /* local num cols              */
		      void *mylocmatvec,    /* b_loc=A_local.x_loc         */
		      void *grid_data       /* grid data for matvec        */
		      );


/*************************************xxt.h************************************
Function: XXT_solve

Input : ADT ptr, b (rhs)
Output: x (soln)
Return:
Description: This function performs x = E^-1.b
Usage: 
XXT_solve(xxt_handle, double *x, double *b)
XXT_solve(xxt_handle, double *x, NULL)
assumes x has been initialized to be b
impl. issue for FORTRAN interface ... punt for now and disallow NULL opt.
**************************************xxt.h***********************************/
extern int XXT_solve(xxt_ADT xxt_handle, double *x, double *b);


/*************************************xxt.h************************************
Function: XXT_stats

Input : handle
Output:
Return:
Description:
factor stats
**************************************xxt.h***********************************/
extern int XXT_stats(xxt_ADT xxt_handle);


/*************************************xxt.h************************************
Function: XXT_sp_1()

Input : pointer to ADT
Output: 
Return: 
Description: sets xxt parameter 1 in xxt_handle
Usage: implement later

void XXT_sp_1(xxt_handle,parameter 1 value)
**************************************xxt.h***********************************/


/*************************************xyt.h************************************
Module Name: xyt
Module Info: need xyt.{c,h} gs.{c,h} comm.{c,h} ivec.{c,h} error.{c,h} 

author:  Henry M. Tufo III
e-mail:  hmt@asci.uchicago.edu
contact:
+--------------------------------+--------------------------------+
|MCS Division - Building 221     |Department of Computer Science  |
|Argonne National Laboratory     |Ryerson 152                     |
|9700 S. Cass Avenue             |The University of Chicago       |
|Argonne, IL  60439              |Chicago, IL  60637              |
|(630) 252-5354/5986 ph/fx       |(773) 702-6019/8487 ph/fx       |
+--------------------------------+--------------------------------+

Last Modification: 3.20.01
**************************************xyt.h***********************************/

/*************************************xyt.h************************************
File Description:
**************************************xyt.h***********************************/

/*************************************xyt.h************************************
Notes on Usage: 
**************************************xyt.h***********************************/



typedef struct xyt_CDT *xyt_ADT;


/*************************************xyt.h************************************
Function: XYT_new()

Input :
Output:
Return: ADT ptr or NULL upon failure.
Description: This function allocates and returns an xyt handle
Usage: xyt_handle = xyt_new();
**************************************xyt.h***********************************/
extern xyt_ADT XYT_new(void);


/*************************************xyt.h************************************
Function: XYT_free()

Input : pointer to ADT.
Output:
Return:
Description: This function frees the storage associated with an xyt handle
Usage: XYT_free(xyt_handle);
**************************************xyt.h***********************************/
EXTERN int XYT_free(xyt_ADT xyt_handle);


/*************************************xyt.h************************************
Function: XYT_factor

Input : ADT ptr,  and pointer to object
Output:
Return: 0 on failure, 1 on success
Description: This function sets the xyt solver 

xyt assumptions: given n rows of global coarse matrix (E_loc) where
   o global dofs N = sum_p(n), p=0,P-1 
   (i.e. row dist. with no dof replication)
   (5.21.00 will handle dif replication case)
   o m is the number of columns in E_loc (m>=n)
   o local2global holds global number of column i (i=0,...,m-1)
   o local2global holds global number of row    i (i=0,...,n-1)
   o mylocmatvec performs E_loc . x_loc where x_loc is an vector of
   length m in 1-1 correspondence with local2global
   (note that gs package takes care of communication).
   (note do not zero out upper m-n entries!)
   o mylocmatvec(void *grid_data, double *in, double *out)

ML beliefs/usage: move this to to ML_XYT_factor routine
   o my_ml holds address of ML struct associated w/E_loc, grid_data, grid_tag
   o grid_tag, grid_data, my_ml used in
      ML_Set_CSolve(my_ml, grid_tag, grid_data, ML_Do_CoarseDirect);
   o grid_data used in 
      A_matvec(grid_data,v,u);

Usage: 
**************************************xyt.h***********************************/
extern int XYT_factor(xyt_ADT xyt_handle,   /* prev. allocated xyt  handle */
                      int *local2global,    /* global column mapping       */
		      int n,                /* local num rows              */
		      int m,                /* local num cols              */
		      void *mylocmatvec,    /* b_loc=A_local.x_loc         */
		      void *grid_data       /* grid data for matvec        */
		      );


/*************************************xyt.h************************************
Function: XYT_solve

Input : ADT ptr, b (rhs)
Output: x (soln)
Return:
Description: This function performs x = E^-1.b
Usage: XYT_solve(xyt_handle, double *x, double *b)
**************************************xyt.h***********************************/
extern int XYT_solve(xyt_ADT xyt_handle, double *x, double *b);


/*************************************xyt.h************************************
Function: XYT_stats

Input : handle
Output:
Return:
Description:
factor stats
**************************************xyt.h***********************************/
extern int XYT_stats(xyt_ADT xyt_handle);


/*************************************xyt.h************************************
Function: XYT_sp_1()

Input : pointer to ADT
Output: 
Return: 
Description: sets xyt parameter 1 in xyt_handle
Usage: implement later

PetscErrorCode XYT_sp_1(xyt_handle,parameter 1 value)
**************************************xyt.h***********************************/

/********************************bit_mask.h************************************

Author: Henry M. Tufo III

e-mail: hmt@cs.brown.edu

snail-mail:
Division of Applied Mathematics
Brown University
Providence, RI 02912

Last Modification: 
11.21.97
*********************************bit_mask.h***********************************/

/********************************bit_mask.h************************************
File Description:
-----------------

*********************************bit_mask.h***********************************/


/********************************bit_mask.h************************************
Function:

Input : 
Output: 
Return: 
Description: 
Usage: 
*********************************bit_mask.h***********************************/
extern int div_ceil(int numin, int denom);
extern PetscErrorCode set_bit_mask(int *bm, int len, int val);
extern int len_bit_mask(int num_items);
extern int ct_bits(char *ptr, int n);
extern PetscErrorCode bm_to_proc(char *ptr, int p_mask, int *msg_list);
extern int len_buf(int item_size, int num_items);

#endif

