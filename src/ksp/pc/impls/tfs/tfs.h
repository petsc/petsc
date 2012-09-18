
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
#include <petscsys.h>
#include <petscblaslapack.h>
#if defined(PETSC_HAVE_STDLIB_H)
#include <stdlib.h>
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
#define TOP_BIT         PETSC_MIN_INT

#define C		0


#define MAX_VEC		1674
#define FORMAT		30
#define MAX_COL_LEN    	100
#define MAX_LINE	FORMAT*MAX_COL_LEN
#define   DELIM         " \n \t"
#define LINE		12
#define C_LINE		80

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

#define MAX_FABS(x,y)   (PetscAbsScalar(x)>PetscAbsScalar(y)) ? ((PetscScalar)x) : ((PetscScalar)y)
#define MIN_FABS(x,y)   (PetscAbsScalar(x)<PetscAbsScalar(y)) ? ((PetscScalar)x) : ((PetscScalar)y)

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

typedef PetscErrorCode (*vfp)(void*,void*,PetscInt,...);
typedef PetscErrorCode (*rbfp)(PetscScalar *, PetscScalar *, PetscInt len);
typedef PetscInt (*bfp)(void*, void *, PetscInt *len, MPI_Datatype *dt);

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
extern PetscMPIInt PCTFS_my_id;
extern PetscMPIInt PCTFS_num_nodes;
extern PetscMPIInt PCTFS_floor_num_nodes;
extern PetscMPIInt PCTFS_i_log2_num_nodes;

extern PetscErrorCode PCTFS_giop(PetscInt *vals, PetscInt *work, PetscInt n, PetscInt *oprs);
extern PetscErrorCode PCTFS_grop(PetscScalar *vals, PetscScalar *work, PetscInt n, PetscInt *oprs);
extern PetscErrorCode PCTFS_comm_init(void);
extern PetscErrorCode PCTFS_giop_hc(PetscInt *vals, PetscInt *work, PetscInt n, PetscInt *oprs, PetscInt dim);
extern PetscErrorCode PCTFS_grop_hc(PetscScalar *vals, PetscScalar *work, PetscInt n, PetscInt *oprs, PetscInt dim);
extern PetscErrorCode PCTFS_ssgl_radd(PetscScalar *vals, PetscScalar *work, PetscInt level, PetscInt *segs);

#define MSGTAG0 101
#define MSGTAG1 1001
#define MSGTAG2 76207
#define MSGTAG3 100001
#define MSGTAG4 163841
#define MSGTAG5 249439
#define MSGTAG6 10000001

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

extern PetscInt *PCTFS_ivec_copy(PetscInt *arg1, PetscInt *arg2, PetscInt n);

extern PetscErrorCode PCTFS_ivec_zero(PetscInt *arg1, PetscInt n);
extern PetscErrorCode PCTFS_ivec_set(PetscInt *arg1, PetscInt arg2, PetscInt n);

extern PetscInt PCTFS_ivec_lb(PetscInt *work, PetscInt n);
extern PetscInt PCTFS_ivec_ub(PetscInt *work, PetscInt n);
extern PetscInt PCTFS_ivec_sum(PetscInt *arg1, PetscInt n);

extern vfp PCTFS_ivec_fct_addr(PetscInt type);

extern PetscErrorCode PCTFS_ivec_non_uniform(PetscInt *arg1, PetscInt *arg2, PetscInt n, PetscInt *arg3);
extern PetscErrorCode PCTFS_ivec_max(PetscInt *arg1, PetscInt *arg2, PetscInt n);
extern PetscErrorCode PCTFS_ivec_min(PetscInt *arg1, PetscInt *arg2, PetscInt n);
extern PetscErrorCode PCTFS_ivec_mult(PetscInt *arg1, PetscInt *arg2, PetscInt n);
extern PetscErrorCode PCTFS_ivec_add(PetscInt *arg1, PetscInt *arg2, PetscInt n);
extern PetscErrorCode PCTFS_ivec_xor(PetscInt *arg1, PetscInt *arg2, PetscInt n);
extern PetscErrorCode PCTFS_ivec_or(PetscInt *arg1, PetscInt *arg2, PetscInt len);
extern PetscErrorCode PCTFS_ivec_and(PetscInt *arg1, PetscInt *arg2, PetscInt len);
extern PetscErrorCode PCTFS_ivec_lxor(PetscInt *arg1, PetscInt *arg2, PetscInt n);
extern PetscErrorCode PCTFS_ivec_lor(PetscInt *arg1, PetscInt *arg2, PetscInt len);
extern PetscErrorCode PCTFS_ivec_land(PetscInt *arg1, PetscInt *arg2, PetscInt len);
extern PetscErrorCode PCTFS_ivec_and3( PetscInt *arg1,  PetscInt *arg2,  PetscInt *arg3, PetscInt n);

extern PetscErrorCode PCTFS_ivec_sort_companion(PetscInt *ar, PetscInt *ar2, PetscInt size);
extern PetscErrorCode PCTFS_ivec_sort(PetscInt *ar, PetscInt size);
extern PetscErrorCode PCTFS_SMI_sort(void *ar1, void *ar2, PetscInt size, PetscInt type);
extern PetscInt PCTFS_ivec_binary_search(PetscInt item, PetscInt *list, PetscInt n);
extern PetscInt PCTFS_ivec_linear_search(PetscInt item, PetscInt *list, PetscInt n);

extern PetscErrorCode PCTFS_ivec_sort_companion_hack(PetscInt *ar, PetscInt **ar2, PetscInt size);

#define SORT_INTEGER 1
#define SORT_INT_PTR 2

extern PetscErrorCode PCTFS_rvec_zero(PetscScalar *arg1, PetscInt n);
extern PetscErrorCode PCTFS_rvec_one(PetscScalar *arg1, PetscInt n);
extern PetscErrorCode PCTFS_rvec_set(PetscScalar *arg1, PetscScalar arg2, PetscInt n);
extern PetscErrorCode PCTFS_rvec_copy(PetscScalar *arg1, PetscScalar *arg2, PetscInt n);
extern PetscErrorCode PCTFS_rvec_scale(PetscScalar *arg1, PetscScalar arg2, PetscInt n);

extern vfp PCTFS_rvec_fct_addr(PetscInt type);
extern PetscErrorCode PCTFS_rvec_add(PetscScalar *arg1, PetscScalar *arg2, PetscInt n);
extern PetscErrorCode PCTFS_rvec_mult(PetscScalar *arg1, PetscScalar *arg2, PetscInt n);
extern PetscErrorCode PCTFS_rvec_max(PetscScalar *arg1, PetscScalar *arg2, PetscInt n);
extern PetscErrorCode PCTFS_rvec_max_abs(PetscScalar *arg1, PetscScalar *arg2, PetscInt n);
extern PetscErrorCode PCTFS_rvec_min(PetscScalar *arg1, PetscScalar *arg2, PetscInt n);
extern PetscErrorCode PCTFS_rvec_min_abs(PetscScalar *arg1, PetscScalar *arg2, PetscInt n);
extern PetscErrorCode PCTFS_vec_exists(PetscScalar *arg1, PetscScalar *arg2, PetscInt n);

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

typedef struct gather_scatter_id *PCTFS_gs_ADT;
typedef PetscErrorCode (*Rbfp)(PetscScalar *, PetscScalar *, PetscInt len);

extern PCTFS_gs_ADT PCTFS_gs_init(PetscInt *elms, PetscInt nel, PetscInt level);
extern PetscErrorCode   PCTFS_gs_gop_vec(PCTFS_gs_ADT PCTFS_gs_handle, PetscScalar *vals, const char *op, PetscInt step);
extern PetscErrorCode   PCTFS_gs_gop_binary(PCTFS_gs_ADT gs, PetscScalar *vals, Rbfp fct);
extern PetscErrorCode   PCTFS_gs_gop_hc(PCTFS_gs_ADT PCTFS_gs_handle, PetscScalar *vals, const char *op, PetscInt dim);
extern PetscErrorCode   PCTFS_gs_free(PCTFS_gs_ADT PCTFS_gs_handle);
extern PetscErrorCode   PCTFS_gs_init_msg_buf_sz(PetscInt buf_size);
extern PetscErrorCode   PCTFS_gs_init_vec_sz(PetscInt size);

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

typedef struct xxt_CDT *xxt_ADT;


/*************************************xxt.h************************************
Function: XXT_new()

Return: ADT ptr or NULL upon failure.
Description: This function allocates and returns an xxt handle
Usage: xxt_handle = xxt_new();
**************************************xxt.h***********************************/
extern xxt_ADT XXT_new(void);


/*************************************xxt.h************************************
Function: XXT_free()

Input : pointer to ADT.

Description: This function frees the storage associated with an xxt handle
Usage: XXT_free(xxt_handle);
**************************************xxt.h***********************************/
extern PetscInt XXT_free(xxt_ADT xxt_handle);


/*************************************xxt.h************************************
Function: XXT_factor

Input : ADT ptr,  and pointer to object
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
extern PetscInt XXT_factor(xxt_ADT xxt_handle,   /* prev. allocated xxt  handle */
                      PetscInt *local2global,    /* global column mapping       */
		      PetscInt n,                /* local num rows              */
		      PetscInt m,                /* local num cols              */
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
**************************************xxt.h***********************************/
extern PetscInt XXT_solve(xxt_ADT xxt_handle, PetscScalar *x, PetscScalar *b);

/*************************************xxt.h************************************
Function: XXT_stats

Input : handle
**************************************xxt.h***********************************/
extern PetscInt XXT_stats(xxt_ADT xxt_handle);


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

typedef struct xyt_CDT *xyt_ADT;


/*************************************xyt.h************************************
Function: XYT_new()

Return: ADT ptr or NULL upon failure.
Description: This function allocates and returns an xyt handle
Usage: xyt_handle = xyt_new();
**************************************xyt.h***********************************/
extern xyt_ADT XYT_new(void);


/*************************************xyt.h************************************
Function: XYT_free()

Input : pointer to ADT.
Description: This function frees the storage associated with an xyt handle
Usage: XYT_free(xyt_handle);
**************************************xyt.h***********************************/
extern PetscInt XYT_free(xyt_ADT xyt_handle);


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
extern PetscInt XYT_factor(xyt_ADT xyt_handle,   /* prev. allocated xyt  handle */
                      PetscInt *local2global,    /* global column mapping       */
		      PetscInt n,                /* local num rows              */
		      PetscInt m,                /* local num cols              */
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
extern PetscInt XYT_solve(xyt_ADT xyt_handle, PetscScalar *x, PetscScalar *b);


/*************************************xyt.h************************************
Function: XYT_stats

Input : handle
**************************************xyt.h***********************************/
extern PetscInt XYT_stats(xyt_ADT xyt_handle);


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
extern PetscInt PCTFS_div_ceil(PetscInt numin, PetscInt denom);
extern PetscErrorCode PCTFS_set_bit_mask(PetscInt *bm, PetscInt len, PetscInt val);
extern PetscInt PCTFS_len_bit_mask(PetscInt num_items);
extern PetscInt PCTFS_ct_bits(char *ptr, PetscInt n);
extern PetscErrorCode PCTFS_bm_to_proc(char *ptr, PetscInt p_mask, PetscInt *msg_list);
extern PetscInt PCTFS_len_buf(PetscInt item_size, PetscInt num_items);

#endif

