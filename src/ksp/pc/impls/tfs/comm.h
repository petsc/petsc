
/***********************************comm.h*************************************
SPARSE GATHER-SCATTER PACKAGE: bss_malloc bss_malloc ivec error comm gs queue

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
#ifndef _comm_h
#define _comm_h


/***********************************comm.h*************************************
Function:

Input : 
Output: 
Return: 
Description: 
Usage: 
***********************************comm.h*************************************/
extern int my_id;
extern int num_nodes;
extern int floor_num_nodes;
extern int i_log2_num_nodes;

extern void giop(int *vals, int *work, int n, int *oprs);
extern void grop(REAL *vals, REAL *work, int n, int *oprs);
extern void gfop(void *vals, void *wk, int n, vbfp fp, DATA_TYPE dt, int comm_type);
extern void comm_init(void);
extern void giop_hc(int *vals, int *work, int n, int *oprs, int dim);
extern void grop_hc(REAL *vals, REAL *work, int n, int *oprs, int dim);
extern void grop_hc_vvl(REAL *vals, REAL *work, int *n, int *oprs, int dim);
extern void ssgl_radd(REAL *vals, REAL *work, int level, int *segs);

#if defined(_CRAY)
#define MSGTAG0 101
#define MSGTAG1 1001
#define MSGTAG2 30002
#define MSGTAG3 10001
#define MSGTAG4 12003
#define MSGTAG5 17001
#define MSGTAG6 22002
#else
#define MSGTAG0 101
#define MSGTAG1 1001
#define MSGTAG2 76207
#define MSGTAG3 100001
#define MSGTAG4 163841
#define MSGTAG5 249439
#define MSGTAG6 10000001
#endif
#endif

