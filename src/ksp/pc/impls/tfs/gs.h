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
#ifndef _gs_h
#define _gs_h

/***********************************gs.h***************************************
Type: gs_ADT
------------

************************************gs.h**************************************/

typedef struct gather_scatter_id *gs_ADT;
typedef void (*Rbfp)(PetscScalar *, PetscScalar *, int len);

/***********************************gs.h***************************************
Function:

Input : 
Output: 
Return: 
Description: 
Usage: 
************************************gs.h**************************************/
extern gs_ADT gs_init(int *elms, int nel, int level);
extern void   gs_gop(gs_ADT gs_handle, PetscScalar *vals, const char *op);
extern void   gs_gop_vec(gs_ADT gs_handle, PetscScalar *vals, const char *op, int step);
extern void   gs_gop_binary(gs_ADT gs, PetscScalar *vals, Rbfp fct);
extern void   gs_gop_hc(gs_ADT gs_handle, PetscScalar *vals, const char *op, int dim);
extern void   gs_free(gs_ADT gs_handle);
extern void   gs_init_msg_buf_sz(int buf_size);
extern void   gs_init_vec_sz(int size);

#endif

