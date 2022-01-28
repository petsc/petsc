
/********************************bit_mask.c************************************

Author: Henry M. Tufo III

e-mail: hmt@cs.brown.edu

snail-mail:
Division of Applied Mathematics
Brown University
Providence, RI 02912

Last Modification:
11.21.97
*********************************bit_mask.c***********************************/
#include <../src/ksp/pc/impls/tfs/tfs.h>

/*********************************bit_mask.c***********************************/
PetscErrorCode PCTFS_bm_to_proc(char *ptr, PetscInt p_mask,  PetscInt *msg_list)
{
  PetscInt i, tmp;

  PetscFunctionBegin;
  if (msg_list) {
    /* low to high */
    ptr+=(p_mask-1);
    for (i=p_mask-1;i>=0;i--) {
      tmp = BYTE*(p_mask-i-1);
      if (*ptr&BIT_0) {
        *msg_list = tmp; msg_list++;
      }
      if (*ptr&BIT_1) {
        *msg_list = tmp+1; msg_list++;
      }
      if (*ptr&BIT_2) {
        *msg_list = tmp+2; msg_list++;
      }
      if (*ptr&BIT_3) {
        *msg_list = tmp+3; msg_list++;
      }
      if (*ptr&BIT_4) {
        *msg_list = tmp+4; msg_list++;
      }
      if (*ptr&BIT_5) {
        *msg_list = tmp+5; msg_list++;
      }
      if (*ptr&BIT_6) {
        *msg_list = tmp+6; msg_list++;
      }
      if (*ptr&BIT_7) {
        *msg_list = tmp+7; msg_list++;
      }
      ptr--;
    }
  }
  PetscFunctionReturn(0);
}

/*********************************bit_mask.c***********************************/
PetscInt PCTFS_ct_bits(char *ptr, PetscInt n)
{
  PetscInt i, tmp=0;

  for (i=0;i<n;i++) {
    if (*ptr&128) tmp++;
    if (*ptr&64)  tmp++;
    if (*ptr&32)  tmp++;
    if (*ptr&16)  tmp++;
    if (*ptr&8)   tmp++;
    if (*ptr&4)   tmp++;
    if (*ptr&2)   tmp++;
    if (*ptr&1)   tmp++;
    ptr++;
  }
  return(tmp);
}

/*********************************bit_mask.c***********************************/
PetscInt PCTFS_div_ceil(PetscInt numer,  PetscInt denom)
{
  if ((numer<0)||(denom<=0)) SETERRABORT(PETSC_COMM_SELF,PETSC_ERR_PLIB,"PCTFS_div_ceil() :: numer=%" PetscInt_FMT " ! >=0, denom=%" PetscInt_FMT " ! >0",numer,denom);
  return(PetscCeilInt(numer,denom));
}

/*********************************bit_mask.c***********************************/
PetscInt PCTFS_len_bit_mask(PetscInt num_items)
{
  PetscInt rt_val, tmp;

  PetscAssertFalse(num_items<0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Value Sent To PCTFS_len_bit_mask() Must be >= 0!");

  rt_val = PetscCeilInt(num_items,BYTE);
  /* make multiple of sizeof PetscInt */
  if ((tmp=rt_val%sizeof(PetscInt))) rt_val+=(sizeof(PetscInt)-tmp);
  return(rt_val);
}

/*********************************bit_mask.c***********************************/
PetscErrorCode PCTFS_set_bit_mask(PetscInt *bm, PetscInt len, PetscInt val)
{
  PetscInt i, offset;
  char     mask = 1;
  char     *cptr;

   PetscFunctionBegin;
  PetscAssertFalse(PCTFS_len_bit_mask(val)>len,PETSC_COMM_SELF,PETSC_ERR_PLIB,"The Bit Mask Isn't That Large!");

  cptr = (char*) bm;

  offset = len/sizeof(PetscInt);
  for (i=0; i<offset; i++) {
    *bm=0;
    bm++;
  }

  offset = val%BYTE;
  for (i=0;i<offset;i++) {
    mask <<= 1;
  }

  offset       = len - val/BYTE - 1;
  cptr[offset] = mask;
  PetscFunctionReturn(0);
}

/*********************************bit_mask.c***********************************/
PetscInt PCTFS_len_buf(PetscInt item_size, PetscInt num_items)
{
  PetscInt rt_val, tmp;

  rt_val = item_size * num_items;

  /*  double precision align for now ... consider page later */
  if ((tmp = (rt_val%(PetscInt)sizeof(double)))) rt_val += (sizeof(double) - tmp);
  return(rt_val);
}

