#define PETSCKSP_DLL

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
#include "../src/ksp/pc/impls/tfs/tfs.h"


/*********************************bit_mask.c***********************************/
PetscErrorCode bm_to_proc( char *ptr, PetscInt p_mask,  PetscInt *msg_list)
{
   PetscInt i, tmp;

   PetscFunctionBegin;
   if (msg_list)
    {
      /* low to high */
      ptr+=(p_mask-1);
      for (i=p_mask-1;i>=0;i--)
	{
	  tmp = BYTE*(p_mask-i-1);
	  if (*ptr&BIT_0) 
	    {*msg_list = tmp; msg_list++;}
	  if (*ptr&BIT_1) 
	    {*msg_list = tmp+1; msg_list++;}
	  if (*ptr&BIT_2) 
	    {*msg_list = tmp+2; msg_list++;}
	  if (*ptr&BIT_3) 
	    {*msg_list = tmp+3; msg_list++;}
	  if (*ptr&BIT_4)
	    {*msg_list = tmp+4; msg_list++;}
	  if (*ptr&BIT_5)
	    {*msg_list = tmp+5; msg_list++;}
	  if (*ptr&BIT_6)
	    {*msg_list = tmp+6; msg_list++;}
	  if (*ptr&BIT_7) 
	    {*msg_list = tmp+7; msg_list++;}
	  ptr --;
	}
  }
  PetscFunctionReturn(0);
}

/*********************************bit_mask.c***********************************/
PetscInt ct_bits( char *ptr, PetscInt n)
{
   PetscInt i, tmp=0;

   PetscFunctionBegin;
  for(i=0;i<n;i++)
    {
      if (*ptr&128) {tmp++;}
      if (*ptr&64)  {tmp++;}
      if (*ptr&32)  {tmp++;}
      if (*ptr&16)  {tmp++;}
      if (*ptr&8)   {tmp++;}
      if (*ptr&4)   {tmp++;}
      if (*ptr&2)   {tmp++;}
      if (*ptr&1)   {tmp++;}
      ptr++;
    }

  return(tmp);
}

/*********************************bit_mask.c***********************************/ 
PetscInt
div_ceil( PetscInt numer,  PetscInt denom)
{
   PetscInt rt_val;

  if ((numer<0)||(denom<=0))
    {SETERRQ2(PETSC_ERR_PLIB,"div_ceil() :: numer=%D ! >=0, denom=%D ! >0",numer,denom);}

  /* if integer division remainder then increment */
  rt_val = numer/denom;
  if (numer%denom) 
    {rt_val++;}
  
  return(rt_val);
}

/*********************************bit_mask.c***********************************/ 
PetscInt
len_bit_mask( PetscInt num_items)
{
   PetscInt rt_val, tmp;

  if (num_items<0)
    {SETERRQ(PETSC_ERR_PLIB,"Value Sent To len_bit_mask() Must be >= 0!");}

  /* mod BYTE ceiling function */
  rt_val = num_items/BYTE;
  if (num_items%BYTE) 
    {rt_val++;}
  
  /* make mults of sizeof int */
  if ((tmp=rt_val%sizeof(PetscInt))) 
    {rt_val+=(sizeof(PetscInt)-tmp);}

  return(rt_val);
}

/*********************************bit_mask.c***********************************/
PetscErrorCode set_bit_mask( PetscInt *bm, PetscInt len, PetscInt val)
{
   PetscInt i, offset;
   char mask = 1;
  char *cptr;


  if (len_bit_mask(val)>len)
    {SETERRQ(PETSC_ERR_PLIB,"The Bit Mask Isn't That Large!");}

  cptr = (char *) bm;

  offset = len/sizeof(PetscInt);
  for (i=0;i<offset;i++)
    {*bm=0; bm++;}

  offset = val%BYTE;
  for (i=0;i<offset;i++)
    {mask <<= 1;}

  offset = len - val/BYTE - 1;
  cptr[offset] = mask;
  PetscFunctionReturn(0);
}

/*********************************bit_mask.c***********************************/
PetscInt len_buf(PetscInt item_size, PetscInt num_items)
{
   PetscInt rt_val, tmp;

   PetscFunctionBegin;
  rt_val = item_size * num_items;

  /*  double precision align for now ... consider page later */
  if ((tmp = (rt_val%(PetscInt)sizeof(double))))
    {rt_val += (sizeof(double) - tmp);}

  return(rt_val);
}



