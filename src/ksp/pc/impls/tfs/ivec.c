#define PETSCKSP_DLL

/**********************************ivec.c**************************************

Author: Henry M. Tufo III

e-mail: hmt@cs.brown.edu

snail-mail:
Division of Applied Mathematics
Brown University
Providence, RI 02912

Last Modification: 
6.21.97
***********************************ivec.c*************************************/

#include "../src/ksp/pc/impls/tfs/tfs.h"

/* sorting args ivec.c ivec.c ... */
#define   SORT_OPT	6     
#define   SORT_STACK	50000


/* allocate an address and size stack for sorter(s) */
static void *offset_stack[2*SORT_STACK];
static PetscInt   size_stack[SORT_STACK];

/***********************************ivec.c*************************************/
PetscInt *ivec_copy( PetscInt *arg1,  PetscInt *arg2,  PetscInt n)
{
  while (n--)  {*arg1++ = *arg2++;}
  return(arg1);
}

/***********************************ivec.c*************************************/
PetscErrorCode ivec_zero( PetscInt *arg1,  PetscInt n)
{
  PetscFunctionBegin;
  while (n--)  {*arg1++ = 0;}
  PetscFunctionReturn(0);
}

/***********************************ivec.c*************************************/
PetscErrorCode ivec_set( PetscInt *arg1,  PetscInt arg2,  PetscInt n)
{
  PetscFunctionBegin;
  while (n--)  {*arg1++ = arg2;}
  PetscFunctionReturn(0);
}

/***********************************ivec.c*************************************/
PetscErrorCode ivec_max( PetscInt *arg1,  PetscInt *arg2,  PetscInt n)
{
  PetscFunctionBegin;
  while (n--)  {*arg1 = PetscMax(*arg1,*arg2); arg1++; arg2++;}
  PetscFunctionReturn(0);
}

/***********************************ivec.c*************************************/
PetscErrorCode ivec_min( PetscInt *arg1,  PetscInt *arg2,  PetscInt n)
{
  PetscFunctionBegin;
  while (n--)  {*(arg1) = PetscMin(*arg1,*arg2); arg1++; arg2++;}
  PetscFunctionReturn(0);
}

/***********************************ivec.c*************************************/
PetscErrorCode ivec_mult( PetscInt *arg1,  PetscInt *arg2,  PetscInt n)
{
  PetscFunctionBegin;
  while (n--)  {*arg1++ *= *arg2++;}
  PetscFunctionReturn(0);
}

/***********************************ivec.c*************************************/
PetscErrorCode ivec_add( PetscInt *arg1,  PetscInt *arg2,  PetscInt n)
{
  PetscFunctionBegin;
  while (n--)  {*arg1++ += *arg2++;}
  PetscFunctionReturn(0);
}

/***********************************ivec.c*************************************/
PetscErrorCode ivec_lxor( PetscInt *arg1,  PetscInt *arg2,  PetscInt n)
{
  PetscFunctionBegin;
  while (n--) {*arg1=((*arg1 || *arg2) && !(*arg1 && *arg2)); arg1++; arg2++;}
  PetscFunctionReturn(0);
}

/***********************************ivec.c*************************************/
PetscErrorCode ivec_xor( PetscInt *arg1,  PetscInt *arg2,  PetscInt n)
{
  PetscFunctionBegin;
  while (n--)  {*arg1++ ^= *arg2++;}
  PetscFunctionReturn(0);
}

/***********************************ivec.c*************************************/
PetscErrorCode ivec_or( PetscInt *arg1,  PetscInt *arg2,  PetscInt n)
{
  PetscFunctionBegin;
  while (n--)  {*arg1++ |= *arg2++;}
  PetscFunctionReturn(0);
}

/***********************************ivec.c*************************************/
PetscErrorCode ivec_lor( PetscInt *arg1,  PetscInt *arg2,  PetscInt n)
{
  PetscFunctionBegin;
  while (n--)  {*arg1 = (*arg1 || *arg2); arg1++; arg2++;} 
  PetscFunctionReturn(0);
}

/***********************************ivec.c*************************************/
PetscErrorCode ivec_and( PetscInt *arg1,  PetscInt *arg2,  PetscInt n)
{
  PetscFunctionBegin;
  while (n--)  {*arg1++ &= *arg2++;}
  PetscFunctionReturn(0);
}

/***********************************ivec.c*************************************/
PetscErrorCode ivec_land( PetscInt *arg1,  PetscInt *arg2,  PetscInt n)
{
  PetscFunctionBegin;
  while (n--) {*arg1 = (*arg1 && *arg2); arg1++; arg2++;} 
  PetscFunctionReturn(0);
}

/***********************************ivec.c*************************************/
PetscErrorCode ivec_and3( PetscInt *arg1,  PetscInt *arg2,  PetscInt *arg3, PetscInt n)
{
  PetscFunctionBegin;
  while (n--)  {*arg1++ = (*arg2++ & *arg3++);}
  PetscFunctionReturn(0);
}

/***********************************ivec.c*************************************/
PetscInt ivec_sum( PetscInt *arg1,  PetscInt n)
{
   PetscInt tmp = 0;


  while (n--) {tmp += *arg1++;}
  return(tmp);
}

/***********************************ivec.c*************************************/
PetscErrorCode ivec_non_uniform(PetscInt *arg1, PetscInt *arg2,  PetscInt n,  PetscInt *arg3)
{
   PetscInt i, j, type;


  PetscFunctionBegin;
  /* LATER: if we're really motivated we can sort and then unsort */
  for (i=0;i<n;)
    {
      /* clump 'em for now */
      j=i+1;
      type = arg3[i];
      while ((j<n)&&(arg3[j]==type))
	{j++;}
      
      /* how many together */
      j -= i;

      /* call appropriate ivec function */
      if (type == GL_MAX)
	{ivec_max(arg1,arg2,j);}
      else if (type == GL_MIN)
	{ivec_min(arg1,arg2,j);}
      else if (type == GL_MULT)
	{ivec_mult(arg1,arg2,j);}
      else if (type == GL_ADD)
	{ivec_add(arg1,arg2,j);}
      else if (type == GL_B_XOR)
	{ivec_xor(arg1,arg2,j);}
      else if (type == GL_B_OR)
	{ivec_or(arg1,arg2,j);}
      else if (type == GL_B_AND)  
	{ivec_and(arg1,arg2,j);}
      else if (type == GL_L_XOR)
	{ivec_lxor(arg1,arg2,j);}
      else if (type == GL_L_OR)
	{ivec_lor(arg1,arg2,j);}
      else if (type == GL_L_AND)   
	{ivec_land(arg1,arg2,j);}
      else
	{SETERRQ(PETSC_ERR_PLIB,"unrecognized type passed to ivec_non_uniform()!");}

      arg1+=j; arg2+=j; i+=j;
    }
  PetscFunctionReturn(0);
}

/***********************************ivec.c*************************************/
vfp ivec_fct_addr( PetscInt type)
{
  PetscFunctionBegin;
  if (type == NON_UNIFORM)
    {return((PetscErrorCode (*)(void*, void *, PetscInt, ...))&ivec_non_uniform);}
  else if (type == GL_MAX)
    {return((PetscErrorCode (*)(void*, void *, PetscInt, ...))&ivec_max);}
  else if (type == GL_MIN)
    {return((PetscErrorCode (*)(void*, void *, PetscInt, ...))&ivec_min);}
  else if (type == GL_MULT)
    {return((PetscErrorCode (*)(void*, void *, PetscInt, ...))&ivec_mult);}
  else if (type == GL_ADD)
    {return((PetscErrorCode (*)(void*, void *, PetscInt, ...))&ivec_add);}
  else if (type == GL_B_XOR)
    {return((PetscErrorCode (*)(void*, void *, PetscInt, ...))&ivec_xor);}
  else if (type == GL_B_OR)
    {return((PetscErrorCode (*)(void*, void *, PetscInt, ...))&ivec_or);}
  else if (type == GL_B_AND)  
    {return((PetscErrorCode (*)(void*, void *, PetscInt, ...))&ivec_and);}
  else if (type == GL_L_XOR)
    {return((PetscErrorCode (*)(void*, void *, PetscInt, ...))&ivec_lxor);}
  else if (type == GL_L_OR)
    {return((PetscErrorCode (*)(void*, void *, PetscInt, ...))&ivec_lor);}
  else if (type == GL_L_AND)   
    {return((PetscErrorCode (*)(void*, void *, PetscInt, ...))&ivec_land);}

  /* catch all ... not good if we get here */
  return(NULL);
}

/******************************************************************************/
PetscErrorCode ivec_sort( PetscInt *ar,  PetscInt size)
{
   PetscInt *pi, *pj, temp;
   PetscInt **top_a = (PetscInt **) offset_stack;
   PetscInt *top_s = size_stack, *bottom_s = size_stack; 


  /* we're really interested in the offset of the last element */
  /* ==> length of the list is now size + 1                    */
  size--;

  /* do until we're done ... return when stack is exhausted */
  for (;;)
    {
      /* if list is large enough use quicksort partition exchange code */
      if (size > SORT_OPT)
	{	
	  /* start up pointer at element 1 and down at size     */  
	  pi = ar+1;
	  pj = ar+size;

	  /* find middle element in list and swap w/ element 1 */
	  SWAP(*(ar+(size>>1)),*pi)

	  /* order element 0,1,size-1 st {M,L,...,U} w/L<=M<=U */
	  /* note ==> pivot_value in index 0                   */
	  if (*pi > *pj)  
	    {SWAP(*pi,*pj)}
	  if (*ar > *pj) 
	    {SWAP(*ar,*pj)}
	  else if (*pi > *ar)   
	    {SWAP(*(ar),*(ar+1))}

	  /* partition about pivot_value ...  	                    */
	  /* note lists of length 2 are not guaranteed to be sorted */
	  for(;;)
	    {
	      /* walk up ... and down ... swap if equal to pivot! */
	      do pi++; while (*pi<*ar);
	      do pj--; while (*pj>*ar);

	      /* if we've crossed we're done */
	      if (pj<pi) break;

	      /* else swap */
	      SWAP(*pi,*pj)
	    }

	  /* place pivot_value in it's correct location */
	  SWAP(*ar,*pj)

	  /* test stack_size to see if we've exhausted our stack */
	  if (top_s-bottom_s >= SORT_STACK)
	    {SETERRQ(PETSC_ERR_PLIB,"ivec_sort() :: STACK EXHAUSTED!!!");}

	  /* push right hand child iff length > 1 */
	  if ((*top_s = size-((PetscInt) (pi-ar))))
	    {
	      *(top_a++) = pi;
	      size -= *top_s+2;  
	      top_s++;
	    }
	  /* set up for next loop iff there is something to do */
	  else if (size -= *top_s+2) 
	    {;}
	  /* might as well pop - note NR_OPT >=2 ==> we're ok! */
	  else
	    {
	      ar = *(--top_a);
	      size = *(--top_s);
	    }
	}

      /* else sort small list directly then pop another off stack */
      else
	{
	  /* insertion sort for bottom */
          for (pj=ar+1;pj<=ar+size;pj++)
            {
              temp = *pj;
              for (pi=pj-1;pi>=ar;pi--)
                {
                  if (*pi <= temp) break;
                  *(pi+1)=*pi;
                }
              *(pi+1)=temp;
	    }

	  /* check to see if stack is exhausted ==> DONE */
	  if (top_s==bottom_s)   PetscFunctionReturn(0);
	  
	  /* else pop another list from the stack */
	  ar = *(--top_a);
	  size = *(--top_s);
	}
    }
  PetscFunctionReturn(0);
}

/******************************************************************************/
PetscErrorCode ivec_sort_companion( PetscInt *ar,  PetscInt *ar2,  PetscInt size)
{
   PetscInt *pi, *pj, temp, temp2;
   PetscInt **top_a = (PetscInt **)offset_stack;
   PetscInt *top_s = size_stack, *bottom_s = size_stack; 
   PetscInt *pi2, *pj2;
   PetscInt mid;

  PetscFunctionBegin;
  /* we're really interested in the offset of the last element */
  /* ==> length of the list is now size + 1                    */
  size--;

  /* do until we're done ... return when stack is exhausted */
  for (;;)
    {
      /* if list is large enough use quicksort partition exchange code */
      if (size > SORT_OPT)
	{	
	  /* start up pointer at element 1 and down at size     */  
	  mid = size>>1;
	  pi = ar+1;
	  pj = ar+mid;
	  pi2 = ar2+1;
	  pj2 = ar2+mid;

	  /* find middle element in list and swap w/ element 1 */
	  SWAP(*pi,*pj)
	  SWAP(*pi2,*pj2)

	  /* order element 0,1,size-1 st {M,L,...,U} w/L<=M<=U */
	  /* note ==> pivot_value in index 0                   */
	  pj = ar+size;
	  pj2 = ar2+size;
	  if (*pi > *pj)  
	    {SWAP(*pi,*pj) SWAP(*pi2,*pj2)}
	  if (*ar > *pj) 
	    {SWAP(*ar,*pj) SWAP(*ar2,*pj2)}
	  else if (*pi > *ar)   
	    {SWAP(*(ar),*(ar+1)) SWAP(*(ar2),*(ar2+1))}

	  /* partition about pivot_value ...  	                    */
	  /* note lists of length 2 are not guaranteed to be sorted */
	  for(;;)
	    {
	      /* walk up ... and down ... swap if equal to pivot! */
	      do {pi++; pi2++;} while (*pi<*ar);
	      do {pj--; pj2--;} while (*pj>*ar);

	      /* if we've crossed we're done */
	      if (pj<pi) break;

	      /* else swap */
	      SWAP(*pi,*pj)
	      SWAP(*pi2,*pj2)
	    }

	  /* place pivot_value in it's correct location */
	  SWAP(*ar,*pj)
	  SWAP(*ar2,*pj2)

	  /* test stack_size to see if we've exhausted our stack */
	  if (top_s-bottom_s >= SORT_STACK)
	    {SETERRQ(PETSC_ERR_PLIB,"ivec_sort_companion() :: STACK EXHAUSTED!!!");}

	  /* push right hand child iff length > 1 */
	  if ((*top_s = size-((PetscInt) (pi-ar))))
	    {
	      *(top_a++) = pi;
	      *(top_a++) = pi2;
	      size -= *top_s+2;  
	      top_s++;
	    }
	  /* set up for next loop iff there is something to do */
	  else if (size -= *top_s+2) 
	    {;}
	  /* might as well pop - note NR_OPT >=2 ==> we're ok! */
	  else
	    {
	      ar2 = *(--top_a);
	      ar  = *(--top_a);
	      size = *(--top_s);
	    }
	}

      /* else sort small list directly then pop another off stack */
      else
	{
	  /* insertion sort for bottom */
          for (pj=ar+1, pj2=ar2+1;pj<=ar+size;pj++,pj2++)
            {
              temp = *pj;
              temp2 = *pj2;
              for (pi=pj-1,pi2=pj2-1;pi>=ar;pi--,pi2--)
                {
                  if (*pi <= temp) break;
                  *(pi+1)=*pi;
                  *(pi2+1)=*pi2;
                }
              *(pi+1)=temp;
              *(pi2+1)=temp2;
	    }

	  /* check to see if stack is exhausted ==> DONE */
	  if (top_s==bottom_s)   PetscFunctionReturn(0);
	  
	  /* else pop another list from the stack */
	  ar2 = *(--top_a);
	  ar  = *(--top_a);
	  size = *(--top_s);
	}
    }
  PetscFunctionReturn(0);
}

/******************************************************************************/
PetscErrorCode ivec_sort_companion_hack( PetscInt *ar,  PetscInt **ar2, PetscInt size)
{
   PetscInt *pi, *pj, temp, *ptr;
   PetscInt **top_a = (PetscInt **)offset_stack;
   PetscInt *top_s = size_stack, *bottom_s = size_stack; 
   PetscInt **pi2, **pj2;
   PetscInt mid;

  PetscFunctionBegin;
  /* we're really interested in the offset of the last element */
  /* ==> length of the list is now size + 1                    */
  size--;

  /* do until we're done ... return when stack is exhausted */
  for (;;)
    {
      /* if list is large enough use quicksort partition exchange code */
      if (size > SORT_OPT)
	{	
	  /* start up pointer at element 1 and down at size     */  
	  mid = size>>1;
	  pi = ar+1;
	  pj = ar+mid;
	  pi2 = ar2+1;
	  pj2 = ar2+mid;

	  /* find middle element in list and swap w/ element 1 */
	  SWAP(*pi,*pj)
	  P_SWAP(*pi2,*pj2)

	  /* order element 0,1,size-1 st {M,L,...,U} w/L<=M<=U */
	  /* note ==> pivot_value in index 0                   */
	  pj = ar+size;
	  pj2 = ar2+size;
	  if (*pi > *pj)  
	    {SWAP(*pi,*pj) P_SWAP(*pi2,*pj2)}
	  if (*ar > *pj) 
	    {SWAP(*ar,*pj) P_SWAP(*ar2,*pj2)}
	  else if (*pi > *ar)   
	    {SWAP(*(ar),*(ar+1)) P_SWAP(*(ar2),*(ar2+1))}

	  /* partition about pivot_value ...  	                    */
	  /* note lists of length 2 are not guaranteed to be sorted */
	  for(;;)
	    {
	      /* walk up ... and down ... swap if equal to pivot! */
	      do {pi++; pi2++;} while (*pi<*ar);
	      do {pj--; pj2--;} while (*pj>*ar);

	      /* if we've crossed we're done */
	      if (pj<pi) break;

	      /* else swap */
	      SWAP(*pi,*pj)
	      P_SWAP(*pi2,*pj2)
	    }

	  /* place pivot_value in it's correct location */
	  SWAP(*ar,*pj)
	  P_SWAP(*ar2,*pj2)

	  /* test stack_size to see if we've exhausted our stack */
	  if (top_s-bottom_s >= SORT_STACK)
         {SETERRQ(PETSC_ERR_PLIB,"ivec_sort_companion_hack() :: STACK EXHAUSTED!!!");}

	  /* push right hand child iff length > 1 */
	  if ((*top_s = size-((PetscInt) (pi-ar))))
	    {
	      *(top_a++) = pi;
	      *(top_a++) = (PetscInt*) pi2;
	      size -= *top_s+2;  
	      top_s++;
	    }
	  /* set up for next loop iff there is something to do */
	  else if (size -= *top_s+2) 
	    {;}
	  /* might as well pop - note NR_OPT >=2 ==> we're ok! */
	  else
	    {
	      ar2 = (PetscInt **) *(--top_a);
	      ar  = *(--top_a);
	      size = *(--top_s);
	    }
	}

      /* else sort small list directly then pop another off stack */
      else
	{
	  /* insertion sort for bottom */
          for (pj=ar+1, pj2=ar2+1;pj<=ar+size;pj++,pj2++)
            {
              temp = *pj;
              ptr = *pj2;
              for (pi=pj-1,pi2=pj2-1;pi>=ar;pi--,pi2--)
                {
                  if (*pi <= temp) break;
                  *(pi+1)=*pi;
                  *(pi2+1)=*pi2;
                }
              *(pi+1)=temp;
              *(pi2+1)=ptr;
	    }

	  /* check to see if stack is exhausted ==> DONE */
	  if (top_s==bottom_s)   PetscFunctionReturn(0);
	  
	  /* else pop another list from the stack */
	  ar2 = (PetscInt **)*(--top_a);
	  ar  = *(--top_a);
	  size = *(--top_s);
	}
    }
  PetscFunctionReturn(0);
}

/******************************************************************************/
PetscErrorCode SMI_sort(void *ar1, void *ar2, PetscInt size, PetscInt type)
{
  PetscFunctionBegin;
  if (type == SORT_INTEGER)
    {
      if (ar2)
	{ivec_sort_companion((PetscInt*)ar1,(PetscInt*)ar2,size);}
      else
	{ivec_sort((PetscInt*)ar1,size);}
    }
  else if (type == SORT_INT_PTR)
    {
      if (ar2)
	{ivec_sort_companion_hack((PetscInt*)ar1,(PetscInt **)ar2,size);}
      else
	{ivec_sort((PetscInt*)ar1,size);}
    }

  else
    {
      SETERRQ(PETSC_ERR_PLIB,"SMI_sort only does SORT_INTEGER!");
    }
  PetscFunctionReturn(0);
}

/***********************************ivec.c*************************************/
PetscInt ivec_linear_search( PetscInt item,  PetscInt *list,  PetscInt n)
{
   PetscInt tmp = n-1;
  PetscFunctionBegin;
  while (n--)  {if (*list++ == item) {return(tmp-n);}}
  return(-1);
}

/***********************************ivec.c*************************************/
PetscInt ivec_binary_search( PetscInt item,  PetscInt *list,  PetscInt rh)
{
   PetscInt mid, lh=0;

  rh--;
  while (lh<=rh)
    {
      mid = (lh+rh)>>1;
      if (*(list+mid) == item) 
	{return(mid);}
      if (*(list+mid) > item)  
	{rh = mid-1;}
      else 
	{lh = mid+1;}
    }
  return(-1);
}

/*********************************ivec.c*************************************/
PetscErrorCode rvec_copy( PetscScalar *arg1,  PetscScalar *arg2,  PetscInt n)
{
  PetscFunctionBegin;
  while (n--)  {*arg1++ = *arg2++;}
  PetscFunctionReturn(0);
}

/*********************************ivec.c*************************************/
PetscErrorCode rvec_zero( PetscScalar *arg1,  PetscInt n)
{
  PetscFunctionBegin;
  while (n--)  {*arg1++ = 0.0;}
  PetscFunctionReturn(0);
}

/***********************************ivec.c*************************************/
PetscErrorCode rvec_one( PetscScalar *arg1,  PetscInt n)
{
  PetscFunctionBegin;
  while (n--)  {*arg1++ = 1.0;}
  PetscFunctionReturn(0);
}

/***********************************ivec.c*************************************/
PetscErrorCode rvec_set( PetscScalar *arg1,  PetscScalar arg2,  PetscInt n)
{
  PetscFunctionBegin;
  while (n--)  {*arg1++ = arg2;}
  PetscFunctionReturn(0);
}

/***********************************ivec.c*************************************/
PetscErrorCode rvec_scale( PetscScalar *arg1,  PetscScalar arg2,  PetscInt n)
{
  PetscFunctionBegin;
  while (n--)  {*arg1++ *= arg2;}
  PetscFunctionReturn(0);
}

/*********************************ivec.c*************************************/
PetscErrorCode rvec_add( PetscScalar *arg1,  PetscScalar *arg2,  PetscInt n)
{
  PetscFunctionBegin;
  while (n--)  {*arg1++ += *arg2++;}
  PetscFunctionReturn(0);
}

/*********************************ivec.c*************************************/
PetscErrorCode rvec_mult( PetscScalar *arg1,  PetscScalar *arg2,  PetscInt n)
{
  PetscFunctionBegin;
  while (n--)  {*arg1++ *= *arg2++;}
  PetscFunctionReturn(0);
}

/*********************************ivec.c*************************************/
PetscErrorCode rvec_max( PetscScalar *arg1,  PetscScalar *arg2,  PetscInt n)
{
  PetscFunctionBegin;
  while (n--)  {*arg1 = PetscMax(*arg1,*arg2); arg1++; arg2++;}
  PetscFunctionReturn(0);
}

/*********************************ivec.c*************************************/
PetscErrorCode rvec_max_abs( PetscScalar *arg1,  PetscScalar *arg2,  PetscInt n)
{
  PetscFunctionBegin;
  while (n--)  {*arg1 = MAX_FABS(*arg1,*arg2); arg1++; arg2++;}
  PetscFunctionReturn(0);
}

/*********************************ivec.c*************************************/
PetscErrorCode rvec_min( PetscScalar *arg1,  PetscScalar *arg2,  PetscInt n)
{
  PetscFunctionBegin;
  while (n--)  {*arg1 = PetscMin(*arg1,*arg2); arg1++; arg2++;}
  PetscFunctionReturn(0);
}

/*********************************ivec.c*************************************/
PetscErrorCode rvec_min_abs( PetscScalar *arg1,  PetscScalar *arg2,  PetscInt n)
{
  PetscFunctionBegin;
  while (n--)  {*arg1 = MIN_FABS(*arg1,*arg2); arg1++; arg2++;}
  PetscFunctionReturn(0);
}

/*********************************ivec.c*************************************/
PetscErrorCode rvec_exists( PetscScalar *arg1,  PetscScalar *arg2,  PetscInt n)
{
  PetscFunctionBegin;
  while (n--)  {*arg1 = EXISTS(*arg1,*arg2); arg1++; arg2++;}
  PetscFunctionReturn(0);
}

/***********************************ivec.c*************************************/
PetscErrorCode rvec_non_uniform(PetscScalar *arg1, PetscScalar *arg2,  PetscInt n,  PetscInt *arg3)
{
   PetscInt i, j, type;

  PetscFunctionBegin;
  /* LATER: if we're really motivated we can sort and then unsort */
  for (i=0;i<n;)
    {
      /* clump 'em for now */
      j=i+1;
      type = arg3[i];
      while ((j<n)&&(arg3[j]==type))
	{j++;}
      
      /* how many together */
      j -= i;

      /* call appropriate ivec function */
      if (type == GL_MAX)
	{rvec_max(arg1,arg2,j);}
      else if (type == GL_MIN)
	{rvec_min(arg1,arg2,j);}
      else if (type == GL_MULT)
	{rvec_mult(arg1,arg2,j);}
      else if (type == GL_ADD)
	{rvec_add(arg1,arg2,j);}
      else if (type == GL_MAX_ABS)
	{rvec_max_abs(arg1,arg2,j);}
      else if (type == GL_MIN_ABS)
	{rvec_min_abs(arg1,arg2,j);}
      else if (type == GL_EXISTS)
	{rvec_exists(arg1,arg2,j);}
      else
	{SETERRQ(PETSC_ERR_PLIB,"unrecognized type passed to rvec_non_uniform()!");}

      arg1+=j; arg2+=j; i+=j;
    }
  PetscFunctionReturn(0);
}

/***********************************ivec.c*************************************/
vfp rvec_fct_addr( PetscInt type)
{
  if (type == NON_UNIFORM)
    {return((PetscErrorCode (*)(void*, void *, PetscInt, ...))&rvec_non_uniform);}
  else if (type == GL_MAX)
    {return((PetscErrorCode (*)(void*, void *, PetscInt, ...))&rvec_max);}
  else if (type == GL_MIN)
    {return((PetscErrorCode (*)(void*, void *, PetscInt, ...))&rvec_min);}
  else if (type == GL_MULT)
    {return((PetscErrorCode (*)(void*, void *, PetscInt, ...))&rvec_mult);}
  else if (type == GL_ADD)
    {return((PetscErrorCode (*)(void*, void *, PetscInt, ...))&rvec_add);}
  else if (type == GL_MAX_ABS)
    {return((PetscErrorCode (*)(void*, void *, PetscInt, ...))&rvec_max_abs);}
  else if (type == GL_MIN_ABS)
    {return((PetscErrorCode (*)(void*, void *, PetscInt, ...))&rvec_min_abs);}
  else if (type == GL_EXISTS)
    {return((PetscErrorCode (*)(void*, void *, PetscInt, ...))&rvec_exists);}

  /* catch all ... not good if we get here */
  return(NULL);
}






