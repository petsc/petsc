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

/**********************************ivec.c**************************************
File Description:
-----------------

***********************************ivec.c*************************************/
#include "src/ksp/pc/impls/tfs/tfs.h"

/* sorting args ivec.c ivec.c ... */
#define   SORT_OPT	6     
#define   SORT_STACK	50000


/* allocate an address and size stack for sorter(s) */
static void *offset_stack[2*SORT_STACK];
static int   size_stack[SORT_STACK];
static long psize_stack[SORT_STACK];



/**********************************ivec.c**************************************
Function ivec_dump()

Input : 
Output: 
Return: 
Description: 
***********************************ivec.c*************************************/
void
ivec_dump(int *v, int n, int tag, int tag2, char * s)
{
  int i;
  printf("%2d %2d %s %2d :: ",tag,tag2,s,my_id);
  for (i=0;i<n;i++)
    {printf("%2d ",v[i]);}
  printf("\n");
  fflush(stdout);
}



/**********************************ivec.c**************************************
Function ivec_lb_ub()

Input : 
Output: 
Return: 
Description: 
***********************************ivec.c*************************************/
void
ivec_lb_ub( int *arg1,  int n, int *lb, int *ub)
{
   int min = INT_MAX;
   int max = INT_MIN;

  while (n--)  
    {
     min = PetscMin(min,*arg1);
     max = PetscMax(max,*arg1);
     arg1++;
    }

  *lb=min;
  *ub=max;
}



/**********************************ivec.c**************************************
Function ivec_copy()

Input : 
Output: 
Return: 
Description: 
***********************************ivec.c*************************************/
int *ivec_copy( int *arg1,  int *arg2,  int n)
{
  while (n--)  {*arg1++ = *arg2++;}
  return(arg1);
}



/**********************************ivec.c**************************************
Function ivec_zero()

Input : 
Output: 
Return: 
Description: 
***********************************ivec.c*************************************/
void 
ivec_zero( int *arg1,  int n)
{
  while (n--)  {*arg1++ = 0;}
}



/**********************************ivec.c**************************************
Function ivec_comp()

Input : 
Output: 
Return: 
Description: 
***********************************ivec.c*************************************/
void 
ivec_comp( int *arg1,  int n)
{
  while (n--)  {*arg1 = ~*arg1; arg1++;}
}



/**********************************ivec.c**************************************
Function ivec_neg_one()

Input : 
Output: 
Return: 
Description: 
***********************************ivec.c*************************************/
void 
ivec_neg_one( int *arg1,  int n)
{
  while (n--)  {*arg1++ = -1;}
}



/**********************************ivec.c**************************************
Function ivec_pos_one()

Input : 
Output: 
Return: 
Description: 
***********************************ivec.c*************************************/
void 
ivec_pos_one( int *arg1,  int n)
{
  while (n--)  {*arg1++ = 1;}
}



/**********************************ivec.c**************************************
Function ivec_c_index()

Input : 
Output: 
Return: 
Description: 
***********************************ivec.c*************************************/
void 
ivec_c_index( int *arg1,  int n)
{
   int i=0;


  while (n--)  {*arg1++ = i++;}
}



/**********************************ivec.c**************************************
Function ivec_fortran_index()

Input : 
Output: 
Return: 
Description: 
***********************************ivec.c*************************************/
void 
ivec_fortran_index( int *arg1,  int n)
{
   int i=0;


  while (n--)  {*arg1++ = ++i;}
}



/**********************************ivec.c**************************************
Function ivec_set()

Input : 
Output: 
Return: 
Description: 
***********************************ivec.c*************************************/
void 
ivec_set( int *arg1,  int arg2,  int n)
{
  while (n--)  {*arg1++ = arg2;}
}



/**********************************ivec.c**************************************
Function ivec_cmp()

Input : 
Output: 
Return: 
Description: 
***********************************ivec.c*************************************/
int
ivec_cmp( int *arg1,  int *arg2,  int n)
{
  while (n--)  {if (*arg1++ != *arg2++)  {return(FALSE);}}
  return(TRUE);
}



/**********************************ivec.c**************************************
Function ivec_max()

Input : 
Output: 
Return: 
Description: 
***********************************ivec.c*************************************/
void 
ivec_max( int *arg1,  int *arg2,  int n)
{
  while (n--)  {*arg1 = PetscMax(*arg1,*arg2); arg1++; arg2++;}
}



/**********************************ivec.c**************************************
Function ivec_min()

Input : 
Output: 
Return: 
Description: 
***********************************ivec.c*************************************/
void 
ivec_min( int *arg1,  int *arg2,  int n)
{
  while (n--)  {*(arg1) = PetscMin(*arg1,*arg2); arg1++; arg2++;}
}



/**********************************ivec.c**************************************
Function ivec_mult()

Input : 
Output: 
Return: 
Description: 
***********************************ivec.c*************************************/
void 
ivec_mult( int *arg1,  int *arg2,  int n)
{
  while (n--)  {*arg1++ *= *arg2++;}
}



/**********************************ivec.c**************************************
Function ivec_add()

Input : 
Output: 
Return: 
Description: 
***********************************ivec.c*************************************/
void 
ivec_add( int *arg1,  int *arg2,  int n)
{
  while (n--)  {*arg1++ += *arg2++;}
}



/**********************************ivec.c**************************************
Function ivec_lxor()

Input : 
Output: 
Return: 
Description: 
***********************************ivec.c*************************************/
void 
ivec_lxor( int *arg1,  int *arg2,  int n)
{
  while (n--) {*arg1=((*arg1 || *arg2) && !(*arg1 && *arg2)); arg1++; arg2++;}
}



/**********************************ivec.c**************************************
Function ivec_xor()

Input : 
Output: 
Return: 
Description: 
***********************************ivec.c*************************************/
void 
ivec_xor( int *arg1,  int *arg2,  int n)
{
  while (n--)  {*arg1++ ^= *arg2++;}
}



/**********************************ivec.c**************************************
Function ivec_or()

Input : 
Output: 
Return: 
Description: 
***********************************ivec.c*************************************/
void 
ivec_or( int *arg1,  int *arg2,  int n)
{
  while (n--)  {*arg1++ |= *arg2++;}
}



/**********************************ivec.c**************************************
Function ivec_lor()

Input : 
Output: 
Return: 
Description: 
***********************************ivec.c*************************************/
void 
ivec_lor( int *arg1,  int *arg2,  int n)
{
  while (n--)  {*arg1 = (*arg1 || *arg2); arg1++; arg2++;} 
}



/**********************************ivec.c**************************************
Function ivec_or3()

Input : 
Output: 
Return: 
Description: 
***********************************ivec.c*************************************/
void 
ivec_or3( int *arg1,  int *arg2,  int *arg3, 
	  int n)
{
  while (n--)  {*arg1++ = (*arg2++ | *arg3++);}
}



/**********************************ivec.c**************************************
Function ivec_and()

Input : 
Output: 
Return: 
Description: 
***********************************ivec.c*************************************/
void 
ivec_and( int *arg1,  int *arg2,  int n)
{
  while (n--)  {*arg1++ &= *arg2++;}
}



/**********************************ivec.c**************************************
Function ivec_land()

Input : 
Output: 
Return: 
Description: 
***********************************ivec.c*************************************/
void 
ivec_land( int *arg1,  int *arg2,  int n)
{
  while (n--) {*arg1 = (*arg1 && *arg2); arg1++; arg2++;} 
}



/**********************************ivec.c**************************************
Function ivec_and3()

Input : 
Output: 
Return: 
Description: 
***********************************ivec.c*************************************/
void 
ivec_and3( int *arg1,  int *arg2,  int *arg3, 
	   int n)
{
  while (n--)  {*arg1++ = (*arg2++ & *arg3++);}
}



/**********************************ivec.c**************************************
Function ivec_sum

Input : 
Output: 
Return: 
Description: 
***********************************ivec.c*************************************/
int ivec_sum( int *arg1,  int n)
{
   int tmp = 0;


  while (n--) {tmp += *arg1++;}
  return(tmp);
}



/**********************************ivec.c**************************************
Function ivec_reduce_and

Input : 
Output: 
Return: 
Description: 
***********************************ivec.c*************************************/
int ivec_reduce_and( int *arg1,  int n)
{
   int tmp = ALL_ONES;


  while (n--) {tmp &= *arg1++;}
  return(tmp);
}



/**********************************ivec.c**************************************
Function ivec_reduce_or

Input : 
Output: 
Return: 
Description: 
***********************************ivec.c*************************************/
int ivec_reduce_or( int *arg1,  int n)
{
   int tmp = 0;


  while (n--) {tmp |= *arg1++;}
  return(tmp);
}



/**********************************ivec.c**************************************
Function ivec_prod

Input : 
Output: 
Return: 
Description: 
***********************************ivec.c*************************************/
int ivec_prod( int *arg1,  int n)
{
   int tmp = 1;


  while (n--)  {tmp *= *arg1++;}
  return(tmp);
}



/**********************************ivec.c**************************************
Function ivec_u_sum

Input : 
Output: 
Return: 
Description: 
***********************************ivec.c*************************************/
int ivec_u_sum( unsigned *arg1,  int n)
{
   unsigned tmp = 0;


  while (n--)  {tmp += *arg1++;}
  return(tmp);
}



/**********************************ivec.c**************************************
Function ivec_lb()

Input : 
Output: 
Return: 
Description: 
***********************************ivec.c*************************************/
int 
ivec_lb( int *arg1,  int n)
{
   int min = INT_MAX;


  while (n--)  {min = PetscMin(min,*arg1); arg1++;}
  return(min);
}



/**********************************ivec.c**************************************
Function ivec_ub()

Input : 
Output: 
Return: 
Description: 
***********************************ivec.c*************************************/
int 
ivec_ub( int *arg1,  int n)
{
   int max = INT_MIN;


  while (n--)  {max = PetscMax(max,*arg1); arg1++;}
  return(max);
}



/**********************************ivec.c**************************************
Function split_buf()

Input : 
Output: 
Return: 
Description: 

assumes that sizeof(int) == 4bytes!!!
***********************************ivec.c*************************************/
int
ivec_split_buf(int *buf1, int **buf2,  int size)
{
  *buf2 = (buf1 + (size>>3));
  return(size);
}



/**********************************ivec.c**************************************
Function ivec_non_uniform()

Input : 
Output: 
Return: 
Description: 
***********************************ivec.c*************************************/
void 
ivec_non_uniform(int *arg1, int *arg2,  int n,  int *arg3)
{
   int i, j, type;


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
	{error_msg_fatal("unrecognized type passed to ivec_non_uniform()!");}

      arg1+=j; arg2+=j; i+=j;
    }
}



/**********************************ivec.c**************************************
Function ivec_addr()

Input : 
Output: 
Return: 
Description: 
***********************************ivec.c*************************************/
vfp ivec_fct_addr( int type)
{
  if (type == NON_UNIFORM)
    {return((void (*)(void*, void *, int, ...))&ivec_non_uniform);}
  else if (type == GL_MAX)
    {return((void (*)(void*, void *, int, ...))&ivec_max);}
  else if (type == GL_MIN)
    {return((void (*)(void*, void *, int, ...))&ivec_min);}
  else if (type == GL_MULT)
    {return((void (*)(void*, void *, int, ...))&ivec_mult);}
  else if (type == GL_ADD)
    {return((void (*)(void*, void *, int, ...))&ivec_add);}
  else if (type == GL_B_XOR)
    {return((void (*)(void*, void *, int, ...))&ivec_xor);}
  else if (type == GL_B_OR)
    {return((void (*)(void*, void *, int, ...))&ivec_or);}
  else if (type == GL_B_AND)  
    {return((void (*)(void*, void *, int, ...))&ivec_and);}
  else if (type == GL_L_XOR)
    {return((void (*)(void*, void *, int, ...))&ivec_lxor);}
  else if (type == GL_L_OR)
    {return((void (*)(void*, void *, int, ...))&ivec_lor);}
  else if (type == GL_L_AND)   
    {return((void (*)(void*, void *, int, ...))&ivec_land);}

  /* catch all ... not good if we get here */
  return(NULL);
}


/**********************************ivec.c**************************************
Function ct_bits()

Input : 
Output: 
Return: 
Description: MUST FIX THIS!!!
***********************************ivec.c*************************************/
#if defined(notusing)
static
int 
ivec_ct_bits( int *ptr,  int n)
{
   int tmp=0;


  /* should expand to full 32 bit */
  while (n--)
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
#endif


/******************************************************************************
Function: ivec_sort().

Input : offset of list to be sorted, number of elements to be sorted.
Output: sorted list (in ascending order).
Return: none.
Description: stack based (nonrecursive) quicksort w/brute-shell bottom. 
******************************************************************************/
void
ivec_sort( int *ar,  int size)
{
   int *pi, *pj, temp;
   int **top_a = (int **) offset_stack;
   int *top_s = size_stack, *bottom_s = size_stack; 


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
	    {error_msg_fatal("ivec_sort() :: STACK EXHAUSTED!!!");}

	  /* push right hand child iff length > 1 */
	  if ((*top_s = size-((int) (pi-ar))))
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
	  if (top_s==bottom_s) return;
	  
	  /* else pop another list from the stack */
	  ar = *(--top_a);
	  size = *(--top_s);
	}
    }
}



/******************************************************************************
Function: ivec_sort_companion().

Input : offset of list to be sorted, number of elements to be sorted.
Output: sorted list (in ascending order).
Return: none.
Description: stack based (nonrecursive) quicksort w/brute-shell bottom. 
******************************************************************************/
void
ivec_sort_companion( int *ar,  int *ar2,  int size)
{
   int *pi, *pj, temp, temp2;
   int **top_a = (int **)offset_stack;
   int *top_s = size_stack, *bottom_s = size_stack; 
   int *pi2, *pj2;
   int mid;


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
	    {error_msg_fatal("ivec_sort_companion() :: STACK EXHAUSTED!!!");}

	  /* push right hand child iff length > 1 */
	  if ((*top_s = size-((int) (pi-ar))))
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
	  if (top_s==bottom_s) return;
	  
	  /* else pop another list from the stack */
	  ar2 = *(--top_a);
	  ar  = *(--top_a);
	  size = *(--top_s);
	}
    }
}



/******************************************************************************
Function: ivec_sort_companion_hack().

Input : offset of list to be sorted, number of elements to be sorted.
Output: sorted list (in ascending order).
Return: none.
Description: stack based (nonrecursive) quicksort w/brute-shell bottom. 
******************************************************************************/
void
ivec_sort_companion_hack( int *ar,  int **ar2, 
			  int size)
{
   int *pi, *pj, temp, *ptr;
   int **top_a = (int **)offset_stack;
   int *top_s = size_stack, *bottom_s = size_stack; 
   int **pi2, **pj2;
   int mid;


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
         {error_msg_fatal("ivec_sort_companion_hack() :: STACK EXHAUSTED!!!");}

	  /* push right hand child iff length > 1 */
	  if ((*top_s = size-((int) (pi-ar))))
	    {
	      *(top_a++) = pi;
	      *(top_a++) = (int*) pi2;
	      size -= *top_s+2;  
	      top_s++;
	    }
	  /* set up for next loop iff there is something to do */
	  else if (size -= *top_s+2) 
	    {;}
	  /* might as well pop - note NR_OPT >=2 ==> we're ok! */
	  else
	    {
	      ar2 = (int **) *(--top_a);
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
	  if (top_s==bottom_s) return;
	  
	  /* else pop another list from the stack */
	  ar2 = (int **)*(--top_a);
	  ar  = *(--top_a);
	  size = *(--top_s);
	}
    }
}



/******************************************************************************
Function: SMI_sort().
Input : offset of list to be sorted, number of elements to be sorted.
Output: sorted list (in ascending order).
Return: none.
Description: stack based (nonrecursive) quicksort w/brute-shell bottom. 
******************************************************************************/
void
SMI_sort(void *ar1, void *ar2, int size, int type)
{
  if (type == SORT_INTEGER)
    {
      if (ar2)
	{ivec_sort_companion((int*)ar1,(int*)ar2,size);}
      else
	{ivec_sort((int*)ar1,size);}
    }
  else if (type == SORT_INT_PTR)
    {
      if (ar2)
	{ivec_sort_companion_hack((int*)ar1,(int **)ar2,size);}
      else
	{ivec_sort((int*)ar1,size);}
    }

  else
    {
      error_msg_fatal("SMI_sort only does SORT_INTEGER!");
    }
/*
  if (type == SORT_REAL)
    {
      if (ar2)
	{rvec_sort_companion(ar2,ar1,size);}
      else
	{rvec_sort(ar1,size);}
    }
*/
}



/**********************************ivec.c**************************************
Function ivec_linear_search()

Input : 
Output: 
Return: 
Description: 
***********************************ivec.c*************************************/
int
ivec_linear_search( int item,  int *list,  int n)
{
   int tmp = n-1;

  while (n--)  {if (*list++ == item) {return(tmp-n);}}
  return(-1);
}



/**********************************ivec.c**************************************
Function ivec_binary_search()

Input : 
Output: 
Return: 
Description: 
***********************************ivec.c*************************************/
int
ivec_binary_search( int item,  int *list,  int rh)
{
   int mid, lh=0;

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



/**********************************ivec.c**************************************
Function rvec_dump

Input : 
Output: 
Return: 
Description: 
***********************************ivec.c*************************************/
void
rvec_dump(PetscScalar *v, int n, int tag, int tag2, char * s)
{
  int i;
  printf("%2d %2d %s %2d :: ",tag,tag2,s,my_id);
  for (i=0;i<n;i++)
    {printf("%f ",v[i]);}
  printf("\n");
  fflush(stdout);
}



/**********************************ivec.c**************************************
Function rvec_lb_ub()

Input : 
Output: 
Return: 
Description: 
***********************************ivec.c*************************************/
void
rvec_lb_ub( PetscScalar *arg1,  int n, PetscScalar *lb, PetscScalar *ub)
{
   PetscScalar min =  REAL_MAX;
   PetscScalar max = -REAL_MAX;

  while (n--)  
    {
     min = PetscMin(min,*arg1);
     max = PetscMax(max,*arg1);
     arg1++;
    }

  *lb=min;
  *ub=max;
}



/********************************ivec.c**************************************
Function rvec_copy()

Input : 
Output: 
Return: 
Description: 
*********************************ivec.c*************************************/
void 
rvec_copy( PetscScalar *arg1,  PetscScalar *arg2,  int n)
{
  while (n--)  {*arg1++ = *arg2++;}
}



/********************************ivec.c**************************************
Function rvec_zero()

Input : 
Output: 
Return: 
Description: 
*********************************ivec.c*************************************/
void 
rvec_zero( PetscScalar *arg1,  int n)
{
  while (n--)  {*arg1++ = 0.0;}
}



/**********************************ivec.c**************************************
Function rvec_one()

Input : 
Output: 
Return: 
Description: 
***********************************ivec.c*************************************/
void 
rvec_one( PetscScalar *arg1,  int n)
{
  while (n--)  {*arg1++ = 1.0;}
}



/**********************************ivec.c**************************************
Function rvec_neg_one()

Input : 
Output: 
Return: 
Description: 
***********************************ivec.c*************************************/
void 
rvec_neg_one( PetscScalar *arg1,  int n)
{
  while (n--)  {*arg1++ = -1.0;}
}



/**********************************ivec.c**************************************
Function rvec_set()

Input : 
Output: 
Return: 
Description: 
***********************************ivec.c*************************************/
void
rvec_set( PetscScalar *arg1,  PetscScalar arg2,  int n)
{
  while (n--)  {*arg1++ = arg2;}
}



/**********************************ivec.c**************************************
Function rvec_scale()

Input : 
Output: 
Return: 
Description: 
***********************************ivec.c*************************************/
void
rvec_scale( PetscScalar *arg1,  PetscScalar arg2,  int n)
{
  while (n--)  {*arg1++ *= arg2;}
}



/********************************ivec.c**************************************
Function rvec_add()

Input : 
Output: 
Return: 
Description: 
*********************************ivec.c*************************************/
void 
rvec_add( PetscScalar *arg1,  PetscScalar *arg2,  int n)
{
  while (n--)  {*arg1++ += *arg2++;}
}



/********************************ivec.c**************************************
Function rvec_dot()

Input : 
Output: 
Return: 
Description: 
*********************************ivec.c*************************************/
PetscScalar
rvec_dot( PetscScalar *arg1,  PetscScalar *arg2,  int n)
{
  PetscScalar dot=0.0;

  while (n--)  {dot+= *arg1++ * *arg2++;}

  return(dot);
}



/********************************ivec.c**************************************
Function rvec_axpy()

Input : 
Output: 
Return: 
Description: 
*********************************ivec.c*************************************/
void
rvec_axpy( PetscScalar *arg1,  PetscScalar *arg2,  PetscScalar scale, 
	   int n)
{
  while (n--)  {*arg1++ += scale * *arg2++;}
}


/********************************ivec.c**************************************
Function rvec_mult()

Input : 
Output: 
Return: 
Description: 
*********************************ivec.c*************************************/
void 
rvec_mult( PetscScalar *arg1,  PetscScalar *arg2,  int n)
{
  while (n--)  {*arg1++ *= *arg2++;}
}



/********************************ivec.c**************************************
Function rvec_max()

Input : 
Output: 
Return: 
Description: 
*********************************ivec.c*************************************/
void 
rvec_max( PetscScalar *arg1,  PetscScalar *arg2,  int n)
{
  while (n--)  {*arg1 = PetscMax(*arg1,*arg2); arg1++; arg2++;}
}



/********************************ivec.c**************************************
Function rvec_max_abs()

Input : 
Output: 
Return: 
Description: 
*********************************ivec.c*************************************/
void 
rvec_max_abs( PetscScalar *arg1,  PetscScalar *arg2,  int n)
{
  while (n--)  {*arg1 = MAX_FABS(*arg1,*arg2); arg1++; arg2++;}
}



/********************************ivec.c**************************************
Function rvec_min()

Input : 
Output: 
Return: 
Description: 
*********************************ivec.c*************************************/
void 
rvec_min( PetscScalar *arg1,  PetscScalar *arg2,  int n)
{
  while (n--)  {*arg1 = PetscMin(*arg1,*arg2); arg1++; arg2++;}
}



/********************************ivec.c**************************************
Function rvec_min_abs()

Input : 
Output: 
Return: 
Description: 
*********************************ivec.c*************************************/
void 
rvec_min_abs( PetscScalar *arg1,  PetscScalar *arg2,  int n)
{
  while (n--)  {*arg1 = MIN_FABS(*arg1,*arg2); arg1++; arg2++;}
}



/********************************ivec.c**************************************
Function rvec_exists()

Input : 
Output: 
Return: 
Description: 
*********************************ivec.c*************************************/
void 
rvec_exists( PetscScalar *arg1,  PetscScalar *arg2,  int n)
{
  while (n--)  {*arg1 = EXISTS(*arg1,*arg2); arg1++; arg2++;}
}



/**********************************ivec.c**************************************
Function rvec_non_uniform()

Input : 
Output: 
Return: 
Description: 
***********************************ivec.c*************************************/
void 
rvec_non_uniform(PetscScalar *arg1, PetscScalar *arg2,  int n,  int *arg3)
{
   int i, j, type;


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
	{error_msg_fatal("unrecognized type passed to rvec_non_uniform()!");}

      arg1+=j; arg2+=j; i+=j;
    }
}



/**********************************ivec.c**************************************
Function rvec_fct_addr()

Input : 
Output: 
Return: 
Description: 
***********************************ivec.c*************************************/
vfp rvec_fct_addr( int type)
{
  if (type == NON_UNIFORM)
    {return((void (*)(void*, void *, int, ...))&rvec_non_uniform);}
  else if (type == GL_MAX)
    {return((void (*)(void*, void *, int, ...))&rvec_max);}
  else if (type == GL_MIN)
    {return((void (*)(void*, void *, int, ...))&rvec_min);}
  else if (type == GL_MULT)
    {return((void (*)(void*, void *, int, ...))&rvec_mult);}
  else if (type == GL_ADD)
    {return((void (*)(void*, void *, int, ...))&rvec_add);}
  else if (type == GL_MAX_ABS)
    {return((void (*)(void*, void *, int, ...))&rvec_max_abs);}
  else if (type == GL_MIN_ABS)
    {return((void (*)(void*, void *, int, ...))&rvec_min_abs);}
  else if (type == GL_EXISTS)
    {return((void (*)(void*, void *, int, ...))&rvec_exists);}

  /* catch all ... not good if we get here */
  return(NULL);
}


/******************************************************************************
Function: my_sort().
Input : offset of list to be sorted, number of elements to be sorted.
Output: sorted list (in ascending order).
Return: none.
Description: stack based (nonrecursive) quicksort w/brute-shell bottom. 
******************************************************************************/
void
rvec_sort( PetscScalar *ar,  int Size)
{
   PetscScalar *pi, *pj, temp;
   PetscScalar **top_a = (PetscScalar **)offset_stack;
   long *top_s = psize_stack, *bottom_s = psize_stack; 
   long size = (long) Size;

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

	  pj = ar+size; 

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
	    {error_msg_fatal("\nSTACK EXHAUSTED!!!\n");}

	  /* push right hand child iff length > 1 */
	  if ((*top_s = size-(pi-ar)))
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
	  if (top_s==bottom_s) return;
	  
	  /* else pop another list from the stack */
	  ar = *(--top_a);
	  size = *(--top_s);
	}
    }
}



/******************************************************************************
Function: my_sort().
Input : offset of list to be sorted, number of elements to be sorted.
Output: sorted list (in ascending order).
Return: none.
Description: stack based (nonrecursive) quicksort w/brute-shell bottom. 
******************************************************************************/
void
rvec_sort_companion( PetscScalar *ar,  int *ar2,  int Size)
{
   PetscScalar *pi, *pj, temp;
   PetscScalar **top_a = (PetscScalar **)offset_stack;
   long *top_s = psize_stack, *bottom_s = psize_stack; 
   long size = (long) Size;

   int *pi2, *pj2;
   int ptr;
   long mid;


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
	    {error_msg_fatal("\nSTACK EXHAUSTED!!!\n");}

	  /* push right hand child iff length > 1 */
	  if ((*top_s = size-(pi-ar)))
	    {
	      *(top_a++) = pi;
	      *(top_a++) = (PetscScalar *) pi2;
	      size -= *top_s+2;  
	      top_s++;
	    }
	  /* set up for next loop iff there is something to do */
	  else if (size -= *top_s+2) 
	    {;}
	  /* might as well pop - note NR_OPT >=2 ==> we're ok! */
	  else
	    {
	      ar2 = (int*) *(--top_a);
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
	  if (top_s==bottom_s) return;
	  
	  /* else pop another list from the stack */
	  ar2 = (int*) *(--top_a);
	  ar  = *(--top_a);
	  size = *(--top_s);
	}
    }
}





/**********************************ivec.c**************************************
Function ivec_binary_search()

Input : 
Output: 
Return: 
Description: 
***********************************ivec.c*************************************/
int
rvec_binary_search( PetscScalar item,  PetscScalar *list,  int rh)
{
  int mid, lh=0;

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





