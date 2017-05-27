/* numsrt.f -- translated by f2c (version of 25 March 1992  12:58:56). */

#include <../src/mat/color/impls/minpack/color.h>

PetscErrorCode MINPACKnumsrt(PetscInt *n,PetscInt *nmax,PetscInt *num,PetscInt *mode,PetscInt *idex,PetscInt *last,PetscInt *next)
{
  /* System generated locals */
  PetscInt i__1, i__2;

  /* Local variables */
  PetscInt jinc, i, j, k, l, jl, ju;

/*     Given a sequence of integers, this subroutine groups */
/*     together those indices with the same sequence value */
/*     and, optionally, sorts the sequence into either */
/*     ascending or descending order. */
/*     The sequence of integers is defined by the array num, */
/*     and it is assumed that the integers are each from the set */
/*     0,1,...,nmax. On output the indices k such that num(k) = l */
/*     for any l = 0,1,...,nmax can be obtained from the arrays */
/*     last and next as follows. */
/*           k = last(l) */
/*           while (k .ne. 0) k = next(k) */
/*     Optionally, the subroutine produces an array index so that */
/*     the sequence num(index(i)), i = 1,2,...,n is sorted. */
/*     The subroutine statement is */
/*       subroutine numsrt(n,nmax,num,mode,index,last,next) */
/*     where */
/*       n is a positive integer input variable. */
/*       nmax is a positive integer input variable. */
/*       num is an input array of length n which contains the */
/*         sequence of integers to be grouped and sorted. It */
/*         is assumed that the integers are each from the set */
/*         0,1,...,nmax. */
/*       mode is an integer input variable. The sequence num is */
/*         sorted in ascending order if mode is positive and in */
/*         descending order if mode is negative. If mode is 0, */
/*         no sorting is done. */
/*       index is an integer output array of length n set so */
/*         that the sequence */
/*               num(index(i)), i = 1,2,...,n */
/*         is sorted according to the setting of mode. If mode */
/*         is 0, index is not referenced. */
/*       last is an integer output array of length nmax + 1. The */
/*         index of num for the last occurrence of l is last(l) */
/*         for any l = 0,1,...,nmax unless last(l) = 0. In */
/*         this case l does not appear in num. */
/*       next is an integer output array of length n. If */
/*         num(k) = l, then the index of num for the previous */
/*         occurrence of l is next(k) for any l = 0,1,...,nmax */
/*         unless next(k) = 0. In this case there is no previous */
/*         occurrence of l in num. */
/*     Argonne National Laboratory. MINPACK Project. July 1983. */
/*     Thomas F. Coleman, Burton S. Garbow, Jorge J. More' */

  /* Parameter adjustments */
  PetscFunctionBegin;
  --next;
  --idex;
  --num;

  i__1 = *nmax;
  for (i = 0; i <= i__1; ++i) last[i] = 0;

  i__1 = *n;
  for (k = 1; k <= i__1; ++k) {
    l       = num[k];
    next[k] = last[l];
    last[l] = k;
  }
  if (!*mode) PetscFunctionReturn(0);

/*     Store the pointers to the sorted array in index. */

  i = 1;
  if (*mode > 0) {
    jl   = 0;
    ju   = *nmax;
    jinc = 1;
  } else {
    jl   = *nmax;
    ju   = 0;
    jinc = -1;
  }
  i__1 = ju;
  i__2 = jinc;
  for (j = jl; i__2 < 0 ? j >= i__1 : j <= i__1; j += i__2) {
    k = last[j];
L30:
    if (!k) goto L40;
    idex[i] = k;
    ++i;
    k = next[k];
    goto L30;
L40:
    ;
  }
  PetscFunctionReturn(0);
}

