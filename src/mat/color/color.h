==> degr.c <==
/* degr.f -- translated by f2c (version of 25 March 1992  12:58:56). */


int MINPACKdegr(int *n,int * indrow,int * jpntr,int * indcol,int * ipntr,int * ndeg,int * iwa)
{
    /* System generated locals */
    int i__1, i__2, i__3;

    /* Local variables */
    int jcol, ic, ip, jp, ir;

==> dsm.c <==
/* dsm.f -- translated by f2c (version of 25 March 1992  12:58:56). */

static integer c_n1 = -1;

int MINPACKdsm(int *m,int *n,int *npairs,int *indrow,int *indcol,int *ngrp,int *maxgrp,
               int *mingrp,int *info,int *ipntr,int *jpntr,int *iwa,int *liwa)
{
    /* System generated locals */
    int i__1, i__2, i__3;


==> ido.c <==
/* ido.f -- translated by f2c (version of 25 March 1992  12:58:56).*/

static integer c_n1 = -1;

int MINPACKido(int *m,int * n,int * indrow,int * jpntr,int * indcol,int * ipntr,int * ndeg,
               int *list,int *maxclq, int *iwa1, int *iwa2, int *iwa3, int *iwa4)
{
    /* System generated locals */
    int i__1, i__2, i__3, i__4;


==> numsrt.c <==
/* numsrt.f -- translated by f2c (version of 25 March 1992  12:58:56). */

int MINPACKnumsrt(int *n,int *nmax,int *num,int *mode,int *index,int *last,int *next)
{
    /* System generated locals */
    int i__1, i__2;

    /* Local variables */
    int jinc, i, j, k, l, jl, ju;


==> seq.c <==
/* seq.f -- translated by f2c (version of 25 March 1992  12:58:56).*/

int MINPACKseq(int *n,int *indrow,int *jpntr,int *indcol,int *ipntr,int *list,int *ngrp,
               int *maxgrp,int *iwa)
{
    /* System generated locals */
    int i__1, i__2, i__3;

    /* Local variables */
    int jcol, j, ic, ip, jp, ir;

==> setr.c <==
/* setr.f -- translated by f2c (version of 25 March 1992  12:58:56). */

int MINPACKsetr(int*m,int* n,int* indrow,int* jpntr,int* indcol, int*ipntr,int* iwa)
{
    /* System generated locals */
    int i__1, i__2;

    /* Local variables */
    int jcol, jp, ir;


==> slo.c <==
/* slo.f -- translated by f2c (version of 25 March 1992  12:58:56).*/

int MINPACKslo(int *n,int * indrow,int * jpntr,int * indcol, iint *pntr, nint *deg,int * list,
         int * maxclq,	 int *iwa1,int * iwa2,int * iwa3,int * iwa4)
{
    /* System generated locals */
    int i__1, i__2, i__3, i__4;

    /* Local variables */
    int jcol, ic, ip, jp, ir, mindeg, numdeg, numord;
