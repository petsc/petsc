#define PETSCMAT_DLL

/* ilut.f -- translated by f2c (version of 25 March 1992  12:58:56).

     The Fortran version of this code was developed by Yousef Saad.
  This code is copyrighted by Yousef Saad with the 

		    GNU GENERAL PUBLIC LICENSE
		       Version 2, June 1991

 Copyright (C) 1989, 1991 Free Software Foundation, Inc.
                          675 Mass Ave, Cambridge, MA 02139, USA
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.

			    Preamble

  The licenses for most software are designed to take away your
freedom to share and change it.  By contrast, the GNU General Public
License is intended to guarantee your freedom to share and change free
software--to make sure the software is free for all its users.  This
General Public License applies to most of the Free Software
Foundation's software and to any other program whose authors commit to
using it.  (Some other Free Software Foundation software is covered by
the GNU Library General Public License instead.)  You can apply it to
your programs, too.

  When we speak of free software, we are referring to freedom, not
price.  Our General Public Licenses are designed to make sure that you
have the freedom to distribute copies of free software (and charge for
this service if you wish), that you receive source code or can get it
if you want it, that you can change the software or use pieces of it
in new free programs; and that you know you can do these things.

  To protect your rights, we need to make restrictions that forbid
anyone to deny you these rights or to ask you to surrender the rights.
These restrictions translate to certain responsibilities for you if you
distribute copies of the software, or if you modify it.

  For example, if you distribute copies of such a program, whether
gratis or for a fee, you must give the recipients all the rights that
you have.  You must make sure that they, too, receive or can get the
source code.  And you must show them these terms so they know their
rights.

  We protect your rights with two steps: (1) copyright the software, and
(2) offer you this license which gives you legal permission to copy,
distribute and/or modify the software.

  Also, for each author's protection and ours, we want to make certain
that everyone understands that there is no warranty for this free
software.  If the software is modified by someone else and passed on, we
want its recipients to know that what they have is not the original, so
that any problems introduced by others will not reflect on the original
authors' reputations.

  Finally, any free program is threatened constantly by software
patents.  We wish to avoid the danger that redistributors of a free
program will individually obtain patent licenses, in effect making the
program proprietary.  To prevent this, we have made it clear that any
patent must be licensed for everyone's free use or not licensed at all.

  The precise terms and conditions for copying, distribution and
modification follow.

		    GNU GENERAL PUBLIC LICENSE
   TERMS AND CONDITIONS FOR COPYING, DISTRIBUTION AND MODIFICATION

  0. This License applies to any program or other work which contains
a notice placed by the copyright holder saying it may be distributed
under the terms of this General Public License.  The "Program", below,
refers to any such program or work, and a "work based on the Program"
means either the Program or any derivative work under copyright law:
that is to say, a work containing the Program or a portion of it,
either verbatim or with modifications and/or translated into another
language.  (Hereinafter, translation is included without limitation in
the term "modification".)  Each licensee is addressed as "you".

Activities other than copying, distribution and modification are not
covered by this License; they are outside its scope.  The act of
running the Program is not restricted, and the output from the Program
is covered only if its contents constitute a work based on the
Program (independent of having been made by running the Program).
Whether that is true depends on what the Program does.

  1. You may copy and distribute verbatim copies of the Program's
source code as you receive it, in any medium, provided that you
conspicuously and appropriately publish on each copy an appropriate
copyright notice and disclaimer of warranty; keep intact all the
notices that refer to this License and to the absence of any warranty;
and give any other recipients of the Program a copy of this License
along with the Program.

You may charge a fee for the physical act of transferring a copy, and
you may at your option offer warranty protection in exchange for a fee.

  2. You may modify your copy or copies of the Program or any portion
of it, thus forming a work based on the Program, and copy and
distribute such modifications or work under the terms of Section 1
above, provided that you also meet all of these conditions:

    a) You must cause the modified files to carry prominent notices
    stating that you changed the files and the date of any change.

    b) You must cause any work that you distribute or publish, that in
    whole or in part contains or is derived from the Program or any
    part thereof, to be licensed as a whole at no charge to all third
    parties under the terms of this License.

    c) If the modified program normally reads commands interactively
    when run, you must cause it, when started running for such
    interactive use in the most ordinary way, to print or display an
    announcement including an appropriate copyright notice and a
    notice that there is no warranty (or else, saying that you provide
    a warranty) and that users may redistribute the program under
    these conditions, and telling the user how to view a copy of this
    License.  (Exception: if the Program itself is interactive but
    does not normally print such an announcement, your work based on
    the Program is not required to print an announcement.)

These requirements apply to the modified work as a whole.  If
identifiable sections of that work are not derived from the Program,
and can be reasonably considered independent and separate works in
themselves, then this License, and its terms, do not apply to those
sections when you distribute them as separate works.  But when you
distribute the same sections as part of a whole which is a work based
on the Program, the distribution of the whole must be on the terms of
this License, whose permissions for other licensees extend to the
entire whole, and thus to each and every part regardless of who wrote it.

Thus, it is not the intent of this section to claim rights or contest
your rights to work written entirely by you; rather, the intent is to
exercise the right to control the distribution of derivative or
collective works based on the Program.

In addition, mere aggregation of another work not based on the Program
with the Program (or with a work based on the Program) on a volume of
a storage or distribution medium does not bring the other work under
the scope of this License.

  3. You may copy and distribute the Program (or a work based on it,
under Section 2) in object code or executable form under the terms of
Sections 1 and 2 above provided that you also do one of the following:

    a) Accompany it with the complete corresponding machine-readable
    source code, which must be distributed under the terms of Sections
    1 and 2 above on a medium customarily used for software interchange; or,

    b) Accompany it with a written offer, valid for at least three
    years, to give any third party, for a charge no more than your
    cost of physically performing source distribution, a complete
    machine-readable copy of the corresponding source code, to be
    distributed under the terms of Sections 1 and 2 above on a medium
    customarily used for software interchange; or,

    c) Accompany it with the information you received as to the offer
    to distribute corresponding source code.  (This alternative is
    allowed only for noncommercial distribution and only if you
    received the program in object code or executable form with such
    an offer, in accord with Subsection b above.)

The source code for a work means the preferred form of the work for
making modifications to it.  For an executable work, complete source
code means all the source code for all modules it contains, plus any
associated interface definition files, plus the scripts used to
control compilation and installation of the executable.  However, as a
special exception, the source code distributed need not include
anything that is normally distributed (in either source or binary
form) with the major components (compiler, kernel, and so on) of the
operating system on which the executable runs, unless that component
itself accompanies the executable.

If distribution of executable or object code is made by offering
access to copy from a designated place, then offering equivalent
access to copy the source code from the same place counts as
distribution of the source code, even though third parties are not
compelled to copy the source along with the object code.

  4. You may not copy, modify, sublicense, or distribute the Program
except as expressly provided under this License.  Any attempt
otherwise to copy, modify, sublicense or distribute the Program is
void, and will automatically terminate your rights under this License.
However, parties who have received copies, or rights, from you under
this License will not have their licenses terminated so long as such
parties remain in full compliance.

  5. You are not required to accept this License, since you have not
signed it.  However, nothing else grants you permission to modify or
distribute the Program or its derivative works.  These actions are
prohibited by law if you do not accept this License.  Therefore, by
modifying or distributing the Program (or any work based on the
Program), you indicate your acceptance of this License to do so, and
all its terms and conditions for copying, distributing or modifying
the Program or works based on it.

  6. Each time you redistribute the Program (or any work based on the
Program), the recipient automatically receives a license from the
original licensor to copy, distribute or modify the Program subject to
these terms and conditions.  You may not impose any further
restrictions on the recipients' exercise of the rights granted herein.
You are not responsible for enforcing compliance by third parties to
this License.

  7. If, as a consequence of a court judgment or allegation of patent
infringement or for any other reason (not limited to patent issues),
conditions are imposed on you (whether by court order, agreement or
otherwise) that contradict the conditions of this License, they do not
excuse you from the conditions of this License.  If you cannot
distribute so as to satisfy simultaneously your obligations under this
License and any other pertinent obligations, then as a consequence you
may not distribute the Program at all.  For example, if a patent
license would not permit royalty-free redistribution of the Program by
all those who receive copies directly or indirectly through you, then
the only way you could satisfy both it and this License would be to
refrain entirely from distribution of the Program.

If any portion of this section is held invalid or unenforceable under
any particular circumstance, the balance of the section is intended to
apply and the section as a whole is intended to apply in other
circumstances.

It is not the purpose of this section to induce you to infringe any
patents or other property right claims or to contest validity of any
such claims; this section has the sole purpose of protecting the
integrity of the free software distribution system, which is
implemented by public license practices.  Many people have made
generous contributions to the wide range of software distributed
through that system in reliance on consistent application of that
system; it is up to the author/donor to decide if he or she is willing
to distribute software through any other system and a licensee cannot
impose that choice.

This section is intended to make thoroughly clear what is believed to
be a consequence of the rest of this License.

  8. If the distribution and/or use of the Program is restricted in
certain countries either by patents or by copyrighted interfaces, the
original copyright holder who places the Program under this License
may add an explicit geographical distribution limitation excluding
those countries, so that distribution is permitted only in or among
countries not thus excluded.  In such case, this License incorporates
the limitation as if written in the body of this License.

  9. The Free Software Foundation may publish revised and/or new versions
of the General Public License from time to time.  Such new versions will
be similar in spirit to the present version, but may differ in detail to
address new problems or concerns.

Each version is given a distinguishing version number.  If the Program
specifies a version number of this License which applies to it and "any
later version", you have the option of following the terms and conditions
either of that version or of any later version published by the Free
Software Foundation.  If the Program does not specify a version number of
this License, you may choose any version ever published by the Free Software
Foundation.

  10. If you wish to incorporate parts of the Program into other free
programs whose distribution conditions are different, write to the author
to ask for permission.  For software which is copyrighted by the Free
Software Foundation, write to the Free Software Foundation; we sometimes
make exceptions for this.  Our decision will be guided by the two goals
of preserving the free status of all derivatives of our free software and
of promoting the sharing and reuse of software generally.

			    NO WARRANTY

  11. BECAUSE THE PROGRAM IS LICENSED FREE OF CHARGE, THERE IS NO WARRANTY
FOR THE PROGRAM, TO THE EXTENT PERMITTED BY APPLICABLE LAW.  EXCEPT WHEN
OTHERWISE STATED IN WRITING THE COPYRIGHT HOLDERS AND/OR OTHER PARTIES
PROVIDE THE PROGRAM "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER EXPRESSED
OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  THE ENTIRE RISK AS
TO THE QUALITY AND PERFORMANCE OF THE PROGRAM IS WITH YOU.  SHOULD THE
PROGRAM PROVE DEFECTIVE, YOU ASSUME THE COST OF ALL NECESSARY SERVICING,
REPAIR OR CORRECTION.

  12. IN NO EVENT UNLESS REQUIRED BY APPLICABLE LAW OR AGREED TO IN WRITING
WILL ANY COPYRIGHT HOLDER, OR ANY OTHER PARTY WHO MAY MODIFY AND/OR
REDISTRIBUTE THE PROGRAM AS PERMITTED ABOVE, BE LIABLE TO YOU FOR DAMAGES,
INCLUDING ANY GENERAL, SPECIAL, INCIDENTAL OR CONSEQUENTIAL DAMAGES ARISING
OUT OF THE USE OR INABILITY TO USE THE PROGRAM (INCLUDING BUT NOT LIMITED
TO LOSS OF DATA OR DATA BEING RENDERED INACCURATE OR LOSSES SUSTAINED BY
YOU OR THIRD PARTIES OR A FAILURE OF THE PROGRAM TO OPERATE WITH ANY OTHER
PROGRAMS), EVEN IF SUCH HOLDER OR OTHER PARTY HAS BEEN ADVISED OF THE
POSSIBILITY OF SUCH DAMAGES.

		     END OF TERMS AND CONDITIONS

	Appendix: How to Apply These Terms to Your New Programs

  If you develop a new program, and you want it to be of the greatest
possible use to the public, the best way to achieve this is to make it
free software which everyone can redistribute and change under these terms.

  To do so, attach the following notices to the program.  It is safest
to attach them to the start of each source file to most effectively
convey the exclusion of warranty; and each file should have at least
the "copyright" line and a pointer to where the full notice is found.

    <one line to give the program's name and a brief idea of what it does.>
    Copyright (C) 19yy  <name of author>

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

Also add information on how to contact you by electronic and paper mail.

If the program is interactive, make it output a short notice like this
when it starts in an interactive mode:

    Gnomovision version 69, Copyright (C) 19yy name of author
    Gnomovision comes with ABSOLUTELY NO WARRANTY; for details type `show w'.
    This is free software, and you are welcome to redistribute it
    under certain conditions; type `show c' for details.

The hypothetical commands `show w' and `show c' should show the appropriate
parts of the General Public License.  Of course, the commands you use may
be called something other than `show w' and `show c'; they could even be
mouse-clicks or menu items--whatever suits your program.

You should also get your employer (if you work as a programmer) or your
school, if any, to sign a "copyright disclaimer" for the program, if
necessary.  Here is a sample; alter the names:

  Yoyodyne, Inc., hereby disclaims all copyright interest in the program
  `Gnomovision' (which makes passes at compilers) written by James Hacker.

  <signature of Ty Coon>, 1 April 1989
  Ty Coon, President of Vice

This General Public License does not permit incorporating your program into
proprietary programs.  If your program is a subroutine library, you may
consider it more useful to permit linking proprietary applications with the
library.  If this is what you want to do, use the GNU Library General
Public License instead of this License.

*/
#include "petsc.h"

static PetscErrorCode SPARSEKIT2qsplit(PetscScalar *a,PetscInt *ind,PetscInt *n,PetscInt *ncut)
{
    /* System generated locals */
    PetscInt i__1;
    PetscScalar d__1;

    /* Local variables */
    PetscInt last,itmp,j,first;
    PetscReal abskey;
    PetscInt mid;
    PetscScalar tmp;

/* -----------------------------------------------------------------------
 */
/*     does a quick-sort split of a real array. */
/*     on input a(1:n). is a real array */
/*     on output a(1:n) is permuted such that its elements satisfy: */

/*     abs(a(i)) .ge. abs(a(ncut)) for i .lt. ncut and */
/*     abs(a(i)) .le. abs(a(ncut)) for i .gt. ncut */

/*    ind(1:n) is an integer array which permuted in the same way as a(*).
*/
/* -----------------------------------------------------------------------
 */
/* ----- */
    /* Parameter adjustments */
    --ind;
    --a;

    /* Function Body */
    first = 1;
    last = *n;
    if (*ncut < first || *ncut > last) {
	return 0;
    }

/*     outer loop -- while mid .ne. ncut do */

L1:
    mid = first;
    abskey = (d__1 = a[mid],PetscAbsScalar(d__1));
    i__1 = last;
    for (j = first + 1; j <= i__1; ++j) {
	if ((d__1 = a[j],PetscAbsScalar(d__1)) > abskey) {
	    ++mid;
/*     interchange */
	    tmp = a[mid];
	    itmp = ind[mid];
	    a[mid] = a[j];
	    ind[mid] = ind[j];
	    a[j] = tmp;
	    ind[j] = itmp;
	}
/* L2: */
    }

/*     interchange */

    tmp = a[mid];
    a[mid] = a[first];
    a[first] = tmp;

    itmp = ind[mid];
    ind[mid] = ind[first];
    ind[first] = itmp;

/*     test for while loop */

    if (mid == *ncut) {
	return 0;
    }
    if (mid > *ncut) {
	last = mid - 1;
    } else {
	first = mid + 1;
    }
    goto L1;
/* ----------------end-of-qsplit------------------------------------------
 */
/* -----------------------------------------------------------------------
 */
} /* qsplit_ */


/* ---------------------------------------------------------------------- */
PetscErrorCode SPARSEKIT2ilutp(PetscInt *n,PetscScalar *a,PetscInt *ja,PetscInt * ia,PetscInt *lfil,PetscReal droptol,PetscReal *permtol,PetscInt *mbloc,PetscScalar *alu,
	PetscInt *jlu,PetscInt *ju,PetscInt *iwk,PetscScalar *w,PetscInt *jw,  PetscInt *iperm,PetscErrorCode *ierr)
{
    /* System generated locals */
    PetscInt i__1,i__2;
    PetscScalar d__1;

    /* Local variables */
    PetscScalar fact;
    PetscInt lenl,imax,lenu,icut,jpos;
    PetscReal xmax;
    PetscInt jrow;
    PetscReal xmax0;
    PetscInt i,j,k;
    PetscScalar s,t;
    PetscInt j_1,j2;
    PetscReal tnorm,t1;
    PetscInt ii,jj;
    PetscInt ju0,len;
    PetscScalar tmp;

/* -----------------------------------------------------------------------
 */
/*     implicit none */
/* ----------------------------------------------------------------------*
 */
/*       *** ILUTP preconditioner -- ILUT with pivoting  ***            * 
*/
/*      incomplete LU factorization with dual truncation mechanism      * 
*/
/* ----------------------------------------------------------------------*
 */
/* author Yousef Saad *Sep 8, 1993 -- Latest revision, August 1996.     * 
*/
/* ----------------------------------------------------------------------*
 */
/* on entry: */
/* ========== */
/* n       = integer. The dimension of the matrix A. */

/* a,ja,ia = matrix stored in Compressed Sparse Row format. */
/*           ON RETURN THE COLUMNS OF A ARE PERMUTED. SEE BELOW FOR */
/*           DETAILS. */

/* lfil    = integer. The fill-in parameter. Each row of L and each row */

/*           of U will have a maximum of lfil elements (excluding the */
/*           diagonal element). lfil must be .ge. 0. */
/*           ** WARNING: THE MEANING OF LFIL HAS CHANGED WITH RESPECT TO 
*/
/*           EARLIER VERSIONS. */

/* droptol = real*8. Sets the threshold for dropping small terms in the */

/*           factorization. See below for details on dropping strategy. */


/* lfil    = integer. The fill-in parameter. Each row of L and */
/*           each row of U will have a maximum of lfil elements. */
/*           WARNING: THE MEANING OF LFIL HAS CHANGED WITH RESPECT TO */
/*           EARLIER VERSIONS. */
/*           lfil must be .ge. 0. */

/* permtol = tolerance ratio used to  determne whether or not to permute 
*/
/*           two columns.  At step i columns i and j are permuted when */

/*                     abs(a(i,j))*permtol .gt. abs(a(i,i)) */

/*           [0 --> never permute; good values 0.1 to 0.01] */

/* mbloc   = if desired, permuting can be done only within the diagonal */

/*           blocks of size mbloc. Useful for PDE problems with several */

/*           degrees of freedom.. If feature not wanted take mbloc=n. */


/* iwk     = integer. The lengths of arrays alu and jlu. If the arrays */
/*           are not big enough to store the ILU factorizations, ilut */
/*           will stop with an error message. */

/* On return: */
/* =========== */

/* alu,jlu = matrix stored in Modified Sparse Row (MSR) format containing 
*/
/*           the L and U factors together. The diagonal (stored in */
/*           alu(1:n)) is inverted. Each i-th row of the alu,jlu matrix 
*/
/*           contains the i-th row of L (excluding the diagonal entry=1) 
*/
/*           followed by the i-th row of U. */

/* ju      = integer array of length n containing the pointers to */
/*           the beginning of each row of U in the matrix alu,jlu. */

/* iperm   = contains the permutation arrays. */
/*           iperm(1:n) = old numbers of unknowns */
/*           iperm(n+1:2*n) = reverse permutation = new unknowns. */

/* ierr    = integer. Error message with the following meaning. */
/*           ierr  = 0    --> successful return. */
/*           ierr .gt. 0  --> zero pivot encountered at step number ierr. 
*/
/*           ierr  = -1   --> Error. input matrix may be wrong. */
/*                            (The elimination process has generated a */
/*                            row in L or U whose length is .gt.  n.) */
/*           ierr  = -2   --> The matrix L overflows the array al. */
/*           ierr  = -3   --> The matrix U overflows the array alu. */
/*           ierr  = -4   --> Illegal value for lfil. */
/*           ierr  = -5   --> zero row encountered. */

/* work arrays: */
/* ============= */
/* jw      = integer work array of length 2*n. */
/* w       = real work array of length n */

/* IMPORTANR NOTE: */
/* -------------- */
/* TO AVOID PERMUTING THE SOLUTION VECTORS ARRAYS FOR EACH LU-SOLVE, */
/* THE MATRIX A IS PERMUTED ON RETURN. [all column indices are */
/* changed]. SIMILARLY FOR THE U MATRIX. */
/* To permute the matrix back to its original state use the loop: */

/*      do k=ia(1), ia(n+1)-1 */
/*         ja(k) = iperm(ja(k)) */
/*      enddo */

/* -----------------------------------------------------------------------
 */
/*     local variables */


    /* Parameter adjustments */
    --iperm;
    --jw;
    --w;
    --ju;
    --jlu;
    --alu;
    --ia;
    --ja;
    --a;

    /* Function Body */
    if (*lfil < 0) {
	goto L998;
    }
/* -----------------------------------------------------------------------
 */
/*     initialize ju0 (points to next element to be added to alu,jlu) */
/*     and pointer array. */
/* -----------------------------------------------------------------------
 */
    ju0 = *n + 2;
    jlu[1] = ju0;

/*  integer PetscReal pointer array. */

    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	jw[*n + j] = 0;
	iperm[j] = j;
	iperm[*n + j] = j;
/* L1: */
    }
/* -----------------------------------------------------------------------
 */
/*     beginning of main loop. */
/* -----------------------------------------------------------------------
 */
    i__1 = *n;
    for (ii = 1; ii <= i__1; ++ii) {
	j_1 = ia[ii];
	j2 = ia[ii + 1] - 1;
	tnorm = 0.;
	i__2 = j2;
	for (k = j_1; k <= i__2; ++k) {
	    tnorm += (d__1 = a[k], PetscAbsScalar(d__1));
/* L501: */
	}
	if (!tnorm) {
	    goto L999;
	}
	tnorm /= j2 - j_1 + 1;

/*     unpack L-part and U-part of row of A in arrays  w  -- */

	lenu = 1;
	lenl = 0;
	jw[ii] = ii;
	w[ii] = (float)0.;
	jw[*n + ii] = ii;

	i__2 = j2;
	for (j = j_1; j <= i__2; ++j) {
	    k = iperm[*n + ja[j]];
	    t = a[j];
	    if (k < ii) {
		++lenl;
		jw[lenl] = k;
		w[lenl] = t;
		jw[*n + k] = lenl;
	    } else if (k == ii) {
		w[ii] = t;
	    } else {
		++lenu;
		jpos = ii + lenu - 1;
		jw[jpos] = k;
		w[jpos] = t;
		jw[*n + k] = jpos;
	    }
/* L170: */
	}
	jj = 0;
	len = 0;

/*     eliminate previous rows */

L150:
	++jj;
	if (jj > lenl) {
	    goto L160;
	}
/* ------------------------------------------------------------------
----- */
/*     in order to do the elimination in the correct order we must sel
ect */
/*     the smallest column index among jw(k), k=jj+1, ..., lenl. */
/* ------------------------------------------------------------------
----- */
	jrow = jw[jj];
	k = jj;

/*     determine smallest column index */

	i__2 = lenl;
	for (j = jj + 1; j <= i__2; ++j) {
	    if (jw[j] < jrow) {
		jrow = jw[j];
		k = j;
	    }
/* L151: */
	}

	if (k != jj) {
/*     exchange in jw */
	    j = jw[jj];
	    jw[jj] = jw[k];
	    jw[k] = j;
/*     exchange in jr */
	    jw[*n + jrow] = jj;
	    jw[*n + j] = k;
/*     exchange in w */
	    s = w[jj];
	    w[jj] = w[k];
	    w[k] = s;
	}

/*     zero out element in row by resetting jw(n+jrow) to zero. */

	jw[*n + jrow] = 0;

/*     get the multiplier for row to be eliminated: jrow */

	fact = w[jj] * alu[jrow];

/*     drop term if small */

	if (PetscAbsScalar(fact) <= droptol) {
	    goto L150;
	}

/*     combine current row and row jrow */

	i__2 = jlu[jrow + 1] - 1;
	for (k = ju[jrow]; k <= i__2; ++k) {
	    s = fact * alu[k];
/*     new column number */
	    j = iperm[*n + jlu[k]];
	    jpos = jw[*n + j];
	    if (j >= ii) {

/*     dealing with upper part. */

		if (!jpos) {

/*     this is a fill-in element */

		    ++lenu;
		    i = ii + lenu - 1;
		    if (lenu > *n) {
			goto L995;
		    }
		    jw[i] = j;
		    jw[*n + j] = i;
		    w[i] = -s;
		} else {
/*     no fill-in element -- */
		    w[jpos] -= s;
		}
	    } else {

/*     dealing with lower part. */

		if (!jpos) {

/*     this is a fill-in element */

		    ++lenl;
		    if (lenl > *n) {
			goto L995;
		    }
		    jw[lenl] = j;
		    jw[*n + j] = lenl;
		    w[lenl] = -s;
		} else {

/*     this is not a fill-in element */

		    w[jpos] -= s;
		}
	    }
/* L203: */
	}

/*     store this pivot element -- (from left to right -- no danger of
 */
/*     overlap with the working elements in L (pivots). */

	++len;
	w[len] = fact;
	jw[len] = jrow;
	goto L150;
L160:

/*     reset double-pointer to zero (U-part) */

	i__2 = lenu;
	for (k = 1; k <= i__2; ++k) {
	    jw[*n + jw[ii + k - 1]] = 0;
/* L308: */
	}

/*     update L-matrix */

	lenl = len;
	len = PetscMin(lenl,*lfil);

/*     sort by quick-split */

	SPARSEKIT2qsplit(&w[1], &jw[1], &lenl, &len);

/*     store L-part -- in original coordinates .. */

	i__2 = len;
	for (k = 1; k <= i__2; ++k) {
	    if (ju0 > *iwk) {
		goto L996;
	    }
	    alu[ju0] = w[k];
	    jlu[ju0] = iperm[jw[k]];
	    ++ju0;
/* L204: */
	}

/*     save pointer to beginning of row ii of U */

	ju[ii] = ju0;

/*     update U-matrix -- first apply dropping strategy */

	len = 0;
	i__2 = lenu - 1;
	for (k = 1; k <= i__2; ++k) {
	    if ((d__1 = w[ii + k], PetscAbsScalar(d__1)) > droptol * tnorm) {
		++len;
		w[ii + len] = w[ii + k];
		jw[ii + len] = jw[ii + k];
	    }
	}
	lenu = len + 1;
	len = PetscMin(lenu,*lfil);
	i__2 = lenu - 1;
	SPARSEKIT2qsplit(&w[ii + 1], &jw[ii + 1], &i__2, &len);

/*     determine next pivot -- */

	imax = ii;
	xmax = (d__1 = w[imax], PetscAbsScalar(d__1));
	xmax0 = xmax;
	icut = ii - 1 + *mbloc - (ii - 1) % *mbloc;
	i__2 = ii + len - 1;
	for (k = ii + 1; k <= i__2; ++k) {
	    t1 = (d__1 = w[k], PetscAbsScalar(d__1));
	    if (t1 > xmax && t1 * *permtol > xmax0 && jw[k] <= icut) {
		imax = k;
		xmax = t1;
	    }
	}

/*     exchange w's */

	tmp = w[ii];
	w[ii] = w[imax];
	w[imax] = tmp;

/*     update iperm and reverse iperm */

	j = jw[imax];
	i = iperm[ii];
	iperm[ii] = iperm[j];
	iperm[j] = i;

/*     reverse iperm */

	iperm[*n + iperm[ii]] = ii;
	iperm[*n + iperm[j]] = j;
/* ------------------------------------------------------------------
----- */

	if (len + ju0 > *iwk) {
	    goto L997;
	}

/*     copy U-part in original coordinates */

	i__2 = ii + len - 1;
	for (k = ii + 1; k <= i__2; ++k) {
	    jlu[ju0] = iperm[jw[k]];
	    alu[ju0] = w[k];
	    ++ju0;
/* L302: */
	}

/*     store inverse of diagonal element of u */

	if (w[ii] == 0.0) {
	    w[ii] = (droptol + 1e-4) * tnorm;
	}
	alu[ii] = 1. / w[ii];

/*     update pointer to beginning of next row of U. */

	jlu[ii + 1] = ju0;
/* ------------------------------------------------------------------
----- */
/*     end main loop */
/* ------------------------------------------------------------------
----- */
/* L500: */
    }

/*     permute all column indices of LU ... */

    i__1 = jlu[*n + 1] - 1;
    for (k = jlu[1]; k <= i__1; ++k) {
	jlu[k] = iperm[*n + jlu[k]];
    }

/*     ...and of A */

    i__1 = ia[*n + 1] - 1;
    for (k = ia[1]; k <= i__1; ++k) {
	ja[k] = iperm[*n + ja[k]];
    }

    *ierr = 0;
    return 0;

/*     incomprehensible error. Matrix must be wrong. */

L995:
    *ierr = -1;
    return 0;

/*     insufficient storage in L. */

L996:
    *ierr = -2;
    return 0;

/*     insufficient storage in U. */

L997:
    *ierr = -3;
    return 0;

/*     illegal lfil entered. */

L998:
    *ierr = -4;
    return 0;

/*     zero row encountered */

L999:
    *ierr = -5;
    return 0;
/* ----------------end-of-ilutp-------------------------------------------
 */
/* -----------------------------------------------------------------------
 */
} /* ilutp_ */

