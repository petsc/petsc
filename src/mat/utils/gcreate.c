#ifndef lint
static char vcid[] = "$Id: gcreate.c,v 1.54 1995/10/24 15:18:29 curfman Exp curfman $";
#endif

#include "sys.h"
#include "mat.h"       /*I "mat.h"  I*/

/*@C
   MatGetFormatFromOptions - Determines from the options database what matrix
   format the user has specified.

   Input Parameter:
.  comm - the MPI communicator
.  type - the type of matrix desired, for example MATSEQAIJ, MATMPIAIJ
.  pre - optional string to prepend to the name

   Output Parameters:
.  set - flag indicating whether user set matrix type option.

   Note:
   This routine is automatically called within MatCreate().

.keywords: matrix, get, format, from, options

.seealso: MatCreate()
@*/

int MatGetFormatFromOptions(MPI_Comm comm,char *pre,MatType *type,int *set)
{
  int size;
  char *p = "-";
  if (pre) p = pre;
  MPI_Comm_size(comm,&size);
  if (OptionsHasName(0,"-help")) {
    MPIU_printf(comm,"Matrix format options:\n");
    MPIU_printf(comm,"  %smat_aij, %smat_seqaij, %smat_mpiaij\n",p,p,p);
    MPIU_printf(comm,"  %smat_row, %smat_seqrow, %smat_mpirow\n",p,p,p);
    MPIU_printf(comm,"  %smat_dense, %smat_seqdense, %smat_mpidense\n",p,p,p);
    MPIU_printf(comm,"  %smat_mpirowbs, %smat_bdiag, %smat_seqbdiag, %smat_mpibdiag\n",p,p,p,p); 
    /* We need to move the following to MatPrintHelp or some analogous routine */
    MPIU_printf(comm,"More matrix options:\n");
    MPIU_printf(comm,"  %smat_view_info : view basic matrix info during MatAssemblyEnd()\n",p);
    MPIU_printf(comm,"  %smat_view_info_detailed : view detailed matrix info during MatAssemblyEnd()\n",p);
    MPIU_printf(comm,"  %smat_view_draw : draw nonzero matrix structure during MatAssemblyEnd()\n",p);
    MPIU_printf(comm,"      -pause <sec> : set seconds of display pause\n");
    MPIU_printf(comm,"      -display <name> : set alternate display\n");
  }
  if (OptionsHasName(pre,"-mat_seqdense")) {
    *type = MATSEQDENSE;
    *set = 1;
  }
  else if (OptionsHasName(pre,"-mat_mpidense")) {
    *type = MATMPIDENSE;
    *set = 1;
  }
  else if (OptionsHasName(pre,"-mat_seqbdiag")) {
    *type = MATSEQBDIAG;
    *set = 1;
  }
  else if (OptionsHasName(pre,"-mat_mpibdiag")) {
    *type = MATMPIBDIAG;
    *set = 1;
  }
  else if (OptionsHasName(pre,"-mat_mpirowbs")) {
    *type = MATMPIROWBS;
    *set = 1;
  }
  else if (OptionsHasName(pre,"-mat_mpirow")) {
    *type = MATMPIROW;
    *set = 1;
  }
  else if (OptionsHasName(pre,"-mat_seqrow")){
    *type = MATSEQROW;
    *set = 1;
  }
  else if (OptionsHasName(pre,"-mat_mpiaij")) {
    *type = MATMPIAIJ;
    *set = 1;
  }
  else if (OptionsHasName(pre,"-mat_seqaij")){
    *type = MATSEQAIJ;
    *set = 1;
  }
  else if (OptionsHasName(pre,"-mat_aij")){
    if (size == 1) *type = MATSEQAIJ;
    else *type = MATMPIAIJ;
    *set = 1;
  }  
  else if (OptionsHasName(pre,"-mat_row")){
    if (size == 1) *type = MATSEQROW;
    else *type = MATMPIROW;
    *set = 1;
  }  
  else if (OptionsHasName(pre,"-mat_bdiag")){
    if (size == 1) *type = MATSEQBDIAG;
    else *type = MATMPIBDIAG;
    *set = 1;
  }  
  else if (OptionsHasName(pre,"-mat_dense")){
    if (size == 1) *type = MATSEQDENSE;
    else *type = MATMPIDENSE;
    *set = 1;
  }  
  else {
    if (size == 1) *type = MATSEQAIJ;
    else *type = MATMPIAIJ;
    *set = 0;
  }
  return 0;
}

/*@C
   MatCreate - Creates a matrix, where the type is determined
   from the options database. Generates a parallel MPI matrix if the
   communicator has more than one processor.

   Input Parameters:
.  m - number of global rows
.  n - number of global columns
.  comm - MPI communicator
 
   Output Parameter:
.  V - location to stash resulting matrix

   Options Database Keywords:
$  -mat_seqaij   : AIJ type, uses MatCreateSeqAIJ
$  -mat_mpiaij   : AIJ type, uses MatCreateMPIAIJ
$  -mat_aij      : AIJ type, (Seq or MPI depending on comm) 
$  -mat_seqrow   : row type, uses MatCreateSeqRow()
$  -mat_mpirow   : MatCreateMPIRow()
$  -mat_row      : row type, (Seq or MPI depending on comm)  
$  -mat_seqbdiag : block diagonal type, uses 
$                  MatCreateSeqBDiag()
$  -mat_mpibdiag : block diagonal type, uses 
$                  MatCreateMPIBDiag()
$  -mat_bdiag    : block diagonal type, 
$                  (Seq or MPI depending on comm)
$  -mat_mpirowbs : rowbs type, uses MatCreateMPIRowbs()
$  -mat_dense    : dense type, (Seq or MPI depending on comm)
$  -mat_mpidense : dense type, uses MatCreateSeqDense()
$  -mat_mpidense : dense type, uses MatCreateMPIDense()

   Notes:
   The default matrix type is AIJ, using MatCreateSeqAIJ() and
   MatCreateMPIAIJ(). 

.keywords: matrix, create, initial

.seealso: MatCreateSeqAIJ((), MatCreateMPIAIJ(), 
          MatCreateSeqRow(), MatCreateMPIRow(), 
          MatCreateSeqBDiag(),MatCreateMPIBDiag(),
          MatCreateSeqDense(), MatCreateMPIDense(), 
          MatCreateMPIRowbs(), MatConvert()
          MatGetFormatFromOptions()
 @*/
int MatCreate(MPI_Comm comm,int m,int n,Mat *V)
{
  MatType type;
  int     set,ierr;

  ierr = MatGetFormatFromOptions(comm,0,&type,&set); CHKERRQ(ierr);
  if (type == MATSEQDENSE) {
    return MatCreateSeqDense(comm,m,n,V);
  }
  if (type == MATSEQBDIAG || type == MATMPIBDIAG) {
    int nb = 1, ndiag = 0, ndiag2 = 0, *d = 0;
    if (OptionsHasName(0,"-help")) {
      MPIU_printf(comm,"Options with -mat_bdiag: -mat_bdiag_bsize <block_size>\n");
      MPIU_printf(comm,"  -mat_bdiag_ndiag <number_diags> \n"); 
      MPIU_printf(comm,"  -mat_bdiag_dvals <d1,d2,d3,...> (diagonal numbers)\n"); 
      MPIU_printf(comm,"   (for example) -mat_bdiag_dvals -5,-1,0,1,5\n"); 
    }
    OptionsGetInt(0,"-mat_bdiag_bsize",&nb);
    OptionsGetInt(0,"-mat_bdiag_ndiag",&ndiag);
    if (ndiag) {
      d = (int *)PETSCMALLOC( ndiag * sizeof(int) ); CHKPTRQ(d);
      ndiag2 = ndiag;
      OptionsGetIntArray(0,"-mat_bdiag_dvals",d,&ndiag2);
      if (ndiag2 != ndiag)
        SETERRQ(1,"MatCreate: Incompatible number of diags and diagonal vals");
    } else if (OptionsHasName(0,"-mat_bdiag_dvals")) {
      SETERRQ(1,"MatCreate: Must specify number of diagonals with -mat_bdiag_ndiag");
    }
    if (type == MATMPIBDIAG) {
      ierr = MatCreateMPIBDiag(comm,PETSC_DECIDE,m,n,ndiag,nb,d,0,V); CHKERRQ(ierr);
    } else {
      ierr = MatCreateSeqBDiag(comm,m,n,ndiag,nb,d,0,V); CHKERRQ(ierr);
    }
    if (d) PETSCFREE(d);
    return 0;
  }
  if (type == MATMPIROWBS) {
    return MatCreateMPIRowbs(comm,PETSC_DECIDE,m,5,0,0,V);
  }
  if (type == MATMPIROW) {
    return MatCreateMPIRow(comm,PETSC_DECIDE,PETSC_DECIDE,m,n,5,0,0,0,V);
  }
  if (type == MATSEQROW) {
    return MatCreateSeqRow(comm,m,n,10,0,V);
  }
  if (type == MATMPIDENSE) {
    return MatCreateMPIDense(comm,PETSC_DECIDE,PETSC_DECIDE,m,n,V);
  }
  if (type == MATMPIAIJ) { 
    return MatCreateMPIAIJ(comm,PETSC_DECIDE,PETSC_DECIDE,m,n,5,0,0,0,V);
  }
  return MatCreateSeqAIJ(comm,m,n,10,0,V); 
}

#include "matimpl.h"
/*@C
   MatGetName - Gets the matrix type name (as a string) from the matrix.

   Input Parameter:
.  mat - the matrix

   Output Parameter:
.  name - name of matrix type

.keywords: matrix, get, name

.seealso:  MatGetType()
@*/
int MatGetName(Mat mat,char **name)
{
  int  itype = (int)mat->type;
  char *matname[10];
  /* Note:  Be sure that this list corresponds to the enum in mat.h */
  matname[0] = "MATSEQDENSE";
  matname[1] = "MATSEQAIJ";
  matname[2] = "MATMPIAIJ";
  matname[3] = "MATSHELL";
  matname[4] = "MATSEQROW";
  matname[5] = "MATMPIROW";
  matname[6] = "MATMPIROWBS";
  matname[7] = "MATSEQBDIAG";
  matname[8] = "MATMPIBDIAG";
  matname[9] = "MATMPIDENSE";
  if (itype < 0 || itype > 9) *name = "Unknown matrix type";
  else                        *name = matname[itype];
  return 0;
}
 
