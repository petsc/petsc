#ifndef lint
static char vcid[] = "$Id: gcreate.c,v 1.67 1996/01/12 22:08:29 bsmith Exp bsmith $";
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
  int  size,flg1,flg2,flg3,flg4,flg5,flg8,flg9,flg10,flg12,flg13,ierr;
  char p[64];

  PetscStrcpy(p,"-");
  if (pre) PetscStrcat(p,pre);

  MPI_Comm_size(comm,&size);
  ierr = OptionsHasName(PETSC_NULL,"-help",&flg1); CHKERRQ(ierr);
  if (flg1) {
    MPIU_printf(comm,"Matrix format options:\n");
    MPIU_printf(comm,"  %smat_aij, %smat_seqaij, %smat_mpiaij\n",p,p,p);
    MPIU_printf(comm,"  %smat_dense, %smat_seqdense, %smat_mpidense\n",p,p,p);
    MPIU_printf(comm,"  %smat_mpirowbs, %smat_bdiag, %smat_seqbdiag, %smat_mpibdiag\n",p,p,p,p); 
  }
  ierr = OptionsHasName(pre,"-mat_seqdense",&flg1); CHKERRQ(ierr);
  ierr = OptionsHasName(pre,"-mat_mpidense",&flg2); CHKERRQ(ierr);
  ierr = OptionsHasName(pre,"-mat_seqbdiag",&flg3); CHKERRQ(ierr);
  ierr = OptionsHasName(pre,"-mat_mpibdiag",&flg4); CHKERRQ(ierr);
  ierr = OptionsHasName(pre,"-mat_mpirowbs",&flg5); CHKERRQ(ierr);
  ierr = OptionsHasName(pre,"-mat_mpiaij",&flg8); CHKERRQ(ierr);
  ierr = OptionsHasName(pre,"-mat_seqaij",&flg9); CHKERRQ(ierr);
  ierr = OptionsHasName(pre,"-mat_aij",&flg10); CHKERRQ(ierr);
  ierr = OptionsHasName(pre,"-mat_bdiag",&flg12); CHKERRQ(ierr);
  ierr = OptionsHasName(pre,"-mat_dense",&flg13); CHKERRQ(ierr);
  if (flg1) {
    *type = MATSEQDENSE;
    *set = 1;
  }
  else if (flg2) {
    *type = MATMPIDENSE;
    *set = 1;
  }
  else if (flg3) {
    *type = MATSEQBDIAG;
    *set = 1;
  }
  else if (flg4) {
    *type = MATMPIBDIAG;
    *set = 1;
  }
  else if (flg5) {
    *type = MATMPIROWBS;
    *set = 1;
  }
  else if (flg8) {
    *type = MATMPIAIJ;
    *set = 1;
  }
  else if (flg9){
    *type = MATSEQAIJ;
    *set = 1;
  }
  else if (flg10){
    if (size == 1) *type = MATSEQAIJ;
    else *type = MATMPIAIJ;
    *set = 1;
  }  
  else if (flg12){
    if (size == 1) *type = MATSEQBDIAG;
    else *type = MATMPIBDIAG;
    *set = 1;
  }  
  else if (flg13){
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
  if (type == MATSEQDENSE)
    return MatCreateSeqDense(comm,m,n,PETSC_NULL,V);
  if (type == MATMPIBDIAG)
    return MatCreateMPIBDiag(comm,PETSC_DECIDE,m,n,PETSC_DEFAULT,PETSC_DEFAULT,
           PETSC_NULL,PETSC_NULL,V);
  if (type == MATSEQBDIAG)
    return MatCreateSeqBDiag(comm,m,n,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_NULL,
           PETSC_NULL,V);
  if (type == MATMPIROWBS)
    return MatCreateMPIRowbs(comm,PETSC_DECIDE,m,PETSC_DEFAULT,PETSC_NULL,PETSC_NULL,V);
  if (type == MATMPIDENSE)
    return MatCreateMPIDense(comm,PETSC_DECIDE,PETSC_DECIDE,m,n,PETSC_NULL,V);
  if (type == MATMPIAIJ)
    return MatCreateMPIAIJ(comm,PETSC_DECIDE,PETSC_DECIDE,m,n,PETSC_DEFAULT,
           PETSC_NULL,PETSC_DEFAULT,PETSC_NULL,V);
  return MatCreateSeqAIJ(comm,m,n,PETSC_DEFAULT,PETSC_NULL,V); 
}

#include "matimpl.h"
/*@C
   MatGetType - Gets the matrix type and name (as a string) from the matrix.

   Input Parameter:
.  mat - the matrix

   Output Parameter:
.  type - the matrix type (or use PETSC_NULL)
.  name - name of matrix type (or use PETSC_NULL)

.keywords: matrix, get, name
@*/
int MatGetType(Mat mat,MatType *type,char **name)
{
  int  itype = (int)mat->type;
  char *matname[10];

  if (type) *type = (MatType) mat->type;
  if (name) {
    /* Note:  Be sure that this list corresponds to the enum in mat.h */
    matname[0] = "MATSEQDENSE";
    matname[1] = "MATSEQAIJ";
    matname[2] = "MATMPIAIJ";
    matname[3] = "MATSHELL";
    matname[4] = "MATMPIROWBS";
    matname[5] = "MATSEQBDIAG";
    matname[6] = "MATMPIBDIAG";
    matname[7] = "MATMPIDENSE";
    if (itype < 0 || itype > 7) *name = "Unknown matrix type";
    else                        *name = matname[itype];
  }
  return 0;
}
 


