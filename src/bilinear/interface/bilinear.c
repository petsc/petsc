#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: bilinear.c,v 1.3 2000/01/10 03:12:26 knepley Exp $";
#endif

/*
   This is where the abstract bilinear operations are defined
*/

#include "src/bilinear/bilinearimpl.h"        /*I "bilinear.h" I*/
#include "include/private/vecimpl.h"
#include "include/private/matimpl.h"

/* Logging support */
int BILINEAR_COOKIE;
int BILINEAR_Copy, BILINEAR_Convert, BILINEAR_SetValues, BILINEAR_AssemblyBegin, BILINEAR_AssemblyEnd;
int BILINEAR_ZeroEntries, BILINEAR_Mult, BILINEAR_FullMult, BILINEAR_Diamond, BILINEAR_LUFactor, BILINEAR_CholeskyFactor;

/*--------------------------------------------- Basic Functions -----------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "BilinearView"
/*@C
  BilinearView - Visualizes a bilinear operator object.

  Input Parameters:
. B      - The bilinear operator
. viewer - [Optional] The visualization context

   Notes:
   The available visualization contexts include
$     PETSC_VIEWER_STDOUT_SELF - standard output (default)
$     PETSC_VIEWER_STDOUT_WORLD - synchronized standard
$       output where only the first processor opens
$       the file.  All other processors send their 
$       data to the first processor to print. 
$     PETSC_VIEWER_DRAWX_WORLD - graphical display of nonzero structure

   The user can open alternative vistualization contexts with
$    ViewerFileOpenASCII() - output to a specified file
$    ViewerFileOpenBinary() - output in binary to a
$         specified file; corresponding input uses BilinearLoad()
$    ViewerDrawOpenX() - output nonzero bilinear operator structure to 
$         an X window display
$    ViewerMatlabOpen() - output bilinear operator to Matlab viewer.
$         Currently only the sequential dense and AIJ
$         bilinear operator types support the Matlab viewer.
$    ViewerMathematicaOpen() - output bilinear operator to Mathematica.

   The user can call ViewerSetFormat() to specify the output
   format of ASCII printed objects (when using PETSC_VIEWER_STDOUT_SELF,
   PETSC_VIEWER_STDOUT_WORLD and ViewerFileOpenASCII).  Available formats include
$    PETSC_VIEWER_FORMAT_ASCII_DEFAULT - default, prints bilinear operator contents
$    PETSC_VIEWER_FORMAT_ASCII_MATLAB - Matlab format
$    PETSC_VIEWER_FORMAT_ASCII_IMPL - implementation-specific format
$      (which is in many cases the same as the default)
$    PETSC_VIEWER_FORMAT_ASCII_INFO - basic information about the bilinear operator
$      size and structure (not the bilinear operator entries)
$    PETSC_VIEWER_FORMAT_ASCII_INFO_LONG - more detailed information about the 
$      bilinear operator structure

.keywords: bilinear operator, view, visualize, output, print, write, draw

.seealso: ViewerSetFormat(), ViewerFileOpenASCII(), ViewerDrawOpenX(), 
          ViewerMatlabOpen(), PetscViewerMathematicaOpen(), PetscViewerFileOpenBinary()
@*/
int BilinearView(Bilinear B, PetscViewer viewer)
{
  int        format;
  int        n_i, n_j, n_k;
  char      *cstr;
  PetscTruth isascii;
  int        ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, BILINEAR_COOKIE);
  if (!viewer)
    viewer = PETSC_VIEWER_STDOUT_SELF;
  else
    PetscValidHeaderSpecific(viewer, PETSC_VIEWER_COOKIE);
  if (!B->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "Not for unassembled bilinear operator");

  ierr = PetscTypeCompare((PetscObject) viewer, PETSC_VIEWER_ASCII, &isascii);                            CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerGetFormat(viewer, &format);                                                         CHKERRQ(ierr);  
    if (format == PETSC_VIEWER_FORMAT_ASCII_INFO || format == PETSC_VIEWER_FORMAT_ASCII_INFO_LONG) {
      PetscViewerASCIIPrintf(viewer, " Bilinear Object:\n");
      ierr = BilinearGetType(B, PETSC_NULL, &cstr);                                                       CHKERRQ(ierr);
      ierr = BilinearGetSize(B, &n_i, &n_j, &n_k);                                                        CHKERRQ(ierr);
      PetscViewerASCIIPrintf(viewer, "  type=%s, n_i=%d, n_j=%d, n_k=%d\n", cstr, n_i, n_j, n_k);
      if (B->ops->getinfo) {
        BilinearInfo info;

        ierr = BilinearGetInfo(B, INFO_GLOBAL_SUM, &info);                                                CHKERRQ(ierr);
        PetscViewerASCIIPrintf(viewer, "  total: nonzeros=%d, allocated nonzeros=%d\n", (int) info.nz_used, (int) info.nz_allocated);
      }
    }
  }
  if (B->ops->view) {
    ierr = PetscViewerASCIIPushTab(viewer);                                                                    CHKERRQ(ierr);
    ierr = (*B->ops->view)(B, viewer);                                                                    CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);                                                                     CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "BilinearView_Private"
/*
  Processes command line options to determine if/how a bilinear operator is to be viewed.
  Called by BilinearAssemblyEnd()
*/
int BilinearView_Private(Bilinear B)
{
  PetscTruth opt;
  int        ierr;

  PetscFunctionBegin;
  ierr = OptionsHasName(PETSC_NULL, "-bilinear_view_info", &opt);                                         CHKERRQ(ierr);
  if (opt == PETSC_TRUE) {
    ierr = PetscViewerPushFormat(VIEWER_STDOUT_(B->comm), PETSC_VIEWER_FORMAT_ASCII_INFO, 0);             CHKERRQ(ierr);
    ierr = BilinearView(B, VIEWER_STDOUT_(B->comm));                                                      CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(VIEWER_STDOUT_(B->comm));                                                 CHKERRQ(ierr);
  }
  ierr = OptionsHasName(PETSC_NULL, "-bilinear_view_info_detailed", &opt);                                CHKERRQ(ierr);
  if (opt == PETSC_TRUE) {
    ierr = PetscViewerPushFormat(VIEWER_STDOUT_(B->comm), PETSC_VIEWER_FORMAT_ASCII_INFO_LONG, 0);        CHKERRQ(ierr);
    ierr = BilinearView(B, VIEWER_STDOUT_(B->comm));                                                      CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(VIEWER_STDOUT_(B->comm));                                                 CHKERRQ(ierr);
  }
  ierr = OptionsHasName(PETSC_NULL, "-bilinear_view", &opt);                                              CHKERRQ(ierr);
  if (opt == PETSC_TRUE) {
    ierr = BilinearView(B, VIEWER_STDOUT_(B->comm));                                                      CHKERRQ(ierr);
  }
#if 0
  ierr = OptionsHasName(PETSC_NULL, "-bilinear_view_mathematica", &opt);                                  CHKERRQ(ierr);
  if (opt == PETSC_TRUE) {
    ierr = PetscViewerPushFormat(VIEWER_STDOUT_(B->comm), PETSC_VIEWER_FORMAT_ASCII_MATHEMATICA, "B");    CHKERRQ(ierr);
    ierr = BilinearView(B, VIEWER_STDOUT_(B->comm));                                                      CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(VIEWER_STDOUT_(B->comm));                                                 CHKERRQ(ierr);
  }
#endif
  ierr = OptionsHasName(PETSC_NULL, "-bilinear_view_draw", &opt);                                         CHKERRQ(ierr);
  if (opt == PETSC_TRUE) {
    ierr = OptionsHasName(PETSC_NULL, "-bilinear_view_contour", &opt);                                    CHKERRQ(ierr);
    if (opt == PETSC_TRUE) {
      PetscViewerPushFormat(VIEWER_DRAW_(B->comm), PETSC_VIEWER_FORMAT_DRAW_CONTOUR, 0);                  CHKERRQ(ierr);
    }
    ierr = BilinearView(B, VIEWER_DRAW_(B->comm));                                                        CHKERRQ(ierr);
    ierr = PetscViewerFlush(VIEWER_DRAW_(B->comm));                                                       CHKERRQ(ierr);
    if (opt == PETSC_TRUE) {
      PetscViewerPopFormat(VIEWER_DRAW_(B->comm));                                                             CHKERRQ(ierr);
    }
  }
  ierr = OptionsHasName(PETSC_NULL, "-bilinear_view_socket", &opt);                                       CHKERRQ(ierr);
  if (opt == PETSC_TRUE) {
    ierr = BilinearView(B, VIEWER_SOCKET_(B->comm));                                                      CHKERRQ(ierr);
    ierr = PetscViewerFlush(VIEWER_SOCKET_(B->comm));                                                          CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "BilinearDestroy"
/*@C
   BilinearDestroy - Frees space taken by a bilinear operator.
  
   Input Parameter:
.  B - the bilinear operator

.keywords: bilinear operator, destroy
@*/
int BilinearDestroy(Bilinear B)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, BILINEAR_COOKIE);
  if (--B->refct > 0) PetscFunctionReturn(0);
  ierr = (*B->ops->destroy)(B);                                                                           CHKERRQ(ierr);
  PetscLogObjectDestroy(B);
  PetscHeaderDestroy(B); 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "BilinearPrintHelp"
/*@
  BilinearPrintHelp - Prints all the options for the bilinear operator.

  Input Parameter:
. B - the bilinear operator 

  Options Database Keys:
$ -help, -h

.keywords: Bilinear, help
.seealso: BilinearCreate(), BilinearCreateXXX()
@*/
int BilinearPrintHelp(Bilinear B)
{
  static int called = 0;
  MPI_Comm   comm;
  int        ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, BILINEAR_COOKIE);

  comm = B->comm;
  if (!called) {
    (*PetscHelpPrintf)(comm,"General bilinear operator options:\n");
    (*PetscHelpPrintf)(comm,"  -bilinear_view_info: view basic bilinear info during BilinearAssemblyEnd()\n");
    (*PetscHelpPrintf)(comm,"  -bilinear_view_info_detailed: view detailed bilinear info during BilinearAssemblyEnd()\n");
    (*PetscHelpPrintf)(comm,"  -bilinear_view_draw: draw nonzero bilinear structure during BilinearAssemblyEnd()\n");
    (*PetscHelpPrintf)(comm,"      -draw_pause <sec>: set seconds of display pause\n");
    (*PetscHelpPrintf)(comm,"      -display <name>: set alternate display\n");
    called = 1;
  }
  if (B->ops->printhelp) {
    ierr = (*B->ops->printhelp)(B);                                                                       CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "BilinearValid"
/*@
  BilinearValid - Checks whether a bilinear operator object is valid.

  Input Parameter:
. B     - The bilinear operator to check 

  Output Parameter:
. valid - The flag indicating bilinear operator status, either
$     PETSC_TRUE if bilinear operator is valid;
$     PETSC_FALSE otherwise.

.keywords: bilinear operator, valid
@*/
int BilinearValid(Bilinear B, PetscTruth *valid)
{
  PetscFunctionBegin;
  PetscValidIntPointer(valid);
  if (!B)
    *valid = PETSC_FALSE;
  else if (B->cookie != BILINEAR_COOKIE)
    *valid = PETSC_FALSE;
  else
    *valid = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "BilinearCopy_Basic"
/*
      Default bilinear operator copy routine.
*/
int BilinearCopy_Basic(Bilinear A, Bilinear B)
{
  int     rowStart, rowEnd;
#if 0
  int    *subcols;
  int     numSubcols;
  PetscScalar *values;
#endif
  int     row;
  int     ierr;

  PetscFunctionBegin;
  ierr = BilinearZeroEntries(B);                                                                          CHKERRQ(ierr);
  ierr = BilinearGetOwnershipRange(A, &rowStart, &rowEnd);                                                CHKERRQ(ierr);
  for(row = rowStart; row < rowEnd; row++) {
#if 0
    ierr = BilinearGetRow(A, row, col, &numSubcols, &subcols, &values);                                   CHKERRQ(ierr);
    ierr = BilinearSetValues(B, 1, &row, 1, &col, numSubcols, subcols, values, INSERT_VALUES);            CHKERRQ(ierr);
    ierr = BilinearRestoreRow(A, row, col, &numSubcols, &subcols, &values);                               CHKERRQ(ierr);
#endif
  }
  ierr = BilinearAssemblyBegin(B, MAT_FINAL_ASSEMBLY);                                                    CHKERRQ(ierr);
  ierr = BilinearAssemblyEnd(B, MAT_FINAL_ASSEMBLY);                                                      CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "BilinearCopy"
/*@C  
   BilinearCopy - Copys a bilinear operator to another bilinear operator.

   Input Parameters:
.  A - The bilinear operator

   Output Parameter:
.  B - The copy

   Notes:
   BilinearCopy() copies the bilinear operator entries of a bilinear operator to another existing
   bilinear operator (after first zeroing the second bilinear operator).  A related routine is
   BilinearConvert(), which first creates a new bilinear operator and then copies the data.
   
.keywords: bilinear operator, copy, convert

.seealso: BilinearConvert()
@*/
int BilinearCopy(Bilinear A, Bilinear B)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, BILINEAR_COOKIE);
  PetscValidHeaderSpecific(B, BILINEAR_COOKIE);
  if (!A->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "Not for unassembled bilinear operator");
  if (A->factor)     SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "Not for factored bilinear operator"); 
  if (A->N_i != B->N_i || A->N_j != B->N_j || A->N_k != B->N_k) {
    SETERRQ6(PETSC_ERR_ARG_SIZ, "Bilinear global dim: (%d,%d,%d) vs (%d,%d,%d)", A->N_i, A->N_j, A->N_k, B->N_i, B->N_j, B->N_k);
  }

  PetscLogEventBegin(BILINEAR_Copy, A, B, 0, 0);
  if (A->ops->copy) { 
    ierr = (*A->ops->copy)(A, B);                                                                         CHKERRQ(ierr);
  }
  else { /* generic conversion */
    ierr = BilinearCopy_Basic(A, B);                                                                      CHKERRQ(ierr);
  }
  PetscLogEventEnd(BILINEAR_Copy, A, B, 0, 0);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "BilinearDuplicate"
/*@C  
  BilinearDuplicate - Duplicates a bilinear operator including the non-zero structure, but 
  does not copy over the values.

  Input Parameters:
. B - The bilinear operator

  Output Parameter:
. C - The new bilinear operator

.keywords: bilinear operator, copy, convert, duplicate

.seealso: BilinearCopy(), BilinearDuplicate(), BilinearConvert()
@*/
int BilinearDuplicate(Bilinear B, Bilinear *C)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, BILINEAR_COOKIE);
  PetscValidPointer(C);
  if (!B->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "Not for unassembled bilinear operator");
  if (B->factor)     SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "Not for factored bilinear operator"); 

  *C = PETSC_NULL;
  BilinearLogEventBegin(BILINEAR_Convert, B, 0, 0, 0);
  if (!B->ops->convertsametype) {
    SETERRQ(PETSC_ERR_SUP, "Not written for this bilinear operator type");
  }
  ierr = (*B->ops->convertsametype)(B, C, DO_NOT_COPY_VALUES);                                            CHKERRQ(ierr);
  BilinearLogEventEnd(BILINEAR_Convert, B, 0, 0, 0);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "BilinearSetOption"
/*@
   BilinearSetOption - Sets a parameter option for a bilinear operator. Some options
   may be specific to certain storage formats.  Some options determine how values will
   be inserted (or added). Sorted, row-oriented input will generally assemble the fastest.
   The default is row-oriented, nonsorted input. 

   Input Parameters:
.  B      - The bilinear operator 
.  option - The option, one of those listed below (and possibly others),
             e.g., BILINEAR_NEW_NONZERO_LOCATION_ERROR

   Options Describing Bilinear Structure:
$    BILINEAR_SYMMETRIC - symmetric in terms of both structure and value

   Options For Use with BilinearSetValues():
   When (re)assembling a bilinear operator, we can restrict the input for
   efficiency/debugging purposes.

$    BILINEAR_NO_NEW_NONZERO_LOCATIONS - additional insertions will not be
        allowed if they generate a new nonzero
$    BILINEAR_YES_NEW_NONZERO_LOCATIONS - additional insertions will be allowed
$    BILINEAR_IGNORE_OFF_PROC_ENTRIES - drop off-processor entries
$    BILINEAR_NEW_NONZERO_LOCATION_ERROR - generate error for new bilinear operator entry

   Notes:
   Some options are relevant only for particular bilinear operator types and
   are thus ignored by others.  Other options are not supported by
   certain bilinear operator types and will generate an error message if set.

   BILINEAR_NO_NEW_NONZERO_LOCATIONS indicates that any add or insertion 
   that would generate a new entry in the nonzero structure is instead
   ignored.  Thus, if memory has not alredy been allocated for this particular 
   data, then the insertion is ignored. For dense Brices, in which
   the entire array is allocated, no entries are ever ignored. 

   BILINEAR_NEW_NONZERO_LOCATION_ERROR indicates that any add or insertion 
   that would generate a new entry in the nonzero structure instead produces 
   an error. (Currently supported for AIJ and BAIJ forBs only.)
   This is a useful flag when using SAME_NONZERO_PATTERN in calling
   SLESSetOperators() to ensure that the nonzero pattern truely does 
   remain unchanged.

   BILINEAR_NEW_NONZERO_ALLOCATION_ERROR indicates that any add or insertion 
   that would generate a new entry that has not been preallocated will 
   instead produce an error. (Currently supported for AIJ and BAIJ forBs
   only.) This is a useful flag when debugging bilinear operator memory preallocation.

   BILINEAR_IGNORE_OFF_PROC_ENTRIES indicates entries destined for 
   other processors should be dropped, rather than stashed.
   This is useful if you know that the "owning" processor is also 
   always generating the correct bilinear operator entries, so that PETSc need
   not transfer duplicate entries generated on another processor.

.keywords: bilinear operator, option, row-oriented, column-oriented, sorted, nonzero
@*/
int BilinearSetOption(Bilinear B, BilinearOption op)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, BILINEAR_COOKIE);
  if (B->ops->setoption) {
    ierr = (*B->ops->setoption)(B, op);                                                                   CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "BilinearGetTypeFromOptions"
/*@C
  BilinearGetTypeFromOptions - Determines from the options database what
  bilinear operator format the user has specified.

  Input Parameter:
. comm - The MPI communicator
. type - The type of bilienar operator desired, for example BILINEAR_DENSE_SEQ
. pre  - [Optional] string to prepend to the name

  Output Parameters:
. set  - The flag indicating whether user set matrix type option.

  Basic Options Database Keys:
  These options return BILINEAR_xxx_SEQ or BILIENAR_xxx_MPI, depending on the communicator, comm.
$    -bilinear_dense    : dense type

   More Options Database Keys:
$    -bilinear_seqdense : BILINEAR_DENSE_SEQ
$    -bilinear_mpidense : BILINEAR_DENSE_MPI

  Note:
  This routine is automatically called within BilinearCreate().

.keywords: Bilinear, get, format, from, options
.seealso: BilinearCreate()
@*/
int BilinearGetTypeFromOptions(MPI_Comm comm, char *pre, BilinearType *type, PetscTruth *set)
{
  static int helpPrinted = 0;
  char       p[256];
  int        numProcs;
  PetscTruth opt;
  int        ierr;

  PetscFunctionBegin;
  PetscValidIntPointer(type);
  PetscValidIntPointer(set);

  ierr = PetscStrcpy(p, "-");                                                                             CHKERRQ(ierr);
  if (pre) {
    ierr = PetscStrcat(p, pre);                                                                           CHKERRQ(ierr);
  }

  ierr = MPI_Comm_size(comm, &numProcs);                                                                  CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL, "-help", &opt);                                                       CHKERRQ(ierr);
  if ((opt == PETSC_TRUE) && (!helpPrinted)) {
    (*PetscHelpPrintf)(comm,"Bilinear operator format options:\n");
    (*PetscHelpPrintf)(comm,"  %sbilinear_dense, %sbilinear_seqdense, %sbilinear_mpidense\n", p, p, p);
    helpPrinted = 1;
  }
  /* Default settings */
  *set = PETSC_FALSE;
  if (numProcs == 1) *type = BILINEAR_DENSE_SEQ;
  else               *type = BILINEAR_DENSE_MPI;
  /* Check options */
  ierr = OptionsHasName(pre, "-bilinear_seqdense", &opt);                                                 CHKERRQ(ierr);
  if (opt == PETSC_TRUE) {*type = BILINEAR_SEQDENSE; *set = PETSC_TRUE;}
  ierr = OptionsHasName(pre, "-bilinear_mpidense", &opt);                                                 CHKERRQ(ierr);
  if (opt == PETSC_TRUE) {*type = BILINEAR_MPIDENSE; *set = PETSC_TRUE;}
  ierr = OptionsHasName(pre, "-bilinear_dense",    &opt);                                                 CHKERRQ(ierr);
  if (opt == PETSC_TRUE) {
    if (numProcs == 1) *type = BILINEAR_SEQDENSE;
    else               *type = BILINEAR_MPIDENSE;
    *set = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

/*------------------------------------- Generation and Assembly Functions -------------------------------------------*/
/*MC
  BilinearSetValue - Set a single entry into a bilinear operator.

  Input Parameters:
. B      - The bilinear operator
. row    - The row location of the entry
. col    - The column location of the entry
. subcol - The subcolumn location of the entry
. value  - The value to insert
. mode   - The insertion mode, either INSERT_VALUES or ADD_VALUES

  Synopsis:
  void BilinearSetValue(Bilinear B, int row, int col, int subcol, PetscScalar value, InsertMode mode);

  Notes: For efficiency one should use BilinearSetValues() and set several or many values simultaneously.

.keywords: assembly, bilinear
.seealso: BilinearSetValues()
M*/

#undef __FUNC__  
#define __FUNC__ "BilinearSetValues"
/*@ 
  BilinearSetValues - Inserts or adds a block of values into a bilinear operator.
  These values may be cached, so BilinearAssemblyBegin() and BilinearAssemblyEnd() 
  MUST be called after all calls to BilinearSetValues() have been completed.

  Input Parameters:
. B       - The bilinear operator
. i, idxi - The number of rows and their global indices 
. j, idxj - The number of columns and their global indices
. k, idxk - The number of subcolumns and their global indices
. v       - The logically three-dimensional array of values
. addv    - The insertion mode, either ADD_VALUES or INSERT_VALUES, where
$     ADD_VALUES - adds values to any existing entries
$     INSERT_VALUES - replaces existing entries with new values

  Notes:
  By default the values, v, are row-oriented and unsorted.
  See BilinearSetOption() for other options.

  Calls to BilinearSetValues() with the INSERT_VALUES and ADD_VALUES 
  options cannot be mixed without intervening calls to the assembly
  routines.

  BilinearSetValues() uses 0-based row and column numbers in Fortran  as well as in C.

.keywords: bilinear operator, insert, add, set, values
.seealso: BilinearSetOption(), BilinearAssemblyBegin(), BilinearAssemblyEnd()
@*/
int BilinearSetValues(Bilinear B, int i, int *idxi, int j, int *idxj, int k, int *idxk, PetscScalar *v, InsertMode addv)
{
  int ierr;

  PetscFunctionBegin;
  if (!i || !j || !k) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(B, BILINEAR_COOKIE);
  PetscValidIntPointer(idxi);
  PetscValidIntPointer(idxj);
  PetscValidIntPointer(idxk);
  PetscValidScalarPointer(v);
  if (B->insertmode == NOT_SET_VALUES) {
    B->insertmode = addv;
  } else if (B->insertmode != addv) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "Cannot mix add values and insert values");
  }
  if (B->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "Not for factored bilinear operator"); 

  if (B->assembled) {
    B->was_assembled = PETSC_TRUE; 
    B->assembled     = PETSC_FALSE;
  }
  BilinearLogEventBegin(BILINEAR_SetValues, B, 0, 0, 0);
  ierr = (*B->ops->setvalues)(B, i, idxi, j, idxj, k, idxk, v, addv);                                     CHKERRQ(ierr);
  BilinearLogEventEnd(BILINEAR_SetValues, B, 0, 0, 0);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "BilinearGetArray"
/*@C
  BilinearGetArray - Returns a pointer to the element values in the bilinear operator.
  This routine  is implementation dependent, and may not even work for 
  certain bilinear operator types. You MUST call BilinearRestoreArray() when you no 
  longer need to access the array.

  Input Parameter:
. B - The bilinear operator

  Output Parameter:
. v - The location of the values

.keywords: bilinear operator, array, elements, values
.seealso: BilinearRestoreArray()
@*/
int BilinearGetArray(Bilinear B, PetscScalar **v)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, BILINEAR_COOKIE);
  PetscValidPointer(v);
  if (!B->ops->getarray) SETERRQ(PETSC_ERR_SUP, "");
  ierr = (*B->ops->getarray)(B, v);                                                                       CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "BilinearRestoreArray"
/*@C
  BilinearRestoreArray - Restores the bilinear operator after BilinearGetArray has been called.

  Input Parameter:
. B - The bilinear operator
. v - The location of the values

.keywords: bilinear operator, array, elements, values, restore
.seealso: BilinearGetArray()
@*/
int BilinearRestoreArray(Bilinear B, PetscScalar **v)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, BILINEAR_COOKIE);
  PetscValidPointer(v);
  if (!B->ops->restorearray) SETERRQ(PETSC_ERR_SUP, "");
  ierr = (*B->ops->restorearray)(B, v);                                                                   CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* 
   This variable is used to prevent counting of BilinearAssemblyBegin() that are called from within a BilinearAssemblyEnd().
*/
static int BilinearAssemblyEnd_InUse = 0;
#undef __FUNC__  
#define __FUNC__ "BilinearAssemblyBegin"
/*@
  BilinearAssemblyBegin - Begins assembling the bilinear operator. This routine should
  be called after completing all calls to BilinearSetValues().

  Input Parameters:
. B    - The bilinear operator 
. type - The type of assembly, either MAT_FLUSH_ASSEMBLY or MAT_FINAL_ASSEMBLY
 
  Notes: 
  BilinearSetValues() generally caches the values.  The bilinear operator is ready to
  use only after BilinearAssemblyBegin() and BilinearAssemblyEnd() have been called.
  Use MAT_FLUSH_ASSEMBLY when switching between ADD_VALUES and INSERT_VALUES
  in BilinearSetValues(); use MAT_FINAL_ASSEMBLY for the final assembly before
  using the bilinear operator.

.keywords: bilinear operator, assembly, assemble, begin
.seealso: BilinearAssemblyEnd(), BilinearSetValues()
@*/
int BilinearAssemblyBegin(Bilinear B, MatAssemblyType type)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, BILINEAR_COOKIE);
  if (B->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "Not for factored bilinear operator"); 
  if (B->assembled) {
    B->was_assembled = PETSC_TRUE; 
    B->assembled     = PETSC_FALSE;
  }
  if (!BilinearAssemblyEnd_InUse) {
    BilinearLogEventBegin(BILINEAR_AssemblyBegin, B, 0, 0, 0);
    if (B->ops->assemblybegin){
      ierr = (*B->ops->assemblybegin)(B, type);                                                           CHKERRQ(ierr);
    }
    BilinearLogEventEnd(BILINEAR_AssemblyBegin, B, 0, 0, 0);
  } else {
    if (B->ops->assemblybegin) {
      ierr = (*B->ops->assemblybegin)(B, type);                                                           CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "BilinearAssemblyEnd"
/*@
  BilinearAssemblyEnd - Completes assembling the bilinear operator.  This routine should
  be called after BilinearAssemblyBegin().

  Input Parameters:
. B    - The bilinear operator 
. type - The type of assembly, either MAT_FLUSH_ASSEMBLY or MAT_FINAL_ASSEMBLY

  Options Database Keys:
$  -bilinear_view_info : Prints info on bilinear operator at
$      conclusion of BilinearEndAssembly()
$  -bilinear_view_info_detailed: Prints more detailed info.
$  -bilinear_view : Prints bilinear operator in ASCII forB.
$  -bilinear_view_Blab : Prints bilinear operator in Bilinearlab forB.
$  -bilinear_view_draw : Draws nonzero structure of bilinear operator,
$      using BilinearView() and DrawOpenX().
$  -display <name> : Set display name (default is host)
$  -draw_pause <sec> : Set number of seconds to pause after display

  Notes: 
  BilinearSetValues() generally caches the values.  The bilinear operator is ready to
  use only after BilinearAssemblyBegin() and BilinearAssemblyEnd() have been called.
  Use MAT_FLUSH_ASSEMBLY when switching between ADD_VALUES and INSERT_VALUES
  in BilinearSetValues(); use MAT_FINAL_ASSEMBLY for the final assembly before
  using the bilinear operator.

.keywords: bilinear operator, assembly, assemble, end
.seealso: BilinearAssemblyBegin(), BilinearSetValues(), DrawOpenX(), BilinearView()
@*/
int BilinearAssemblyEnd(Bilinear B, MatAssemblyType type)
{
  static int inassm = 0;
  int        ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, BILINEAR_COOKIE);

  inassm++;
  BilinearAssemblyEnd_InUse++;
  BilinearLogEventBegin(BILINEAR_AssemblyEnd, B, 0, 0, 0);
  if (B->ops->assemblyend) {
    ierr = (*B->ops->assemblyend)(B, type);                                                               CHKERRQ(ierr);
  }

  /* Flush assembly is not a true assembly */
  if (type != MAT_FLUSH_ASSEMBLY) {
    B->assembled = PETSC_TRUE;
    B->num_ass++;
  }
  B->insertmode = NOT_SET_VALUES;
  BilinearLogEventEnd(BILINEAR_AssemblyEnd, B, 0, 0, 0);
  BilinearAssemblyEnd_InUse--;

  if (inassm == 1 && type != MAT_FLUSH_ASSEMBLY) {
    ierr = BilinearView_Private(B);                                                                       CHKERRQ(ierr);
  }
  inassm--;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "BilinearZeroEntries"
/*@
  BilinearZeroEntries - Zeros all entries of a bilinear operator.  For sparse bilinear
  operators this routine retains the old nonzero structure.

  Input Parameters:
. B - The bilinear operator 

.keywords: bilinear operator, zero, entries

.seealso: BilinearZeroRows()
@*/
int BilinearZeroEntries(Bilinear B)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, BILINEAR_COOKIE);
  if (B->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "Not for factored bilinear operator"); 
  if (!B->ops->zeroentries) SETERRQ(PETSC_ERR_SUP, "");

  BilinearLogEventBegin(BILINEAR_ZeroEntries, B, 0, 0, 0);
  ierr = (*B->ops->zeroentries)(B);                                                                       CHKERRQ(ierr);
  BilinearLogEventEnd(BILINEAR_ZeroEntries, B, 0, 0, 0);
  PetscFunctionReturn(0);
}

/*-------------------------------------------- Application Functions ------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "BilinearMult"
/*@
  BilinearMult - Computes the bilinear operator-vector product, M = Bx.

  Input Parameters:
. B - The bilinear operator
. x - The vector to be multilplied

  Output Parameter:
. M - The resulting matrix

.keywords: bilinear operator, multiply, bilinear operator-vector product
.seealso: BilinearFullMult(), BilinearDiamond()
@*/
int BilinearMult(Bilinear B, Vec x, Mat M)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, BILINEAR_COOKIE);
  PetscValidHeaderSpecific(x, VEC_COOKIE);
  PetscValidHeaderSpecific(M, MAT_COOKIE);
  if (!B->assembled)  SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "Not for unassembled bilinear operator");
  if (B->factor)      SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "Not for factored bilinear operator");
  if (B->N_k != x->N) SETERRQ2(PETSC_ERR_ARG_SIZ, "Bilinear B, Vec x: global dim: %d vs %d", B->N_k, x->N);
  if (B->N_i != M->M) SETERRQ2(PETSC_ERR_ARG_SIZ, "Bilinear B, Mat M: global dim: %d vs %d", B->N_i, M->M);
  if (B->N_j != M->N) SETERRQ2(PETSC_ERR_ARG_SIZ, "Bilinear B, Mat M: global dim: %d vs %d", B->N_j, M->N);
  if (B->n_k != x->n) SETERRQ2(PETSC_ERR_ARG_SIZ, "Bilinear B, Vec x: global dim: %d vs %d", B->n_k, x->n);
  if (B->n_i != M->m) SETERRQ2(PETSC_ERR_ARG_SIZ, "Bilinear B, Mat M: local dim: %d vs %d", B->n_i, M->m);
  if (B->n_j != M->n) SETERRQ2(PETSC_ERR_ARG_SIZ, "Bilinear B, Mat M: local dim: %d vs %d", B->n_j, M->n);

  BilinearLogEventBegin(BILINEAR_Mult, B, x, M, 0);
  ierr = (*B->ops->mult)(B, x, M);                                                                        CHKERRQ(ierr);
  BilinearLogEventEnd(BILINEAR_Mult, B, x, M, 0);

  PetscFunctionReturn(0);
}   

#undef __FUNC__  
#define __FUNC__ "BilinearFullMult"
/*@
  BilinearFullMult - Computes the bilinear operator-vector product, z = Bxy.

  Input Parameters:
. B   - The bilinear operator
. x,y - The vectors to be multilplied

  Output Parameter:
. z   - The resulting vector

.keywords: bilinear operator, multiply, bilinear operator-vector product
.seealso: BilinearMult(), BilinearDiamond()
@*/
int BilinearFullMult(Bilinear B, Vec x, Vec y, Vec z)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, BILINEAR_COOKIE);
  PetscValidHeaderSpecific(x, VEC_COOKIE);
  PetscValidHeaderSpecific(y, VEC_COOKIE);
  PetscValidHeaderSpecific(z, VEC_COOKIE);
  if (!B->assembled)  SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "Not for unassembled bilinear operator");
  if (B->factor)      SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "Not for factored bilinear operator");
  if (B->N_j != x->N) SETERRQ2(PETSC_ERR_ARG_SIZ, "Bilinear B, Vec x: global dim: %d vs %d", B->N_j, x->N);
  if (B->N_k != y->N) SETERRQ2(PETSC_ERR_ARG_SIZ, "Bilinear B, Vec y: global dim: %d vs %d", B->N_k, y->N);
  if (B->N_i != z->N) SETERRQ2(PETSC_ERR_ARG_SIZ, "Bilinear B, Vec z: global dim: %d vs %d", B->N_i, z->N);
  if (B->n_j != x->n) SETERRQ2(PETSC_ERR_ARG_SIZ, "Bilinear B, Vec x: local dim: %d vs %d", B->n_j, x->n);
  if (B->n_k != y->n) SETERRQ2(PETSC_ERR_ARG_SIZ, "Bilinear B, Vec y: local dim: %d vs %d", B->n_k, y->n);
  if (B->n_i != z->n) SETERRQ2(PETSC_ERR_ARG_SIZ, "Bilinear B, Vec z: local dim: %d vs %d", B->n_i, z->n);

  BilinearLogEventBegin(BILINEAR_FullMult, B, x, y, z);
  ierr = (*B->ops->fullmult)(B, x, y, z);                                                                 CHKERRQ(ierr);
  BilinearLogEventEnd(BILINEAR_FullMult, B, x, y, z);

  PetscFunctionReturn(0);
}   

#undef __FUNC__  
#define __FUNC__ "BilinearDiamond"
/*@
  BilinearDiamond - Computes the bilinear operator-vector product, y = Bxx = B \diamond x.

  Input Parameters:
. B - The bilinear operator
. x - The vector to be multilplied

  Output Parameter:
. y - The resulting vector

.keywords: bilinear operator, multiply, bilinear operator-vector product
.seealso: BilinearMult(), BilinearFullMult()
@*/
int BilinearDiamond(Bilinear B, Vec x, Vec y)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, BILINEAR_COOKIE);
  PetscValidHeaderSpecific(x, VEC_COOKIE);
  PetscValidHeaderSpecific(y, VEC_COOKIE);
  if (!B->assembled)  SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "Not for unassembled bilinear operator");
  if (B->factor)      SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "Not for factored bilinear operator");
  if (B->N_j != x->N) SETERRQ2(PETSC_ERR_ARG_SIZ, "Bilinear B, Vec x: global dim: %d vs %d", B->N_j, x->N);
  if (B->N_k != x->N) SETERRQ2(PETSC_ERR_ARG_SIZ, "Bilinear B, Vec x: global dim: %d vs %d", B->N_k, x->N);
  if (B->N_i != y->N) SETERRQ2(PETSC_ERR_ARG_SIZ, "Bilinear B, Vec y: global dim: %d vs %d", B->N_i, y->N);
  if (B->n_j != x->n) SETERRQ2(PETSC_ERR_ARG_SIZ, "Bilinear B, Vec x: local dim: %d vs %d", B->n_j, x->n);
  if (B->n_k != x->n) SETERRQ2(PETSC_ERR_ARG_SIZ, "Bilinear B, Vec x: local dim: %d vs %d", B->n_k, x->n);
  if (B->n_i != y->n) SETERRQ2(PETSC_ERR_ARG_SIZ, "Bilinear B, Vec y: local dim: %d vs %d", B->n_i, y->n);

  BilinearLogEventBegin(BILINEAR_Diamond, B, x, y, 0);
  ierr = (*B->ops->diamond)(B, x, y);                                                                     CHKERRQ(ierr);
  BilinearLogEventEnd(BILINEAR_Diamond, B, x, y, 0);

  PetscFunctionReturn(0);
}

/*------------------------------------------- Interrogation Functions -----------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "BilinearGetInfo"
/*@C
  BilinearGetInfo - Returns inforBion about bilinear operator storage (number of nonzeros, memory, etc.).

  Input Parameters:
. B    - The bilinear operator

  Output Parameters:
. type - The type of parameters to be returned
$    flag = INFO_LOCAL:      local bilinear operator
$    flag = INFO_GLOBAL_MAX: maximum over all processors
$    flag = INFO_GLOBAL_SUM: sum over all processors
. info - The bilinear operator information context

  Notes:
  The BilinearInfo context contains a variety of bilinear operator data, including
  number of nonzeros allocated and used, number of mallocs during bilinear operator
  assembly, etc.  Additional information for factored operators is provided (such as
  the fill ratio, number of mallocs during  factorization, etc.).  Much of this info
  is printed to STDOUT when using the runtime options 
$   -log_info -bilinear_view_info

  Example for C/C++ Users:
  See the file $(PETSC_DIR)/include/B.h for a complete list of data within the
  BilinearInfo context. For example, 
$
$      BilinearInfo info;
$      Bilinear     A;
$      double  mal, nz_a, nz_u;
$
$      BilinearGetInfo(A, INFO_LOCAL, &info);
$      mal  = info.mallocs;
$      nz_a = info.nz_allocated;
$

.keywords: bilinear operator, get, info, storage, nonzeros, memory, fill
@*/
int BilinearGetInfo(Bilinear B, InfoType type, BilinearInfo *info)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, BILINEAR_COOKIE);
  PetscValidPointer(info);
  if (!B->ops->getinfo) SETERRQ(PETSC_ERR_SUP, "");
  ierr = (*B->ops->getinfo)(B, type, info);                                                               CHKERRQ(ierr);
  PetscFunctionReturn(0);
}   
       
#undef __FUNC__  
#define __FUNC__ "BilinearHasOperation"
/*@
  BilinearHasOperation - Determines if the given matrix supports the particular operation.

  Input Parameters:
. mat - The matrix
. op  - The operation, for example, BILINEAROP_MULT

  Output Parameter:
. has - The flag indicating the presence of a function

  Notes:
  See the file petsc/include/bilinear.h for a complete list of bilinear
  operations, which all have the form BILINEAROP_<OPERATION>, where
  <OPERATION> is the name (in all capital letters) of the  user-level
  routine,  e.g., BilinearNorm() -> BILINEAROP_NORM.

.keywords: bilinear, has, operation
@*/
int BilinearHasOperation(Bilinear B, BilinearOperation op, PetscTruth *has)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, BILINEAR_COOKIE);
  if (((void **) B->ops)[op])
    *has = PETSC_TRUE;
  else
    *has = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "BilinearGetSize"
/*@
  BilinearGetSize - Returns the numbers of rows and columns in a bilinear operator.

  Input Parameter:
. B   - The bilinear operator

  Output Parameters:
. N_i - The number of global rows
. N_j - The number of global columns
. N_k - The number of global subcolumns

.keywords: bilinear operator, dimension, size, rows, columns, subcolumns, global, get
.seealso: BilinearGetLocalSize()
@*/
int BilinearGetSize(Bilinear B, int *N_i, int *N_j, int *N_k)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, BILINEAR_COOKIE);
  PetscValidIntPointer(N_i);
  PetscValidIntPointer(N_j);
  PetscValidIntPointer(N_k);
  ierr = (*B->ops->getsize)(B, N_i, N_j, N_k);                                                            CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "BilinearGetLocalSize"
/*@
  BilinearGetLocalSize - Returns the number of rows, columns, and subcolumns
  in a bilinear operator stored locally.  This information may be implementation
  dependent, so use with care.

  Input Parameters:
. B   - The bilinear operator

  Output Parameters:
. n_i - The number of local rows
. n_j - The number of local columns
. n_k - The number of local subcolumns

.keywords: bilinear operator, dimension, size, local, rows, columns, subcolumns, get
.seealso: BilinearGetSize()
@*/
int BilinearGetLocalSize(Bilinear B, int *n_i, int *n_j, int *n_k)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, BILINEAR_COOKIE);
  PetscValidIntPointer(n_i);
  PetscValidIntPointer(n_j);
  PetscValidIntPointer(n_k);
  ierr = (*B->ops->getlocalsize)(B, n_i, n_j, n_k);                                                       CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "BilinearGetOwnershipRange"
/*@
  BilinearGetOwnershipRange - Returns the range of bilinear operator rows owned by
  this processor, assuming that the bilinear operator is laid out with the first
  n1 rows on the first processor, the next n2 rows on the second, etc.
  For certain parallel layouts this range may not be well defined.

  Input Parameters:
. B        - The bilinear operator

  Output Parameters:
. rowStart - The global index of the first local row
. rowEnd   - The global index of the last local row + 1

.keywords: bilinear operator, get, range, ownership
@*/
int BilinearGetOwnershipRange(Bilinear B, int *rowStart, int *rowEnd)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, BILINEAR_COOKIE);
  PetscValidIntPointer(rowStart);
  PetscValidIntPointer(rowEnd);
  if (!B->ops->getownershiprange) SETERRQ(PETSC_ERR_SUP, "");
  ierr = (*B->ops->getownershiprange)(B, rowStart, rowEnd);                                               CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "BilinearEqual"
/*@
  BilinearEqual - Compares two bilinear operators.

  Input Parameters:
. A  - The first bilinear operator
. B  - The second bilinear operator

  Output Parameter:
. eq - The result, PETSC_TRUE if the operators are equal, PETSC_FALSE otherwise.

.keywords: bilinear operator, equal, equivalent
@*/
int BilinearEqual(Bilinear A, Bilinear B, PetscTruth *eq)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, BILINEAR_COOKIE);
  PetscValidHeaderSpecific(B, BILINEAR_COOKIE);
  PetscValidIntPointer(eq);
  if (!A->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "Not for unassembled bilinear operator");
  if (!B->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "Not for unassembled bilinear operator");
  if (A->N_i != B->N_i || A->N_j != B->N_j || A->N_k != B->N_k) {
    SETERRQ6(PETSC_ERR_ARG_SIZ, "Bilinear global dim: (%d,%d,%d) vs (%d,%d,%d)", A->N_i, A->N_j, A->N_k, B->N_i, B->N_j, B->N_k);
  }
  if (!A->ops->equal) SETERRQ(PETSC_ERR_SUP, "");
  ierr = (*A->ops->equal)(A, B, eq);                                                                      CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "BilinearNorm"
/*@ 
  BilinearNorm - Calculates various norms of a bilinear operator.

  Input Parameters:
. B    - The bilinear operator
. type - The type of norm, NORM_1, NORM_2, NORM_FROBENIUS, NORM_INFINITY

  Output Parameters:
. norm - The resulting norm 

.keywords: bilinear operator, norm, Frobenius
@*/
int BilinearNorm(Bilinear B, NormType type, double *norm)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, BILINEAR_COOKIE);
  PetscValidScalarPointer(norm);

  if (!B->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "Not for unassembled bilinear operator");
  if (B->factor)     SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "Not for factored bilinear operator"); 
  if (!B->ops->norm) SETERRQ(PETSC_ERR_SUP, "Not for this bilinear operator type");
  ierr = (*B->ops->norm)(B, type, norm);                                                                  CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------- Factorization Functions -----------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "BilinearLUFactor"
/*@  
  BilinearLUFactor - Performs in-place LU factorization of bilinear operator.

  Input Parameters:
. B      - The bilinear operator
. row    - [Optional] The row permutation
. col    - [Optional] The column permutation
. subcol - [Optional] The subcolumn permutation
. fill   - The expected fill as ratio of original fill.

.keywords: bilinear operator, factor, LU, in-place
.seealso: BilinearCholeskyFactor()
@*/
int BilinearLUFactor(Bilinear B, IS row, IS col, IS subcol, double f)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, BILINEAR_COOKIE);
  if ((B->N_i != B->N_j) || (B->N_j != B->N_k)) SETERRQ(PETSC_ERR_ARG_WRONG, "Bilinear operator must be square");
  if (!B->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "Not for unassembled bilinear operator");
  if (B->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "Not for factored bilinear operator"); 
  if (!B->ops->lufactor) SETERRQ(PETSC_ERR_SUP, "");

  BilinearLogEventBegin(BILINEAR_LUFactor, B, row, col, subcol);
  ierr = (*B->ops->lufactor)(B, row, col, subcol, f);                                                     CHKERRQ(ierr);
  BilinearLogEventEnd(BILINEAR_LUFactor, B, row, col, subcol);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "BilinearCholeskyFactor"
/*@  
  BilinearCholeskyFactor - Performs in-place Cholesky factorization of a symmetric bilinear operator. 

  Input Parameters:
. B    - The bilinear operator
. perm - The row, column, and subcolumn permutations
. f    - The expected fill as ratio of original fill

  Notes:
  See BilinearLUFactor() for the nonsymmetric case.

.keywords: bilinear operator, factor, in-place, Cholesky
.seealso: BilinearLUFactor()
@*/
int BilinearCholeskyFactor(Bilinear B, IS perm, double f)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, BILINEAR_COOKIE);
  if ((B->N_i != B->N_j) || (B->N_j != B->N_k)) SETERRQ(PETSC_ERR_ARG_WRONG, "Bilinear operator must be square");
  if (!B->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "Not for unassembled bilinear operator");
  if (B->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "Not for factored bilinear operator"); 
  if (!B->ops->choleskyfactor) SETERRQ(PETSC_ERR_SUP, "");

  BilinearLogEventBegin(BILINEAR_CholeskyFactor, B, perm, 0, 0);
  ierr = (*B->ops->choleskyfactor)(B, perm, f);                                                           CHKERRQ(ierr);
  BilinearLogEventEnd(BILINEAR_CholeskyFactor, B, perm, 0, 0);
  PetscFunctionReturn(0);
}

/*MC
  BilinearSerializeRegister - Adds a serialization method to the vector package.

  Synopsis:

  BilinearSerializeRegister(char *serialize_name, char *path, char *serialize_func_name,
                            int (*serialize_func)(MPI_Comm, Bilinear *, PetscViewer, PetscTruth))

  Not Collective

  Input Parameters:
+ serialize_name      - The name of a new user-defined serialization routine
. path                - The path (either absolute or relative) of the library containing this routine
. serialize_func_name - The name of routine to create method context
- serialize_func      - The serialization routine itself

   Notes:
   BilinearSerializeRegister() may be called multiple times to add several user-defined solvers.

   If dynamic libraries are used, then the fourth input argument (serialize_func) is ignored.

   Sample usage:
.vb
   BilinearSerializeRegister("my_store", /home/username/my_lib/lib/libO/solaris/mylib.a, "MyStoreFunc", MyStoreFunc);
.ve

   Then, your serialization can be chosen with the procedural interface via
$     BilinearSetSerializeType(B, "my_store")
   or at runtime via the option
$     -bilinear_serialize_type my_store

   Level: advanced

   $PETSC_ARCH and $BOPT occuring in pathname will be replaced with appropriate values.

.keywords: Bilinear, register

.seealso: BilinearSerializeRegisterAll(), BilinearSerializeRegisterDestroy()
M*/
#undef __FUNC__  
#define __FUNC__ "BilinearSerializeRegister_Private"
int BilinearSerializeRegister_Private(const char *sname,const char *path,const char *name,int (*function)(MPI_Comm, Bilinear *, PetscViewer, PetscTruth))
{
  char fullname[256];
  int  ierr;

  PetscFunctionBegin;
  ierr = PetscStrcpy(fullname, path);                                                                     CHKERRQ(ierr);
  ierr = PetscStrcat(fullname, ":");                                                                      CHKERRQ(ierr);
  ierr = PetscStrcat(fullname, name);                                                                     CHKERRQ(ierr);
  ierr = FListAdd_Private(&BilinearSerializeList, sname, fullname, (int (*)(void*)) function);            CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
