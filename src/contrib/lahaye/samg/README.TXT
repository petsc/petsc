*******************************************************************************
The Level 1 SAMG-PETSc interface has a call tree like: 

int SamgShellPCCreate(SamgShellPC **shell)
int SamgShellPCSetUp(SamgShellPC *shell, Mat pmat)
    int SamgGetParam(SAMG_PARAM *samg_param)
    drvsamg_(...)
int SamgShellPCApply(void *ctx, Vec r, Vec z)
    drvsamg_(...)
int SamgShellPCDestroy(SamgShellPC *shell)

Remarks: 
** Unlike in the RAMG-PETSc interface, the SAMG is not called by immediately 
by PETSc routines. The PETSc routines call drvsamg, and drvsamg on its turn
calls SAMG. The reason for this intermediate step is the required use of 
the Fortran90 "use" statement in drvsamg. 

*******************************************************************************
Given the PETSc data structures

/*....Maximum number of levels to be used in SAMG....*/ 
#define MAX_LEVELS 25 

and 

typedef struct{
  /*..Implementation notes
    1 - The menber A is not stored on level 1 (the finest level in SAMG 
        ordering) to avoid unnecessary memory useage. 
  */ 
  SLES sles_pre;  
  SLES sles_post;
  Mat  A, B, C;
  Mat  Interp; 
  Vec  x, b, upd_b, r, y, b_y, r_y; 
  int  size; /*..Number of variables on level..*/ 
  //  int  debug;
} GridCtx; 

the Level 2 SAMG-PETSc interface has a call tree like:

int SamgShellPCCreate(SamgShellPC **shell)
int SamgShellPCSetUp(SamgShellPC *shell, Mat pmat)
    int SamgGetParam(SAMG_PARAM *samg_param)
    drvsamg_(...)
int SamgGetGrid(int levels, int numnodes, int numnonzero, 
                       GridCtx* grid, void* ctx); 
[...PETSc multigrid routines...]
int SamgShellPCDestroy(SamgShellPC *shell)

SamgGetGrid parses the SAMG hierarchy to the PETSc data structure of type 
GridCtx. It can broken up into following steps: 

int SamgGetGrid(int levels, int numnodes, int numnonzero, 
                       GridCtx* grid, void* ctx); 
   int SamgGetCoarseMat(int level, int ia_shift, int ja_shift,
                        Mat* coarsemat, void* ctx);
      samggetdimmat_(...)
      samggetmat_(...)     
   int SamgGetInterpolation(int level, int iw_shift, int jw_shift,
                            Mat* interpolation, void* ctx) ;
      samggetdimint_(...)
      samggetdimmat_(...)
      samggetint_(...)

Remarks: 
** The SamgGetCoarseMat and SamgGetInterpolation fetch the coarser grid 
matrices and the interpolation operators from the SAMG common blocks. This 
fetching is performed in two steps. 
1) In the first steps the dimension of the coarse grid matrix and 
   interpolation operator are looked in the common block (calls to 
   samggetdimmat and samggetdimint). The reason for calling a getdim routine 
   twice for the interpolation is that the interpolation is a rectangular 
   matrix, and the number of rows and collumns are looked up seperately. 
2) Given the dimension of the matrices, PETSc allocates memory from them. 
   Having allocated the necessary memory, the matrix valued are fetched. 
  
** CAUTION: THERE IS A VERY TRICKY ROUTINE, CALLED appy_shift HIDDEN 
IN SamgGetCoarseMat AND SamgGetInterpolation. The reason for it being there is 
the following. The coarse grid matrices and the interpolation operators on the 
consecutive levels in SAMG are stored continguously in a long work array (like
in the good old Fortran77 days). (SAMG inheritted this structure from RAMG). 
The values of the ia and ja pointers in SAMG are thus pointers in this large 
work array. In PETSc we would the ia and ja pointers to be pointers in 
seperate data structures for each level. Only then we can convert the tupple 
(a,ia,ja) into a PETSc matrix. The shifting of pointers to global work array 
to pointers in local data structures is done in the appy_shift routine. 

** The routine SamgCheckGalerkin was written for debugging purposes only. It 
checks whether the SAMG hierarchy is parsed OK to the PETSc data structures 
by checking whether A^H = I_h^H A^h I_H^h. The functions used in this routine 
are defined in PetscTools.C. 
*******************************************************************************
