#define PETSC_DLL

#include "private/viewerimpl.h"
#include "mat.h"

/*MC
   PETSC_VIEWER_MATLAB - A viewer that saves the variables into a Matlab .mat file that may be read into Matlab
       with load('filename').

   Level: intermediate

       Note: Currently can only save PETSc vectors to .mat files, not matrices (use the PETSC_VIEWER_BINARY and 
             ${PETSC_DIR}/bin/matlab/PetscBinaryRead.m to read matrices into matlab).

             For parallel vectors obtained with DACreateGlobalVector() or DAGetGlobalVector() the vectors are saved to
             the .mat file in natural ordering. You can use DAView() to save the DA information to the .mat file
             the fields in the Matlab loaded da variable give the array dimensions so you can reshape the Matlab
             vector to the same multidimensional shape as it had in PETSc for plotting etc. For example,

$             In your PETSc C/C++ code (assuming a two dimensional DA with one degree of freedom per node)
$                PetscObjectSetName((PetscObject)x,"x");
$                VecView(x,PETSC_VIEWER_MATLAB_WORLD);
$                PetscObjectSetName((PetscObject)da,"da");
$                DAView(x,PETSC_VIEWER_MATLAB_WORLD);
$             Then from Matlab
$                load('matlaboutput.mat')   % matlaboutput.mat is the default filename
$                xnew = zeros(da.n,da.m);
$                xnew(:) = x;    % reshape one dimensional vector back to two dimensions

              If you wish to put the same variable into the .mat file several times you need to give it a new
              name before each call to view.

              Use PetscViewerMatlabPutArray() to just put an array of doubles into the .mat file

.seealso:  PETSC_VIEWER_MATLAB_(),PETSC_VIEWER_MATLAB_SELF(), PETSC_VIEWER_MATLAB_WORLD(),PetscViewerCreate(),
           PetscViewerMatlabOpen(), VecView(), DAView(), PetscViewerMatlabPutArray(), PETSC_VIEWER_BINARY,
           PETSC_ASCII_VIEWER, DAView(), PetscViewerFileSetName(), PetscViewerFileSetMode()

M*/

typedef struct {
  MATFile       *ep;
  PetscMPIInt   rank;
  PetscFileMode btype;
} PetscViewer_Matlab;

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerMatlabPutArray"
/*@C
    PetscViewerMatlabPutArray - Puts an array into the Matlab viewer.

      Not collective: only processor zero saves the array

    Input Parameters:
+    mfile - the viewer
.    m,n - the dimensions of the array
.    array - the array (represented in one dimension)
-    name - the name of the array

   Level: advanced

     Notes: Only writes array values on processor 0.

@*/
PetscErrorCode PETSC_DLLEXPORT PetscViewerMatlabPutArray(PetscViewer mfile,int m,int n,PetscScalar *array,char *name)
{
  PetscErrorCode     ierr;
  PetscViewer_Matlab *ml = (PetscViewer_Matlab*)mfile->data; 
  mxArray            *mat;
  
  PetscFunctionBegin;  
  if (!ml->rank) {
    ierr = PetscInfo1(mfile,"Putting Matlab array %s\n",name);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
    mat  = mxCreateDoubleMatrix(m,n,mxREAL);
#else
    mat  = mxCreateDoubleMatrix(m,n,mxCOMPLEX);
#endif
    ierr = PetscMemcpy(mxGetPr(mat),array,m*n*sizeof(PetscScalar));CHKERRQ(ierr);
    matPutVariable(ml->ep,name,mat);

    ierr = PetscInfo1(mfile,"Put Matlab array %s\n",name);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerMatlabPutVariable"
PetscErrorCode PETSC_DLLEXPORT PetscViewerMatlabPutVariable(PetscViewer viewer,const char* name,void* mat)
{
  PetscViewer_Matlab *ml = (PetscViewer_Matlab*)viewer->data; ;

  PetscFunctionBegin;
  matPutVariable(ml->ep,name,(mxArray*)mat);
  PetscFunctionReturn(0);
}
  
#undef __FUNCT__  
#define __FUNCT__ "PetscViewerMatlabGetArray"
/*@C
    PetscViewerMatlabGetArray - Gets a variable from a Matlab viewer into an array

    Not Collective; only processor zero reads in the array

    Input Parameters:
+    mfile - the Matlab file viewer
.    m,n - the dimensions of the array
.    array - the array (represented in one dimension)
-    name - the name of the array

   Level: advanced

     Notes: Only reads in array values on processor 0.

@*/
PetscErrorCode PETSC_DLLEXPORT PetscViewerMatlabGetArray(PetscViewer mfile,int m,int n,PetscScalar *array,char *name)
{
  PetscErrorCode     ierr;
  PetscViewer_Matlab *ml = (PetscViewer_Matlab*)mfile->data; 
  mxArray            *mat;
  
  PetscFunctionBegin;  
  if (!ml->rank) {
    ierr = PetscInfo1(mfile,"Getting Matlab array %s\n",name);CHKERRQ(ierr);
    mat  = matGetVariable(ml->ep,name);
    if (!mat) SETERRQ1(PETSC_ERR_LIB,"Unable to get array %s from matlab",name);
    ierr = PetscMemcpy(array,mxGetPr(mat),m*n*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = PetscInfo1(mfile,"Got Matlab array %s\n",name);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PetscViewerFileSetMode_Matlab" 
PetscErrorCode PETSC_DLLEXPORT PetscViewerFileSetMode_Matlab(PetscViewer viewer,PetscFileMode type)
{
  PetscViewer_Matlab *vmatlab = (PetscViewer_Matlab*)viewer->data;

  PetscFunctionBegin;
  vmatlab->btype = type;
  PetscFunctionReturn(0);
}
EXTERN_C_END

/*
        Actually opens the file 
*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PetscViewerFileSetName_Matlab" 
PetscErrorCode PETSC_DLLEXPORT PetscViewerFileSetName_Matlab(PetscViewer viewer,const char name[])
{
  PetscViewer_Matlab  *vmatlab = (PetscViewer_Matlab*)viewer->data;
  PetscFileMode       type = vmatlab->btype;

  PetscFunctionBegin;
  if (type == (PetscFileMode) -1) {
    SETERRQ(PETSC_ERR_ORDER,"Must call PetscViewerFileSetMode() before PetscViewerFileSetName()");
  }

  /* only first processor opens file */
  if (!vmatlab->rank){
    if (type == FILE_MODE_READ){
      vmatlab->ep = matOpen(name,"r");
    } else if (type == FILE_MODE_WRITE || type == FILE_MODE_WRITE) {
      vmatlab->ep = matOpen(name,"w");
    } else {
      SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Unknown file type");
    }
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerDestroy_Matlab"
PetscErrorCode PetscViewerDestroy_Matlab(PetscViewer v)
{
  PetscErrorCode     ierr;
  PetscViewer_Matlab *vf = (PetscViewer_Matlab*)v->data; 

  PetscFunctionBegin;
  if (vf->ep) matClose(vf->ep);
  ierr = PetscFree(vf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PetscViewerCreate_Matlab"
PetscErrorCode PETSC_DLLEXPORT PetscViewerCreate_Matlab(PetscViewer viewer)
{
  PetscErrorCode     ierr;
  PetscViewer_Matlab *e;

  PetscFunctionBegin;
  ierr         = PetscNewLog(viewer,PetscViewer_Matlab,&e);CHKERRQ(ierr);
  ierr         = MPI_Comm_rank(((PetscObject)viewer)->comm,&e->rank);CHKERRQ(ierr);
  e->btype     = (PetscFileMode)-1;
  viewer->data = (void*) e;
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)viewer,"PetscViewerFileSetName_C","PetscViewerFileSetName_Matlab",
                                     PetscViewerFileSetName_Matlab);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)viewer,"PetscViewerFileSetMode_C","PetscViewerFileSetMode_Matlab",
                                     PetscViewerFileSetMode_Matlab);CHKERRQ(ierr);
  viewer->ops->destroy = PetscViewerDestroy_Matlab;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerMatlabOpen" 
/*@C
   PetscViewerMatlabOpen - Opens a Matlab .mat file for input or output.

   Collective on MPI_Comm

   Input Parameters:
+  comm - MPI communicator
.  name - name of file 
-  type - type of file
$    FILE_MODE_WRITE - create new file for Matlab output
$    FILE_MODE_READ - open existing file for Matlab input
$    FILE_MODE_WRITE - open existing file for Matlab output

   Output Parameter:
.  binv - PetscViewer for Matlab input/output to use with the specified file

   Level: beginner

   Note:
   This PetscViewer should be destroyed with PetscViewerDestroy().

    For writing files it only opens the file on processor 0 in the communicator.
    For readable files it opens the file on all nodes that have the file. If 
    node 0 does not have the file it generates an error even if other nodes
    do have the file.

   Concepts: Matlab .mat files
   Concepts: PetscViewerMatlab^creating

.seealso: PetscViewerASCIIOpen(), PetscViewerSetFormat(), PetscViewerDestroy(),
          VecView(), MatView(), VecLoad(), MatLoad()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscViewerMatlabOpen(MPI_Comm comm,const char name[],PetscFileMode type,PetscViewer *binv)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = PetscViewerCreate(comm,binv);CHKERRQ(ierr);
  ierr = PetscViewerSetType(*binv,PETSC_VIEWER_MATLAB);CHKERRQ(ierr);
  ierr = PetscViewerFileSetMode(*binv,type);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(*binv,name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscMPIInt Petsc_Viewer_Matlab_keyval = MPI_KEYVAL_INVALID;

#undef __FUNCT__  
#define __FUNCT__ "PETSC_VIEWER_MATLAB_"  
/*@C
     PETSC_VIEWER_MATLAB_ - Creates a Matlab PetscViewer shared by all processors 
                     in a communicator.

     Collective on MPI_Comm

     Input Parameter:
.    comm - the MPI communicator to share the Matlab PetscViewer
    
     Level: intermediate

   Options Database Keys:
$    -viewer_matlab_filename <name>

   Environmental variables:
-   PETSC_VIEWER_MATLAB_FILENAME

     Notes:
     Unlike almost all other PETSc routines, PETSC_VIEWER_MATLAB_ does not return 
     an error code.  The matlab PetscViewer is usually used in the form
$       XXXView(XXX object,PETSC_VIEWER_MATLAB_(comm));

     Use PETSC_VIEWER_SOCKET_() or PetscViewerSocketOpen() to communicator with an interactive Matlab session.

.seealso: PETSC_VIEWER_MATLAB_WORLD, PETSC_VIEWER_MATLAB_SELF, PetscViewerMatlabOpen(), PetscViewerCreate(),
          PetscViewerDestroy()
@*/
PetscViewer PETSC_DLLEXPORT PETSC_VIEWER_MATLAB_(MPI_Comm comm)
{
  PetscErrorCode ierr;
  PetscTruth     flg;
  PetscViewer    viewer;
  char           fname[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  if (Petsc_Viewer_Matlab_keyval == MPI_KEYVAL_INVALID) {
    ierr = MPI_Keyval_create(MPI_NULL_COPY_FN,MPI_NULL_DELETE_FN,&Petsc_Viewer_Matlab_keyval,0);
    if (ierr) {PetscError(__LINE__,"PETSC_VIEWER_MATLAB_",__FILE__,__SDIR__,1,1," ");PetscFunctionReturn(0);}
  }
  ierr = MPI_Attr_get(comm,Petsc_Viewer_Matlab_keyval,(void **)&viewer,(int*)&flg);
  if (ierr) {PetscError(__LINE__,"PETSC_VIEWER_MATLAB_",__FILE__,__SDIR__,1,1," ");PetscFunctionReturn(0);}
  if (!flg) { /* PetscViewer not yet created */
    ierr = PetscOptionsGetenv(comm,"PETSC_VIEWER_MATLAB_FILENAME",fname,PETSC_MAX_PATH_LEN,&flg);
    if (ierr) {PetscError(__LINE__,"PETSC_VIEWER_MATLAB_",__FILE__,__SDIR__,1,1," ");PetscFunctionReturn(0);}
    if (!flg) {
      ierr = PetscStrcpy(fname,"matlaboutput.mat");
      if (ierr) {PetscError(__LINE__,"PETSC_VIEWER_MATLAB_",__FILE__,__SDIR__,1,1," ");PetscFunctionReturn(0);}
    }
    ierr = PetscViewerMatlabOpen(comm,fname,FILE_MODE_WRITE,&viewer); 
    if (ierr) {PetscError(__LINE__,"PETSC_VIEWER_MATLAB_",__FILE__,__SDIR__,1,1," ");PetscFunctionReturn(0);}
    ierr = PetscObjectRegisterDestroy((PetscObject)viewer);
    if (ierr) {PetscError(__LINE__,"PETSC_VIEWER_MATLAB_",__FILE__,__SDIR__,1,1," ");PetscFunctionReturn(0);}
    ierr = MPI_Attr_put(comm,Petsc_Viewer_Matlab_keyval,(void*)viewer);
    if (ierr) {PetscError(__LINE__,"PETSC_VIEWER_MATLAB_",__FILE__,__SDIR__,1,1," ");PetscFunctionReturn(0);}
  } 
  PetscFunctionReturn(viewer);
}





