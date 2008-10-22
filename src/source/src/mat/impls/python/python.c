/* -------------------------------------------------------------------------- */

#include "src/inline/python.h"
#include "include/private/matimpl.h"

/* -------------------------------------------------------------------------- */

/* backward compatibility hacks */

#if (PETSC_VERSION_MAJOR    == 2 &&	\
     PETSC_VERSION_MINOR    == 3 &&	\
     PETSC_VERSION_SUBMINOR == 2 &&	\
     PETSC_VERSION_RELEASE  == 1)
#define MAT_PYTHON_FIX_SETFROMOPTIONS
#endif

#if (PETSC_VERSION_MAJOR    == 2 &&	\
     PETSC_VERSION_MINOR    == 3 &&	\
     PETSC_VERSION_SUBMINOR == 3 &&	\
     PETSC_VERSION_RELEASE  == 1)
#define MAT_PYTHON_FIX_SETFROMOPTIONS
#endif

#define MAT_PYTHON_FIX_SETFROMOPTIONS

/* -------------------------------------------------------------------------- */

#define MATPYTHON "python"

PETSC_EXTERN_C_BEGIN
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreatePython(MPI_Comm,const char*,const char*,Mat*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatPythonSetContext(Mat,void*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatPythonGetContext(Mat,void**);
PETSC_EXTERN_C_END

/* -------------------------------------------------------------------------- */

typedef struct {
  PyObject *self;
  char     *module;
  char     *factory;
  PetscTruth  scale,shift;
  PetscScalar vscale,vshift;
} Mat_Py;

/* -------------------------------------------------------------------------- */

#define Mat_Py_Self(mat) (((Mat_Py*)(mat)->data)->self)

#define MAT_PYTHON_CALL_HEAD(mat, PyMethod)	\
  PETSC_PYTHON_CALL_HEAD(Mat_Py_Self(mat), PyMethod)
#define MAT_PYTHON_CALL_JUMP(mat, LABEL)	\
  PETSC_PYTHON_CALL_JUMP(LABEL)
#define MAT_PYTHON_CALL_BODY(mat, ARGS)		\
  PETSC_PYTHON_CALL_BODY(ARGS)
#define MAT_PYTHON_CALL_TAIL(mat, PyMethod)	\
  PETSC_PYTHON_CALL_TAIL()

#define MAT_PYTHON_CALL(mat, PyMethod, ARGS)	\
  MAT_PYTHON_CALL_HEAD(mat, PyMethod);		\
  MAT_PYTHON_CALL_BODY(mat, ARGS);		\
  MAT_PYTHON_CALL_TAIL(mat, PyMethod)		\
/**/

#define MAT_PYTHON_CALL_MAYBE(mat, PyMethod, ARGS, LABEL)	\
  MAT_PYTHON_CALL_HEAD(mat, PyMethod);				\
  MAT_PYTHON_CALL_JUMP(mat, LABEL);				\
  MAT_PYTHON_CALL_BODY(mat, ARGS);				\
  MAT_PYTHON_CALL_TAIL(mat, PyMethod)				\
/**/

#define MAT_PYTHON_CALL_NOARGS(mat, PyMethod)				\
  MAT_PYTHON_CALL_HEAD(mat, PyMethod);					\
  MAT_PYTHON_CALL_BODY(mat, ("", NULL));				\
  MAT_PYTHON_CALL_TAIL(mat, PyMethod)					\
/**/

#define MAT_PYTHON_MATARG(mat) ("O&",PyPetscMat_New,mat)

#define MAT_PYTHON_CALL_MATARG(mat, PyMethod)				\
  MAT_PYTHON_CALL_HEAD(mat, PyMethod);					\
  MAT_PYTHON_CALL_BODY(mat, ("O&", PyPetscMat_New, mat));		\
  MAT_PYTHON_CALL_TAIL(mat, PyMethod)					\
/**/

#define MAT_PYTHON_SETERRSUP(mat, PyMethod)				\
  SETERRQ1(PETSC_ERR_SUP,"method %s() not implemented",PyMethod);	\
  PetscFunctionReturn(PETSC_ERR_SUP)					\
/**/

/* -------------------------------------------------------------------------- */
#if 0
static int MatPythonHasOperation(Mat mat, const char operation[])
{
  Mat_Py   *py   = (Mat_Py *) mat->data;
  PyObject *attr = NULL;
  if (py->self == NULL || py->self == Py_None) return 0;
  attr = PyObject_GetAttrString(py->self, operation);
  if      (attr == NULL)    { PyErr_Clear();   return 0; }
  else if (attr == Py_None) { Py_DecRef(attr); return 0; }
  return 1;
}

#undef __FUNCT__  
#define __FUNCT__ "MatPythonFillOperations"
static PetscErrorCode MatPythonFillOperations(Mat mat)
{
  PetscFunctionBegin;
  if (MatPythonHasOperation(mat, "multTranspose"))
    mat->ops->multtranspose = MatApplySymmetricLeft_Python;
  PetscFunctionReturn(0);
}
#endif
/* -------------------------------------------------------------------------- */

#undef  __FUNCT__
#define __FUNCT__ "MatSetFromOptions_Python"
static PetscErrorCode MatSetFromOptions_Python(Mat mat)
{
  char           *modcls[2] = {0, 0};
  PetscInt       nmax = 2;
  PetscTruth     flg;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscOptionsBegin(((PetscObject)mat)->comm,((PetscObject)mat)->prefix,"Matrix options","Mat");CHKERRQ(ierr);
  ierr = PetscOptionsHead("Python options");CHKERRQ(ierr);
  ierr = PetscOptionsStringArray("-mat_python","Python module and class/factory",
				 "MatCreatePython", modcls,&nmax,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if (flg) {
    if (nmax == 2) {
      PyObject *self = NULL;
      ierr = PetscCreatePythonObject(modcls[0],modcls[1],&self);CHKERRQ(ierr);
      ierr = MatPythonSetContext(mat,self);Py_DecRef(self);CHKERRQ(ierr);
    }
    ierr = PetscStrfree(modcls[0]);CHKERRQ(ierr);
    ierr = PetscStrfree(modcls[1]);CHKERRQ(ierr);
  }
  MAT_PYTHON_CALL_MATARG(mat, "setFromOptions");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatView_Python"
static PetscErrorCode MatView_Python(Mat mat,PetscViewer viewer)
{
  Mat_Py         *py = (Mat_Py*)mat->data;
  PetscTruth     isascii,isstring;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&isascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_STRING,&isstring);CHKERRQ(ierr);
  if (isascii || isstring) {
    ierr = PetscStrfree(py->module);CHKERRQ(ierr); 
    ierr = PetscStrfree(py->factory);CHKERRQ(ierr);
    ierr = PetscPythonGetModuleAndClass(py->self,&py->module,&py->factory);CHKERRQ(ierr);
  }
  if (isascii) {
    const char* module  = py->module  ? py->module  : "no yet set";
    const char* factory = py->factory ? py->factory : "no yet set";
    ierr = PetscViewerASCIIPrintf(viewer,"  module:  %s\n",module);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  factory: %s\n",factory);CHKERRQ(ierr);
  }
  if (isstring) {
    const char* module  = py->module  ? py->module  : "<module>";
    const char* factory = py->factory ? py->factory : "<factory>";
    ierr = PetscViewerStringSPrintf(viewer,"%s.%s",module,factory);CHKERRQ(ierr);
  }
  MAT_PYTHON_CALL(mat, "view", ("O&O&",
				PyPetscMat_New,     mat,
				PyPetscViewer_New,  viewer));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_Python"
static PetscErrorCode MatDestroy_Python(Mat mat)
{
  Mat_Py         *py   = (Mat_Py *)mat->data;
  PyObject       *self = py->self;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (Py_IsInitialized()) {
    MAT_PYTHON_CALL_NOARGS(mat, "destroy");
    py->self = NULL; Py_DecRef(self);
  }
  ierr = PetscStrfree(py->module);CHKERRQ(ierr);
  ierr = PetscStrfree(py->factory);CHKERRQ(ierr);
  ierr = PetscFree(mat->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */

#undef __FUNCT__
#define __FUNCT__ "MatSetOption_Python"
static PetscErrorCode MatSetOption_Python(Mat mat,MatOption op,PetscTruth flag)
{
  PetscFunctionBegin;
  MAT_PYTHON_CALL(mat, "setOption", ("O&ii",
				     PyPetscMat_New, mat,
				     (int)           op,
				     (int)           flag));
  PetscFunctionReturn(0);
}

#if (PETSC_VERSION_MAJOR    == 2 &&	\
     PETSC_VERSION_MINOR    == 3 &&	\
     (PETSC_VERSION_SUBMINOR == 2  ||	\
      PETSC_VERSION_SUBMINOR == 3) &&	\
     PETSC_VERSION_RELEASE  == 1)
static PetscErrorCode MatSetOption_Python_old(Mat mat,MatOption op)
{ return MatSetOption_Python(mat,op,PETSC_TRUE); }
#define MatSetOption_Python MatSetOption_Python_old
#endif

#undef  __FUNCT__
#define __FUNCT__ "MatSetUpPreallocation_Python"
static PetscErrorCode MatSetUpPreallocation_Python(Mat mat)
{
  PetscFunctionBegin;
  /* MatDestroy() calls MatPreallocated() !!! */
  if (!Py_IsInitialized()) PetscFunctionReturn(0);
#if defined(MAT_PYTHON_FIX_SETFROMOPTIONS)
  { 
    PyObject       *self = Mat_Py_Self(mat);
    PetscErrorCode ierr;
    if (self == NULL || self == Py_None) {
      ierr = MatSetFromOptions_Python(mat);CHKERRQ(ierr); 
    }
  }
#endif
  mat->preallocated = PETSC_TRUE;
  MAT_PYTHON_CALL_MATARG(mat, "setUpPreallocation");
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MatZeroEntries_Python"
static PetscErrorCode MatZeroEntries_Python(Mat mat)
{
  Mat_Py *py = (Mat_Py *) mat->data;
  PetscFunctionBegin;
  MAT_PYTHON_CALL_MAYBE(mat, "zeroEntries", 
			MAT_PYTHON_MATARG(mat),
			notimplemented);
  py->scale = PETSC_FALSE; py->vscale = 1;
  py->shift = PETSC_FALSE; py->vshift = 0;
  PetscFunctionReturn(0);
 notimplemented: /* MatZeroEntries */
  MAT_PYTHON_SETERRSUP(mat, "zeroEntries");
}

#undef __FUNCT__
#define __FUNCT__ "MatScale_Python"
static PetscErrorCode MatScale_Python(Mat mat,PetscScalar a)
{
  Mat_Py *py = (Mat_Py *) mat->data;
#if defined(PETSC_USE_COMPLEX)
  Py_complex ca;
#endif
  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)
  ca.real = PetscRealPart(a);
  ca.imag = PetscImaginaryPart(a);
#endif
#if defined(PETSC_USE_COMPLEX)
  MAT_PYTHON_CALL_MAYBE(mat, "scale", ("O&D",
				       PyPetscMat_New, mat,
				       ca),
			scale);
#else
  MAT_PYTHON_CALL_MAYBE(mat, "scale", ("O&d",
				       PyPetscMat_New, mat,
				       (double)        a),
			scale);
#endif
  py->scale  = PETSC_FALSE;
  py->vscale = 1;
  PetscFunctionReturn(0);
 scale: /* MatScale */
  py->scale  = PETSC_TRUE;
  py->vscale = a;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatShift_Python"
static PetscErrorCode MatShift_Python(Mat mat,PetscScalar a)
{
  Mat_Py *py = (Mat_Py *) mat->data;
#if defined(PETSC_USE_COMPLEX)
  Py_complex ca;
#endif
  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)
  ca.real = PetscRealPart(a);
  ca.imag = PetscImaginaryPart(a);
#endif
#if defined(PETSC_USE_COMPLEX)
  MAT_PYTHON_CALL_MAYBE(mat, "shift", ("O&D",PetscInt
				       PyPetscMat_New, mat,
				       ca),
			shift);
#else
  MAT_PYTHON_CALL_MAYBE(mat, "shift", ("O&d",
				       PyPetscMat_New, mat,
				       (double)        a),
			shift);
#endif
  py->shift  = PETSC_FALSE;
  py->vshift = 0;
  PetscFunctionReturn(0);
 shift: /* MatShift */
  py->shift  = PETSC_TRUE;
  py->vshift = a;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MatAssemblyBegin_Python"
static PetscErrorCode MatAssemblyBegin_Python(Mat mat,MatAssemblyType type)
{
  PetscFunctionBegin;
  MAT_PYTHON_CALL(mat, "assemblyBegin", ("O&i",
					 PyPetscMat_New, mat,
					 (int)           type));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatAssemblyEnd_Python"
static PetscErrorCode MatAssemblyEnd_Python(Mat mat,MatAssemblyType type)
{
  Mat_Py         *py = (Mat_Py *) mat->data;
  PetscFunctionBegin;
  MAT_PYTHON_CALL(mat, "assemblyEnd", ("O&i",
				       PyPetscMat_New, mat,
				       (int)           type));
  if (type == MAT_FINAL_ASSEMBLY) {
    py->scale = PETSC_FALSE; py->vscale = 1.0;
    py->shift = PETSC_FALSE; py->vshift = 0.0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMult_Python"
static PetscErrorCode MatMult_Python(Mat mat,Vec x,Vec y)
{
  Mat_Py         *py = (Mat_Py *) mat->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  MAT_PYTHON_CALL_MAYBE(mat, "mult", ("O&O&O&",
				      PyPetscMat_New, mat,
				      PyPetscVec_New, x,
				      PyPetscVec_New, y),
			notimplemented);
  /* shift and scale */
  if (py->shift && py->scale) {
    ierr = VecAXPBY(y,py->vshift,py->vscale,x);CHKERRQ(ierr);
  } else if (py->scale) {
    ierr = VecScale(y,py->vscale);CHKERRQ(ierr);
  } else if (py->shift) {
    ierr = VecAXPY(y,py->vshift,x);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
 notimplemented: /* MatMult */
  MAT_PYTHON_SETERRSUP(mat, "mult");
}

#undef __FUNCT__
#define __FUNCT__ "MatMultAdd_Python"
static PetscErrorCode MatMultAdd_Python(Mat mat,Vec x,Vec v,Vec y)
{
  Mat_Py         *py = (Mat_Py *) mat->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (py->scale || py->shift) goto multadd;
  MAT_PYTHON_CALL_MAYBE(mat, "multAdd", ("O&O&O&O&",
					 PyPetscMat_New, mat,
					 PyPetscVec_New, x,
					 PyPetscVec_New, v,
					 PyPetscVec_New, y),
			multadd);
  PetscFunctionReturn(0);
 multadd: /* MatMultAdd */
  ierr = MatMult_Python(mat,x,y);CHKERRQ(ierr);
  ierr = VecAXPY(y, 1, v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultTranspose_Python"
static PetscErrorCode MatMultTranspose_Python(Mat mat,Vec x, Vec y)
{
  Mat_Py         *py = (Mat_Py *) mat->data;
  PetscTruth     symmset,symmknown;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  MAT_PYTHON_CALL_MAYBE(mat, "multTranspose", ("O&O&O&",
					       PyPetscMat_New, mat,
					       PyPetscVec_New, x,
					       PyPetscVec_New, y),
			mult);
  /* shift and scale */
  if (py->shift && py->scale) {
    ierr = VecAXPBY(y,py->vshift,py->vscale,x);CHKERRQ(ierr);
  } else if (py->scale) {
    ierr = VecScale(y,py->vscale);CHKERRQ(ierr);
  } else if (py->shift) {
    ierr = VecAXPY(y,py->vshift,x);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
 mult:
  ierr = MatIsSymmetricKnown(mat,&symmset,&symmknown);CHKERRQ(ierr);
  if (!symmset || !symmknown) goto notimplemented;
  ierr = MatMult_Python(mat,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
 notimplemented: /* MatMultTranspose */
  MAT_PYTHON_SETERRSUP(mat, "multTranspose");
}


#undef __FUNCT__
#define __FUNCT__ "MatMultTransposeAdd_Python"
static PetscErrorCode MatMultTransposeAdd_Python(Mat mat,Vec x,Vec v,Vec y)
{
  Mat_Py         *py = (Mat_Py *) mat->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (py->scale || py->shift) goto multtransposeadd;
  MAT_PYTHON_CALL_MAYBE(mat, "multTransposeAdd", ("O&O&O&O&",
						  PyPetscMat_New, mat,
						  PyPetscVec_New, x,
						  PyPetscVec_New, v,
						  PyPetscVec_New, y),
			multtransposeadd);
  PetscFunctionReturn(0);
 multtransposeadd: /* MatMultTransposeAdd */
  ierr = MatMultTranspose_Python(mat,x,y);CHKERRQ(ierr);
  ierr = VecAXPY(y, 1, v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSolve_Python"
static PetscErrorCode MatSolve_Python(Mat mat,Vec b,Vec x)
{
  Mat_Py         *py = (Mat_Py *) mat->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  /*  shift */
  if (py->shift) goto notimplemented;
  MAT_PYTHON_CALL_MAYBE(mat, "solve", ("O&O&O&",
				       PyPetscMat_New, mat,
				       PyPetscVec_New, b,
				       PyPetscVec_New, x),
			notimplemented);
  /*  scale */
  if (py->scale) { ierr = VecScale(x,1/py->vscale);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
 notimplemented: /* MatSolve */
  MAT_PYTHON_SETERRSUP(mat, "solve");
}

#undef __FUNCT__
#define __FUNCT__ "MatSolveAdd_Python"
static PetscErrorCode MatSolveAdd_Python(Mat mat,Vec b,Vec v,Vec x)
{
  Mat_Py         *py = (Mat_Py *) mat->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (py->scale || py->shift) goto solveadd;
  MAT_PYTHON_CALL_MAYBE(mat, "solveAdd", ("O&O&O&O&",
					  PyPetscMat_New, mat,
					  PyPetscVec_New, b,
					  PyPetscVec_New, v,
					  PyPetscVec_New, x),
			solveadd);
  PetscFunctionReturn(0);
 solveadd: /* MatSolveAdd */
  ierr = MatSolve_Python(mat,b,x);CHKERRQ(ierr);
  ierr = VecAXPY(x, 1, v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSolveTranspose_Python"
static PetscErrorCode MatSolveTranspose_Python(Mat mat,Vec b, Vec x)
{
  Mat_Py         *py = (Mat_Py *) mat->data;
  PetscTruth     symmset,symmknown;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  /*  shift */
  if (py->shift) goto notimplemented;
  MAT_PYTHON_CALL_MAYBE(mat, "solveTranspose", ("O&O&O&",
						PyPetscMat_New, mat,
						PyPetscVec_New, b,
						PyPetscVec_New, x),
			solve);
  /* scale */
  if (py->scale) { ierr = VecScale(x,1/py->vscale);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
 solve:
  ierr = MatIsSymmetricKnown(mat,&symmset,&symmknown);CHKERRQ(ierr);
  if (!symmset || !symmknown) goto notimplemented;
  ierr = MatSolve_Python(mat,b,x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
 notimplemented: /* MatSolveTranspose */
  MAT_PYTHON_SETERRSUP(mat, "solveTranspose");
}


#undef __FUNCT__
#define __FUNCT__ "MatSolveTransposeAdd_Python"
static PetscErrorCode MatSolveTransposeAdd_Python(Mat mat,Vec b,Vec v,Vec x)
{
  Mat_Py         *py = (Mat_Py *) mat->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (py->scale || py->shift) goto solvetransposeadd;
  MAT_PYTHON_CALL_MAYBE(mat, "solveTransposeAdd", ("O&O&O&O&",
						   PyPetscMat_New, mat,
						   PyPetscVec_New, b,
						   PyPetscVec_New, v,
						   PyPetscVec_New, x),
			solvetransposeadd);
  PetscFunctionReturn(0);
 solvetransposeadd: /* MatSolveTransposeAdd */
  ierr = MatSolveTranspose_Python(mat,b,x);CHKERRQ(ierr);
  ierr = VecAXPY(x, 1, v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatGetDiagonal_Python"
static PetscErrorCode MatGetDiagonal_Python(Mat mat,Vec v)
{
  Mat_Py         *py = (Mat_Py *) mat->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  MAT_PYTHON_CALL_MAYBE(mat, "getDiagonal", ("O&O&",
					     PyPetscMat_New, mat,
					     PyPetscVec_New, v),
			notimplemented);
  if (py->scale) { ierr = VecScale(v,py->vscale);CHKERRQ(ierr); }
  if (py->shift) { ierr = VecShift(v,py->vshift);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
 notimplemented: /* MatGetDiagonal */
  MAT_PYTHON_SETERRSUP(mat, "getDiagonal");
}

#undef __FUNCT__
#define __FUNCT__ "MatGetDiagonal_Python"
static PetscErrorCode MatDiagonalSet_Python(Mat mat,Vec v,InsertMode im)
{
  Mat_Py *py = (Mat_Py *) mat->data;
  PetscFunctionBegin;
  MAT_PYTHON_CALL_MAYBE(mat, "setDiagonal", ("O&O&i",
					     PyPetscMat_New, mat,
					     PyPetscVec_New, v,
					     (int)           im),
			notimplemented);
  if (im != ADD_VALUES) py->shift = PETSC_FALSE; py->vshift = 0;
  PetscFunctionReturn(0);
 notimplemented:  /* MatDiagonalSet */
  MAT_PYTHON_SETERRSUP(mat, "setDiagonal");
}

#undef __FUNCT__
#define __FUNCT__ "MatDiagonalScale_Python"
static PetscErrorCode MatDiagonalScale_Python(Mat mat,Vec l, Vec r)
{
  PetscFunctionBegin;
  MAT_PYTHON_CALL_MAYBE(mat, "diagonalScale", ("O&O&O&",
					       PyPetscMat_New, mat,
					       PyPetscVec_New, l,
					       PyPetscVec_New, r),
			notimplemented);
  PetscFunctionReturn(0);
 notimplemented: /* MatDiagonalScale */
  MAT_PYTHON_SETERRSUP(mat, "diagonalScale");
}

#undef  __FUNCT__
#define __FUNCT__ "MatRealPart_Python"
static PetscErrorCode MatRealPart_Python(Mat mat)
{
  PetscFunctionBegin;
  MAT_PYTHON_CALL_MAYBE(mat, "realPart",
			MAT_PYTHON_MATARG(mat),
			notimplemented);
  PetscFunctionReturn(0);
 notimplemented: /* MatRealPart */
  MAT_PYTHON_SETERRSUP(mat, "realPart");
}

#undef  __FUNCT__
#define __FUNCT__ "MatImaginaryPart_Python"
static PetscErrorCode MatImaginaryPart_Python(Mat mat)
{
  PetscFunctionBegin;
  MAT_PYTHON_CALL_MAYBE(mat, "imagPart", 
			MAT_PYTHON_MATARG(mat),
			notimplemented);
  PetscFunctionReturn(0);
 notimplemented: /* MatImaginaryPart */
  MAT_PYTHON_SETERRSUP(mat, "imagPart");
}

#undef  __FUNCT__
#define __FUNCT__ "MatImaginaryPart_Python"
static PetscErrorCode MatConjugate_Python(Mat mat)
{
  PetscFunctionBegin;
  MAT_PYTHON_CALL_MAYBE(mat, "conjugate", 
			MAT_PYTHON_MATARG(mat),
			notimplemented);
  PetscFunctionReturn(0);
 notimplemented: /* MatConjugate */
  MAT_PYTHON_SETERRSUP(mat, "conjugate");
}

/* -------------------------------------------------------------------------- */

/*MC
   MATPYTHON - .

   Level: intermediate

   Contributed by Lisandro Dalcin <dalcinl at gmail dot com>

.seealso:  Mat, MatCreate(), MatSetType(), MatType (for list of available types)
M*/

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatCreate_Python"
PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_Python(Mat mat)
{
  Mat_Py      *py;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = PetscInitializePython();CHKERRQ(ierr);

  ierr = PetscNew(Mat_Py,&py);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(mat,sizeof(Mat_Py));CHKERRQ(ierr);
  mat->data  = (void*)py;

  /* Python */
  py->self    = NULL;
  py->module  = NULL;
  py->factory = NULL;
  py->scale   = PETSC_FALSE;
  py->vscale  = 1;
  py->shift   = PETSC_FALSE;
  py->vshift  = 0;

  /* PETSc */
  mat->ops->destroy              = MatDestroy_Python;
  mat->ops->view                 = MatView_Python;
  mat->ops->setfromoptions       = MatSetFromOptions_Python;

  mat->ops->setoption            = MatSetOption_Python;
  mat->ops->setuppreallocation   = MatSetUpPreallocation_Python;

  mat->ops->zeroentries          = MatZeroEntries_Python;
  mat->ops->scale                = MatScale_Python;
  mat->ops->shift                = MatShift_Python;
  mat->ops->assemblybegin        = MatAssemblyBegin_Python;
  mat->ops->assemblyend          = MatAssemblyEnd_Python;

  mat->ops->mult                 = MatMult_Python;
  mat->ops->multtranspose        = MatMultTranspose_Python;
  mat->ops->multadd              = MatMultAdd_Python;
  mat->ops->multtransposeadd     = MatMultTransposeAdd_Python;

  mat->ops->solve                = MatSolve_Python;
  mat->ops->solvetranspose       = MatSolveTranspose_Python;
  mat->ops->solveadd             = MatSolveAdd_Python;
  mat->ops->solvetransposeadd    = MatSolveTransposeAdd_Python;

  mat->ops->getdiagonal          = MatGetDiagonal_Python;
  mat->ops->diagonalset          = MatDiagonalSet_Python;
  mat->ops->diagonalscale        = MatDiagonalScale_Python;

  mat->ops->realpart             = MatRealPart_Python;
  mat->ops->imaginarypart        = MatImaginaryPart_Python;
  mat->ops->conjugate            = MatConjugate_Python;

  mat->factor       = 0;
  mat->assembled    = PETSC_TRUE;
  mat->preallocated = PETSC_FALSE;
  ierr = PetscObjectChangeTypeName((PetscObject)mat,MATPYTHON);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END

/* -------------------------------------------------------------------------- */


/* -------------------------------------------------------------------------- */

#undef __FUNCT__
#define __FUNCT__ "MatPythonGetContext"
/*@
   MatPythonGetContext - .

   Input Parameter:
.  mat - Mat object

   Output Parameter:
.  ctx - Python context

   Level: beginner

.keywords: Mat, matrix, create

.seealso: Mat, MatCreate(), MAtSetType(), MATPYTHON
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatPythonGetContext(Mat mat,void **ctx)
{
  Mat_Py         *py;
  PetscTruth     ispython;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidPointer(ctx,2);
  *ctx = NULL;
  ierr = PetscTypeCompare((PetscObject)mat,MATPYTHON,&ispython);CHKERRQ(ierr);
  if (!ispython) PetscFunctionReturn(0);
  py = (Mat_Py *) mat->data;
  *ctx = (void *) py->self;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPythonSetContext"
/*@
   MatPythonSetContext - .

   Collective on PC

   Input Parameters:
.  pc - PC context
.  ctx - Python context

   Level: beginner

.keywords: Mat, matrix, create

.seealso: Mat, MatCreate(), MAtSetType(), MATPYTHON
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatPythonSetContext(Mat mat,void *ctx)
{
  Mat_Py          *py;
  PyObject       *old, *self = (PyObject *) ctx;
  PetscTruth     ispython;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  if (ctx) PetscValidPointer(ctx,2);
  ierr = PetscTypeCompare((PetscObject)mat,MATPYTHON,&ispython);CHKERRQ(ierr);
  if (!ispython) PetscFunctionReturn(0);
  py = (Mat_Py *) mat->data;
  /* do nothing if contexts are the same */
  if (self == Py_None) self = NULL;
  if (py->self == self) PetscFunctionReturn(0);
  /* del previous Python context in the Mat object */
  MAT_PYTHON_CALL_NOARGS(mat, "destroy");
  old = py->self; py->self = NULL; Py_DecRef(old);
  /* set current Python context in the Mat object  */
  py->self = (PyObject *) self; Py_IncRef(py->self);
  MAT_PYTHON_CALL_MATARG(mat, "create");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreatePython"
/*@
   MatCreatePython - Creates a Python matrix context.

   Collective on MPI_Comm

   Input Parameters:
.  comm - MPI communicator 
.  modname - module name
.  clsname - factory/class name

   Output Parameter:
.  mat - location to put the matrix context

   Level: beginner

.keywords: Mat,  create

.seealso: Mat, MatCreate(), MatSetType(), MatPYTHON
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatCreatePython(MPI_Comm comm,
						  const char *modname,
						  const char *clsname,
						  Mat *mat)
{
  PyObject       *self = NULL;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  if (modname) PetscValidCharPointer(modname,2);
  if (clsname) PetscValidCharPointer(clsname,3);
  /* create the Mat context and set its type */
  ierr = MatCreate(comm,mat);CHKERRQ(ierr);
  ierr = MatSetType(*mat,MATPYTHON);CHKERRQ(ierr);
  if (modname == PETSC_NULL) PetscFunctionReturn(0);
  if (clsname == PETSC_NULL) PetscFunctionReturn(0);
  /* create the Python object from module and class/factory  */
  ierr = PetscCreatePythonObject(modname,clsname,&self);CHKERRQ(ierr);
  /* set the created Python object in Mat context */
  ierr = MatPythonSetContext(*mat,self);Py_DecRef(self);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
