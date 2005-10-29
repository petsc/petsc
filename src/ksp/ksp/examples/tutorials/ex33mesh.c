#include <petscda.h>
#include <petscdmmg.h>
#include <ALE.hh>
#include <Sieve.hh>
#include <IndexBundle.hh>

typedef enum {DIRICHLET, NEUMANN} BCType;

typedef struct {
  PetscScalar   nu;
  BCType        bcType;
} UserContext;

#ifndef MESH_3D

#define NUM_QUADRATURE_POINTS 9

/* Quadrature points */
static double points[18] = {
  -0.794564690381,
  -0.822824080975,
  -0.866891864322,
  -0.181066271119,
  -0.952137735426,
  0.575318923522,
  -0.0885879595127,
  -0.822824080975,
  -0.409466864441,
  -0.181066271119,
  -0.787659461761,
  0.575318923522,
  0.617388771355,
  -0.822824080975,
  0.0479581354402,
  -0.181066271119,
  -0.623181188096,
  0.575318923522};

/* Quadrature weights */
static double weights[9] = {
  0.223257681932,
  0.2547123404,
  0.0775855332238,
  0.357212291091,
  0.407539744639,
  0.124136853158,
  0.223257681932,
  0.2547123404,
  0.0775855332238};

#define NUM_BASIS_FUNCTIONS 3

/* Nodal basis function evaluations */
static double Basis[27] = {
  0.808694385678,
  0.10271765481,
  0.0885879595127,
  0.52397906772,
  0.0665540678392,
  0.409466864441,
  0.188409405952,
  0.0239311322871,
  0.787659461761,
  0.455706020244,
  0.455706020244,
  0.0885879595127,
  0.29526656778,
  0.29526656778,
  0.409466864441,
  0.10617026912,
  0.10617026912,
  0.787659461761,
  0.10271765481,
  0.808694385678,
  0.0885879595127,
  0.0665540678392,
  0.52397906772,
  0.409466864441,
  0.0239311322871,
  0.188409405952,
  0.787659461761};

/* Nodal basis function derivative evaluations */
static double BasisDerivatives[54] = {
  -0.5,
  -0.5,
  0.5,
  4.74937635818e-17,
  0.0,
  0.5,
  -0.5,
  -0.5,
  0.5,
  4.74937635818e-17,
  0.0,
  0.5,
  -0.5,
  -0.5,
  0.5,
  4.74937635818e-17,
  0.0,
  0.5,
  -0.5,
  -0.5,
  0.5,
  4.74937635818e-17,
  0.0,
  0.5,
  -0.5,
  -0.5,
  0.5,
  4.74937635818e-17,
  0.0,
  0.5,
  -0.5,
  -0.5,
  0.5,
  4.74937635818e-17,
  0.0,
  0.5,
  -0.5,
  -0.5,
  0.5,
  4.74937635818e-17,
  0.0,
  0.5,
  -0.5,
  -0.5,
  0.5,
  4.74937635818e-17,
  0.0,
  0.5,
  -0.5,
  -0.5,
  0.5,
  4.74937635818e-17,
  0.0,
  0.5};

#else

#define NUM_QUADRATURE_POINTS 27

/* Quadrature points */
static double points[81] = {
  -0.809560240317,
  -0.835756864273,
  -0.854011951854,
  -0.865851516496,
  -0.884304792128,
  -0.305992467923,
  -0.939397037651,
  -0.947733495427,
  0.410004419777,
  -0.876607962782,
  -0.240843539439,
  -0.854011951854,
  -0.913080888692,
  -0.465239359176,
  -0.305992467923,
  -0.960733394129,
  -0.758416359732,
  0.410004419777,
  -0.955631394718,
  0.460330056095,
  -0.854011951854,
  -0.968746121484,
  0.0286773243482,
  -0.305992467923,
  -0.985880737721,
  -0.53528439884,
  0.410004419777,
  -0.155115591937,
  -0.835756864273,
  -0.854011951854,
  -0.404851369974,
  -0.884304792128,
  -0.305992467923,
  -0.731135462175,
  -0.947733495427,
  0.410004419777,
  -0.452572254354,
  -0.240843539439,
  -0.854011951854,
  -0.61438408645,
  -0.465239359176,
  -0.305992467923,
  -0.825794030022,
  -0.758416359732,
  0.410004419777,
  -0.803159052121,
  0.460330056095,
  -0.854011951854,
  -0.861342428212,
  0.0286773243482,
  -0.305992467923,
  -0.937360010468,
  -0.53528439884,
  0.410004419777,
  0.499329056443,
  -0.835756864273,
  -0.854011951854,
  0.0561487765469,
  -0.884304792128,
  -0.305992467923,
  -0.522873886699,
  -0.947733495427,
  0.410004419777,
  -0.0285365459258,
  -0.240843539439,
  -0.854011951854,
  -0.315687284208,
  -0.465239359176,
  -0.305992467923,
  -0.690854665916,
  -0.758416359732,
  0.410004419777,
  -0.650686709523,
  0.460330056095,
  -0.854011951854,
  -0.753938734941,
  0.0286773243482,
  -0.305992467923,
  -0.888839283216,
  -0.53528439884,
  0.410004419777};

/* Quadrature weights */
static double weights[27] = {
  0.0701637994372,
  0.0653012061324,
  0.0133734490519,
  0.0800491405774,
  0.0745014590358,
  0.0152576273199,
  0.0243830167241,
  0.022693189565,
  0.0046474825267,
  0.1122620791,
  0.104481929812,
  0.021397518483,
  0.128078624924,
  0.119202334457,
  0.0244122037118,
  0.0390128267586,
  0.0363091033041,
  0.00743597204272,
  0.0701637994372,
  0.0653012061324,
  0.0133734490519,
  0.0800491405774,
  0.0745014590358,
  0.0152576273199,
  0.0243830167241,
  0.022693189565,
  0.0046474825267};

#define NUM_BASIS_FUNCTIONS 4

/* Nodal basis function evaluations */
static double Basis[108] = {
  0.749664528222,
  0.0952198798417,
  0.0821215678634,
  0.0729940240731,
  0.528074388273,
  0.0670742417521,
  0.0578476039361,
  0.347003766038,
  0.23856305665,
  0.0303014811743,
  0.0261332522867,
  0.705002209888,
  0.485731727037,
  0.0616960186091,
  0.379578230281,
  0.0729940240731,
  0.342156357896,
  0.0434595556538,
  0.267380320412,
  0.347003766038,
  0.154572667042,
  0.0196333029355,
  0.120791820134,
  0.705002209888,
  0.174656645238,
  0.0221843026408,
  0.730165028048,
  0.0729940240731,
  0.12303063253,
  0.0156269392579,
  0.514338662174,
  0.347003766038,
  0.0555803583921,
  0.00705963113955,
  0.23235780058,
  0.705002209888,
  0.422442204032,
  0.422442204032,
  0.0821215678634,
  0.0729940240731,
  0.297574315013,
  0.297574315013,
  0.0578476039361,
  0.347003766038,
  0.134432268912,
  0.134432268912,
  0.0261332522867,
  0.705002209888,
  0.273713872823,
  0.273713872823,
  0.379578230281,
  0.0729940240731,
  0.192807956775,
  0.192807956775,
  0.267380320412,
  0.347003766038,
  0.0871029849888,
  0.0871029849888,
  0.120791820134,
  0.705002209888,
  0.0984204739396,
  0.0984204739396,
  0.730165028048,
  0.0729940240731,
  0.0693287858938,
  0.0693287858938,
  0.514338662174,
  0.347003766038,
  0.0313199947658,
  0.0313199947658,
  0.23235780058,
  0.705002209888,
  0.0952198798417,
  0.749664528222,
  0.0821215678634,
  0.0729940240731,
  0.0670742417521,
  0.528074388273,
  0.0578476039361,
  0.347003766038,
  0.0303014811743,
  0.23856305665,
  0.0261332522867,
  0.705002209888,
  0.0616960186091,
  0.485731727037,
  0.379578230281,
  0.0729940240731,
  0.0434595556538,
  0.342156357896,
  0.267380320412,
  0.347003766038,
  0.0196333029355,
  0.154572667042,
  0.120791820134,
  0.705002209888,
  0.0221843026408,
  0.174656645238,
  0.730165028048,
  0.0729940240731,
  0.0156269392579,
  0.12303063253,
  0.514338662174,
  0.347003766038,
  0.00705963113955,
  0.0555803583921,
  0.23235780058,
  0.705002209888};

/* Nodal basis function derivative evaluations */
static double BasisDerivatives[324] = {
  -0.5,
  -0.5,
  -0.5,
  0.5,
  2.6964953901e-17,
  8.15881875835e-17,
  0.0,
  0.5,
  1.52600313755e-17,
  0.0,
  0.0,
  0.5,
  -0.5,
  -0.5,
  -0.5,
  0.5,
  2.6938349868e-17,
  1.08228622783e-16,
  0.0,
  0.5,
  -1.68114298577e-17,
  0.0,
  0.0,
  0.5,
  -0.5,
  -0.5,
  -0.5,
  0.5,
  2.69035912409e-17,
  1.43034809879e-16,
  0.0,
  0.5,
  -5.87133459397e-17,
  0.0,
  0.0,
  0.5,
  -0.5,
  -0.5,
  -0.5,
  0.5,
  2.6964953901e-17,
  2.8079494593e-17,
  0.0,
  0.5,
  1.52600313755e-17,
  0.0,
  0.0,
  0.5,
  -0.5,
  -0.5,
  -0.5,
  0.5,
  2.6938349868e-17,
  7.0536336094e-17,
  0.0,
  0.5,
  -1.68114298577e-17,
  0.0,
  0.0,
  0.5,
  -0.5,
  -0.5,
  -0.5,
  0.5,
  2.69035912409e-17,
  1.26006930238e-16,
  0.0,
  0.5,
  -5.87133459397e-17,
  0.0,
  0.0,
  0.5,
  -0.5,
  -0.5,
  -0.5,
  0.5,
  2.6964953901e-17,
  -3.49866380524e-17,
  0.0,
  0.5,
  1.52600313755e-17,
  0.0,
  0.0,
  0.5,
  -0.5,
  -0.5,
  -0.5,
  0.5,
  2.6938349868e-17,
  2.61116525673e-17,
  0.0,
  0.5,
  -1.68114298577e-17,
  0.0,
  0.0,
  0.5,
  -0.5,
  -0.5,
  -0.5,
  0.5,
  2.69035912409e-17,
  1.05937620823e-16,
  0.0,
  0.5,
  -5.87133459397e-17,
  0.0,
  0.0,
  0.5,
  -0.5,
  -0.5,
  -0.5,
  0.5,
  2.6964953901e-17,
  1.52807111565e-17,
  0.0,
  0.5,
  1.52600313755e-17,
  0.0,
  0.0,
  0.5,
  -0.5,
  -0.5,
  -0.5,
  0.5,
  2.6938349868e-17,
  6.1520690456e-17,
  0.0,
  0.5,
  -1.68114298577e-17,
  0.0,
  0.0,
  0.5,
  -0.5,
  -0.5,
  -0.5,
  0.5,
  2.69035912409e-17,
  1.21934019246e-16,
  0.0,
  0.5,
  -5.87133459397e-17,
  0.0,
  0.0,
  0.5,
  -0.5,
  -0.5,
  -0.5,
  0.5,
  2.6964953901e-17,
  -1.48832491782e-17,
  0.0,
  0.5,
  1.52600313755e-17,
  0.0,
  0.0,
  0.5,
  -0.5,
  -0.5,
  -0.5,
  0.5,
  2.6938349868e-17,
  4.0272766482e-17,
  0.0,
  0.5,
  -1.68114298577e-17,
  0.0,
  0.0,
  0.5,
  -0.5,
  -0.5,
  -0.5,
  0.5,
  2.69035912409e-17,
  1.1233505023e-16,
  0.0,
  0.5,
  -5.87133459397e-17,
  0.0,
  0.0,
  0.5,
  -0.5,
  -0.5,
  -0.5,
  0.5,
  2.6964953901e-17,
  -5.04349365259e-17,
  0.0,
  0.5,
  1.52600313755e-17,
  0.0,
  0.0,
  0.5,
  -0.5,
  -0.5,
  -0.5,
  0.5,
  2.6938349868e-17,
  1.52296507396e-17,
  0.0,
  0.5,
  -1.68114298577e-17,
  0.0,
  0.0,
  0.5,
  -0.5,
  -0.5,
  -0.5,
  0.5,
  2.69035912409e-17,
  1.01021564153e-16,
  0.0,
  0.5,
  -5.87133459397e-17,
  0.0,
  0.0,
  0.5,
  -0.5,
  -0.5,
  -0.5,
  0.5,
  2.6964953901e-17,
  -5.10267652705e-17,
  0.0,
  0.5,
  1.52600313755e-17,
  0.0,
  0.0,
  0.5,
  -0.5,
  -0.5,
  -0.5,
  0.5,
  2.6938349868e-17,
  1.4812758129e-17,
  0.0,
  0.5,
  -1.68114298577e-17,
  0.0,
  0.0,
  0.5,
  -0.5,
  -0.5,
  -0.5,
  0.5,
  2.69035912409e-17,
  1.00833228612e-16,
  0.0,
  0.5,
  -5.87133459397e-17,
  0.0,
  0.0,
  0.5,
  -0.5,
  -0.5,
  -0.5,
  0.5,
  2.6964953901e-17,
  -5.78459929494e-17,
  0.0,
  0.5,
  1.52600313755e-17,
  0.0,
  0.0,
  0.5,
  -0.5,
  -0.5,
  -0.5,
  0.5,
  2.6938349868e-17,
  1.00091968699e-17,
  0.0,
  0.5,
  -1.68114298577e-17,
  0.0,
  0.0,
  0.5,
  -0.5,
  -0.5,
  -0.5,
  0.5,
  2.69035912409e-17,
  9.86631702223e-17,
  0.0,
  0.5,
  -5.87133459397e-17,
  0.0,
  0.0,
  0.5,
  -0.5,
  -0.5,
  -0.5,
  0.5,
  2.6964953901e-17,
  -6.58832349994e-17,
  0.0,
  0.5,
  1.52600313755e-17,
  0.0,
  0.0,
  0.5,
  -0.5,
  -0.5,
  -0.5,
  0.5,
  2.6938349868e-17,
  4.34764891191e-18,
  0.0,
  0.5,
  -1.68114298577e-17,
  0.0,
  0.0,
  0.5,
  -0.5,
  -0.5,
  -0.5,
  0.5,
  2.69035912409e-17,
  9.61055074835e-17,
  0.0,
  0.5,
  -5.87133459397e-17,
  0.0,
  0.0,
  0.5};

#endif

#undef __FUNCT__
#define __FUNCT__ "ExpandIntervals"
PetscErrorCode ExpandIntervals(ALE::Obj<ALE::Point_array> intervals, PetscInt *indices)
{
  int k = 0;

  PetscFunctionBegin;
  for(ALE::Point_array::iterator i_itor = intervals->begin(); i_itor != intervals->end(); i_itor++) {
    for(int i = 0; i < (*i_itor).index; i++) {
      indices[k++] = (*i_itor).prefix + i;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ExpandSetIntervals"
PetscErrorCode ExpandSetIntervals(ALE::Point_set intervals, PetscInt *indices)
{
  int k = 0;

  PetscFunctionBegin;
  for(ALE::Point_set::iterator i_itor = intervals.begin(); i_itor != intervals.end(); i_itor++) {
    for(int i = 0; i < (*i_itor).index; i++) {
      indices[k++] = (*i_itor).prefix + i;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "restrictField"
PetscErrorCode restrictField(ALE::IndexBundle *bundle, ALE::PreSieve *orientation, PetscScalar *array, ALE::Point e, PetscScalar *values[])
{
  ALE::Point_set             empty;
  ALE::Obj<ALE::Point_array> intervals = bundle->getClosureIndices(orientation->cone(e), empty);
  //ALE::Obj<ALE::Point_array> intervals = bundle->getOverlapOrderedIndices(orientation->cone(e), empty);
  /* This should be done by memory pooling by array size (we have a simple form below) */
  static PetscScalar *vals;
  static PetscInt     numValues = 0;
  static PetscInt    *indices = NULL;
  PetscInt            numIndices = 0;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  for(ALE::Point_array::iterator i_itor = intervals->begin(); i_itor != intervals->end(); i_itor++) {
    numIndices += (*i_itor).index;
  }
  if (numValues && (numValues != numIndices)) {
    ierr = PetscFree(indices); CHKERRQ(ierr);
    indices = NULL;
    ierr = PetscFree(vals); CHKERRQ(ierr);
    vals = NULL;
  }
  if (!indices) {
    numValues = numIndices;
    ierr = PetscMalloc(numValues * sizeof(PetscInt), &indices); CHKERRQ(ierr);
    ierr = PetscMalloc(numValues * sizeof(PetscScalar), &vals); CHKERRQ(ierr);
  }
  for(ALE::Point_array::iterator i_itor = intervals->begin(); i_itor != intervals->end(); i_itor++) {
    printf("indices (%d, %d)\n", (*i_itor).prefix, (*i_itor).index);
  }
  ierr = ExpandIntervals(intervals, indices); CHKERRQ(ierr);
  for(int i = 0; i < numIndices; i++) {
    printf("indices[%d] = %d\n", i, indices[i]);
    vals[i] = array[indices[i]];
  }
  *values = vals;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "assembleField"
PetscErrorCode assembleField(ALE::IndexBundle *bundle, ALE::PreSieve *orientation, Vec b, ALE::Point e, PetscScalar array[], InsertMode mode)
{
  ALE::Point_set   empty;
  ALE::Obj<ALE::Point_array> intervals = bundle->getClosureIndices(orientation->cone(e), empty);
  //ALE::Obj<ALE::Point_array> intervals = bundle->getOverlapOrderedIndices(orientation->cone(e), empty);
  static PetscInt  indicesSize = 0;
  static PetscInt *indices = NULL;
  PetscInt         numIndices = 0;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  for(ALE::Point_array::iterator i_itor = intervals->begin(); i_itor != intervals->end(); i_itor++) {
    numIndices += (*i_itor).index;
  }
  if (indicesSize && (indicesSize != numIndices)) {
    ierr = PetscFree(indices); CHKERRQ(ierr);
    indices = NULL;
  }
  if (!indices) {
    indicesSize = numIndices;
    ierr = PetscMalloc(indicesSize * sizeof(PetscInt), &indices); CHKERRQ(ierr);
  }
  for(ALE::Point_array::iterator i_itor = intervals->begin(); i_itor != intervals->end(); i_itor++) {
    printf("indices (%d, %d)\n", (*i_itor).prefix, (*i_itor).index);
  }
  ierr = ExpandIntervals(intervals, indices); CHKERRQ(ierr);
  for(int i = 0; i < numIndices; i++) {
    printf("indices[%d] = %d\n", i, indices[i]);
  }
  ierr = VecSetValues(b, numIndices, indices, array, mode);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "assembleOperator"
PetscErrorCode assembleOperator(ALE::IndexBundle *bundle, ALE::PreSieve *orientation, Mat A, ALE::Point e, PetscScalar array[], InsertMode mode)
{
  ALE::Point_set   empty;
  ALE::Obj<ALE::Point_array> intervals = bundle->getClosureIndices(orientation->cone(e), empty);
  //ALE::Obj<ALE::Point_array> intervals = bundle->getOverlapOrderedIndices(orientation->cone(e), empty);
  static PetscInt  indicesSize = 0;
  static PetscInt *indices = NULL;
  PetscInt         numIndices = 0;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  for(ALE::Point_array::iterator i_itor = intervals->begin(); i_itor != intervals->end(); i_itor++) {
    numIndices += (*i_itor).index;
  }
  if (indicesSize && (indicesSize != numIndices)) {
    ierr = PetscFree(indices); CHKERRQ(ierr);
    indices = NULL;
  }
  if (!indices) {
    indicesSize = numIndices;
    ierr = PetscMalloc(indicesSize * sizeof(PetscInt), &indices); CHKERRQ(ierr);
  }
  for(ALE::Point_array::iterator i_itor = intervals->begin(); i_itor != intervals->end(); i_itor++) {
    printf("indices (%d, %d)\n", (*i_itor).prefix, (*i_itor).index);
  }
  ierr = ExpandIntervals(intervals, indices); CHKERRQ(ierr);
  for(int i = 0; i < numIndices; i++) {
    printf("indices[%d] = %d\n", i, indices[i]);
  }
  ierr = MatSetValues(A, numIndices, indices, numIndices, indices, array, mode);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateTestMesh"
/*
  CreateTestMesh - Create a simple square mesh

         13
  14--29----31---12
    |\    |\    |
    2 2 5 2 3 7 3
    7  6  8  0  2
    | 4 \ | 6 \ |
    |    \|    \|
  15--20-16-24---11
    |\    |\    |
    1 1 1 2 2 3 2
    8  7  1  2  5
    | 0 \ | 2 \ |
    |    \|    \|
   8--19----23---10
          9
*/
PetscErrorCode CreateTestMesh(MPI_Comm comm, Mesh *mesh)
{
  ALE::IndexBundle *bundle = new ALE::IndexBundle(comm);
  ALE::Sieve         *topology;
  PetscInt            dim = 2;
  PetscInt            faces[24] = {
    0, 1, 7,
    8, 7, 1,
    1, 2, 8,
    3, 8, 2,
    7, 8, 6,
    5, 6, 8,
    8, 3, 5,
    4, 5, 3};
  PetscScalar         vertexCoords[18] = {
    0.0, 0.0,
    1.0, 0.0,
    2.0, 0.0,
    2.0, 1.0,
    2.0, 2.0,
    1.0, 2.0,
    0.0, 2.0,
    0.0, 1.0,
    1.0, 1.0};
  PetscInt            boundaryVertices[8] = {
    0, 1, 2, 3, 4, 5, 6, 7};
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = MeshCreate(comm, mesh);CHKERRQ(ierr);
  ierr = MeshCreateTopology(*mesh, dim, 9, 8, faces);CHKERRQ(ierr);
  ierr = MeshCreateBoundary(*mesh, 8, boundaryVertices); CHKERRQ(ierr);
  /* Create field ordering */
  ierr = MeshGetTopology(*mesh, (void **) &topology);CHKERRQ(ierr);
  bundle->setTopology(topology);
  bundle->setFiberDimensionByDepth(0, 1);
  //bundle->computeGlobalIndices();
  ierr = MeshSetBundle(*mesh, (void *) bundle);CHKERRQ(ierr);
  /* Finish old-style DM construction */
  ALE::Point_set empty;
  ierr = MeshSetGhosts(*mesh, 1, bundle->getBundleDimension(empty), 0, NULL);
  /* Create coordinates */
  ierr = MeshCreateCoordinates(*mesh, vertexCoords);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateTestMesh3"
/*
  CreateTestMesh3 - Create a simple cubic mesh

  

*/
PetscErrorCode CreateTestMesh3(MPI_Comm comm, Mesh *mesh)
{
  ALE::IndexBundle *bundle = new ALE::IndexBundle(comm);
  ALE::Sieve         *topology;
  PetscInt            dim = 3;
  PetscInt            faces[24] = {
    0, 1, 3, 5,
    0, 5, 3, 4,
    4, 5, 3, 7,
    2, 3, 1, 5,
    3, 2, 7, 5,
    6, 7, 2, 5};
  PetscScalar         vertexCoords[24] = {
    0.0, 0.0, 0.0,
    1.0, 0.0, 0.0,
    1.0, 1.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0,
    1.0, 0.0, 1.0,
    1.0, 1.0, 1.0,
    0.0, 1.0, 1.0};
  PetscInt            boundaryVertices[8] = {
    0, 1, 2, 3, 4, 5, 6, 7};
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = MeshCreate(comm, mesh);CHKERRQ(ierr);
  ierr = MeshCreateTopology(*mesh, dim, 8, 6, faces);CHKERRQ(ierr);
  ierr = MeshCreateBoundary(*mesh, 8, boundaryVertices); CHKERRQ(ierr);
  /* Create field ordering */
  ierr = MeshGetTopology(*mesh, (void **) &topology);CHKERRQ(ierr);
  bundle->setTopology(topology);
  bundle->setFiberDimensionByDepth(0, 1);
  //bundle->computeGlobalIndices();
  ierr = MeshSetBundle(*mesh, (void *) bundle);CHKERRQ(ierr);
  /* Finish old-style DM construction */
  ALE::Point_set empty;
  ierr = MeshSetGhosts(*mesh, 1, bundle->getBundleDimension(empty), 0, NULL);
  /* Create coordinates */
  ierr = MeshCreateCoordinates(*mesh, vertexCoords);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ElementGeometry"
PetscErrorCode ElementGeometry(ALE::IndexBundle *coordBundle, ALE::PreSieve *orientation, PetscScalar *coords, ALE::Point e, PetscReal v0[], PetscReal J[], PetscReal invJ[], PetscReal *detJ)
{
  PetscInt       dim = coordBundle->getFiberDimension(*coordBundle->getTopology()->depthStratum(0).begin());
  PetscScalar   *array;
  PetscReal      det, invDet;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = restrictField(coordBundle, orientation, coords, e, &array); CHKERRQ(ierr);
  if (v0) {
    for(int d = 0; d < dim; d++) {
      v0[d] = array[d];
    }
  }
  if (J) {
    for(int d = 0; d < dim; d++) {
      for(int e = 0; e < dim; e++) {
        J[d*dim+e] = 0.5*(array[(e+1)*dim+d] - array[0*dim+d]);
      }
    }
    for(int d = 0; d < dim; d++) {
      if (d == 0) {
        printf("J = /");
      } else if (d == dim-1) {
        printf("    \\");
      } else {
        printf("    |");
      }
      for(int e = 0; e < dim; e++) {
        printf(" %g", J[d*dim+e]);
      }
      if (d == 0) {
        printf(" \\\n");
      } else if (d == dim-1) {
        printf(" /\n");
      } else {
        printf(" |\n");
      }
    }
    if (dim == 2) {
      det = fabs(J[0]*J[3] - J[1]*J[2]);
    } else if (dim == 3) {
      det = fabs(J[0*3+0]*(J[1*3+1]*J[2*3+2] - J[1*3+2]*J[2*3+1]) +
                 J[0*3+1]*(J[1*3+2]*J[2*3+0] - J[1*3+0]*J[2*3+2]) +
                 J[0*3+2]*(J[1*3+0]*J[2*3+1] - J[1*3+1]*J[2*3+0]));
    }
    invDet = 1.0/det;
    if (detJ) {
      *detJ = det;
    }
    if (invJ) {
      if (dim == 2) {
        invJ[0] =  invDet*J[3];
        invJ[1] = -invDet*J[2];
        invJ[2] = -invDet*J[1];
        invJ[3] =  invDet*J[0];
      } else if (dim == 3) {
        invJ[0*3+0] = invDet*(J[1*3+1]*J[2*3+2] - J[1*3+2]*J[2*3+1]);
        invJ[0*3+1] = invDet*(J[1*3+2]*J[2*3+0] - J[1*3+0]*J[2*3+2]);
        invJ[0*3+2] = invDet*(J[1*3+0]*J[2*3+1] - J[1*3+1]*J[2*3+0]);
        invJ[1*3+0] = invDet*(J[0*3+1]*J[2*3+2] - J[0*3+2]*J[2*3+1]);
        invJ[1*3+1] = invDet*(J[0*3+2]*J[2*3+0] - J[0*3+0]*J[2*3+2]);
        invJ[1*3+2] = invDet*(J[0*3+0]*J[2*3+1] - J[0*3+1]*J[2*3+0]);
        invJ[2*3+0] = invDet*(J[0*3+1]*J[1*3+2] - J[0*3+2]*J[1*3+1]);
        invJ[2*3+1] = invDet*(J[0*3+2]*J[1*3+0] - J[0*3+0]*J[1*3+2]);
        invJ[2*3+2] = invDet*(J[0*3+0]*J[1*3+1] - J[0*3+1]*J[1*3+0]);
      }
      for(int d = 0; d < dim; d++) {
        if (d == 0) {
          printf("Jinv = /");
        } else if (d == dim-1) {
          printf("       \\");
        } else {
          printf("       |");
        }
        for(int e = 0; e < dim; e++) {
          printf(" %g", invJ[d*dim+e]);
        }
        if (d == 0) {
          printf(" \\\n");
        } else if (d == dim-1) {
          printf(" /\n");
        } else {
          printf(" |\n");
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ComputeRho"
PetscErrorCode ComputeRho(PetscReal x, PetscReal y, PetscScalar *rho)
{
  PetscFunctionBegin;
  if ((x > 1.0/3.0) && (x < 2.0/3.0) && (y > 1.0/3.0) && (y < 2.0/3.0)) {
    //*rho = 100.0;
    *rho = 1.0;
  } else {
    *rho = 1.0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ComputeBlock"
PetscErrorCode ComputeBlock(DMMG dmmg, Vec u, Vec r, ALE::Point_set block)
{
  Mesh                mesh = (Mesh) dmmg->dm;
  UserContext        *user = (UserContext *) dmmg->user;
  ALE::Sieve         *topology;
  ALE::PreSieve      *orientation;
  ALE::IndexBundle *bundle;
  ALE::IndexBundle *coordBundle;
  ALE::Point_set      elements;
  ALE::Point_set      empty;
  PetscInt            dim;
  Vec                 coordinates;
  PetscScalar        *coords;
  PetscScalar        *array;
  PetscScalar        *field;
  PetscReal           elementVec[NUM_BASIS_FUNCTIONS];
  PetscReal           linearVec[NUM_BASIS_FUNCTIONS];
  PetscReal           *v0, *Jac, *Jinv, *t_der, *b_der;
  PetscReal           xi, eta, x_q, y_q, detJ, rho, funcValue;
  PetscInt            f, g, q;
  PetscErrorCode      ierr;

  ierr = MeshGetTopology(mesh, (void **) &topology);CHKERRQ(ierr);
  ierr = MeshGetOrientation(mesh, (void **) &orientation);CHKERRQ(ierr);
  ierr = MeshGetBundle(mesh, (void **) &bundle);CHKERRQ(ierr);
  ierr = MeshGetCoordinateBundle(mesh, (void **) &coordBundle);CHKERRQ(ierr);
  ierr = MeshGetCoordinates(mesh, &coordinates);CHKERRQ(ierr);
  ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
  ierr = VecGetArray(u, &array); CHKERRQ(ierr);
  dim = coordBundle->getFiberDimension(*coordBundle->getTopology()->depthStratum(0).begin());
  ierr = PetscMalloc(dim * sizeof(PetscReal), &v0);CHKERRQ(ierr);
  ierr = PetscMalloc(dim * sizeof(PetscReal), &t_der);CHKERRQ(ierr);
  ierr = PetscMalloc(dim * sizeof(PetscReal), &b_der);CHKERRQ(ierr);
  ierr = PetscMalloc(dim*dim * sizeof(PetscReal), &Jac);CHKERRQ(ierr);
  ierr = PetscMalloc(dim*dim * sizeof(PetscReal), &Jinv);CHKERRQ(ierr);
  elements = topology->star(block);
  for(ALE::Point_set::iterator element_itor = elements.begin(); element_itor != elements.end(); element_itor++) {
    ALE::Point e = *element_itor;

    ierr = ElementGeometry(coordBundle, orientation, coords, e, v0, Jac, Jinv, &detJ); CHKERRQ(ierr);
    ierr = restrictField(bundle, orientation, array, e, &field); CHKERRQ(ierr);
    /* Element integral */
    ierr = PetscMemzero(elementVec, NUM_BASIS_FUNCTIONS*sizeof(PetscScalar));CHKERRQ(ierr);
    for(q = 0; q < NUM_QUADRATURE_POINTS; q++) {
      xi = points[q*2+0] + 1.0;
      eta = points[q*2+1] + 1.0;
      x_q = Jac[0]*xi + Jac[1]*eta + v0[0];
      y_q = Jac[2]*xi + Jac[3]*eta + v0[1];
      ierr = ComputeRho(x_q, y_q, &rho);CHKERRQ(ierr);
      funcValue = PetscExpScalar(-(x_q*x_q)/user->nu)*PetscExpScalar(-(y_q*y_q)/user->nu);
      for(f = 0; f < NUM_BASIS_FUNCTIONS; f++) {
        t_der[0] = Jinv[0]*BasisDerivatives[(q*NUM_BASIS_FUNCTIONS+f)*2+0] + Jinv[2]*BasisDerivatives[(q*NUM_BASIS_FUNCTIONS+f)*2+1];
        t_der[1] = Jinv[1]*BasisDerivatives[(q*NUM_BASIS_FUNCTIONS+f)*2+0] + Jinv[3]*BasisDerivatives[(q*NUM_BASIS_FUNCTIONS+f)*2+1];
        for(g = 0; g < NUM_BASIS_FUNCTIONS; g++) {
          b_der[0] = Jinv[0]*BasisDerivatives[(q*NUM_BASIS_FUNCTIONS+g)*2+0] + Jinv[2]*BasisDerivatives[(q*NUM_BASIS_FUNCTIONS+g)*2+1];
          b_der[1] = Jinv[1]*BasisDerivatives[(q*NUM_BASIS_FUNCTIONS+g)*2+0] + Jinv[3]*BasisDerivatives[(q*NUM_BASIS_FUNCTIONS+g)*2+1];
          linearVec[f] += rho*(t_der[0]*b_der[0] + t_der[1]*b_der[1])*field[g];
        }
        elementVec[f] += (Basis[q*NUM_BASIS_FUNCTIONS+f]*funcValue - linearVec[f])*weights[q]*detJ;
      }
    }
    printf("elementVec = [%g %g %g]\n", elementVec[0], elementVec[1], elementVec[2]);
    /* Assembly */
    ierr = assembleField(bundle, orientation, r, e, elementVec, ADD_VALUES); CHKERRQ(ierr);
  }
  ierr = PetscFree(v0);CHKERRQ(ierr);
  ierr = PetscFree(t_der);CHKERRQ(ierr);
  ierr = PetscFree(b_der);CHKERRQ(ierr);
  ierr = PetscFree(Jac);CHKERRQ(ierr);
  ierr = PetscFree(Jinv);CHKERRQ(ierr);
  ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
  ierr = VecRestoreArray(u, &array); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ComputeRHS"
PetscErrorCode ComputeRHS(DMMG dmmg, Vec b)
{
  Mesh                mesh = (Mesh) dmmg->dm;
  UserContext        *user = (UserContext *) dmmg->user;
  ALE::Sieve         *topology;
  ALE::PreSieve      *orientation;
  ALE::IndexBundle *bundle;
  ALE::IndexBundle *coordBundle;
  ALE::Point_set      elements;
  ALE::Point_set      empty;
  PetscInt            dim;
  Vec                 coordinates;
  PetscScalar        *coords;
  PetscReal           elementVec[NUM_BASIS_FUNCTIONS];
  PetscReal           *v0, *Jac;
  PetscReal           xi, eta, x_q, y_q, detJ, funcValue;
  PetscInt            f, q;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = MeshGetTopology(mesh, (void **) &topology);CHKERRQ(ierr);
  ierr = MeshGetOrientation(mesh, (void **) &orientation);CHKERRQ(ierr);
  ierr = MeshGetBundle(mesh, (void **) &bundle);CHKERRQ(ierr);
  ierr = MeshGetCoordinateBundle(mesh, (void **) &coordBundle);CHKERRQ(ierr);
  ierr = MeshGetCoordinates(mesh, &coordinates);CHKERRQ(ierr);
  ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
  dim = coordBundle->getFiberDimension(*coordBundle->getTopology()->depthStratum(0).begin());
  ierr = PetscMalloc(dim * sizeof(PetscReal), &v0);CHKERRQ(ierr);
  ierr = PetscMalloc(dim*dim * sizeof(PetscReal), &Jac);CHKERRQ(ierr);
  elements = topology->heightStratum(0);
  for(ALE::Point_set::iterator element_itor = elements.begin(); element_itor != elements.end(); element_itor++) {
    ALE::Point e = *element_itor;

    ierr = ElementGeometry(coordBundle, orientation, coords, e, v0, Jac, PETSC_NULL, &detJ); CHKERRQ(ierr);
    /* Element integral */
    ierr = PetscMemzero(elementVec, NUM_BASIS_FUNCTIONS*sizeof(PetscScalar));CHKERRQ(ierr);
    for(q = 0; q < NUM_QUADRATURE_POINTS; q++) {
      xi = points[q*2+0] + 1.0;
      eta = points[q*2+1] + 1.0;
      x_q = Jac[0]*xi + Jac[1]*eta + v0[0];
      y_q = Jac[2]*xi + Jac[3]*eta + v0[1];
      funcValue = PetscExpScalar(-(x_q*x_q)/user->nu)*PetscExpScalar(-(y_q*y_q)/user->nu);
      for(f = 0; f < NUM_BASIS_FUNCTIONS; f++) {
        elementVec[f] += Basis[q*NUM_BASIS_FUNCTIONS+f]*funcValue*weights[q]*detJ;
      }
    }
    printf("elementVec = [%g %g %g]\n", elementVec[0], elementVec[1], elementVec[2]);
    /* Assembly */
    ierr = assembleField(bundle, orientation, b, e, elementVec, ADD_VALUES); CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
  ierr = PetscFree(v0);CHKERRQ(ierr);
  ierr = PetscFree(Jac);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(b);CHKERRQ(ierr);

  /* force right hand side to be consistent for singular matrix */
  /* note this is really a hack, normally the model would provide you with a consistent right handside */
  if (user->bcType == NEUMANN) {
    MatNullSpace nullspace;

    ierr = KSPGetNullSpace(dmmg->ksp,&nullspace);CHKERRQ(ierr);
    ierr = MatNullSpaceRemove(nullspace,b,PETSC_NULL);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ComputeJacobian"
PetscErrorCode ComputeJacobian(DMMG dmmg, Mat J, Mat jac)
{
  Mesh                mesh = (Mesh) dmmg->dm;
  UserContext        *user = (UserContext *) dmmg->user;
  ALE::Sieve         *topology;
  ALE::Sieve         *boundary;
  ALE::PreSieve      *orientation;
  ALE::IndexBundle *bundle;
  ALE::IndexBundle *coordBundle;
  ALE::Point_set      elements;
  ALE::Point_set      empty;
  PetscInt            dim;
  Vec                 coordinates;
  PetscScalar        *coords;
  PetscReal           elementMat[NUM_BASIS_FUNCTIONS*NUM_BASIS_FUNCTIONS];
  PetscReal           *v0, *Jac, *Jinv, *t_der, *b_der;
  PetscReal           xi, eta, x_q, y_q, detJ, rho;
  PetscInt            f, g, q;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = MeshGetTopology(mesh, (void **) &topology);CHKERRQ(ierr);
  ierr = MeshGetBoundary(mesh, (void **) &boundary);CHKERRQ(ierr);
  ierr = MeshGetOrientation(mesh, (void **) &orientation);CHKERRQ(ierr);
  ierr = MeshGetBundle(mesh, (void **) &bundle);CHKERRQ(ierr);
  ierr = MeshGetCoordinateBundle(mesh, (void **) &coordBundle);CHKERRQ(ierr);
  ierr = MeshGetCoordinates(mesh, &coordinates);CHKERRQ(ierr);
  ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
  dim = coordBundle->getFiberDimension(*coordBundle->getTopology()->depthStratum(0).begin());
  ierr = PetscMalloc(dim * sizeof(PetscReal), &v0);CHKERRQ(ierr);
  ierr = PetscMalloc(dim * sizeof(PetscReal), &t_der);CHKERRQ(ierr);
  ierr = PetscMalloc(dim * sizeof(PetscReal), &b_der);CHKERRQ(ierr);
  ierr = PetscMalloc(dim*dim * sizeof(PetscReal), &Jac);CHKERRQ(ierr);
  ierr = PetscMalloc(dim*dim * sizeof(PetscReal), &Jinv);CHKERRQ(ierr);
  elements = topology->heightStratum(0);
  for(ALE::Point_set::iterator element_itor = elements.begin(); element_itor != elements.end(); element_itor++) {
    ALE::Point e = *element_itor;

    CHKMEMQ;
    ierr = ElementGeometry(coordBundle, orientation, coords, e, v0, Jac, Jinv, &detJ); CHKERRQ(ierr);
    /* Element integral */
    ierr = PetscMemzero(elementMat, NUM_BASIS_FUNCTIONS*NUM_BASIS_FUNCTIONS*sizeof(PetscScalar));CHKERRQ(ierr);
    for(q = 0; q < NUM_QUADRATURE_POINTS; q++) {
      xi = points[q*2+0] + 1.0;
      eta = points[q*2+1] + 1.0;
      x_q = Jac[0]*xi + Jac[1]*eta + v0[0];
      y_q = Jac[2]*xi + Jac[3]*eta + v0[1];
      ierr = ComputeRho(x_q, y_q, &rho);CHKERRQ(ierr);
      for(f = 0; f < NUM_BASIS_FUNCTIONS; f++) {
        t_der[0] = Jinv[0]*BasisDerivatives[(q*NUM_BASIS_FUNCTIONS+f)*2+0] + Jinv[2]*BasisDerivatives[(q*NUM_BASIS_FUNCTIONS+f)*2+1];
        t_der[1] = Jinv[1]*BasisDerivatives[(q*NUM_BASIS_FUNCTIONS+f)*2+0] + Jinv[3]*BasisDerivatives[(q*NUM_BASIS_FUNCTIONS+f)*2+1];
        for(g = 0; g < NUM_BASIS_FUNCTIONS; g++) {
          b_der[0] = Jinv[0]*BasisDerivatives[(q*NUM_BASIS_FUNCTIONS+g)*2+0] + Jinv[2]*BasisDerivatives[(q*NUM_BASIS_FUNCTIONS+g)*2+1];
          b_der[1] = Jinv[1]*BasisDerivatives[(q*NUM_BASIS_FUNCTIONS+g)*2+0] + Jinv[3]*BasisDerivatives[(q*NUM_BASIS_FUNCTIONS+g)*2+1];
          elementMat[f*NUM_BASIS_FUNCTIONS+g] += rho*(t_der[0]*b_der[0] + t_der[1]*b_der[1])*weights[q]*detJ;
        }
      }
    }
    printf("elementMat = [%g %g %g]\n             [%g %g %g]\n             [%g %g %g]\n",
           elementMat[0], elementMat[1], elementMat[2], elementMat[3], elementMat[4], elementMat[5], elementMat[6], elementMat[7], elementMat[8]);
    /* Assembly */
    ierr = assembleOperator(bundle, orientation, jac, e, elementMat, ADD_VALUES); CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
  ierr = PetscFree(v0);CHKERRQ(ierr);
  ierr = PetscFree(t_der);CHKERRQ(ierr);
  ierr = PetscFree(b_der);CHKERRQ(ierr);
  ierr = PetscFree(Jac);CHKERRQ(ierr);
  ierr = PetscFree(Jinv);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (user->bcType == DIRICHLET) {
    /* Zero out BC rows */
    ALE::Point id(0, 1);
    ALE::Point_set boundaryElements = boundary->cone(id);
    int numBoundaryIndices = bundle->getFiberDimension(boundaryElements);
    ALE::Point_set boundaryIntervals = bundle->getFiberIndices(boundaryElements, empty)->cap();
    PetscInt *boundaryIndices;

    ierr = PetscMalloc(numBoundaryIndices * sizeof(PetscInt), &boundaryIndices); CHKERRQ(ierr);
    ierr = ExpandSetIntervals(boundaryIntervals, boundaryIndices); CHKERRQ(ierr);
    for(int i = 0; i < numBoundaryIndices; i++) {
      printf("boundaryIndices[%d] = %d\n", i, boundaryIndices[i]);
    }
    ierr = MatZeroRows(jac, numBoundaryIndices, boundaryIndices, 1.0);CHKERRQ(ierr);
    ierr = PetscFree(boundaryIndices);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
