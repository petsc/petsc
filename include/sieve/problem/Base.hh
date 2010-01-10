#ifndef included_ALE_Problem_Base_hh
#define included_ALE_Problem_Base_hh

#include <DMBuilder.hh>

#include <petscmesh_viewers.hh>
#include <petscdmmg.h>

namespace ALE {
  namespace Problem {
    typedef enum {RUN_FULL, RUN_TEST, RUN_MESH} RunType;
    typedef enum {NEUMANN, DIRICHLET} BCType;
    typedef enum {ASSEMBLY_FULL, ASSEMBLY_STORED, ASSEMBLY_CALCULATED} AssemblyType;
    typedef union {SectionReal section; Vec vec;} ExactSolType;

#if 0
    namespace Functions {
      extern PetscScalar lambda;

      PetscScalar zero(const double x[]);
      PetscScalar constant(const double x[]);
      PetscScalar nonlinear_2d(const double x[]);
      PetscScalar singularity_2d(const double x[]);
      PetscScalar singularity_exact_2d(const double x[]);
      PetscScalar singularity_exact_3d(const double x[]);
      PetscScalar singularity_3d(const double x[]);
      PetscScalar linear_2d(const double x[]);
      PetscScalar quadratic_2d(const double x[]);
      PetscScalar cubic_2d(const double x[]);
      PetscScalar nonlinear_3d(const double x[]);
      PetscScalar linear_3d(const double x[]);
      PetscScalar quadratic_3d(const double x[]);
      PetscScalar cubic_3d(const double x[]);
      PetscScalar cos_x(const double x[]);

      PetscScalar linear_2d_bem(const double x[]);
      PetscScalar linear_nder_2d(const double x[]);
      PetscScalar quadratic_nder_2d(const double x[]);

      PetscErrorCode Function_Structured_2d(DALocalInfo *info, PetscScalar *x[], PetscScalar *f[], void *ctx);
      PetscErrorCode Rhs_Structured_2d_FD(DALocalInfo *info, PetscScalar *x[], PetscScalar *f[], void *ctx);
      PetscErrorCode Jac_Structured_2d_FD(DALocalInfo *info, PetscScalar *x[], Mat J, void *ctx);
      PetscErrorCode Function_Structured_3d(DALocalInfo *info, PetscScalar **x[], PetscScalar **f[], void *ctx);
      PetscErrorCode Rhs_Structured_3d_FD(DALocalInfo *info, PetscScalar **x[], PetscScalar **f[], void *ctx);
      PetscErrorCode Jac_Structured_3d_FD(DALocalInfo *info, PetscScalar **x[], Mat J, void *ctx);
      PetscErrorCode Rhs_Unstructured(::Mesh mesh, SectionReal X, SectionReal section, void *ctx);
      PetscErrorCode Jac_Unstructured(::Mesh mesh, SectionReal section, Mat A, void *ctx);

      PetscErrorCode PointEvaluation(::Mesh mesh, SectionReal X, double coordsx[], double detJx, PetscScalar elemVec[]);
      PetscErrorCode RhsBd_Unstructured(::Mesh mesh, SectionReal X, SectionReal section, void *ctx);
      PetscErrorCode JacBd_Unstructured(::Mesh mesh, SectionReal section, Mat M, void *ctx);
    }
#endif

    typedef struct {
      PetscInt      debug;                       // The debugging level
      RunType       run;                         // The run type
      PetscInt      dim;                         // The topological mesh dimension
      PetscTruth    reentrantMesh;               // Generate a reentrant mesh?
      PetscTruth    circularMesh;                // Generate a circular mesh?
      PetscTruth    refineSingularity;           // Generate an a priori graded mesh for the poisson problem
      PetscTruth    structured;                  // Use a structured mesh
      PetscTruth    generateMesh;                // Generate the unstructure mesh
      PetscTruth    interpolate;                 // Generate intermediate mesh elements
      PetscReal     refinementLimit;             // The largest allowable cell volume
      char          baseFilename[2048];          // The base filename for mesh files
      char          partitioner[2048];           // The graph partitioner
      PetscScalar (*func)(const double []);      // The function to project
      BCType        bcType;                      // The type of boundary conditions
      PetscScalar (*exactFunc)(const double []); // The exact solution function
      ExactSolType  exactSol;                    // The discrete exact solution
      ExactSolType  error;                       // The discrete cell-wise error
      AssemblyType  operatorAssembly;            // The type of operator assembly 
      double (*integrate)(const double *, const double *, const int, double (*)(const double *)); // Basis functional application
      double        lambda;                      // The parameter controlling nonlinearity
      double        reentrant_angle;              // The angle for the reentrant corner.
    } BratuOptions;

    typedef struct {
      PetscInt      debug;                       // The debugging level
      RunType       run;                         // The run type
      PetscInt      dim;                         // The topological mesh dimension
      PetscTruth    reentrantMesh;               // Generate a reentrant mesh?
      PetscTruth    circularMesh;                // Generate a circular mesh?
      PetscTruth    refineSingularity;           // Generate an a priori graded mesh for the poisson problem
      PetscTruth    structured;                  // Use a structured mesh
      PetscTruth    generateMesh;                // Generate the unstructure mesh
      PetscTruth    interpolate;                 // Generate intermediate mesh elements
      PetscReal     refinementLimit;             // The largest allowable cell volume
      char          baseFilename[2048];          // The base filename for mesh files
      char          partitioner[2048];           // The graph partitioner
      PetscScalar (*func)(const double []);      // The function to project
      BCType        bcType;                      // The type of boundary conditions
      PetscScalar (*exactDirichletFunc)(const double []); // The exact solution function for Dirichlet data
      PetscScalar (*exactNeumannFunc)(const double []);   // The exact solution function for Neumann data
      ExactSolType  exactSol;                    // The discrete exact solution
      ExactSolType  error;                       // The discrete cell-wise error
      AssemblyType  operatorAssembly;            // The type of operator assembly 
      double (*integrate)(const double *, const double *, const int, double (*)(const double *)); // Basis functional application
      double        lambda;                      // The parameter controlling nonlinearity
      double        reentrant_angle;             // The angle for the reentrant corner.
      PetscScalar   phiCoefficient;              // Coefficient C for phi = {0 in interior, 0.5 on smooth boundary}
    } LaplaceBEMOptions;
  }
}

#endif
