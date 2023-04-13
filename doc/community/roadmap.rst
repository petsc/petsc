.. _roadmap:

*************
PETSc Roadmap
*************

Planned major focus areas for PETSc development include the following.

* Continue implementing advanced algorithms to provide state-of-the-art problem-solving infrastructure for large-scale nonlinear problems.

* GPU support including NVIDIA, AMD, and Intel systems.

* Differentiable software. That is more support in all components of PETSc and PETSc simulations for providing and conveying
  derivative information and making it available to the user and machine learning software such as PyTorch, JAX, and TensorFlow.
  This can include algorithmic (e.g., ``TSAdjoint``) and automatic differentiation.

* Batching. Efficiently solving structurally similar problems together using potentially multiple levels of parallelism.

* Enhanced language bindings.

  * Easier Python usage including transparent interoperability with **NumPy**, PyTorch, JAX, and TensorFlow.

  * Julia bindings.
  
  * Rust bindings.
