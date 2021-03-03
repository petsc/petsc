.. _doc_taosolve:

======================
Summary of Tao Solvers
======================

Unconstrained
=============

.. list-table::
   :widths: auto
   :align: center
   :header-rows: 1

   * - Algorithm
     - Associated Type
     - Objective
     - Gradient
     - Hessian
     - Constraints
     - Jacobian
   * - Nelder-Mead
     - ``TAONM``
     - X
     -
     -
     -
     -
   * - Conjugate Gradient
     - ``TAOCG``
     - X
     - X
     -
     -
     -
   * - Limited Memory Variable Metric (quasi-Newton)
     - ``TAOLMVM``
     - X
     - X
     -
     -
     -
   * - Orthant-wise Limited Memory (quasi-Newton)
     - ``TAOOWLQN``
     - X
     - X
     -
     -
     -
   * - Bundle Method for Regularized Risk Minimization
     - ``TAOBMRM``
     - X
     - X
     -
     -
     -
   * - Newton Line Search
     - ``TAONLS``
     - X
     - X
     - X
     -
     -
   * - Newton Trust Region
     - ``TAONTR``
     - X
     - X
     - X
     -
     -

----------------------------

Bound Constrained
=================

.. list-table::
   :widths: auto
   :align: center
   :header-rows: 1

   * - Algorithm
     - Associated Type
     - Objective
     - Gradient
     - Hessian
     - Constraints
     - Jacobian
     - Constraint Type
   * - Bounded Conjugate Gradient
     - ``TAOBNCG``
     - X
     - X
     -
     -
     -
     - Box constraints
   * - Bounded Limited Memory Variable Metric (Quasi-Newton)
     - ``TAOBLMVM``
     - X
     - X
     -
     -
     -
     - Box constraints
   * - Bounded Quasi-Newton Line Search
     - ``TAOBQNLS``
     - X
     - X
     -
     -
     -
     - Box constraints
   * - Bounded Newton Line Search
     - ``TAOBNLS``
     - X
     - X
     -
     -
     -
     - Box constraints
   * - Bounded Newton Trust-Region
     - ``TAOBNTR``
     - X
     - X
     -
     -
     -
     - Box constraints
   * - Gradient Projection Conjugate Gradient
     - ``TAOGPCG``
     - X
     - X
     -
     -
     -
     - Box constraints
   * - Bounded Quadratic Interior Point
     - ``TAOBQPIP``
     - X
     - X
     -
     -
     -
     - Box constraints
   * - Tron
     - ``TAOTRON``
     - X
     - X
     - X
     -
     -
     - Box constraints

----------------------------

Complementarity
===============

.. list-table::
   :widths: auto
   :align: center
   :header-rows: 1

   * - Algorithm
     - Associated Type
     - Objective
     - Gradient
     - Hessian
     - Constraints
     - Jacobian
     - Constraint Type
   * - Active-Set Feasible Line Search
     - ``TAOASFLS``
     -
     -
     -
     - X
     - X
     - Complementarity
   * - Active-Set Infeasible Line Search
     - ``TAOASILS``
     -
     -
     -
     - X
     - X
     - Complementarity
   * - Semismooth Feasible Line Search
     - ``TAOSSFLS``
     -
     -
     -
     - X
     - X
     - Complementarity
   * - Semismooth Infeasible Line Searchx
     - ``TAOSSILS``
     -
     -
     -
     - X
     - X
     - Complementarity

----------------------------

Nonlinear Least Squares
=======================

.. list-table::
   :widths: auto
   :align: center
   :header-rows: 1

   * - Algorithm
     - Associated Type
     - Objective
     - Gradient
     - Hessian
     - Constraints
     - Jacobian
     - Constraint Type
   * - POUNDERS
     - ``TAOPOUNDERS``
     - X
     -
     -
     -
     -
     - Box Constraints

----------------------------

PDE-Constrained
===============

.. list-table::
   :widths: auto
   :align: center
   :header-rows: 1

   * - Algorithm
     - Associated Type
     - Objective
     - Gradient
     - Hessian
     - Constraints
     - Jacobian
     - Constraint Type
   * - Linearly Constrained Lagrangian
     - ``TAOLCL``
     - X
     - X
     - X
     - X
     - X
     - PDE Constraints

----------------------------

Constrained
===========

.. list-table::
   :widths: auto
   :align: center
   :header-rows: 1

   * - Algorithm
     - Associated Type
     - Objective
     - Gradient
     - Hessian
     - Constraints
     - Jacobian
     - Constraint Type
   * - Interior Point Method
     - ``TAOIPM``
     - X
     - X
     - X
     - X
     - X
     - General Constraints
   * - Barrier-Based Primal-Dual Interior Point
     - ``TAOPDIPM``
     - X
     - X
     - X
     - X
     - X
     - General Constraints
