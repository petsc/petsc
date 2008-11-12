#include <petscmg.h>      /*I      "petscmg.h"    I*/
#include <petscdmmg.h>    /*I      "petscdmmg.h"  I*/
#include <petscmesh.h>    /*I      "petscmesh.h"  I*/
#include <Selection.hh>

/* Just to set iterations */
#include "private/snesimpl.h"      /*I "petscsnes.h"  I*/

PetscErrorCode DMMGFormFunctionMesh(SNES snes, Vec X, Vec F, void *ptr);

#if 0
PetscErrorCode CreateNullSpace(DMMG dmmg, Vec *nulls) {
  Mesh           mesh = (Mesh) dmmg->dm;
  Vec            nS   = nulls[0];
  SectionReal    nullSpace;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshGetSectionReal(mesh, "nullSpace", &nullSpace);CHKERRQ(ierr);
  {
    ALE::Obj<PETSC_MESH_TYPE> m;
    ALE::Obj<PETSC_MESH_TYPE::real_section_type> s;

    ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
    ierr = SectionRealGetSection(nullSpace, s);CHKERRQ(ierr);
    ALE::Obj<ALE::Discretization> disc = m->getDiscretization("p");
    const int dim = m->getDimension();

    for(int d = 0; d <= dim; ++d) {
      const int numDof = disc->getNumDof(d);

      if (numDof) {
        const ALE::Obj<PETSC_MESH_TYPE::label_sequence>& stratum = m->depthStratum(d);
        const PETSC_MESH_TYPE::label_sequence::iterator  end     = stratum->end();
        double                                    *values  = new double[numDof];

        for(PETSC_MESH_TYPE::label_sequence::iterator p_iter = stratum->begin(); p_iter != end; ++p_iter) {
          for(int i = 0; i < numDof; ++i) values[i] = 1.0;
          s->updatePoint(*p_iter, values);
        }
      }
    }
  }
  ierr = SectionRealToVec(nullSpace, mesh, SCATTER_FORWARD, nS);CHKERRQ(ierr);
  std::cout << "Null space:" << std::endl;
  ierr = VecView(nS, PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = SectionRealDestroy(nullSpace);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

/* Nonlinear relaxation on all the equations with an initial guess in x */
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "Relax_Mesh"
PetscErrorCode PETSCMAT_DLLEXPORT Relax_Mesh(DMMG *dmmg, Mesh mesh, MatSORType flag, int its, Vec X, Vec B)
{
  SectionReal      sectionX, sectionB, cellX;
  Mesh             smallMesh;
  DMMG            *smallDmmg;
  DALocalFunction1 func;
  DALocalFunction1 jac;
  ALE::Obj<PETSC_MESH_TYPE> m;
  ALE::Obj<PETSC_MESH_TYPE::real_section_type> sX;
  ALE::Obj<PETSC_MESH_TYPE::real_section_type> sB;
  PetscTruth       fasDebug;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHasName(dmmg[0]->prefix, "-dmmg_fas_debug", &fasDebug);CHKERRQ(ierr);
  if (fasDebug) {ierr = PetscPrintf(dmmg[0]->comm, "  FAS mesh relaxation\n");CHKERRQ(ierr);}
  if (its <= 0) SETERRQ1(PETSC_ERR_ARG_WRONG, "Relaxation requires global its %D positive", its);
  ierr = MeshCreate(PETSC_COMM_SELF, &smallMesh);CHKERRQ(ierr);
  ierr = DMMGCreate(PETSC_COMM_SELF, -1, PETSC_NULL, &smallDmmg);CHKERRQ(ierr);
  //ierr = DMMGSetMatType(smallDmmg, MATSEQDENSE);CHKERRQ(ierr);
  ierr = DMMGSetOptionsPrefix(smallDmmg, "fas_");CHKERRQ(ierr);
  ierr = DMMGSetUser(smallDmmg, 0, DMMGGetUser(dmmg, 0));CHKERRQ(ierr);
  ierr = DMMGGetSNESLocal(dmmg, &func, &jac);CHKERRQ(ierr);
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  ierr = MeshGetSectionReal(mesh, "default", &sectionX);CHKERRQ(ierr);
  ierr = SectionRealToVec(sectionX, mesh, SCATTER_REVERSE, X);CHKERRQ(ierr);
  ierr = SectionRealGetSection(sectionX, sX);CHKERRQ(ierr);
  ierr = MeshGetSectionReal(mesh, "constant", &sectionB);CHKERRQ(ierr);
  ierr = SectionRealToVec(sectionB, mesh, SCATTER_REVERSE, B);CHKERRQ(ierr);
  ierr = SectionRealGetSection(sectionB, sB);CHKERRQ(ierr);
  ierr = SectionRealCreate(PETSC_COMM_SELF, &cellX);CHKERRQ(ierr);
  //const ALE::Obj<PETSC_MESH_TYPE::sieve_type>&     sieve   = m->getSieve();
  //const ALE::Obj<PETSC_MESH_TYPE::label_sequence>& cells   = m->heightStratum(0);
  //const int                                  depth   = m->depth();
  //const ALE::Obj<PETSC_MESH_TYPE::label_type>&     marker  = m->getLabel("marker");
  //const int                                  cellDof = m->sizeWithBC(sX, *cells->begin());

#ifdef PETSC_OPT_SIEVE
  SETERRQ(PETSC_ERR_SUP, "I am being lazy, bug me.");
#else
  ALE::Obj<PETSC_MESH_TYPE::names_type> fields = m->getDiscretizations();
  std::map<std::string, ALE::Obj<ALE::Discretization> > sDiscs;

  for(PETSC_MESH_TYPE::names_type::iterator f_iter = fields->begin(); f_iter != fields->end(); ++f_iter) {
    const ALE::Obj<ALE::Discretization>& disc  = m->getDiscretization(*f_iter);
    ALE::Obj<ALE::Discretization>        sDisc = new ALE::Discretization(disc->comm(), disc->debug());

    sDisc->setQuadratureSize(disc->getQuadratureSize());
    sDisc->setQuadraturePoints(disc->getQuadraturePoints());
    sDisc->setQuadratureWeights(disc->getQuadratureWeights());
    sDisc->setBasisSize(disc->getBasisSize());
    sDisc->setBasis(disc->getBasis());
    sDisc->setBasisDerivatives(disc->getBasisDerivatives());
    for(int d = 0; d <= m->getDimension(); ++d) {
      sDisc->setNumDof(d, disc->getNumDof(d));
      sDisc->setDofClass(d, disc->getDofClass(d));
    }
    if (disc->getBoundaryConditions()->size()) {
      if (fasDebug) {std::cout << "Adding BC for field " << *f_iter << std::endl;}
      ALE::Obj<ALE::BoundaryCondition> sBC = new ALE::BoundaryCondition(disc->comm(), disc->debug());
      sBC->setLabelName("marker");
      sBC->setMarker(1);
      sBC->setFunction(PETSC_NULL);
      sBC->setDualIntegrator(PETSC_NULL);
      sDisc->setBoundaryCondition(sBC);
    }
    sDiscs[*f_iter] = sDisc;
  }
  while(its--) {
    if (fasDebug) {ierr = PetscPrintf(dmmg[0]->comm, "    forward sweep %d\n", its);CHKERRQ(ierr);}
    if (flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP){
      // Loop over all cells
      //   This is an overlapping block SOR, but it is easier and seems more natural than doing each unknown
      for(PETSC_MESH_TYPE::label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
        ALE::Obj<PETSC_MESH_TYPE::sieve_type::supportSet> cellBlock  = sieve->nSupport(sieve->nCone(*c_iter, depth), depth);
        ALE::Obj<PETSC_MESH_TYPE>                         sm         = ALE::Selection<PETSC_MESH_TYPE>::submesh(m, cellBlock);
        ALE::Obj<PETSC_MESH_TYPE::real_section_type>      ssX        = sm->getRealSection("default");
        const ALE::Obj<PETSC_MESH_TYPE::label_type>&      cellMarker = sm->createLabel("marker");

        if (fasDebug) {ierr = PetscPrintf(dmmg[0]->comm, "    forward sweep cell %d\n", *c_iter);CHKERRQ(ierr);}
        ierr = SectionRealSetSection(cellX, ssX);CHKERRQ(ierr);
        // Assign BC to mesh
        for(PETSC_MESH_TYPE::sieve_type::supportSet::iterator b_iter = cellBlock->begin(); b_iter != cellBlock->end(); ++b_iter) {
          const ALE::Obj<PETSC_MESH_TYPE::coneArray> closure = ALE::SieveAlg<PETSC_MESH_TYPE>::closure(m, *b_iter);
          const PETSC_MESH_TYPE::coneArray::iterator end     = closure->end();
          const bool                           isCell  = *b_iter == *c_iter;

          for(PETSC_MESH_TYPE::coneArray::iterator cl_iter = closure->begin(); cl_iter != end; ++cl_iter) {
            if (isCell) {
              sm->setValue(cellMarker, *cl_iter, m->getValue(marker, *cl_iter));
            } else {
              if (sm->height(*cl_iter) == 0) {
                sm->setValue(cellMarker, *cl_iter, 2);
              } else if (sm->getValue(cellMarker, *cl_iter, -1) < 0) {
                sm->setValue(cellMarker, *cl_iter, 1);
              }
            }
          }
        }
        for(std::map<std::string, ALE::Obj<ALE::Discretization> >::iterator d_iter = sDiscs.begin(); d_iter != sDiscs.end(); ++d_iter) {
          sm->setDiscretization(d_iter->first, d_iter->second);
        }
        // Create field
        sm->setupField(ssX, 2, true);
        // Setup constant
        sm->setRealSection("constant", sB);
        // Setup DMMG
        ierr = MeshSetMesh(smallMesh, sm);CHKERRQ(ierr);
        ierr = DMMGSetDM(smallDmmg, (DM) smallMesh);CHKERRQ(ierr);
        ierr = DMMGSetSNESLocal(smallDmmg, func, jac, 0, 0);CHKERRQ(ierr);
        ierr = DMMGSetFromOptions(smallDmmg);CHKERRQ(ierr);
        // TODO: Construct null space, if necessary
        //ierr = DMMGSetNullSpace(smallDmmg, PETSC_FALSE, 1, CreateNullSpace);CHKERRQ(ierr);
        //ALE::Obj<PETSC_MESH_TYPE::real_section_type> nullSpace = sm->getRealSection("nullSpace");
        //sm->setupField(nullSpace, 2, true);
        // Fill in intial guess with BC values
        for(PETSC_MESH_TYPE::sieve_type::supportSet::iterator b_iter = cellBlock->begin(); b_iter != cellBlock->end(); ++b_iter) {
          sm->updateAll(ssX, *b_iter, m->restrictNew(sX, *b_iter));
        }
        if (fasDebug) {
          sX->view("Initial solution guess");
          ssX->view("Cell solution guess");
        }
        // Solve
        ierr = DMMGSolve(smallDmmg);CHKERRQ(ierr);
        // Update global solution with local solution
        ierr = SectionRealToVec(cellX, smallMesh, SCATTER_REVERSE, DMMGGetx(smallDmmg));CHKERRQ(ierr);
        m->updateAll(sX, *c_iter, sm->restrictNew(ssX, *c_iter));
        if (fasDebug) {
          ssX->view("Cell solution final");
          sX->view("Final solution");
        }
      }
    }
    if (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP){
    }
  }
#endif
  sB->zero();
  ierr = SectionRealToVec(sectionX, mesh, SCATTER_FORWARD, X);CHKERRQ(ierr);
  ierr = SectionRealDestroy(sectionX);CHKERRQ(ierr);
  ierr = SectionRealDestroy(sectionB);CHKERRQ(ierr);
  ierr = SectionRealDestroy(cellX);CHKERRQ(ierr);
  ierr = DMMGDestroy(smallDmmg);CHKERRQ(ierr);
  ierr = MeshDestroy(smallMesh);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

/*
 This is alpha FAS code.

  R is the usual multigrid restriction (e.g. the tranpose of piecewise linear interpolation)
  Q is either a scaled injection or the usual R
*/
#undef __FUNCT__
#define __FUNCT__ "DMMGSolveFAS_Mesh"
PetscErrorCode DMMGSolveFAS_Mesh(DMMG *dmmg, PetscInt level)
{
  SNES           snes = dmmg[level]->snes;
  PetscReal      norm;
  PetscInt       i, j, k;
  PetscTruth     fasDebug;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHasName(dmmg[0]->prefix, "-dmmg_fas_debug", &fasDebug);CHKERRQ(ierr);
  ierr = VecSet(dmmg[level]->r, 0.0);CHKERRQ(ierr);
/*   for(j = 1; j <= level; ++j) { */
/*     if (!dmmg[j]->inject) { */
/*       ierr = DMGetInjection(dmmg[j-1]->dm, dmmg[j]->dm, &dmmg[j]->inject);CHKERRQ(ierr); */
/*     } */
/*   } */

  for(i = 0, snes->iter = 1; i < 100; ++i, ++snes->iter) {
    ierr = PetscPrintf(dmmg[0]->comm, "FAS iteration %d\n", i);CHKERRQ(ierr);
    for(j = level; j > 0; j--) {
      if (dmmg[j]->monitorall) {ierr = PetscPrintf(dmmg[0]->comm, "  FAS level %d\n", j);CHKERRQ(ierr);}
      /* Relax on fine mesh to obtain x^{new}_{fine}, residual^{new}_{fine} = F_{fine}(x^{new}_{fine}) \approx 0 */
      ierr = Relax_Mesh(dmmg, (Mesh) dmmg[j]->dm, SOR_SYMMETRIC_SWEEP, dmmg[j]->presmooth, dmmg[j]->x, dmmg[j]->r);CHKERRQ(ierr);
      ierr = DMMGFormFunctionMesh(0,dmmg[j]->x,dmmg[j]->w,dmmg[j]);CHKERRQ(ierr);

      /* residual^{old}_fine} - residual^{new}_{fine} = F(x^{old}_{fine}) - residual^{new}_{fine} */
      ierr = VecAYPX(dmmg[j]->w,-1.0,dmmg[j]->r);CHKERRQ(ierr);

      if (j == level || dmmg[j]->monitorall) {
        /* norm( residual_fine - f(x_fine) ) */
        ierr = VecNorm(dmmg[j]->w,NORM_2,&norm);CHKERRQ(ierr);
        if (dmmg[j]->monitorall) {
          for (k=0; k<level-j+1; k++) {ierr = PetscPrintf(dmmg[j]->comm,"  ");CHKERRQ(ierr);}
          ierr = PetscPrintf(dmmg[j]->comm,"FAS lvl %d function norm %G\n",j,norm);CHKERRQ(ierr);
        }
        if (j == level) {
          if (norm < dmmg[level]->abstol) goto theend; 
          if (i == 0) {
            dmmg[level]->rrtol = norm*dmmg[level]->rtol;
          } else {
            if (norm < dmmg[level]->rrtol) goto theend;
          }
        }
      }

      /* residual^{new}_{coarse} = R*(residual^{old}_fine} - residual^{new}_{fine}) */
      ierr = MatRestrict(dmmg[j]->R, dmmg[j]->w, dmmg[j-1]->r);CHKERRQ(ierr); 
      
      /* F_{coarse}(R*x^{new}_{fine}) */
      ierr = MatRestrict(dmmg[j]->R, dmmg[j]->x, dmmg[j-1]->x);CHKERRQ(ierr); 
/*       ierr = VecScatterBegin(dmmg[j]->inject,dmmg[j]->x,dmmg[j-1]->x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr); */
/*       ierr = VecScatterEnd(dmmg[j]->inject,dmmg[j]->x,dmmg[j-1]->x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr); */
      ierr = DMMGFormFunctionMesh(0,dmmg[j-1]->x,dmmg[j-1]->w,dmmg[j-1]);CHKERRQ(ierr);

      /* residual_coarse = F_{coarse}(R*x_{fine}) + R*(residual^{old}_fine} - residual^{new}_{fine}) */
      ierr = VecAYPX(dmmg[j-1]->r,1.0,dmmg[j-1]->w);CHKERRQ(ierr);

      /* save R*x^{new}_{fine} into b (needed when interpolating compute x back up) */
      ierr = VecCopy(dmmg[j-1]->x,dmmg[j-1]->b);CHKERRQ(ierr);
    }

    if (dmmg[0]->monitorall) {
      for (k=0; k<level+1; k++) {ierr = PetscPrintf(dmmg[0]->comm,"  ");CHKERRQ(ierr);}
      ierr = PetscPrintf(dmmg[0]->comm, "FAS coarse grid\n");CHKERRQ(ierr);
    }
    if (level == 0) {
      ierr = DMMGFormFunctionMesh(0,dmmg[0]->x,dmmg[0]->w,dmmg[0]);CHKERRQ(ierr);
      ierr = VecAYPX(dmmg[j]->w,-1.0,dmmg[j]->r);CHKERRQ(ierr);
      ierr = VecNorm(dmmg[0]->w,NORM_2,&norm);CHKERRQ(ierr);
      if (norm < dmmg[level]->abstol) goto theend; 
      if (i == 0) {
        dmmg[level]->rrtol = norm*dmmg[level]->rtol;
      }
    }
    ierr = Relax_Mesh(dmmg, (Mesh) dmmg[0]->dm, SOR_SYMMETRIC_SWEEP, dmmg[0]->coarsesmooth, dmmg[0]->x, dmmg[0]->r);CHKERRQ(ierr);
    if (level == 0 || dmmg[0]->monitorall) {
      ierr = DMMGFormFunctionMesh(0,dmmg[0]->x,dmmg[0]->w,dmmg[0]);CHKERRQ(ierr);
      if (fasDebug) {
        SectionReal residual;

        ierr = MeshGetSectionReal((Mesh) dmmg[0]->dm, "default", &residual);CHKERRQ(ierr);
        ierr = SectionRealView(residual, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
        ierr = SectionRealDestroy(residual);CHKERRQ(ierr);
      }
      ierr = VecAXPY(dmmg[0]->w,-1.0,dmmg[0]->r);CHKERRQ(ierr);
      ierr = VecNorm(dmmg[0]->w,NORM_2,&norm);CHKERRQ(ierr);
      for (k=0; k<level+1; k++) {ierr = PetscPrintf(dmmg[0]->comm,"  ");CHKERRQ(ierr);}
      ierr = PetscPrintf(dmmg[0]->comm,"FAS coarse grid function norm %G\n",norm);CHKERRQ(ierr);
      if (level == 0) {
        if (norm < dmmg[level]->abstol) goto theend; 
        if (norm < dmmg[level]->rrtol)  goto theend;
      }
    }

    for (j=1; j<=level; j++) {
      ierr = PetscPrintf(dmmg[0]->comm, "  FAS level %d\n", j);CHKERRQ(ierr);
      /* x^{new}_{coarse} - R*x^{new}_{fine} */
      ierr = VecAXPY(dmmg[j-1]->x,-1.0,dmmg[j-1]->b);CHKERRQ(ierr);
      /* x_fine = x_fine + R'*(x^{new}_{coarse} - R*x^{new}_{fine}) */
      ierr = MatInterpolateAdd(dmmg[j]->R, dmmg[j-1]->x, dmmg[j]->x, dmmg[j]->x);CHKERRQ(ierr);

      if (dmmg[j]->monitorall) {
        /* norm( F(x_fine) - residual_fine ) */
        ierr = DMMGFormFunctionMesh(0,dmmg[j]->x,dmmg[j]->w,dmmg[j]);CHKERRQ(ierr);
        ierr = VecAXPY(dmmg[j]->w,-1.0,dmmg[j]->r);CHKERRQ(ierr);
        ierr = VecNorm(dmmg[j]->w,NORM_2,&norm);CHKERRQ(ierr);
        for (k=0; k<level-j+1; k++) {ierr = PetscPrintf(dmmg[j]->comm,"  ");CHKERRQ(ierr);}
        ierr = PetscPrintf(dmmg[j]->comm,"FAS lvl %d function norm before postsmooth %G\n",j,norm);CHKERRQ(ierr);
      }

      /* Relax residual_fine - F(x_fine)  = 0 */
      for (k=0; k<dmmg[j]->postsmooth; k++) {
        ierr = Relax_Mesh(dmmg, (Mesh) dmmg[j]->dm, SOR_SYMMETRIC_SWEEP, 1, dmmg[j]->x, dmmg[j]->r);CHKERRQ(ierr);
      }

      if ((j == level) || dmmg[j]->monitorall) {
        /* norm( F(x_fine) - residual_fine ) */
        ierr = DMMGFormFunctionMesh(0,dmmg[j]->x,dmmg[j]->w,dmmg[j]);CHKERRQ(ierr);
        ierr = VecAXPY(dmmg[j]->w,-1.0,dmmg[j]->r);CHKERRQ(ierr);
        ierr = VecNorm(dmmg[j]->w,NORM_2,&norm);CHKERRQ(ierr);
        for (k=0; k<level-j+1; k++) {ierr = PetscPrintf(dmmg[j]->comm,"  ");CHKERRQ(ierr);}
        ierr = PetscPrintf(dmmg[j]->comm,"FAS lvl %d function norm %G\n",j,norm);CHKERRQ(ierr);
        if (j == level) {
          if (norm < dmmg[level]->abstol) goto theend; 
          if (norm < dmmg[level]->rrtol) goto theend;
        }
      }
    }

    if (dmmg[level]->monitor){
      ierr = DMMGFormFunctionMesh(0,dmmg[level]->x,dmmg[level]->w,dmmg[level]);CHKERRQ(ierr);
      ierr = VecNorm(dmmg[level]->w,NORM_2,&norm);CHKERRQ(ierr);
      ierr = PetscPrintf(dmmg[level]->comm,"%D FAS function norm %G\n",i+1,norm);CHKERRQ(ierr);
    }
  }
  theend:
  PetscFunctionReturn(0);
}
