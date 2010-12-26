%%
%
%  Solves a linear system where the DM object manages the mesh--solver interactions
%
%   Set the Matlab path and initialize PETSc
path(path,'../../')
PetscInitialize({'-ksp_monitor','-pc_type','none','-ksp_converged_reason'});
%%
%   Create a DM object
da = PetscDMDACreate2d(PetscDM.NONPERIODIC,PetscDM.STENCIL_BOX,4,4,1,1,1,1);
da.SetFunction('rhsfunction');
type rhsfunction
da.SetJacobian('jacobian');
type jacobian
%%
%%
%   Create the linear solver, tell it the DM
ksp = PetscKSP();
ksp.SetType('gmres');
ksp.SetDM(da);
ksp.SetFromOptions();
ksp.Solve();
x = ksp.GetSolution();
x.View;
ksp.View;
%%
%   Free PETSc objects and shutdown PETSc
%
da.Destroy();
ksp.Destroy();
PetscFinalize();
