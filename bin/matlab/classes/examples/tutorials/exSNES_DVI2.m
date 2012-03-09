%%
% Solves a nonlinear variational inequality where the user manages the mesh--solver interactions
%
% This is a translation of minimal surface area problem with a plate 
% as written in src/snes/examples/tests/ex16.c
% 
%
%   Set the Matlab path and initialize PETSc
path(path,'../../')
PetscInitialize({'-snes_monitor','-ksp_monitor'});
%%
%   Open a viewer to display PETSc objects
viewer = PetscViewer();
viewer.SetType('ascii');
%%
%  Create DM to manage the grid and get work vectors
user.mx = 24;user.my = 24;
user.dm = PetscDMDACreate2d(PetscDM.NONPERIODIC,PetscDM.STENCIL_BOX,user.mx,user.my,Petsc.DECIDE,Petsc.DECIDE,1,1);
x  = user.dm.CreateGlobalVector();
r  = x.Duplicate();
J  = user.dm.CreateMatrix('aij');

%% Data for the plate
% 
user.bmx = user.mx/2; user.bmy = user.my/2;user.bheight=0.4;
%%
%  Set Boundary conditions
[user] = MSA_BoundaryConditions(user);
%%
%  Set initial guess
[x] = MSA_InitialGuess(user,x);
%%
%  Create the nonlinear solver 
snes = PetscSNES();
snes.SetType('vi');

%%
%  Set minimum surface area problem function routine
snes.SetFunction(r,'snesdvi_function',user);


%%
%  Set minimum surface area problem jacobian routine
snes.SetJacobian(J,J,'snesdvi_jacobian',user);

%%
%  Set solution monitoring routine
snes.MonitorSet('snesdvi_monitor',user);
figure(1),clf;figure(2),clf;
%%
%   Set VI bounds
xl = x.Duplicate();
xu = x.Duplicate();
[xl,xu] = MSA_Plate(xl,xu,user);

snes.VISetVariableBounds(xl,xu);    

%%
%  Solve the nonlinear system
snes.SetFromOptions();
snes.Solve(x);
x.View(viewer);
snes.View(viewer);

%%
%   Free PETSc objects and Shutdown PETSc
%
user.bottom.Destroy();
user.top.Destroy();
user.right.Destroy();
user.left.Destroy();
r.Destroy();
x.Destroy();
J.Destroy();
snes.Destroy();
viewer.Destroy();
PetscFinalize();
