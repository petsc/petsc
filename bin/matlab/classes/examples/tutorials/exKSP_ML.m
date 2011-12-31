PetscInitialize({'-pc_type','ml','-ksp_monitor'});
v = PetscViewer('/Users/barrysmith/Datafiles/Matrices/medium',Petsc.FILE_MODE_READ);
A = PetscMat;
A.Load(v);
v.Destroy;
ksp = PetscKSP;
ksp.SetOperators(A);
ksp.SetFromOptions;
b = PetscVec(1:181);
x = PetscVec(1:181);
ksp.Solve(b,x);
ksp.Destroy;
A.Destroy;
PetscFinalize;



