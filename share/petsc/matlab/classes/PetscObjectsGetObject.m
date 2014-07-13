function obj = PetscObjectsGetObject(name)
%
%   Given the name of any PetscObject in the PETSc application return the Matlab interface object
%
%  Developer notes: I could not figure out a good way to return a character string from a C program
%  except by listing it explicitly as the returned type
%  cname is the class name of the object, pobj is the C address of that object
[cname,dummy,pobj] = calllib('libpetsc', 'PetscObjectsGetObjectMatlab',name,0);
%  Create the Matlab object based on the class name
if (pobj == 0) 
  obj = 0;
else
  obj = eval(['Petsc' cname '(' num2str(pobj) ', ''pobj'')']);
end


