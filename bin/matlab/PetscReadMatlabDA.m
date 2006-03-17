function Set = PetscReadMatlabDA(filename)
% PETSCREADMATLABDA - Reads and interprets matlab specific lines 
%   from the .info files produced by PetscViewerMatlabDA
%   
%   All matlab specific lines in the .info file are expected to 
%   begin with %Set.
   
   fid=fopen(filename,'r');
   if (fid == -1)
      error(sprintf('PetscReadMatlabDA: cannot load file %s',filename))
      Set = []; return; 
   end
   str=fgets(fid);
   while (ischar(str))
      % check for output lines that start %Set.blah=etc.
      marker = strmatch('%$$',str); 
      if marker
	 eval(str(marker+4:end));
      end
      str=fgets(fid);
   end
   return
