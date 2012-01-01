function ex61genm
%
%  Writes a data file for ex61 that has a single point at the center as a radiation event

fd = PetscOpenFile('ex61.random.1','w');
write(fd,1,'int32');
write(fd,[0.0 .5 .5 1.0],'double');
close(fd);
