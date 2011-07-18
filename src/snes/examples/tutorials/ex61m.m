
f = PetscBinaryRead('perfectmovie','cell');
s = size(f); s = s(2);
clear M
for i=1:s,
  cv = f{i};
  cv = reshape(cv(2:5:5*96*96),96,96);
  pcolor(cv);
  caxis([0 1])
  shading interp;
  colorbar;
M((i-1)/1 + 1) = getframe;
end
movie(M)

