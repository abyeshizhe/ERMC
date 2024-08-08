function x=Sort(x)
%x=[1,4,6;2,3,1;1,1,1];
[r,~]=size(x); 
for i=1:r-1 
  for j=i+1:r 
  if norm(x(i,:))>norm(x(j,:)) 
     temp=x(i,:); 
     x(i,:)=x(j,:); 
     x(j,:)=temp; 
  end 
 end 
end 
