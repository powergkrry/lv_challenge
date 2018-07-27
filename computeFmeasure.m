function [P,R,F,U]=computeFmeasure(G,H)
% G - Ground truth segmentation (binary)
% H - segmentation to evaluate (binary)
 
 
 P1=sum(sum( G.*H )) ./ sum(sum(H));
 R1=sum(sum( G.*H )) ./ sum(sum(G));
 F1=2*P1*R1/(P1+R1);

 U = sum( sum( G.*H ) ) ./ sum( sum( double( G | H ) ) );
 
 P=P1; R=R1; F=F1;
 
 if 0,
   H=1-H;
   P2=sum(sum( G.*H )) ./ sum(sum(H));
   R2=sum(sum( G.*H )) ./ sum(sum(G));
   F2=2*P2*R2/(P2+R2);
   
   if F1>F2
     P=P1; R=R1; F=F1;
   else
     P=P2; R=R2; F=F2;
   end
 end