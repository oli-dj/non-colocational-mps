function [ ] = print_impala_list( list )
%PRINT_IMPALA_LIST Summary of this function goes here
%   Detailed explanation goes here
%Print list

list_length = size(list,1);
template_length = size(list{1,1},2);
num_cat = size(list{1,2},2);

formatSpec = 'L%i = ((';
for i = 1:template_length-1
    formatSpec = [formatSpec '%i,'];
end
formatSpec = [formatSpec '%i),('];
for i = 1:num_cat-1
    formatSpec = [formatSpec '%i,'];
end
formatSpec = [formatSpec '%i))\n'];
for i = 1:list_length
    d = list{i,1};
    c = list{i,2};
    fprintf(formatSpec,i-1,d,c);
    

end

end

