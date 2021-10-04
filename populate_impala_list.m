function [ list ] = populate_impala_list( TI, tau )
%POPULATE_IMPALA_LIST Create IMPALA style pattern library for MPS
%   This implementation, while very fast for smaller TI, slows down a lot
%   as the TI size increases.
%   It naively expects each node in the eroded training image to contain
%   an unique data event, and only later consolidates the counts into a
%   smaller list. This can be performed a lot faster, no doubt, but since
%   this only needs to run once, optimizing for performance was considered
%   low priority.
%
%   Inputs:
%    TI:        Training image 2D or 3D
%    tau:       Data template
%
%   Ouputs:
%    list: cell array with d and c vectors
%        d: data event
%        c: counts
%
%   Oli D. Johannsson, oli@johannsson.dk (2017)

template_length = size(tau,1);
cat = unique(TI(:))';
num_cat = length(cat);
list_counter = 0;

% 2D or 3D
switch length(size(TI))
    case 2 % 2D
        % only take x and y components of tau
        tau = tau(:,1:2);
        
        [num_x,num_y] = size(TI);
        min_x = 1 + abs(min(tau(:,1)));
        max_x = num_x - max(tau(:,1));
        min_y = 1 + abs(min(tau(:,2)));
        max_y = num_y - max(tau(:,2));
        
        list_length = (1+max_x-min_x)*(1+max_y-min_y);
        list = cell(list_length,2);
        
        for i = min_x : max_x
            for j = min_y : max_y
                list_counter = list_counter + 1;
                
                d = zeros(1,template_length);
                
                for h = 1:template_length
                    d(h) = TI(i+tau(h,1),j+tau(h,2));
                    
                end
                
                c = 1.*(cat == TI(i,j));
                
                list{list_counter,1} = d;
                list{list_counter,2} = c;
            end
        end
        
        %Find unique patterns
        [d,~,Id] = unique(cell2mat(list(:,1)),'rows');
        final_list_length = size(d,1);
        
        % Create count matrix
        C = cell2mat(list(:,2));
        
        %Preallocate final count matrix
        c = zeros(final_list_length, num_cat);
        
        %For each unique pattern
        for i = 1:final_list_length
            %Sum counts for each facies
            Idi = Id == i;
            c(i,:) = sum(C(Idi,:),1);
            %Had forgotten ",1) " in sum! :D
            
%              C(Idi,:) = [];
%             if mod(i,1000)==1
%                 C = C(~arrayfun(@isempty, C));
%                 
%                 %Find unique patterns
%                 [d,~,Id] = unique(cell2mat(list(:,1)),'rows');
%             end
            
        end
        
    case 3 % 3D
        tau = tau(:,1:3);
        [num_x,num_y,num_z] = size(TI);
        min_x = 1 + abs(min(tau(:,1)));
        max_x = num_x - abs(max(tau(:,1)));
        min_y = 1 + abs(min(tau(:,2)));
        max_y = num_y - abs(max(tau(:,2)));
        min_z = 1 + abs(min(tau(:,3)));
        max_z = num_z - abs(max(tau(:,3)));
        
        %Preallocate
        %D = NaN(max_x,max_y,max_z,template_length);
        %C = NaN(max_x,max_y,max_z,num_cat);
        
        for i = min_x : max_x
            for j = min_y : max_y
                for k = min_z : max_z
                    %list_counter = list_counter + 1;
                    %list_counter = 1 +...
                    %    (i-min_x)*(1+max_y-min_y)*(1+max_z-min_z) +...
                    %    (j-min_y)*(1+max_z-min_z) + (k-min_z);
                    
                    d = zeros(1,template_length);
                    
                    for h = 1:template_length
                        d(h) = TI(i+tau(h,1),j+tau(h,2),k+tau(h,3));
                        
                    end
                    
                    c = 1.*(cat == TI(i,j,k));
                    
                    %list{list_counter,1} = d;
                    %list{list_counter,2} = c;
                    D(i,j,k,:) = d;
                    C(i,j,k,:) = c;
                    
                end
            end
        end
        
        %Reshape;
        D = reshape(D,[max_x*max_y*max_z,size(tau,1)]);
        C = reshape(C,[max_x*max_y*max_z,length(cat)]);
        
        listLength = size(D,1);
        num_parts = 100; %TODO: move to top or make function of list length
        splits = [1:round(listLength/(num_parts)):listLength listLength+1];
        
        %Dsplit = NaN(num_parts,listLength+1,template_length);
        %Csplit = NaN(num_parts,listLength+1,num_cat);
        
        for i = 1:num_parts
            Dsplit(i,1:splits(i+1)-splits(i),:) = ...
                D(splits(i):splits(i+1)-1,:);
            Csplit(i,1:splits(i+1)-splits(i),:) = ...
                C(splits(i):splits(i+1)-1,:);
        end
        
        dpartial = [];
        cpartial = [];

        %Partial sum
        for i = 1:num_parts
            
            Dtemp(:,:) = Dsplit(i,:,:);
            Ctemp(:,:) = Csplit(i,:,:);

            [dsplit,~,Id] = unique(Dtemp(:,:),'rows');
            
            split_list_length = size(dsplit,1);
            
            csplit = zeros(split_list_length, num_cat);
            
            for j = 1:split_list_length
                %Sum counts for each facies
                csplit(j,:) = sum(Ctemp(Id == j,:),1);
            end
            
            dpartial = vertcat(dpartial,dsplit);
            cpartial = vertcat(cpartial,csplit);
        end        
        % Final sum;
        [d,~,Id] = unique(dpartial(:,:),'rows');
        
        final_list_length = size(d,1);
        c = zeros(final_list_length, num_cat);
        for j = 1:final_list_length
            %Sum counts for each facies
            c(j,:) = sum(cpartial(Id == j,:),1);
        end
end

list = mat2cell([d c],ones(1,final_list_length),[template_length num_cat]);


