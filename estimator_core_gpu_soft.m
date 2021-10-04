function [ CG, tauG, stats] = estimator_core_gpu_soft(SG, SDG, list, path,...
    tau, rand_pre, cat, options)
%IMPALA_CORE Function that runs and IMPALA style MPS-algorithm on GPU
%   Implementation of IMPALA-esque MPS algorithm that searches the list
%   using CUDA kernels on a GPU. Supports trimming and non-colocational
%   soft data, as well as colocational.
%
% Inputs:
%  SG:                  Simulation grid (2D or 3D)
%  SDG:                 Soft Data Grid (SG dim +1)
%  list:                IMPALA list (c and d vectors)
%  path:                pre-calculated path (random or otherwise)
%  tau:                 data template
%  rand_pre:            pre-calculated random numbers
%  cat:                 categories
%  options.threshold:   minimum count in list, else use marginal cpdf.
%         .print:       boolean; 1 shows progress, 0 no print to screen
%         .num_soft_nc  Number nof non-colocated soft data to consider
%                       (default 0, increases processing time dramatically)
%         .trim         boolean, to trim or not
%         .trim_size    how much to trim
%         .trim_trigger how many list misses required to trigger a trim
%         .cap          max number of informed nodes.
%
% Outputs
%  CG:                  Count grid
%  ~ES:                  Estimation Grid
%  ~SG:                  Simlation grid
%  tauG:                local size of data template used
%  stats.               Statistics struct
%
% Oli D. Johannsson, oli@johannsson.dk (2018)

%% TESTING: Cut event length?
cut = 0;

save_d = 0;

%% Get options
print = options.print;
threshold = options.threshold;
cap = options.cap;

% Non colocational soft data.
try
    prob_factor_limit = options.prob_factor_limit;
catch
    prob_factor_limit = 0;
end

% Trimming options
trimming = options.trimming;
trim_size = options.trim_size;
trim_trigger = options.trim_trigger;

% Soft data options
num_soft_nc = options.num_soft_nc;

%% Initialization
formatspec = 'Time elapsed: %1i seconds. %1i percent done ...\n';
num_cat = length(cat);  %number of categories
n_u = size(path,1);     %number of uninformed nodes
dim = length(size(SG)); %dimensionality of simulation grid (2D or 3D)
tau = tau(:,1:dim);     %only use dimensions of tau present in the SG
template_length = size(tau,1); %size of data template

% Init estimator grid
CG = NaN(size(SG,1),size(SG,2),num_cat);

% Copy any hard data
CG(:,:,1) = 1 - SG;
CG(:,:,2) = SG;

% Init stats
tauG = zeros(size(SG));
stats.informed_init = NaN(n_u,1);
stats.informed_final = NaN(n_u,1);
stats.time_elapsed = NaN(n_u,1);

% IDEA/TODO: maybe do this outside of "core" function
% turn pattern library list into pattern and count matrices (for speed).
D = cell2mat(list(:,1));
C = cell2mat(list(:,2));

%% Calculate marginal cpdf
marginal_counts = sum(C,1);
% total counts
marginal_counts_tot = sum(marginal_counts);
% probabilities
marginal_probs = marginal_counts./marginal_counts_tot;
% commulative probabilities
marginal_prob_cum = cumsum(marginal_probs);


%% Get size of list and number of facies
listLength = size(D,1);
eventLength = size(D,2);
numFacies = size(C,2);

%% Reset GPU
g = gpuDevice(1);
reset(g);

% Load the find kernel
%cudaFilename = 'impalaFindStopEarly.cu';
%ptxFilename = 'impalaFindStopEarly.ptx';
cudaFilename = 'impalaFindSmem.cu';
ptxFilename = 'impalaFindSmem.ptx';
kernelName = 'impalaFind';
kernelFind = parallel.gpu.CUDAKernel( ptxFilename, cudaFilename,...
    kernelName);
kernelFind.ThreadBlockSize = [kernelFind.MaxThreadsPerBlock,1,1];
kernelFind.GridSize = [ceil(listLength/kernelFind.MaxThreadsPerBlock),1];

% Load the multiplication kernel
cudaFilename = 'multiplyArray.cu';
ptxFilename = 'multiplyArray.ptx';
kernelName = 'multiplyArray';
kernelMultiplyArray = parallel.gpu.CUDAKernel( ptxFilename,...
    cudaFilename, kernelName);
kernelMultiplyArray.ThreadBlockSize =...
    [kernelMultiplyArray.MaxThreadsPerBlock,1,1];
kernelMultiplyArray.GridSize =...
    [ceil(listLength/kernelMultiplyArray.MaxThreadsPerBlock),1];

%% Prepare list
% make D and C linear
D = D';
D = D(:);
C = C(:);

%% GPU: Copy to GPU
listGPU = gpuArray(single(D));
countsGPU = gpuArray(int32(C));
listLengthGPU = gpuArray(int32(listLength));
numFaciesGPU = gpuArray(int32(numFacies));
eventLengthGPU = gpuArray(int32(eventLength));
%currentEventLengthGPU = gpuArray(int32(eventLength));

%% Initialize arrays on GPU
countsEventGPU = gpuArray(int32(zeros(listLength,numFacies)));
matchesGPU = gpuArray(int32(zeros(listLength,1)));
%sumGPU = gpuArray(int32(zeros(numFacies,1)));
%outGPU = gpuArray(int32(zeros(listLength,1)));
%cur_countsGPU = gpuArray(int32(zeros(numFacies,1)));

%% While uninformed nodes exist
switch dim % Switch for 2D and 3D. TODO: Implement 1D support too.
    case 2 % 2D
        discards = 0;
        trim = 0;
        for i = 1:n_u
            d = NaN(1,template_length);
            %% Get data event
            num_informed = 0;
            if save_d == 1
                d_list_entry =  NaN(cap,2);
            end
            for h = 1:(template_length-trim)
                if num_informed < cap
                    try
                        d(h) = SG(path(i,1)+tau(h,1),path(i,2)+tau(h,2));
                        if ~isnan(d(h))
                            num_informed = num_informed + 1;
                            if save_d == 1
                                d_list_entry(num_informed,:) = [path(i,1)+tau(h,1),path(i,2)+tau(h,2),d(h)];
                            end
                        end
                    catch
                        d(h) = NaN;
                    end
                else
                    d(h) = NaN;
                end
                
            end
%             if save_d == 1
%                 d_list_entry
%             end
            %% Get soft data event with max num_soft_nc elements
            nsd = 0; %Reset number of soft data counter
            if num_soft_nc > 0
                %Preallocate / clear
                d_soft = NaN(num_soft_nc,num_cat);
                h_soft = NaN(num_soft_nc,1);
                
                %Search within template
                for h = 1:(template_length-trim)
                    % ...and only this many and only if node uninformed
                    if (isnan(d(h))) && (nsd < num_soft_nc)
                        try
                            % Check if node has been simulated
                            if isnan(sum(SG(path(i,1)+tau(h,1),...
                                    path(i,2)+tau(h,2))))
                                sd_temp = SDG(path(i,1)+tau(h,1),...
                                    path(i,2)+tau(h,2),:);
                                if ~isnan(sd_temp)
                                    d_soft(nsd+1,:) = sd_temp;
                                    %record relative location
                                    h_soft(nsd+1,:) = h;
                                    
                                    %increase counter
                                    nsd = nsd +1;
                                end
                            end
                        catch
                            % Don't increase counter (redundant)
                            nsd=nsd;
                        end
                    end
                end
                %Prune soft data event
                if nsd < num_soft_nc
                    d_soft = d_soft(1:nsd,:);
                    h_soft = h_soft(1:nsd,:);
                end
            end
            
            % Record the initial number of informed nodes and soft nodes
            stats.informed_init(i) = sum(~isnan(d));
            stats.nsd(i) = nsd;
            
            % GPU: Copy data event to GPU
            dataEventGPU = gpuArray(single(d));
%             if cut
%                 currentEventLength = find(~isnan(d),1,'last');
%                 currentEventLengthGPU = gpuArray(int32(currentEventLength));
%             end
            wait(g);
            
            %% If any informed nodes or softdata
            if (~isempty(find(~isnan(d),1))) || (nsd > 0)
                counts_tot = 0;
                while counts_tot < threshold
                    % Search list for matches with informed nodes
                    matchesGPU = feval( kernelFind, listGPU,...
                        dataEventGPU, listLengthGPU,...
                        eventLengthGPU,...%currentEventLengthGPU,
                        matchesGPU);
                    wait(g);
                    
                    % Multiply by counts
                    countsEventGPU = feval( kernelMultiplyArray,...
                        matchesGPU, countsGPU, listLengthGPU,...
                        numFaciesGPU, countsEventGPU);
                    
                    %Perform summation on gpu
                    % TODO: Reduction on GPU via kernel?
                    cur_countsGPU = sum(countsEventGPU,1);
                    counts = gather(cur_countsGPU);
                    wait(g);
                    
                    counts_tot = sum(counts);
                    
                    % If number of counts below threshold
                    if counts_tot < threshold
                        % ...remove last informed node
                        d(find(~isnan(d),1,'last')) = NaN;
                        
                        % Copy dataEvent to GPU
                        dataEventGPU = gpuArray(single(d));
                        if cut
                            currentEventLength = find(~isnan(d),1,'last');
                            currentEventLengthGPU = gpuArray(int32(currentEventLength));
                        end
                        wait(g);
                    end
                end
                
               %% Non-colocated soft data
                % If any non-colocated soft data:
                if nsd > 0
                    num_combinations = num_cat^nsd;
                    all_counts_soft = NaN(num_combinations,num_cat);
                    %all_probs_soft = NaN(num_combinations,num_cat);
                    d_hard = d; % Save hard data event
                    
                    %For each combination do
                    for l=1:num_combinations
                        %Calculate soft configuration
                        soft_config = id2n(l,num_cat,nsd);
                        
                        %Calculate probability of configuration from soft data
                        prob_factor = 1;
                        for m = 1:nsd
                            prob_factor = prob_factor * ...
                                d_soft(m,soft_config(m) + 1);
                        end
                        %prob_factor
                        
                        %fprintf(num2str(prob_factor));
                        
                        %Low Prob factor culling (NEW!)
                        if prob_factor < prob_factor_limit
                            all_counts_soft(l,:) = zeros(1,num_cat);
                            
                        else
                            % Set data event
                            for m = 1:nsd
                                d(h_soft(m)) = soft_config(m);
                            end
                            
                            % Copy dataEvent to GPU
                            dataEventGPU = gpuArray(single(d));
                            wait(g);
                            
                            % Search Pattern Library
                            matchesGPU = feval( kernelFind, listGPU,...
                                dataEventGPU, listLengthGPU,...
                                eventLengthGPU, matchesGPU);
                            wait(g);
                            
                            % Multiply by counts
                            countsEventGPU = feval( kernelMultiplyArray,...
                                matchesGPU, countsGPU,...
                                listLengthGPU, numFaciesGPU,...
                                countsEventGPU);
                            
                            % Perform summation on gpu
                            % TODO: Reduction on GPU via kernel
                            cur_countsGPU = sum(countsEventGPU,1);
                            counts_soft = gather(cur_countsGPU);
                            wait(g);
                            
                            temps_counts_sum = sum(counts_soft);
                            
                            %if under threshold;
%                             if temps_counts_sum < threshold
%                                 all_counts_soft(l,:) = counts_soft.*0;
%                             else
                               
                                %Normalize
                                %counts_soft = counts_soft./temps_counts_sum;
                                
                                %Multiply with probability factor
                                all_counts_soft(l,:) = counts_soft.*prob_factor;
                                
%                             end
                        end
                        
                        % Retrieve hard data for next combination
                        d = d_hard;
                    end
                    counts = sum(all_counts_soft,1);
                    %Change sum to product.
                    %counts = prod(all_counts_soft);
                    
                    % Sum of counts
                    counts_tot = sum(counts);
                    
                    % Normalize counts
                    counts = counts./counts_tot;
                    
                end
                
                
                
                
                %% Co-locational soft data
                if ~isnan(sum(SDG(path(i,1),path(i,2))))
                    SD(1) = SDG(path(i,1),path(i,2),1);
                    SD(2) = SDG(path(i,1),path(i,2),2);
                    counts = counts.*SD;
                end
                counts_tot = sum(counts);
                
                % If the co-locational soft data is incompatible with
                % the counts found, use it with the marginal distribution
                % instead.
                %TODO: Fix this.
                if (counts_tot < .1) || isnan(counts_tot)
                    if ~isnan(sum(SDG(path(i,1),path(i,2))))
                        counts = marginal_counts.*SD;
                    else
                        counts = marginal_counts;
                        fprintf('.');
                    end
                end
                counts_tot = sum(counts);
                
                probs = counts./counts_tot;
                
                % commulative probabilities
                prob_cum = cumsum(probs);
                
                % Save to Count grid
                for k = 1:num_cat
                    CG(path(i,1),path(i,2),k) = counts(k)./counts_tot;
                end
                
                % draw a value and assign
                %SG(path(i,1),path(i,2)) = cat(find(prob_cum > ...
                %    rand_pre(i),1));
                
                
                % record data event length
                informed = find(~isnan(d));
                tauG(path(i,1),path(i,2)) = length(informed);
                stats.informed_final(i) = sum(~isnan(d));
                discards = discards + stats.informed_init(i) - ...
                    stats.informed_final(i);
                
            else
                %% Draw from marginal distribution
                counts = marginal_counts;
                counts_tot = sum(counts);
                
                %% Co-locational soft data
                if ~isnan(sum(SDG(path(i,1),path(i,2),:)))
                    SD(1) = SDG(path(i,1),path(i,2),1);
                    SD(2) = SDG(path(i,1),path(i,2),2);
                    
                    % use co-locational soft data with marginal
                    % distribution
                    counts = marginal_counts.*SD;
                    counts_tot = sum(counts);
                    probs = counts./counts_tot;
                    % commulative probabilities
                    prob_cum = cumsum(probs);
                    
                    % Save to Count grid
                    for k = 1:num_cat
                        CG(path(i,1),path(i,2),k) = counts(k)./counts_tot;
                    end
                    
                    % draw a value and assign
                    %SG(path(i,1),path(i,2)) = ...
                    %    cat(find(prob_cum > rand_pre(i),1));
                else
                    % Save to Count grid
                    for k = 1:num_cat
                        CG(path(i,1),path(i,2),k) = counts(k)./counts_tot;
                    end
                    
                    %SG(path(i,1),path(i,2)) = ...
                    %    cat(find(marginal_prob_cum > rand_pre(i),1));
                    
                    % Set data event length to zero
                    tauG(path(i,1),path(i,2)) = 0;
                    stats.informed_final(i) = 0;
                end
            end
            if (print && ~mod(100.*i./n_u,5))
                time_elapsed = toc;
                fprintf(formatspec,round(time_elapsed),round(100*(i/n_u)));
            end
            
            stats.time_elapsed(i) = toc;
            stats.template_length(i) = template_length - trim;
            
            %% Trim function
            if trimming && (discards > trim_trigger)
                trim = trim + trim_size;
                discards = 0;
            end
        end
        
    case 3 %3D
        discards = 0;
        trim = 0;
        for i = 1:n_u
            d = NaN(1,template_length);
            %% Get data event
            for h = 1:(template_length-trim)
                try
                    d(h) = SG(path(i,1)+tau(h,1),...
                        path(i,2)+tau(h,2),...
                        path(i,3)+tau(h,3));
                catch
                    d(h) = NaN;
                end
            end
            
            %% Get soft data event with max num_soft_nc elements
            nsd = 0; %Reset number of soft data counter
            if num_soft_nc > 0
                %Preallocate / clear
                d_soft = NaN(num_soft_nc,num_cat);
                h_soft = NaN(num_soft_nc,1);
                
                %Search within template
                for h = 1:(template_length-trim)
                    % ...and only this many and only if node uninformed
                    if (isnan(d(h)) && nsd < num_soft_nc)
                        try
                            % Record soft data
                            sd_temp = SDG(path(i,1)+tau(h,1),...
                                path(i,2)+tau(h,2),...
                                path(i,3)+tau(h,3),:);
                            if ~isnan(sd_temp)
                                d_soft(nsd+1,:) = sd_temp;
                                %record relative location
                                h_soft(nsd+1,:) = h;
                                
                                %increase counter
                                nsd = nsd +1;
                            end
                        catch
                            % Don't increase counter (redundant)
                            nsd=nsd;
                        end
                    end
                end
                %Prune soft data event
                if nsd < num_soft_nc
                    d_soft = d_soft(1:nsd,:);
                    h_soft = h_soft(1:nsd,:);
                end
            end
            
            % Record the initial number of informed nodes and soft nodes
            stats.informed_init(i) = sum(~isnan(d));
            stats.nsd(i) = nsd;
            
            % GPU: Copy data event to GPU
            dataEventGPU = gpuArray(single(d));
            if cut
                eventLengthTemp = find(~isnan(d),1,'last');
                eventLengthGPU = gpuArray(int32(eventLengthTemp));
            end
            wait(g);
            
            %% If any informed nodes or softdata
            if ((~isempty(find(~isnan(d),1))) || (nsd > 0))
                counts_tot = 0;
                while counts_tot < threshold
                    % Search list for matches with informed nodes
                    matchesGPU = feval( kernelFind, listGPU,...
                        dataEventGPU, listLengthGPU,...
                        eventLengthGPU,...%currentEventLengthGPU,
                        matchesGPU);
%                     matchesGPU = feval( kernelFind, listGPU,...
%                         dataEventGPU, listLengthGPU,...
%                         eventLengthGPU, currentEventLengthGPU, matchesGPU);
                    wait(g);
                    
                    % Multiply by counts
                    countsEventGPU = feval( kernelMultiplyArray,...
                        matchesGPU, countsGPU, listLengthGPU,...
                        numFaciesGPU, countsEventGPU);
                    
                    %Perform summation on gpu
                    % TODO: Reduction on GPU via kernel?
                    cur_countsGPU = sum(countsEventGPU,1);
                    counts = gather(cur_countsGPU);
                    wait(g);
                    
                    counts_tot = sum(counts);
                    
                    % If number of counts below threshold
                    if counts_tot < threshold
                        % ...remove last informed node
                        d(find(~isnan(d),1,'last')) = NaN;
                        
                        % Copy dataEvent to GPU
                        dataEventGPU = gpuArray(single(d));
                        if cut
                            currentEventLength = find(~isnan(d),1,'last');
                            currentEventLengthGPU = gpuArray(int32(currentEventLength));
                        end
                        wait(g);
                    end
                end
                
                %% Non-colocated soft data
                % If any non-colocated soft data:
                if nsd > 0
                    num_combinations = num_cat^nsd;
                    all_counts_soft = NaN(num_combinations,num_cat);
                    d_hard = d; % Save hard data event
                    
                    %For each combination do
                    for l=1:num_combinations
                        %Calculate soft configuration
                        soft_config = id2n(l,num_cat,nsd);
                        
                        %Calculate probability of configuration from data
                        prob_factor = 1;
                        for m = 1:nsd
                            prob_factor = prob_factor * ...
                                d_soft(m,soft_config(m) + 1);
                        end
                        
                        % Set data event
                        for m = 1:nsd
                            d(h_soft(m)) = soft_config(m);
                        end
                        
                        % Copy dataEvent to GPU
                        dataEventGPU = gpuArray(single(d));
                        if cut
                            currentEventLength = find(~isnan(d),1,'last');
                            currentEventLengthGPU = gpuArray(int32(currentEventLength));
                        end
                        wait(g);
                        
                        % Search Pattern Library
%                         matchesGPU = feval( kernelFind, listGPU,...
%                             dataEventGPU, listLengthGPU,...
%                             eventLengthGPU, currentEventLengthGPU, matchesGPU);
                    matchesGPU = feval( kernelFind, listGPU,...
                        dataEventGPU, listLengthGPU,...
                        eventLengthGPU,...%currentEventLengthGPU,
                        matchesGPU);
                        wait(g);
                        
                        % Multiply by counts
                        countsEventGPU = feval( kernelMultiplyArray,...
                            matchesGPU, countsGPU,...
                            listLengthGPU, numFaciesGPU,...
                            countsEventGPU);
                        
                        %Perform summation on gpu
                        % TODO: Reduction on GPU via kernel?
                        cur_countsGPU = sum(countsEventGPU,1);
                        counts_soft = gather(cur_countsGPU);
                        wait(g);
                        
                        % Multiply with soft data probability factor
                        counts_soft = counts_soft.*prob_factor;
                        all_counts_soft(l,:) = counts_soft;
                        
                        % Retrieve hard data
                        d = d_hard;
                    end
                    counts = sum(all_counts_soft,1);
                    
                    % Sum of counts
                    counts_tot = sum(counts);
                    
                    % Normalize counts
                    counts = counts./counts_tot;
                end
                
                %% Co-locational soft data
                if ~isnan(sum(SDG(path(i,1),path(i,2),path(i,3))))
                    SD(1) = SDG(path(i,1),path(i,2),path(i,3),1);
                    SD(2) = SDG(path(i,1),path(i,2),path(i,3),2);
                    SD(3) = SDG(path(i,1),path(i,2),path(i,3),3);
                    counts = counts.*SD;
                end
                counts_tot = sum(counts);
                
                % If the co-locational soft data is incompatible with
                % the counts found, use it with the marginal distribution
                % instead.
                if (counts_tot < .1) || isnan(counts_tot)
                    if ~isnan(sum(SDG(path(i,1),path(i,2),path(i,3))))
                        counts = marginal_counts.*SD;
                    else
                        counts = marginal_counts;
                    end
                end
                counts_tot = sum(counts);
                
                probs = counts./counts_tot;
                
                % commulative probabilities
                prob_cum = cumsum(probs);
                
                % Save to Count grid
                for k = 1:num_cat
                    CG(path(i,1),path(i,2),k) = counts(k)./counts_tot;
                end
                
                % draw a value and assign
                %SG(path(i,1),path(i,2),path(i,3)) = cat(find(prob_cum > ...
                %    rand_pre(i),1));
                
                % record data event length
                informed = find(~isnan(d));
                tauG(path(i,1),path(i,2),path(i,3)) = length(informed);
                stats.informed_final(i) = sum(~isnan(d));
                discards = discards + stats.informed_init(i)-...
                    stats.informed_final(i);
                
            else
                %% Draw from marginal distribution
                
                %% Co-locational soft data
                if ~isnan(sum(SDG(path(i,1),path(i,2),path(i,3))))
                    SD(1) = SDG(path(i,1),path(i,2),path(i,3),1);
                    SD(2) = SDG(path(i,1),path(i,2),path(i,3),2);
                    SD(3) = SDG(path(i,1),path(i,2),path(i,3),3);
                    
                    % use co-locational soft data with marginal
                    % distribution
                    counts = marginal_counts.*SD;
                    counts_tot = sum(counts);
                    probs = counts./counts_tot;
                    % commulative probabilities
                    prob_cum = cumsum(probs);
                    
                    % Save to Count grid
                    for k = 1:num_cat
                        CG(path(i,1),path(i,2),k) = counts(k)./counts_tot;
                    end
                    
                    % draw a value and assign
                    %SG(path(i,1),path(i,2),path(i,3)) = ...
                    %    cat(find(prob_cum > rand_pre(i),1));
                else
                    % Save to Count grid
                    for k = 1:num_cat
                        CG(path(i,1),path(i,2),k) = counts(k)./counts_tot;
                    end
                    
                    %SG(path(i,1),path(i,2),path(i,3)) = ...
                    %    cat(find(marginal_prob_cum > rand_pre(i),1));
                end
                
                % Set data event length to zero
                tauG(path(i,1),path(i,2),path(i,3)) = 0;
                stats.informed_final(i) = 0;
            end
            if (print && ~mod(100.*i./n_u,5))
                time_elapsed = toc;
                fprintf(formatspec,round(time_elapsed),round(100*(i/n_u)));
            end
            
            stats.time_elapsed(i) = toc;
            stats.template_length(i) = template_length - trim;
            
            %% Trim function
            if (trimming >0) && (discards > trim_trigger)
                trim = trim + trim_size;
                discards = 0;
            end
        end
end



