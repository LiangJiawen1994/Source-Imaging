function [seed, cellstruct_cls] = SBFCluster(B,L,VertConn,extent,varargin)
if nargin<5
    type = 'spatio_temporal';
else
    type = varargin{1};
end

[seed, cellstruct_cls] = AutoCluster(B,L,VertConn,extent,type);

function [seed,cellstruct_cls] = AutoCluster(M,L,VertConn,extent,type)
%% Normalize Lead Field Matrix
Gstruct = struct;
if isempty(L)
    return
end

Gn = bsxfun(@rdivide, L, sqrt(sum(L.^2, 1)));
[~,lambda,U] = svd(Gn',0);
Gstruct.Gn = single(Gn);
Gstruct.lambda = single(diag(lambda));
Gstruct.U = single(U);
clear Gn lambda U
M = single(M);
%% Clustering

% ==== clustering technique
nbS     = size(L,2);        % Nb of sources
tS      = size(M,2);        % 閏hantillon temp.
SCR     = zeros(nbS,tS);                                        % nb sources fct of time

SVD_threshold = 0.95;
%% MSP Calculation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%1- SVD of the DATA MATRIX
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[U,S,V] = svd(M,0);
s = diag(S);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%2- Threshold to identify the signal subspace (min 3 or 95% inertia)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
inertia = zeros(1, length(s));                                          % Prealocating inertia to increase the speed
for i = 1:length(s)
    inertia(i) = sum(s(1:i).^2)./(sum(s.^2));
end

% cumsum
q = find(inertia>=SVD_threshold,1);

% Ask for standard display with verbose

fprintf('stable clustering: dimension of the signal subspace %d, for inertia > %3.2f\n', q,SVD_threshold);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%3- MSP applied on each principal component of the signal subspace
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmpi(type,'spatio_temporal')
    % SPATIO_TEMPORAL EXTENSION of MSP scores in Signal Subspace (Refer to APPENDIX S1 in Chowdhury et al. 2013 Plos One
    
    APM_temp = zeros(nbS,q);
    scale_prob = max(max(abs(V(:,1:q))));
    for i = 1:q
        
        M_data = U(:,i)*S(i,i)*V(:,i)';
        [APM] = msp( M_data, Gstruct);                 % MSP is calculated here
        
        % SPATIO_TEMPORAL VERSION of MSP scores in Signal Subspace
        APM_temp(:,i) = APM;
    end
    APM_tot = APM_temp*abs(V(:,1:q))'/scale_prob;
    
    APM_tot = APM_tot/(max(APM_tot(:)));
    
    SCR = SCR + APM_tot - SCR.*APM_tot;             %To consider every modalities, we should sum these scores (inspired by be_sliding_clustering.)
    score = mean(SCR,2);
elseif strcmpi(type,'direct')
    score = msp( M, Gstruct);                 % MSP is calculated here
elseif strcmpi(type,'random')
    score = randn(nbS,1);
else
    error('Not validate!!!')
end

% Classification matrix:
% Contains labels ranging from 0 to number of parcels (1 column / time sample) for each sources.

% nb_time = size(SCR,2);
% CLS     = zeros(nbS, nb_time);
seed = []; cellstruct_cls = [];
SCR = double(SCR);
for i = 1:numel(extent)
    [seed1, ~,cellstruct_cls1] = create_clusters(VertConn, score ,extent(i));
    seed = [seed;seed1];
    cellstruct_cls = [cellstruct_cls;cellstruct_cls1'];
end


function [seed, selected_source, cellstruct] = create_clusters(nm, scores,extent)
%CREATE_CLUSTERS Creates clusters with a neighbor matrix.
%   C = CREATE_CLUSTERS(NEIGHBORS, SCORES,OPTIONS)
%   returns a 1xNS  matrix in C with a cluster number for each dipole.
%   Cluster 0 is the null parcel.
%   NEIGHBORS is the neighbor matrix (0s on the diagonal).
%   SCORES is a 1xNS matrix of score for each dipole.
%   OPTIONS.neighborhood_order is the is the level of neighborhood for
%   each cluster, it represents the spatial extent of a cluster.
%   OPTIONS.MSP_scores_threshold is the threshold above wich every dipole
%   will be selected, based on their score.
%   If THRESHOLD is set to 0 every dipole will be part of a cluster, i.e.
%   the null parcel will be empty.
%
%   NOTE1:
%   Clusters will be created until one of the conditions is reached.
%   Conditions are : No more dipoles have a SCORE above the THRESHOLD
%
%   NOTE2:
%   A cluster must have at least 3 dipoles to be created.
%
%% ==============================================
% Copyright (C) 2011 - LATIS Team
%
%  Authors: LATIS team, 2011
%
%% ==============================================
% License
%
% BEst is free software: you can redistribute it and/or modify
%    it under the terms of the GNU General Public License as published by
%    the Free Software Foundation, either version 3 of the License, or
%    (at your option) any later version.
%
%    BEst is distributed in the hope that it will be useful,
%    but WITHOUT ANY WARRANTY; without even the implied warranty of
%    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%    GNU General Public License for more details.
%
%    You should have received a copy of the GNU General Public License
%    along with BEst. If not, see <http://www.gnu.org/licenses/>.
% -------------------------------------------------------------------------

%mem = [OPTIONS.InverseMethod  ', (clustering)'];

% Default options settings
Def_OPTIONS.clustering = struct(...
    'neighborhood_order',   4, ...
    'MSP_scores_threshold', 0 );
OPTIONS.clustering.MSP_scores_threshold = 0;
% Return empty OPTIONS structure
if (nargin == 0)
    OPTIONS = Def_OPTIONS;
    %     clusters = [];
    return
end
% Check field names of passed OPTIONS and fill missing ones with default values

%OPTIONS = be_struct_copy_fields(OPTIONS, Def_OPTIONS, {'clustering'});
clear Def_OPTIONS

%Adds connectivity on the diagonal (there should be zeros on the diagonal of nm
%initially)
nm=nm+speye(size(nm));

% Initialization
% Number of sources
nb_sources = numel(scores);

% Setting the neighbor matrix to the appropriate degree
neighborhood = nm^extent;

% Sort the scores then find which indices are greater than the threshold
[sorted_scores, indices] = sort(scores,'descend');

thresh_index = nb_sources;
if OPTIONS.clustering.MSP_scores_threshold
    thresh_index = find(sorted_scores >= OPTIONS.clustering.MSP_scores_threshold, 1, 'last');
end


% The tresh_index will be empty if no score are greater than the threshold
if isempty(thresh_index)
    thresh_index = -1;
end

% intialization
ii = 1;
selected_source = zeros(nb_sources,1);
cluster_no = 1;
seed = [];
% Cluster creation
% Clusters will be created until one of the condition is reached.
% Conditions are the threshold and the number parameters
while (ii <= thresh_index)
    % The node with the highest score is selected
    node = indices(ii);
    % Verification that the node is not part of a cluster
    if selected_source(node) == 0
        
        % Getting the neighbors of the node
        % neighbors = unique([find(neighborhood(node,:)) node]);
        neighbors = find(neighborhood(node,:));
        
        % If a node is already in a cluster, it stays in the old cluster.
        neighbors(selected_source(neighbors) ~= 0) = [];
        
        
        % Saving the cluster if its big enough (minimum 3 dipoles)
        if numel(neighbors) >= 5
            selected_source(neighbors) = cluster_no;
            %             clusters{1,cluster_no} = neighbors;
            cluster_no = cluster_no + 1;
            seed = [seed;node];
        end
        
    end
    ii = ii + 1;
end

% After the first pass is completed, we make sure that all the dipoles over
% the threshold were selected.
% The 'free' dipoles are merged to the lowest nearest cluster.
free_nodes = indices(selected_source(indices(1:thresh_index))==0);
% free_nodes = find(~selected_source);
while ~isempty(free_nodes)
    for free_node = free_nodes'
        % Getting the neighbors of the free_node
        %neighbors = unique([find(neighborhood(free_nodes(ii),:)) free_nodes(ii)]);
        neighbors = find(nm(free_node,:));
        
        % Removing the cluster 0
        neighbors(~selected_source(neighbors)) = [];
        
        if ~isempty(neighbors)
            % Selecting the lowers closest cluster
            cluster_no = min(selected_source(neighbors));
            % Adding the free node to the cluster
            selected_source(free_node) = cluster_no;
            % Delete the node from free_nodes array
            free_nodes(free_nodes==free_node) = [];
        end
        
    end
end

if nargout > 2
    cellstruct = cell(1, max(selected_source));
    for ii = 1 : max(selected_source)
        cellstruct(ii) = {find(selected_source == ii)'};
    end
end

return





function [scores] = msp(M, Gstruct)
%   BE_MSP returns a vector of MSP scores as proposed by Mattout(2005).
%   [OPTIONS, scores] = BE_MSP(M, Gstruct, OPTIONS) returns the MSP scores
%   of the sources associated, with the data M and the structure Gstruct.
%   M is a matrix of dimension [number of sensors x number of time samples]
%   Gstruct is a structure that contains the column-wise normalized G,
%   the svd factors U and the diagoanal of singular values:
%   Gstruct.G is a matrix of dimension [number of sensors x number of
%   sources].
%   OPTIONS contains a threshold used to filter the data and the forward
%   operator (see be_memsolver_multiM and reference below).
%
%   Output:
%   scores is a vector of dimension [#sources x 1].
%
% 	Reference
%       Mattout, J., M. Pelegrini-Issac, L. Garnero et H. Benali. 2005.
%       Multivariate source prelocalization (MSP): Use of functionally
%       informed basis functions for better conditioning the MEG inverse
%       problem. NeuroImage, vol. 26, no 2, p. 356-373.
%
%% ==============================================
% Copyright (C) 2011 - LATIS Team
%
%  Authors: LATIS team, 2011
%
%% ==============================================
% License
%
% BEst is free software: you can redistribute it and/or modify
%    it under the terms of the GNU General Public License as published by
%    the Free Software Foundation, either version 3 of the License, or
%    (at your option) any later version.
%
%    BEst is distributed in the hope that it will be useful,
%    but WITHOUT ANY WARRANTY; without even the implied warranty of
%    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%    GNU General Public License for more details.
%
%    You should have received a copy of the GNU General Public License
%    along with BEst. If not, see <http://www.gnu.org/licenses/>.
% -------------------------------------------------------------------------


% Default options settings
% Def_OPTIONS.clustering = struct('MSP_R2_threshold', .92);
OPTIONS.clustering.MSP_R2_threshold = 0.92;
% Return empty OPTIONS structure
if (nargin == 0)
    OPTIONS = Def_OPTIONS;
    return
end
% Check field names of passed OPTIONS and fill missing ones with default values
scores  =zeros(size(Gstruct.Gn,2),1);
% clear Def_OPTIONS

% Check for NaN in the data
if sum( any(isnan(M)) )
    return
end

% Normalize the data matrix. Eliminate any resulting NaN.
Mn = bsxfun(@rdivide, M, sqrt(sum(M.^2, 1)));
Mn(isnan(Mn)) = 0;

% Project the normalized data on eigenvectors.
gamma = Gstruct.U'*Mn;

% Calculate the multiple correlation coefficients R2.
% NOTE:
%   This step is different than what is proposed in the reference.
%   R2 is defined as:
%   R2 = diag(gamma * pinv(gamma'*gamma) * gamma');
% Actually, gamma'*gamma is a large matrix close to the Identity. We thus
% neglect it. This is of no consequence on the final scores
R2 = diag(gamma*gamma');

% Reorder the singular values as a function of R2.
[temp, indices] = sort(R2,'descend');
lambda = Gstruct.lambda(indices);

% Select the columns of B as a function of the ordered singular values up
% to the threshold value.
i_T = indices( 1:find(cumsum(lambda)./sum(lambda ) >= OPTIONS.clustering.MSP_R2_threshold,1) );
Ut = Gstruct.U(:,sort(i_T));

% Create the projector.(accelarate the pinv with svd Ms instead of Ms'*Ms)
Ms = Ut*Ut'*Mn;
% Ps = Ms*pinv(Ms'*Ms)*Ms';
[~,S,V] = svd(Ms,'econ');
s = diag(S);
tol = sqrt(max(size(Ms)) * eps(norm(s,inf)));
r = sum(s > tol)+1;s(r:end) = [];V(:,r:end) = [];
s = (1./(s.^2))';
temp = Ms*V;
Ps = temp.*s*temp';

% clear Ms Ut R2 gamma C Mn indices i_T M V S s temp tol

% Calculate the MSP scores.
scores = sum(Ps*Gstruct.Gn.*Gstruct.Gn,1)';
return

% function [W,W1] = spatial_priorw(neighbors)
% %   This function returns the W Green matrix from which the local
% %   covariance matrices will be obtained.
% %
% %   INUPTS:
% % 		- OPTIONS    	: structure of parameters
% %       - neighbors		: neighbor matrix (with 0s on the diag)
% %
% %   OUTPUTS:
% %		- OPTIONS		: Keep track of parameters
% %		- W				: Green Matrix
% %
% % 	Reference
% %       Harrison et al., NIMG 38(4), 677-695 (2007
% %       Friston et al., NIMG 39(3), 1104-1120 (2008)
% %
% %   Authors: LATIS team, 2011.
% %% ==============================================
% % License
% %
% % BEst is free software: you can redistribute it and/or modify
% %    it under the terms of the GNU General Public License as published by
% %    the Free Software Foundation, either version 3 of the License, or
% %    (at your option) any later version.
% %
% %    BEst is distributed in the hope that it will be useful,
% %    but WITHOUT ANY WARRANTY; without even the implied warranty of
% %    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% %    GNU General Public License for more details.
% %
% %    You should have received a copy of the GNU General Public License
% %    along with BEst. If not, see <http://www.gnu.org/licenses/>.
% % -------------------------------------------------------------------------
%
%
% % % Default options settings
% % Def_OPTIONS.solver = struct('spatial_smoothing', 0.6);
% %
% % % Return empty OPTIONS structure
% % if (nargin == 0)
% %     OPTIONS = Def_OPTIONS;
% %     return
% % end
% % % Check field names of passed OPTIONS and fill missing ones with default values
% % OPTIONS = be_struct_copy_fields(OPTIONS, Def_OPTIONS, {'solver'}, 0);
% % clear Def_OPTIONS
% %%
% % Parameters:
% nb_vertices = size(neighbors,2);
% rho = 0.6;%OPTIONS.solver.spatial_smoothing; % scalar that weight the adjacency matrix
% W   = speye(nb_vertices);
%
%
% % % Add comment to result
% % if rho && ~isempty(neighbors)
% %     OPTIONS.automatic.Comment = [OPTIONS.automatic.Comment ' | smooth=' num2str(rho)];
% % else
% %     return
% % end
%
% % Preallocation of the sparse matrix
% A  = neighbors - spdiags(sum(neighbors,2),0,nb_vertices,nb_vertices);
% A0 = rho*A/2;
% for i = 1:7
%     W = W + A0;
%     A0 = rho/2*A0*A / (i+1);
% end
% W1 = W.*(W > exp(-8));
% W = W1'*W1;
% W1 = W1';
% return
