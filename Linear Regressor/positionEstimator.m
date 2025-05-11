function [x, y] = positionEstimator(test_data, modelParameters)
% positionEstimator.m
%
% Decodes hand position by:
%   1) Classifying the trial's direction (1..8) using first 300 ms of spikes.
%   2) Integrating velocity predictions in 20-ms windows.
%   3) Restricting the predicted position to remain within the convex hull
%      corresponding to the chosen direction.
%
% Inputs:
%   test_data: struct with fields:
%       - trialId      : unique trial identifier
%       - startHandPos : [2 x 1], hand position 300 ms before movement onset
%       - spikes       : [numNeurons x T], spike data from t=1..T
%
%   modelParameters: struct from training with fields:
%       - regressor.beta : ((numNeurons+1) x 2)
%       - dt             : 20
%       - avgFRperAngle  : (8 x numNeurons), for classification
%       - first300ms     : 300
%       - directionHulls : cell(8,1), each a [2 x M] hull boundary for that direction
%
% Output:
%   [x, y]: estimated 2D position at time T.

    persistent trialState
    if isempty(trialState)
        trialState = struct();
    end

    tID = test_data.trialId;

    % If new trial, classify direction & init state
    if ~isfield(trialState, ['tid_' num2str(tID)])
        dir_id = classifyDirection(test_data, modelParameters);

        s.direction_id = dir_id;
        s.startHandPos = test_data.startHandPos;   % [2 x 1]
        s.cumulativeDisplacement = [0; 0];
        s.lastProcessedTime = 0;

        % Store in persistent struct
        trialState.(['tid_' num2str(tID)]) = s;
    end

    % Retrieve state
    s = trialState.(['tid_' num2str(tID)]);

    % Process new 20-ms windows
    T_available = size(test_data.spikes, 2);
    dt   = modelParameters.dt;
    beta = modelParameters.regressor.beta;

    while s.lastProcessedTime + dt <= T_available
        wStart = s.lastProcessedTime + 1;
        wEnd   = s.lastProcessedTime + dt;

        % Sum spikes in this 20-ms window
        feature = sum(test_data.spikes(:, wStart:wEnd), 2);
        feature_aug = [1; feature];

        % Predict velocity (2 x 1)
        predicted_velocity = beta' * feature_aug;
        displacement = predicted_velocity * dt;

        % Update cumulative displacement
        s.cumulativeDisplacement = s.cumulativeDisplacement + displacement;
        s.lastProcessedTime = s.lastProcessedTime + dt;
    end

    % Raw predicted position
    rawPos = s.startHandPos + s.cumulativeDisplacement;

    % Restrict to the hull for this direction
    dir_id = s.direction_id;
    hullBoundary = modelParameters.directionHulls{dir_id};  % [2 x M]

    % Check if rawPos is inside that hull
    [in, on] = inpolygon(rawPos(1), rawPos(2), hullBoundary(1,:), hullBoundary(2,:));
    if ~(in || on)
        % If outside, project to the hull
        finalPos = projectToPolygon(rawPos, hullBoundary);
    else
        finalPos = rawPos;
    end

    % Update the cumulative displacement to reflect any clamping
    s.cumulativeDisplacement = finalPos - s.startHandPos;

    % Return final position
    x = finalPos(1);
    y = finalPos(2);

    % Save state
    trialState.(['tid_' num2str(tID)]) = s;
end

%----------------------------------------------------------------------
% Helper: classifyDirection
%----------------------------------------------------------------------
function direction_id = classifyDirection(test_data, modelParams)
    avgFR = modelParams.avgFRperAngle;  % (8 x numNeurons)
    first300 = modelParams.first300ms;

    spikesSoFar = test_data.spikes;       % (numNeurons x T)
    T = size(spikesSoFar,2);
    Tuse = min(first300, T);

    testCount = sum(spikesSoFar(:,1:Tuse), 2);  % (numNeurons x 1)

    dists = zeros(size(avgFR,1),1);
    for k = 1:size(avgFR,1)
        diffVec = avgFR(k,:)' - testCount;
        dists(k) = sum(diffVec.^2);
    end

    [~, direction_id] = min(dists);
end

%----------------------------------------------------------------------
% Helper: projectToPolygon
%   Projects a point onto the nearest point on the polygon boundary
%----------------------------------------------------------------------
function proj = projectToPolygon(point, poly)
    % point: 2 x 1
    % poly : 2 x M (convex hull boundary in order)
    minDist = inf;
    proj = point;
    numVertices = size(poly,2);

    for i = 1:numVertices-1
        p1 = poly(:,i);
        p2 = poly(:,i+1);
        candidate = projectPointOnSegment(point, p1, p2);
        d = norm(point - candidate);
        if d < minDist
            minDist = d;
            proj = candidate;
        end
    end

    % Check last edge from poly(:,end) to poly(:,1)
    p1 = poly(:, end);
    p2 = poly(:, 1);
    candidate = projectPointOnSegment(point, p1, p2);
    d = norm(point - candidate);
    if d < minDist
        proj = candidate;
    end
end

%----------------------------------------------------------------------
% Helper: projectPointOnSegment
%   Projects a point onto the line segment [p1,p2], clamped
%----------------------------------------------------------------------
function candidate = projectPointOnSegment(point, p1, p2)
    v = p2 - p1;
    w = point - p1;
    t = dot(w, v) / dot(v, v);
    t = max(0, min(1, t));
    candidate = p1 + t * v;
end
