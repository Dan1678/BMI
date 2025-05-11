function modelParameters = positionEstimatorTraining(training_data)
% positionEstimatorTraining.m
%
% This function:
%   1) Trains a velocity regressor using 20-ms windows of spikes and hand velocities.
%   2) Computes average firing rates in the first 300 ms for each direction
%      (for naive direction classification).
%   3) Computes a convex hull for the hand positions belonging to each direction,
%      so that we can later restrict decoded positions to that hull.
%
% Arguments:
%   training_data: [n_trials x numAngles] struct array with fields:
%       - trialId
%       - spikes : [numNeurons x T]
%       - handPos: [2 x T]
%     Each trial spans from 300 ms before movement onset to 100 ms after movement end.
%
% Output:
%   modelParameters: struct containing:
%       - regressor.beta: ((numNeurons+1) x 2) velocity regression coefficients
%       - dt: 20 (window size in ms)
%       - avgFRperAngle: (8 x numNeurons) average firing rate in first 300 ms, per direction
%       - first300ms: 300
%       - directionHulls: cell array of size (8 x 1), each is [2 x M] hull boundary
%       - numNeurons: number of neurons

    dt = 20;          % 20-ms window for velocity regression
    first300ms = 300; % 300-ms window for direction classification

    [numTrials, numAngles] = size(training_data);
    numNeurons = size(training_data(1,1).spikes, 1);

    %% 1) Naive Direction Classifier Parameters
    avgFRperAngle = zeros(numAngles, numNeurons);  % (8 x numNeurons)
    for k = 1:numAngles
        spikeCountsAll = zeros(numTrials, numNeurons);
        for n = 1:numTrials
            T = min(size(training_data(n,k).spikes, 2), first300ms);
            spikeCountsAll(n,:) = sum(training_data(n,k).spikes(:, 1:T), 2)';
        end
        avgFRperAngle(k,:) = mean(spikeCountsAll, 1);
    end

    %% 2) Train Velocity Regressor (20-ms windows)
    X = [];  % feature matrix
    Y = [];  % velocity targets

    for k = 1:numAngles
        for n = 1:numTrials
            spikes  = training_data(n,k).spikes;         % [numNeurons x T_total]
            handPos = training_data(n,k).handPos(1:2,:); % [2 x T_total]
            T_total = size(spikes, 2);

            for t = 1:dt:(T_total - dt)
                feature  = sum(spikes(:, t:t+dt-1), 2);    % sum of spikes in [t..t+dt-1]
                posStart = handPos(:, t);
                posEnd   = handPos(:, t+dt);
                velocity = (posEnd - posStart) / dt;       % average velocity in that 20-ms window

                X = [X; feature'];
                Y = [Y; velocity'];
            end
        end
    end

    % Add bias
    X_aug = [ones(size(X,1),1), X];  % [nExamples x (numNeurons+1)]
    beta  = X_aug \ Y;               % ((numNeurons+1) x 2)

    %% 3) Compute a Convex Hull for Each Direction Separately
    directionHulls = cell(numAngles, 1);

    for k = 1:numAngles
        % Gather all positions for direction k
        allPos_k = [];
        for n = 1:numTrials
            allPos_k = [allPos_k, training_data(n,k).handPos(1:2,:)];  % [2 x T_k]
        end

        xPos = allPos_k(1,:);
        yPos = allPos_k(2,:);

        % Compute convex hull for direction k
        if size(allPos_k,2) < 3
            % Edge case: not enough points to form a hull
            % Just store all points as boundary (rare in real data).
            directionHulls{k} = allPos_k;
        else
            hullIndices = convhull(xPos, yPos);
            directionHulls{k} = [xPos(hullIndices); yPos(hullIndices)];  % [2 x M_k]
        end
    end

    %% Store parameters
    modelParameters.regressor.beta = beta;
    modelParameters.dt = dt;
    modelParameters.numNeurons = numNeurons;
    modelParameters.avgFRperAngle = avgFRperAngle;
    modelParameters.first300ms = first300ms;
    modelParameters.directionHulls = directionHulls;
end
