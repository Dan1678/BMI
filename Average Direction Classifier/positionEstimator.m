function [x, y] = positionEstimator(test_data, modelParameters)

    persistent trialState
    
    if isempty(trialState)
        trialState = struct();
    end

    tID = test_data.trialId;
    
    if ~isfield(trialState, ['tid_' num2str(tID)])

        % Classify which angle (k) it is (1..8)
        direction_id = classifyDirection(test_data, modelParameters);

        % Initialize the state for this new trial
        s.direction_id  = direction_id;
        s.startHandPos  = test_data.startHandPos;
        s.initialAvgPos = modelParameters.avgTrajectory{direction_id}(:,1);
        s.currentTime   = 0;

        trialState.(['tid_' num2str(tID)]) = s;
    end
    
    s = trialState.(['tid_' num2str(tID)]);

    T = size(test_data.spikes,2);  % # of ms so far

    direction_id = s.direction_id;
    
    % Get the average trajectory for that direction
    avgTraj = modelParameters.avgTrajectory{direction_id};
    Tangle  = size(avgTraj,2); 

    % If T exceeds Tangle, clamp
    if T > Tangle
        T = Tangle;
    end
    
    % The average offset from avgTraj(:,1) to avgTraj(:,T)
    offset = avgTraj(:,T) - avgTraj(:,1);

    % So the predicted position is (startHandPos + offset)
    x = s.startHandPos(1) + offset(1);
    y = s.startHandPos(2) + offset(2);

    s.currentTime = T;

    trialState.(['tid_' num2str(tID)]) = s;
end


function direction_id = classifyDirection(test_data, modelParams)
% Use the first 300 ms of spiking to guess angle ID (1..8).

    numAngles  = modelParams.numAngles;
    avgFR      = modelParams.avgFRperAngle;
    first300   = modelParams.first300ms;

    spikesSoFar = test_data.spikes;
    T = size(spikesSoFar,2);
    Tuse = min(first300, T);

    % sum across the first ms, for each of the 98 neurons
    testCount = sum(spikesSoFar(:,1:Tuse),2);

    % pick the angle whose average 300-ms firing is "closest" (Euclidean distance)
    dists = zeros(numAngles,1);
    for k=1:numAngles
        diffVec = avgFR(k,:)' - testCount;
        dists(k) = sum(diffVec.^2);
    end

    [~, direction_id] = min(dists);
end
