package players.mctsEAHeuristic;

import core.AbstractGameState;
import core.AbstractPlayer;
import core.interfaces.*;
import evaluation.optimisation.TunableParameters;
import games.root_final.EvolutionaryHeuristicGenerator;
import org.jetbrains.annotations.NotNull;
import players.PlayerParameters;
import players.simple.RandomPlayer;
import utilities.JSONUtils;

import java.util.Arrays;
import java.util.Random;

import static players.mcts.MCTSEnums.Information.Closed_Loop;
import static players.mcts.MCTSEnums.Information.Information_Set;
import static players.mcts.MCTSEnums.MASTType.None;
import static players.mcts.MCTSEnums.OpponentTreePolicy.OneTree;
import static players.mcts.MCTSEnums.RolloutTermination.DEFAULT;
import static players.mcts.MCTSEnums.SelectionPolicy.SIMPLE;
import static players.mcts.MCTSEnums.Strategies.PARAMS;
import static players.mcts.MCTSEnums.Strategies.RANDOM;
import static players.mcts.MCTSEnums.TreePolicy.*;

public class MCTSEAHParams extends PlayerParameters {

    public double K = Math.sqrt(2);
    public int rolloutLength = 10; // assuming we have a good heuristic
    public boolean rolloutLengthPerPlayer = false;  // if true, then rolloutLength is multiplied by the number of players
    public int maxTreeDepth = 1000; // effectively no limit
    public MCTSEAHEnums.Information information = MCTSEAHEnums.Information.Information_Set;  // this should be the default in TAG, given that most games have hidden information
    public MCTSEAHEnums.MASTType MAST = MCTSEAHEnums.MASTType.None;
    public boolean useMAST = false;
    public double MASTGamma = 0.0;
    public double MASTDefaultValue = 0.0;
    public double MASTBoltzmann = 0.1;
    public double exp3Boltzmann = 1.0;
    public double hedgeBoltzmann = 100;
    public boolean useMASTAsActionHeuristic = false;
    public MCTSEAHEnums.SelectionPolicy selectionPolicy = MCTSEAHEnums.SelectionPolicy.SIMPLE;  // In general better than ROBUST
    public MCTSEAHEnums.TreePolicy treePolicy = MCTSEAHEnums.TreePolicy.UCB;
    public MCTSEAHEnums.OpponentTreePolicy opponentTreePolicy = MCTSEAHEnums.OpponentTreePolicy.OneTree;
    public boolean paranoid = false;
    public MCTSEAHEnums.Strategies rolloutType = MCTSEAHEnums.Strategies.RANDOM;
    public MCTSEAHEnums.Strategies oppModelType = MCTSEAHEnums.Strategies.DEFAULT;  // Default is to use the same as rolloutType
    public String rolloutClass, oppModelClass = "";
    public AbstractPlayer rolloutPolicy;
    public ITunableParameters rolloutPolicyParams;
    public AbstractPlayer opponentModel;
    public ITunableParameters opponentModelParams;
    public double exploreEpsilon = 0.1;
    public int omaVisits = 0;
    public boolean normaliseRewards = true;  // This will automatically normalise rewards to be in the range [0,1]
    // so that K does not need to be tuned to the precise scale of reward in a game
    // It also means that at the end of the game (when rewards are possibly closer to each other, they are still scaled to [0, 1]
    public boolean maintainMasterState = false;
    public boolean discardStateAfterEachIteration = true;  // default will remove reference to OpenLoopState in backup(). Saves memory!
    public MCTSEAHEnums.RolloutTermination rolloutTermination = MCTSEAHEnums.RolloutTermination.DEFAULT;
    public IStateHeuristic heuristic = AbstractGameState::getHeuristicScore;
    public IActionKey MASTActionKey;
    public IStateKey MCGSStateKey;
    public boolean MCGSExpandAfterClash = true;
    public double firstPlayUrgency = 1000000000.0;
    @NotNull public IActionHeuristic actionHeuristic = IActionHeuristic.nullReturn;
    public int actionHeuristicRecalculationThreshold = 20;
    public boolean pUCT = false;  // in this case we multiply the exploration value in UCB by the probability that the action heuristic would take the action
    public double pUCTTemperature = 0.0;  // If greater than zero we construct a Boltzmann distribution over actions based on the action heuristic
    // if zero (or less) then we use the action heuristic values directly, setting any negative values to zero)
    public int initialiseVisits = 0;  // This is the number of visits to initialise the MCTS tree with (using the actionHeuristic)
    public double progressiveWideningConstant = 0.0; //  Zero indicates switched off (well, less than 1.0)
    public double progressiveWideningExponent = 0.0;
    public double progressiveBias = 0.0;
    public boolean reuseTree = false;
    public int maxBackupThreshold = 1000000;

    public EvolutionaryHeuristicGenerator.crossoverTypes heuristicCrossover;
    public String heuristicPath;
    public int heuristicPopulationSize;

    public boolean train;


    public MCTSEAHParams() {
        addTunableParameter("K", Math.sqrt(2), Arrays.asList(0.0, 0.1, 1.0, Math.sqrt(2), 3.0, 10.0));
        addTunableParameter("MASTBoltzmann", 0.1);
        addTunableParameter("exp3Boltzmann", 1.0);
        addTunableParameter("hedgeBoltzmann", 100.0);
        addTunableParameter("rolloutLength", 10, Arrays.asList(0, 3, 10, 30, 100));
        addTunableParameter("rolloutLengthPerPlayer", false);
        addTunableParameter("maxTreeDepth", 1000, Arrays.asList(1, 3, 10, 30, 100));
        addTunableParameter("rolloutType", MCTSEAHEnums.Strategies.RANDOM, Arrays.asList(MCTSEAHEnums.Strategies.values()));
        addTunableParameter("oppModelType", MCTSEAHEnums.Strategies.RANDOM, Arrays.asList(MCTSEAHEnums.Strategies.values()));
        addTunableParameter("rolloutClass", "");
        addTunableParameter("oppModelClass", "");
        addTunableParameter("rolloutPolicyParams", ITunableParameters.class);
        addTunableParameter("rolloutTermination", MCTSEAHEnums.RolloutTermination.DEFAULT, Arrays.asList(MCTSEAHEnums.RolloutTermination.values()));
        addTunableParameter("opponentModelParams", ITunableParameters.class);
        addTunableParameter("opponentModel", new RandomPlayer());
        addTunableParameter("information", MCTSEAHEnums.Information.Information_Set, Arrays.asList(MCTSEAHEnums.Information.values()));
        addTunableParameter("selectionPolicy", MCTSEAHEnums.SelectionPolicy.SIMPLE, Arrays.asList(MCTSEAHEnums.SelectionPolicy.values()));
        addTunableParameter("treePolicy", MCTSEAHEnums.TreePolicy.UCB, Arrays.asList(MCTSEAHEnums.TreePolicy.values()));
        addTunableParameter("opponentTreePolicy", MCTSEAHEnums.OpponentTreePolicy.OneTree, Arrays.asList(MCTSEAHEnums.OpponentTreePolicy.values()));
        addTunableParameter("exploreEpsilon", 0.1);
        addTunableParameter("heuristic", (IStateHeuristic) AbstractGameState::getHeuristicScore);
        addTunableParameter("opponentHeuristic", (IStateHeuristic) AbstractGameState::getHeuristicScore);
        addTunableParameter("MAST", MCTSEAHEnums.MASTType.None, Arrays.asList(MCTSEAHEnums.MASTType.values()));
        addTunableParameter("MASTGamma", 0.0, Arrays.asList(0.0, 0.5, 0.9, 1.0));
        addTunableParameter("useMASTAsActionHeuristic", false);
        addTunableParameter("progressiveWideningConstant", 0.0, Arrays.asList(0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0));
        addTunableParameter("progressiveWideningExponent", 0.0, Arrays.asList(0.0, 0.1, 0.2, 0.3, 0.5));
        addTunableParameter("normaliseRewards", true);
        addTunableParameter("maintainMasterState", false);
        addTunableParameter("discardStateAfterEachIteration", true);
        addTunableParameter("omaVisits", 0);
        addTunableParameter("paranoid", false);
        addTunableParameter("MASTActionKey", IActionKey.class);
        addTunableParameter("MASTDefaultValue", 0.0);
        addTunableParameter("MCGSStateKey", IStateKey.class);
        addTunableParameter("MCGSExpandAfterClash", true);
        addTunableParameter("FPU", 1000000000.0);
        addTunableParameter("actionHeuristic",  IActionHeuristic.nullReturn);
        addTunableParameter("progressiveBias", 0.0);
        addTunableParameter("pUCT", false);
        addTunableParameter("pUCTTemperature", 0.0);
        addTunableParameter("initialiseVisits", 0);
        addTunableParameter("actionHeuristicRecalculation", 20);
        addTunableParameter("reuseTree", false);
        addTunableParameter("maxBackupThreshold", 1000000);
        addTunableParameter("heuristicCrossover", EvolutionaryHeuristicGenerator.crossoverTypes.UNIFORM, Arrays.asList(EvolutionaryHeuristicGenerator.crossoverTypes.values()));
        addTunableParameter("heuristicPath", "");
        addTunableParameter("heuristicPopulationSize", 15);
        addTunableParameter("train", true);
    }

    @Override
    public void _reset() {
        super._reset();
        K = (double) getParameterValue("K");
        rolloutLength = (int) getParameterValue("rolloutLength");
        rolloutLengthPerPlayer = (boolean) getParameterValue("rolloutLengthPerPlayer");
        maxTreeDepth = (int) getParameterValue("maxTreeDepth");
        rolloutType = (MCTSEAHEnums.Strategies) getParameterValue("rolloutType");
        rolloutTermination = (MCTSEAHEnums.RolloutTermination) getParameterValue("rolloutTermination");
        oppModelType = (MCTSEAHEnums.Strategies) getParameterValue("oppModelType");
        information = (MCTSEAHEnums.Information) getParameterValue("information");
        treePolicy = (MCTSEAHEnums.TreePolicy) getParameterValue("treePolicy");
        selectionPolicy = (MCTSEAHEnums.SelectionPolicy) getParameterValue("selectionPolicy");
        if (selectionPolicy == MCTSEAHEnums.SelectionPolicy.TREE &&
                (treePolicy == MCTSEAHEnums.TreePolicy.UCB || treePolicy == MCTSEAHEnums.TreePolicy.UCB_Tuned || treePolicy == MCTSEAHEnums.TreePolicy.AlphaGo)) {
            // in this case TREE is equivalent to SIMPLE
            selectionPolicy = MCTSEAHEnums.SelectionPolicy.SIMPLE;
        }
        opponentTreePolicy = (MCTSEAHEnums.OpponentTreePolicy) getParameterValue("opponentTreePolicy");
        exploreEpsilon = (double) getParameterValue("exploreEpsilon");
        MASTBoltzmann = (double) getParameterValue("MASTBoltzmann");
        MAST = (MCTSEAHEnums.MASTType) getParameterValue("MAST");
        MASTGamma = (double) getParameterValue("MASTGamma");
        exp3Boltzmann = (double) getParameterValue("exp3Boltzmann");
        hedgeBoltzmann = (double) getParameterValue("hedgeBoltzmann");
        rolloutClass = (String) getParameterValue("rolloutClass");
        oppModelClass = (String) getParameterValue("oppModelClass");

        progressiveBias = (double) getParameterValue("progressiveBias");
        omaVisits = (int) getParameterValue("omaVisits");
        progressiveWideningConstant = (double) getParameterValue("progressiveWideningConstant");
        progressiveWideningExponent = (double) getParameterValue("progressiveWideningExponent");
        normaliseRewards = (boolean) getParameterValue("normaliseRewards");
        maintainMasterState = (boolean) getParameterValue("maintainMasterState");
        paranoid = (boolean) getParameterValue("paranoid");
        discardStateAfterEachIteration = (boolean) getParameterValue("discardStateAfterEachIteration");
        pUCT = (boolean) getParameterValue("pUCT");
        pUCTTemperature = (double) getParameterValue("pUCTTemperature");
        if (information == MCTSEAHEnums.Information.Closed_Loop)
            discardStateAfterEachIteration = false;

        MASTActionKey = (IActionKey) getParameterValue("MASTActionKey");
        MASTDefaultValue = (double) getParameterValue("MASTDefaultValue");

        actionHeuristic = (IActionHeuristic) getParameterValue("actionHeuristic");
        heuristic = (IStateHeuristic) getParameterValue("heuristic");
        MCGSStateKey = (IStateKey) getParameterValue("MCGSStateKey");
        MCGSExpandAfterClash = (boolean) getParameterValue("MCGSExpandAfterClash");
        rolloutPolicyParams = (TunableParameters) getParameterValue("rolloutPolicyParams");
        opponentModelParams = (TunableParameters) getParameterValue("opponentModelParams");
        // we then null those elements of params which are constructed (lazily) from the above
        firstPlayUrgency = (double) getParameterValue("FPU");
        initialiseVisits = (int) getParameterValue("initialiseVisits");
        actionHeuristicRecalculationThreshold = (int) getParameterValue("actionHeuristicRecalculation");
        reuseTree = (boolean) getParameterValue("reuseTree");
        maxBackupThreshold = (int) getParameterValue("maxBackupThreshold");
        opponentModel = null;
        rolloutPolicy = null;
        useMASTAsActionHeuristic = (boolean) getParameterValue("useMASTAsActionHeuristic");
        useMAST = MAST != MCTSEAHEnums.MASTType.None;
        heuristicCrossover = (EvolutionaryHeuristicGenerator.crossoverTypes) getParameterValue("heuristicCrossover");
        heuristicPath = (String) getParameterValue("heuristicPath");
        heuristicPopulationSize = (int) getParameterValue("heuristicPopulationSize");
        train = (boolean) getParameterValue("train");
    }

    @Override
    protected MCTSEAHParams _copy() {
        // All the copying is done in TunableParameters.copy()
        // Note that any *local* changes of parameters will not be copied
        // unless they have been 'registered' with setParameterValue("name", value)
        return new MCTSEAHParams();
    }

    public AbstractPlayer getOpponentModel() {
        if (opponentModel == null) {
            if (oppModelType == MCTSEAHEnums.Strategies.PARAMS)
                opponentModel = (AbstractPlayer) opponentModelParams.instantiate();
            else if (oppModelType == MCTSEAHEnums.Strategies.DEFAULT)
                opponentModel = getRolloutStrategy();
            else
                opponentModel = constructStrategy(oppModelType, oppModelClass);
            opponentModel.getParameters().actionSpace = actionSpace;  // TODO makes sense?
        }
        return opponentModel;
    }

    public AbstractPlayer getRolloutStrategy() {
        if (rolloutPolicy == null) {
            if (rolloutType == MCTSEAHEnums.Strategies.PARAMS)
                rolloutPolicy = (AbstractPlayer) rolloutPolicyParams.instantiate();
            else
                rolloutPolicy = constructStrategy(rolloutType, rolloutClass);
            rolloutPolicy.getParameters().actionSpace = actionSpace;  // TODO makes sense?
        }
        return rolloutPolicy;
    }

    private AbstractPlayer constructStrategy(MCTSEAHEnums.Strategies type, String details) {
        switch (type) {
            case RANDOM:
                return new RandomPlayer(new Random(getRandomSeed()));
            case MAST:
                return new MASTPlayer(MASTActionKey, MASTBoltzmann, 0.0, getRandomSeed(), MASTDefaultValue);
            case CLASS:
                // we have a bespoke Class to instantiate
                return JSONUtils.loadClassFromString(details);
            case PARAMS:
                throw new AssertionError("PolicyParameters have not been set");
            default:
                throw new AssertionError("Unknown strategy type : " + type);
        }
    }

    public IStateHeuristic getHeuristic() {
        return heuristic;
    }

    @Override
    public MCTSEAHPlayer instantiate() {
        if (!useMAST && (useMASTAsActionHeuristic || rolloutType == MCTSEAHEnums.Strategies.MAST)) {
            throw new AssertionError("MAST data not being collected, but MAST is being used as the rollout policy or as the action heuristic. Set MAST parameter.");
        }
        return new MCTSEAHPlayer((MCTSEAHParams) this.copy());
    }

}
