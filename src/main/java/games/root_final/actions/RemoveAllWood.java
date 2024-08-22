package games.root_final.actions;

import core.AbstractGameState;
import core.actions.AbstractAction;
import games.root_final.RootGameState;
import games.root_final.RootParameters;
import games.root_final.components.RootBoardNodeWithRootEdges;

import java.util.Objects;

public class RemoveAllWood extends AbstractAction {
    public final int playerID;

    public RemoveAllWood(int playerID){
        this.playerID = playerID;
    }
    @Override
    public boolean execute(AbstractGameState gs) {
        RootGameState currentState = (RootGameState) gs;
        if (currentState.getCurrentPlayer() == playerID && currentState.getPlayerFaction(playerID) == RootParameters.Factions.MarquiseDeCat){
            for (RootBoardNodeWithRootEdges location: currentState.getGameMap().getNonForrestBoardNodes()){
                if (location.getWood() > 0){
                    for (int i = 0; i < location.getWood(); i++){
                        location.removeToken(RootParameters.TokenType.Wood);
                        currentState.addToken(RootParameters.TokenType.Wood);
                    }
                }
            }
            return true;
        }
        return false;
    }

    @Override
    public AbstractAction copy() {
        return this;
    }

    @Override
    public boolean equals(Object obj) {
        if (obj == this){return true;}
        if (obj instanceof RemoveAllWood r){
            return playerID == r.playerID;
        }
        return false;
    }

    @Override
    public int hashCode() {
        return Objects.hash("RemoveAllWood", playerID);
    }

    @Override
    public String getString(AbstractGameState gameState) {
        RootGameState gs = (RootGameState) gameState;
        return gs.getPlayerFaction(playerID).toString() + " uses all wood on the map";
    }
}
