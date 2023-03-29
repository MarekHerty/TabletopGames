package games.hanabi.gui;

import core.components.Deck;
import core.components.PartialObservableDeck;
import games.hanabi.*;


import javax.swing.*;
import java.awt.*;

import static games.hanabi.gui.HanabiGUIManager.*;

public class HanabiPlayerView extends JComponent {

    // ID of player showing
    int playerId;
    // Number of points player has
    int nPoints;
    HanabiDeckView playerHandView;
    // Border offsets
    int border = 5;
    int borderBottom = 20;
    int width, height;

    public HanabiPlayerView(PartialObservableDeck<HanabiCard> d, int playerId, int humanId, String dataPath) {
        this.width = playerAreaWidth + border*2;
        this.height = playerAreaHeight + border + borderBottom;
        this.playerId = playerId;
        this.playerHandView = new HanabiDeckView(humanId, d, true, dataPath, new Rectangle(border, border, playerAreaWidth, hanabiCardHeight));
    }

    /**
     * Draws the player's hand and their number of points.
     * @param g - Graphics object.
     */
    @Override
    protected void paintComponent(Graphics g) {
        playerHandView.drawDeck((Graphics2D) g);
        g.setColor(Color.black);
        g.drawString(nPoints + " points", border+playerAreaWidth/2 - 20, border+hanabiCardHeight + 10);
    }

    @Override
    public Dimension getPreferredSize() {
        return new Dimension(width, height);
    }

    /**
     * Updates information
     * @param gameState - current game state
     */
    public void update(HanabiGameState gameState) {
        playerHandView.updateComponent(gameState.getPlayerDecks().get(playerId));
    }
}
