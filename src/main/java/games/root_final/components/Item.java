package games.root_final.components;

import core.CoreConstants;
import core.components.Component;
import games.catan.components.Building;

import java.util.Objects;

public class Item extends Component {
    public enum ItemType{
        sword,
        boot,
        crossbow,
        torch,
        hammer,
        tea,
        coin,
        bag

    }
    public final ItemType itemType;
    public boolean refreshed = true;

    public boolean damaged = false;

    public Item(CoreConstants.ComponentType type, ItemType itemType){
        super(type, itemType.toString());
        this.itemType = itemType;
    }

    public Item(CoreConstants.ComponentType type, ItemType itemType, int componentID){
        super(type, itemType.toString(), componentID);
        this.itemType = itemType;

    }
    @Override
    public Item copy() {
        Item item =  new Item(type, itemType, componentID);
        item.damaged = damaged;
        item.refreshed = refreshed;
        return  item;
    }
    @Override
    public boolean equals(Object obj){
        if (this == obj) return true;
        if (!(obj instanceof games.root_final.components.Item)) return false;
        if (!super.equals(obj)) return false;
        Item that = (Item) obj;
        return itemType == that.itemType && damaged == that.damaged && refreshed == that.refreshed && componentID == that.componentID;
    }

    @Override
    public int hashCode(){
        return Objects.hash(super.hashCode(),itemType, refreshed, damaged);
    }
}
