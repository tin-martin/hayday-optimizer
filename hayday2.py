import json

# Load your JSON file
with open("hayday_goods.json", "r") as f:
    goods_data = json.load(f)

def base_ingredients(name):
    from treelib import Node, Tree

    tree = Tree()
    tree.create_node(name,name)  # No parent means its the root node
    base_ings = {}
    queue = []

    # Add initial ingredients with their quantities
    for ing, ing_qty in goods_data[name]['ingredients'].items():
        temp = tree.create_node(ing+ " ("+ str(ing_qty) + ")", parent=name)
        queue.append((ing, ing_qty,0, temp.identifier))
        

    record = []
    while queue:
        current, qty, depth, identifier = queue.pop()
        if(len(record) == depth):
            record.append({current:qty})
        else:
            if current in record[depth]:
                record[depth][current] += qty
            else:
                 record[depth][current] = qty
        
        # If current is not in goods_data → it's a base item
        if current not in goods_data:
            base_ings[current] = base_ings.get(current, 0) + qty
        elif current in goods_data[current]['ingredients'] or len(goods_data[current]['ingredients'].keys())== 0:
            base_ings[current] = base_ings.get(current, 0) + qty
        else:
            # Break down further: multiply quantity
            for ing, ing_qty in goods_data[current]['ingredients'].items():
                temp = tree.create_node(ing+ " ("+ str(ing_qty) + ")", parent=identifier)
                queue.append((ing , ing_qty * qty,depth+1, temp.identifier))
            
    #tree.show()
    return record, base_ings, tree
# Example usage
#_, base_ings = base_ingredients("Bacon fondue")

#print(base_ings)
"""
sorted_list = []
for name in goods_data.keys():
    _, base_ings, tree = base_ingredients(name)
    sum = 0
    for ings in base_ings:
        sum += base_ings[ings]
    sorted_list.append([tree,sum])
   
    

sorted_list = sorted(sorted_list, key=lambda x: x[1])
for i in range(1,10):
    sorted_list[-i][0].show()
 #   print(sorted_list[-i][1])
"""
import matplotlib.pyplot as plt
hist_data = []
for name in goods_data.keys():
    _, base_ings, tree = base_ingredients(name)
    sum = 0
    for ings in base_ings:
        sum += base_ings[ings]
    hist_data.append(sum)

plt.hist(hist_data)
