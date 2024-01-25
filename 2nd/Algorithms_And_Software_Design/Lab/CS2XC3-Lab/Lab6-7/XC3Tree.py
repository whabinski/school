
values = [-i for i in range(-100,0)]

class XC3Tree:
    
    def __init__(self, degree):
        #self.value = values.pop()
        self.degree = degree
        self.parent = None
        self.children = []
        self.make_children()
        
    def make_children(self):
        for x in range(self.degree):
            x = x + 1
            degree = (x - 2) if (x > 2) else 0
            self.x = XC3Tree(degree)
            self.children.append(self.x)
            self.x.parent = self
    
    def count_nodes(self):
        count = 1
        for child in self.children:
            count += child.count_nodes()
        return count
    

# TESTING ******************************************************************************

#t = XC3Tree(6)

#print(t.count_nodes())