
class DAGLongestPath:
    """Calculate the longest path in a directed acyclic graph (DAG) in terms of node weights
    
    Use this class to get (one of) the paths with the largest sum of node weights
    in a directed acyclic graph (DAG). After constructing the empty object,
    use `add_node(label, weight)` and `add_edge(label1, label2)` to build the graph, 
    and then call `longest_path` to retrieve the path and the sum of the weights.
    This latter operation is destructive and will delete the graph.
    """
    
    def __init__(self):
        """Construct a new empty graph."""
        self.nodes = {}  # Dictionary {<label>:<weight>, ...}
        self.edges = {}  # Dictionary of sets dict{ <source_label>: set{<target_label>, ...}, ...}
        self.rev_edges = {}  # Dictionary of sets
        self.unseen_sources = set()  # Labels of all nodes not processed yet that have no incoming edges
        self.longest_in_weight = {}  # Dictionary {<label>:<weight>, ...}
        self.longest_in_route = {}   # Dictionary {<label>:[<label>, ...], ...}
        self.longest_route = None;   # The longest route (in weights) we have seen
        self.longest_route_weight = None;  # The larges weight we have seen
    
    def add_node(self, label, weight):
        """Add a node to a graph.
        
        # Arguments
            label: a scalar label for the node
            weight: a nonnegative number
        """
        if weight < 0: raise ValueError("weight cannot be negative")
        self.nodes[label] = weight
        self.edges[label] = set()
        self.rev_edges[label] = set()
        self.unseen_sources.add(label)
        
    def add_edge(self, source, target):
        """Add an edge to a graph.
        
        # Arguments
            source: the label of the source node; it should already exist in the graph
            target: the label of the target node; it should already exist in the graph
        """
        if source not in self.nodes: raise ValueError("source {} not a node".format(source))
        if target not in self.nodes: raise ValueError("target {} not a node".format(target))
        self.edges[source].add(target)
        self.rev_edges[target].add(source)
        self.unseen_sources.discard(target)
        
    def __del_edges_from(self, source):
        """Private method to delete all outgoing edges from a node."""
        targets = self.edges[source]
        self.edges[source] = set()
        for target in targets:
            self.rev_edges[target].discard(source)
            if len(self.rev_edges[target]) == 0: # no incoming edges
                self.unseen_sources.add(target)
                
    #def __print(self):
    def pub_print(self):
        """Private method to print information about the graph."""
        print("Nodes, Edges")
        for id, w in self.nodes.items():
            print("  {}{} = {} -> {}".format(
                's' if id in self.unseen_sources else ' ', 
                id, 
                w,
                ",".join([str(x) for x in self.edges[id]])
            ))
        print("Rev-Edges")
        for id, source in self.rev_edges.items():
            print("  {} <- {}".format(id, ",".join([str(x) for x in source])))
        print("Longest in")
        for id, w in self.nodes.items():
            print("  {} : {} = {}".format(
                id,
                str(self.longest_in_weight.get(id, 0)),
                ",".join([str(x) for x in self.longest_in_route.get(id, [])])
            ))        
        print("")
        
        
    def longest_path(self):
        """Return the longest path in the graph in terms of the node weights.
        
        Warning: This operation is destructive and will delete the graph.
        
        # Returns
            An array of the route (array of labels), and the sum of the weights along the route.
        """
        while len(self.unseen_sources) > 0:
            sourcenode = self.unseen_sources.pop()
            
            #new_weight = self.longest_in_weight.get(sourcenode, 0) + self.nodes[sourcenode]
            new_weight = self.longest_in_weight.get(sourcenode, 1) * self.nodes[sourcenode]
            new_route = self.longest_in_route.get(sourcenode, []) + [sourcenode]

            if len(self.edges[sourcenode]) == 0: # no outgoing edges; isolated node
                if self.longest_route is None or self.longest_route_weight < new_weight:
                    self.longest_route = new_route
                    self.longest_route_weight = new_weight
                continue
            
            # There are outgoing edges            
            for target in self.edges[sourcenode]:
                
                if self.longest_in_weight.get(target, 0) < new_weight:
                    self.longest_in_weight[target] = new_weight
                    self.longest_in_route[target] = new_route
                
            self.__del_edges_from(sourcenode)
            
        return (self.longest_route, self.longest_route_weight)


if __name__ == '__main__':

    dag = DAGLongestPath()
    # Four groupings
    
    #Shell group
    def shell_group(sh, input):
        dag.add_node(sh+1, 1.41) # sh_p
        dag.add_node(sh+2, .61) # sh_w
        dag.add_node(sh+3, 2.08) # sh_sn
        dag.add_edge(input, sh+1)
        dag.add_edge(input, sh+2)
        dag.add_edge(input, sh+3)

    #Pizza group
    def pizza_group(p, input):
        dag.add_node(p+1, .48) # p_w
        dag.add_node(p+2, 1.52) # p_sn
        dag.add_node(p+3, .71) # p_sh
        dag.add_edge(input, p+1)
        dag.add_edge(input, p+2)
        dag.add_edge(input, p+3)

    #Wasabi group
    def wasasbi_group(w, input):
        dag.add_node(w+1, 2.05) # w_p
        dag.add_node(w+2, 3.26) # w_sn
        dag.add_node(w+3, 1.56) # w_sh
        dag.add_edge(input, w+1)
        dag.add_edge(input, w+2)
        dag.add_edge(input, w+3)
    
    #Snow group
    def snow_group(sn, input):
        dag.add_node(sn+1, .64) # sn_p
        dag.add_node(sn+2, .3) # sn_w
        dag.add_node(sn+3, .46) # sn_sh
        dag.add_edge(input, sn+1)
        dag.add_edge(input, sn+2)
        dag.add_edge(input, sn+3)

    def close_pizza(p):
        dag.add_edge(p+1, 1001)
        dag.add_edge(p+2, 1002)

    def close_wasabi(w):
        dag.add_edge(w+1, 1000)
        dag.add_edge(w+2, 1002)
        
    def close_snow(sn):
        dag.add_edge(sn+1, 1000)
        dag.add_edge(sn+2, 1001)

    def close_shell(sh):
        dag.add_edge(sh+1, 1000)
        dag.add_edge(sh+2, 1001)
        dag.add_edge(sh+3, 1002)

    
    # Assuming 5 is best
    
    # trade 1
    dag.add_node(0, 1)
    shell_group(0, 0)

    # trade 2
    pizza_group(4, 1)
    wasasbi_group(8, 2)
    snow_group(12, 3)

    # trade 3
    wasasbi_group(16, 5)
    snow_group(20, 6)
    shell_group(24, 7)    

    pizza_group(28, 9)
    snow_group(32, 10)
    shell_group(36, 11)    

    pizza_group(40, 13)
    wasasbi_group(44, 14)
    shell_group(48, 15)    

    # trade 4

    # first 3
    pizza_group(52, 17)
    snow_group(56, 18)
    shell_group(60, 19)

    pizza_group(64, 21)
    wasasbi_group(68, 22)
    shell_group(72, 23)   

    pizza_group(76, 25)
    wasasbi_group(80, 26)
    snow_group(84, 27)

    # second 3
    wasasbi_group(88, 29)
    snow_group(92, 30)
    shell_group(96, 31)

    pizza_group(100, 33)
    wasasbi_group(104, 34)
    shell_group(108, 35)   

    pizza_group(112, 37)
    wasasbi_group(116, 38)
    snow_group(120, 39)
    
    # third 3
    wasasbi_group(124, 41)
    snow_group(128, 42)
    shell_group(132, 43)
    
    pizza_group(136, 45)
    snow_group(140, 46)
    shell_group(144, 47)

    pizza_group(148, 49)
    wasasbi_group(152, 50)
    snow_group(156, 51)

    
    # trade 5
    dag.add_node(1000, .71) #pizza to shell
    dag.add_node(1001, 1.56) #wasabi to shell
    dag.add_node(1002, .46) #snow to shell
    
    #1
    close_pizza(52)
    close_snow(56)
    close_shell(60)
    
    close_pizza(64)
    close_wasabi(68)
    close_shell(72)

    close_pizza(76)
    close_wasabi(80)
    close_snow(84)

    #2
    close_wasabi(88)
    close_snow(92)
    close_shell(96)

    close_pizza(100)
    close_wasabi(104)
    close_shell(108)

    close_pizza(112)
    close_wasabi(116)
    close_snow(120)

    #3
    close_wasabi(124)
    close_snow(128)
    close_shell(132)

    close_pizza(136)
    close_snow(140)
    close_shell(144)

    close_pizza(148)
    close_wasabi(152)
    close_snow(156)
    # calc ans is ([0, 1, 5, 19, 61, 1000], 1.0569693888)
    

    """
    # Assuming 2 is best
    # Beginning reqs
    dag.add_node(0, 1)
    dag.add_node(1000, .71) #pizza to shell
    dag.add_node(1001, 1.56) #wasabi to shell
    dag.add_node(1002, .46) #snow to shell
    
    # trade 1
    shell_group(0, 0)

    # trade 2
    close_shell(0)
    # calc ans is ([0, 1, 1000], 1.0010999999999999)
    """

    """
    # Assuming 3 is best
    # Beginning reqs
    dag.add_node(0, 1)
    dag.add_node(1000, .71) #pizza to shell
    dag.add_node(1001, 1.56) #wasabi to shell
    dag.add_node(1002, .46) #snow to shell

    # trade 1
    shell_group(0, 0)

    # trade 2
    pizza_group(4, 1)
    wasasbi_group(8, 2)
    snow_group(12, 3)

    # trade 3
    close_pizza(4)
    close_wasabi(8)
    close_snow(12)
    #calc ans is ([0, 1, 5, 1001], 1.055808)
    """
    
    """
    # Assuming 4 is best
    # Beginning reqs
    dag.add_node(0, 1)
    dag.add_node(1000, .71) #pizza to shell
    dag.add_node(1001, 1.56) #wasabi to shell
    dag.add_node(1002, .46) #snow to shell

    # trade 1
    shell_group(0, 0)

    # trade 2
    pizza_group(4, 1)
    wasasbi_group(8, 2)
    snow_group(12, 3)

    # trade 3
    wasasbi_group(16, 5)
    snow_group(20, 6)
    shell_group(24, 7)    

    pizza_group(28, 9)
    snow_group(32, 10)
    shell_group(36, 11)    

    pizza_group(40, 13)
    wasasbi_group(44, 14)
    shell_group(48, 15)

    # trade 4
    close_wasabi(16)
    close_snow(20)
    close_shell(24)

    close_pizza(28)
    close_snow(32)
    close_shell(36)

    close_pizza(40)
    close_wasabi(44)
    close_shell(48)
    # calc ans is ([0, 1, 5, 19], 1.055808)
    """
    print(dag.longest_path())
