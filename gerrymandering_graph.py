import math
import random
from typing import Callable,Tuple
import time

class Graph:

    """
    Class representation of a graph using an adjacency list representation

    This class contains methods relating to partitioning the graph into subgraphs.
    This includes: 
     - checking if subgraphs are contiguous
     - getting a set of nodes which are adjacent to a subgraph
     - removing a node from a subgraph such that the largest contiguous subgraph remains
     - checking if a set of subgraphs is of equal size and is contiguous
     - generating a set of random equally sized contiguous subgraphs

    Attributes:
    nodes : the number of nodes in the graph
    edges : a list of sets of nodes such that edge[node] is the set of nodes connected to by node
    color_node : a passable function to get the ANSI color codes for a node

    Methods:
    isContiguious(subgraph) checks if the given set of nodes is a contiguous subgraph
    getConnectedNodes(subgraph) returns a set of nodes connected to the given subgraph
    randomEqualContiguoussubgraphs(rand,k,iter=0) attempts to create k equally sized and contiguous 
        subgraphs
    contiguoussubgraphRemove(subgraph,node,assigned) removes the node n from the contiguous subgraph, 
        leaving the largest remaining contiguous region
    isValidEqualContiguoussubgraphs(subgraphs) checks if the list of subgraphs is a valid set 
        of equally sized contiguous subgraphs which cover the entire graph
    """

    nodes : int
    edges : list[set[int]]

    class SmallestSubgraphQueue:
        """
        Class to keep track of the smallest of k subgraphs

        Allows for the size of a subgraph to be changed arbitrarially

        Attributes:
        heap tracks the smallest subgraph
        size tracks the size of each subgraph
        index tracks where in the heap each subgraph is for fast lookup

        Methods:
        get() returns the smallest subgraph
        update(subgraph,size) updates the size of the given subgraph and readjusts the heap
        """
        heap : list[int] # the heap to track which subgraph is smallest
        size : list[int] # the list to track the size of each subgraph
        index: list[int] # the list to track the index of each subgraph in the heap to reduce lookup time

        def __init__(self, k:int):
            self.heap = [i for i in range(k)]
            self.index = [i for i in range(k)]
            self.size = [0 for i in range(k)]
        
        def get(self):
            """
            gets the index of the smallest subgraph

            Returns:
                An integer representing the subgraph
            """
            return self.heap[0]
        
        def update(self, subgraph:int, size:int):
            """
            updates the size of a subgraph then updates the heap

            Args:
                subgraph: the index of the subgraph which has had its size changed
                size: the new size of the subgraph
            """
            # update the size of the subgraph
            if(size<self.size[subgraph]):
                # if the size is smaller than it was move it up the heap
                self.__decrease(subgraph,size)
            elif(size>self.size[subgraph]):
                # if the size is larger move it up the heap
                self.__increase(subgraph,size)
        
        def __increase(self, subgraph:int, size:int):
            """
            handles updates where the size of the subgraph increases

            Args:
                subgraph: the index of the subgraph which has had its size changed
                size: the new size of the subgraph
            """
            # set the new size and then downheap
            self.size[subgraph] = size
            index : int = self.index[subgraph]
            self.__down(index)
        
        def __down(self, index: int):
            """
            Performs the downheap operation to maintain the minheap property

            Args:
                index: the index of the changed node
            """
            # move the node at the index down the heap
            smallest : int = index
            left : int = smallest*2+1
            right : int = smallest*2+2
            if(left<len(self.heap) and self.size[self.heap[left]] < self.size[self.heap[smallest]]):
                smallest = left
            if(right<len(self.heap) and self.size[self.heap[right]] < self.size[self.heap[smallest]]):
                smallest = right
            if(smallest!=index):
                temp : int = self.heap[index]
                self.heap[index] = self.heap[smallest]
                self.heap[smallest] = temp
                self.index[self.heap[index]] = index
                self.index[self.heap[smallest]] = smallest
                self.__down(smallest)
                
            
        def __decrease(self, subgraph:int, size:int):
            """
            Decreases the size of a given subgraph then performs the upheap operation

            Args:
                subgraph: index of the subgraph
                size: new size of the subgraph  
            """
            self.size[subgraph] = size
            index : int = self.index[subgraph]
            self.__up(index)

        def __up(self, index:int):
            """
            Performs upheap operation to maintain minheap property after a decrease

            Args:
                index: the index of the node being bubbled up
            """
            if(index==0):
                return
            subgraph = self.heap[index]
            # move the node up the heap if it is smaller
            prev: int = index//2
            if(self.size[self.heap[prev]]>self.size[subgraph]):
                self.index[subgraph] = prev
                self.index[self.heap[prev]] = index
                self.heap[index]=self.heap[prev]
                self.heap[prev]=subgraph
                self.__up(prev)
    
    color_node : Callable[[int],Tuple[str,str]]

    def __init__(self, nodes: int, edges: list[set[int]], color_node : Callable[[int],Tuple[str,str]] = lambda a : ("","")):
        """
        initializes a graph with given nodes and edges

        Args:
            nodes: number of nodes in the graph
            edges: list of sets of edges such that edges[node] is the set of nodes connected to by the node
            color_node:  function to get ANSI color codes for each node defaults to empty
        """
        self.nodes = nodes
        self.edges = edges
        self.color_node = color_node

    def isContiguous(self, subgraph: set[int])->bool:
        """
        Checks if the given set of nodes forms a contiguous subgraph

        Args:
            subgraph: a set of ints (referencing nodes) representing a subgraph

        Returns:
            A boolean representing if the given subgraph is contiguous
        """
        unvisited: set[int] = set(subgraph)
        stack:list[int] = [unvisited.pop()]
        while(len(stack)>0):
            node:int = stack.pop()
            # for each node connected to the current node which is not visited add it to the stack
            # and remove it from the unvisited set
            for connected in self.edges[node]&unvisited:
                stack.append(connected)
                unvisited.remove(connected)
        # if no nodes in the subgraph were unvisited then the graph was connected
        return len(unvisited)==0
    
    def getConnectedNodes(self, subgraph: set[int])->set[int]:
        """
        Gets all nodes connected to a subgraph which are not in that subgraph

        Args:
            subgraph: a set of ints representing a subgraph

        Returns:
            A set of ints representing all nodes connected to the subgraph
        """
        if(len(subgraph)==0):
            return {i for i in range(self.nodes)}
        connected:set[int] = set()
        # add all nodes connected to the subgraph to the connected set
        for node in subgraph:
            connected |= self.edges[node]
        # remove all nodes already in the subgraph
        return connected-subgraph
    
    def randomEqualContiguousSubgraphs(self, k : int, rand:random.Random=random.Random(), iter: int = 0)->list[set[int]]:
        """
        Gets a list of k random subgraphs which are of equal size

        Args:
            k: the number of subgraphs to divide the graph into
            rand: the random object to use for randomization
            iter: the number of random seedings we already tried

        Returns:
            A list of k sets of integers representing k equal sized contiguous subgraphs

        Raises:
            ValueError: k must divide the number of nodes and each contiguous regions of the graph
            TimeoutError: because we are brute forcing, the process may timeout
        """
        
        if(self.nodes<k):
            raise ValueError(f"k ({k}) must be less than nodes ({self.nodes})")
        if(self.nodes%k!=0):
            raise ValueError(f"k ({k}) must divide nodes ({self.nodes})")
        n : int = self.nodes/k
        subgraphs: list[set[int]] = [set() for i in range(k)]
        assigned: dict[int,int] = dict()
        smallest: Graph.SmallestSubgraphQueue = Graph.SmallestSubgraphQueue(k)
        timeout: int = 0
        # this method of generating subgraphs is more brute force than I would like, 
        # but I haven't been able to find a more optimal method of generating k equally 
        # sized random contiguous subgraphs
        while(not self.isValidEqualContiguousSubgraphs(subgraphs)):
            subgraph : int = smallest.get()
            connected = self.getConnectedNodes(subgraphs[subgraph])
            if(len(connected)==0):
                # if this is the smallest, but has no more connections, we cannot build the subgraphs
                # TODO: It may make sense to make a connected and disconnected graph class then to make
                # the subgraphs in a disconnected graph you could defer to the connected subgraphs
                raise ValueError("Could not generate solution")
            if(not connected<=assigned.keys()):
                # if there are unassigned nodes we can connect with we should prioritize those to avoid 
                # overwriting pre existing subgraphs
                connected -= assigned.keys()
            # select a random node and assign it to the subgraph
            node:int = rand.choice(list(connected))
            subgraphs[subgraph].add(node)
            if(node in assigned):
                # if we already assigned this node to another subgraph remove it from the 
                # subgraph it is currently in
                temp : int = assigned[node]
                self.contiguousSubgraphRemove(subgraphs[temp],node,assigned)
                smallest.update(temp,len(subgraphs[temp]))
            assigned[node] = subgraph
            subgraphs[subgraph].add(node)
            smallest.update(subgraph,len(subgraphs[subgraph]))
            # because it is brute force we want a reasonable timeout
            if(timeout>k*self.nodes*100):
                # try with new random seeding a few times before totally failing
                if(iter>100):
                    raise TimeoutError("Could not generate solution in a reasonable time")
                return self.randomEqualContiguousSubgraphs(k,rand,iter+1)
            timeout+=1
        return subgraphs

    def contiguousSubgraphRemove(self, subgraph:set[int], node:int, assigned:dict[int,int]={}):
        """
        Removes a node from a contiguous subgraph leaving the largest contiguous subgraph of the remainder

        Args:
            subgraph: the set of nodes which make up the contiguous subgraph
            node: the node to remove from the subgraph
            assigned: the nodes which have already been assigned
        """
        unvisited: set[int] = set(subgraph)
        unvisited.remove(node)
        assigned.pop(node)
        subgraph.remove(node)
        directions: set[int] = self.edges[node]
        best: set[int] = set()
        # from the removed node check each connected node
        for direction in directions & unvisited:
            # build the sub - subgraph from the initial node
            sub: set[int] = set()
            sub.add(direction)
            for next_node in self.getConnectedNodes(sub)&unvisited:
                unvisited.remove(next_node)
                sub.add(next_node)
            # remove the smaller of the two sets from the subgraph
            if(len(sub)>len(best)):
                for worse_node in best:
                    assigned.pop(worse_node)
                subgraph -= best
                best = sub
            else:
                for worse_node in sub:
                    assigned.pop(worse_node)
                subgraph -= sub

    def isValidEqualContiguousSubgraphs(self, subgraphs:list[set[int]])->bool:
        """
        Checks if the list of subgraphs is a valid set of equally sized contiguous 
        subgraphs which covers the entire graph

        Args:
            subgraphs: the list of sets of ints which represents the list of subgraphs
        
        Returns:
            True if the list of sets passed are all of equal size, large enough to cover the entire graph
        """
        n : int = self.nodes//len(subgraphs)
        # First check that all the subgraphs are of the same size
        for subgraph in subgraphs:
            if(len(subgraph)!=n):
                return False
        # Next do the more expensive check that each subgraph is contiguous
        for subgraph in subgraphs:
            if(not self.isContiguous(subgraph)):
               return False
        return True
    
    def __str__(self):
        """
        converts the graph to a string listing each node and its edges on a new line

        Returns:
            String representation of the graph
        """

        ans = ""
        for i in range(self.nodes):
            ans += self.node_string(i)
        return ans

    def node_string(self, node:int) -> str:
        """
        String representation of a given node

        Args: 
            node: an integer representing a node of the graph

        Returns: 
            A string representation of the node (with ANSI colors courtsey of color_node)
        """

        #TODO: May be cleaner to have color node take in the node and the string and just return the colored string
        return f"{self.color_node(node)[0]}{node}: {self.edges[node]}{self.color_node(node)[1]}\n"

class Rectangle(Graph):
    """
    A graph which is in the shape of a rectangle

    Attributes:
        n: an integer representing the height of the rectangle
        m: an integer representing the width of the rectangle
    """

    n:int
    m:int
    
    def __init__(self, n:int, m:int):
        """
        Initialize a rectangular graph of height n and width m

        Args:
            n: height
            m: width
        """
        
        self.n = n
        self.m = m
        edges : list[set[int]] = [set() for i in range(n*m)]
        for i in range(n):
            for j in range(m):
                if(i>0):
                    edges[i*m+j].add((i-1)*m+j)
                if(i+1<n):
                    edges[i*m+j].add((i+1)*m+j)
                if(j>0):
                    edges[i*m+j].add(i*m+j-1)
                if(j+1<m):
                    edges[i*m+j].add(i*m+j+1)
        super().__init__(n*m, edges)

    def __str__(self):
        """
        String representation of the graph placing nodes in a grid

        Returns:
            string representing the graph
        """
        
        ans:str = ""
        for i in range(self.n):
            for j  in range (self.m):
                ans += self.node_string(i*self.m+j)
            ans += "\n"
        return ans

    def node_string(self, node:int)->str:
        """
        String representing a given node

        Returns:
            string representing the node. Its just the number in hex.
        """
        
        if(self.nodes<=16):
            return f"{self.color_node(node)[0]}{node:01x}{self.color_node(node)[1]} "
        if(self.nodes<=256):
            return f"{self.color_node(node)[0]}{node:02x}{self.color_node(node)[1]} "
        return f"{self.color_node(node)[0]}{node:01x}{self.color_node(node)[1]} "
        
class Image(Graph):
    """
    A graph initialized from a 2D boolean array

    Attributes:
        image: the image which initializes the graph
        coordToNode: a dict which converts an i,j coordinate to a node index
    """

    image : list[list[int]]
    coordToNode : dict[(int,int),int]
    def __init__(self, image : list[list[int]]):
        """
        Initializes a graph from an image

        Args:
            image: a 2D array of ints with 1 representing a populated cell
        """
        # image could probably be a bool array instead, but I wanted to leave it as an 
        # int in case I wanted to have greater populations on nodes
        self.image = image
        nodes : int = 0
        self.coordToNode = dict()
        edges : list[set[int]] = []
        for i in range(len(image)):
            for j in range(len(image[i])):
                if(image[i][j]==1):
                    self.coordToNode[(i,j)] = nodes
                    edges.append(set())
                    nodes += 1
                    if((i-1,j) in self.coordToNode):
                        edges[self.coordToNode[(i,j)]].add(self.coordToNode[(i-1,j)])
                        edges[self.coordToNode[(i-1,j)]].add(self.coordToNode[(i,j)])
                    if((i,j-1) in self.coordToNode):
                        edges[self.coordToNode[(i,j)]].add(self.coordToNode[(i,j-1)])
                        edges[self.coordToNode[(i,j-1)]].add(self.coordToNode[(i,j)])
        super().__init__(nodes, edges)

    def __str__(self):
        """
        Convert the image graph into a string

        Returns:
            A string representation of the image with each pixel either being a node number or blank spaces
        """
        
        ans:str = ""
        for i in range(len(self.image)):
            for j  in range (len(self.image[i])):
                if((i,j) in self.coordToNode):
                    ans += self.node_string(self.coordToNode[(i,j)])
                else:
                    ans += self.empty_string()
                # ans += self.node_string(i*self.m+j)
            ans += "\n"
        return ans
    

    def empty_string(self) -> str:
        """
        Returns a string of whitespace to represent empty pixels in the image
        """
        
        if(self.nodes<=16):
            return "  "
        if(self.nodes<=256):
            return f"   "
        return f"    "

    def node_string(self, node:int) -> str:
        """
        String representing a given node

        Returns:
            string representing the node. Its just the number in hex.
        """
        
        if(self.nodes<=16):
            return f"{self.color_node(node)[0]}{node:01x}{self.color_node(node)[1]} "
        if(self.nodes<=256):
            return f"{self.color_node(node)[0]}{node:02x}{self.color_node(node)[1]} "
        return f"{self.color_node(node)[0]}{node:01x}{self.color_node(node)[1]} "

class Puzzle:
    """
    A gerrymandering puzzle object. The graph can be divided into k equally sized districts such that:
     - there is a plurality winner in each district
     - there is a plurality winner in the overall number of districts
     - the minority party can be the overall plurality winner
     - there are p competing parties

    Attributes:
        g: the underlying graph of the puzzle
        parties: the party of each node i.e. parties[node] is the party of the node
        node_to_district: a list which maps the nodes onto the districts
        districts: the list of sets which makes up the current guess of the districts
        minority: the party which needs to win plurality to mark the puzzle solved
        k: the number of districts
        p: the number of parties
        solution: the auto generated solution
    """

    g : Graph
    # ANSI vibrant foreground color codes
    party_colors : list[int] = [91, 92, 93, 94, 95, 96, 97, 90]
    # ANSI background color codes
    district_colors : list[int] = [41, 42, 43, 44, 45, 46, 47, 40]
    parties : list[int]
    node_to_district : list[int]
    districts : list[set[int]]
    minority: int
    k : int
    p : int
    solution : list[set[int]]

    def __init__(self,g:Graph, k:int, p:int, rand:random.Random = random.Random()):
        """
        Initialize a puzzle on graph g with k districts and p parties

        Args:
            g: the graph to build the puzzle on
            k: the number of districts
            p: the number of parties
            rand: the random object with which to initialize the puzzle

        Raises:
            ValueError: k and p must be less than 7 due to rendering limitations
            ValueError: k and p must be chosen such that we can have a plurality winning minority district
        """

        # The system could handle more than 7 districts or parties, but it cannot render them due to the number of ANSI colors
        # Both issues can probably be solved with some improved (not terminal based) rendering code
        if(k>7):
            # NOTE: It may be possible to use 4 colors (at least for 2d graphs) but it would be hard for the user to distinguish
            raise ValueError("more than 7 districts currently unsupported")
        if(p>7):
            # NOTE: Maybe change the adressing code so you can display the party number rather than the address
            raise ValueError("more than 7 parties currently unsupported")


        # We multiply the minimum number of nodes to win a district by the minimum number of districts
        # to get the minimum number of nodes needed to win a plurality call this value s
        # We want a minority district winning the plurality election so we need n (the total number of nodes)
        # minus s and divided by the remaining number of parties p-1 to be less than s
        # (n-s)/(p-1)<s ==> n-s<ps-s ==> n<ps
        # I will explain how I calculate the minimum number of districts and nodes at their calculations
        if(math.ceil((k-1+p)/p)*math.ceil((g.nodes//k-1+p)/p)*p>g.nodes):
            raise ValueError("Cannot make minority which wins plurality")
        self.g = g
        self.k = k
        self.p = p
        self.node_to_district = [-1 for i in range(g.nodes)]
        self.districts = [set() for i in range(k)]
        solution = g.randomEqualContiguousSubgraphs(k,rand)
        self.solution = solution
        party_districts = [set() for i in range(p)]
        self.parties = [-1 for i in range(g.nodes)]
        for i in range(k):
            # Lets call the minimum number of districts required to win the plurality i
            # i must be greater than the number of districts won by other parties which 
            # can be calculated as the ceiling of the remaining districts (k-i) divided by
            # the remaining parties (p-1)
            # i>ceil((k-i)/(p-1))
            # because all values are integers i-1>=ceil((k-i)/(p-1))
            # because ceil(x)>=x: i-1>=(k-i)/(p-1)
            # i-1>=(k-i)/(p-1) ==> pi-p-i+1>=k-i ==> pi >= k+p-1 ==> i>=(k+p-1)/p
            # because we want the minimal i and it is an integer i=ceil((k+p-1)/p)
            if(i<math.ceil((k-1+p)/p)):
                party_districts[0].add(i)
            else:
                party_districts[1+(i%(p-1))].add(i)
        party_order = list(range(p))
        rand.shuffle(party_order)
        for i in range(p):
            for district in party_districts[i]:
                j = 0
                district_order = list(solution[district])
                rand.shuffle(district_order)
                for node in district_order:
                    # This follows the same logic as the minimum number of districts to win
                    # we just replace k with n/k as that is the number of nodes in the district
                    if(j<math.ceil((g.nodes//k-1+p)/p)):
                        self.parties[node]=party_order[i]
                    else:
                        self.parties[node]=party_order[1+(j%(p-1))]
                    j+=1
        self.minority = party_order[0]
        g.color_node = self.color_node
        
    def __str__(self):
        """
        Writes the current state of the puzzle to a string

        Returns:
            A string reperesentation of the puzzle
        """

        ans = f"Make party {self.color_code(self.minority,-1)[0]}{self.minority}\033[0m win\nparties:"
        for i in range(self.p):
            ans+=f"{self.color_code(i,-1)[0]}{i}\033[0m"
        ans+="\ndistricts:"
        for i in range(self.k):
            ans+=f"{self.color_code(-1,i)[0]}{i}\033[0m"
        ans+=f"\neach district will have {self.g.nodes//self.k} cells\nmap:\n"
        ans+=str(self.g)
        return ans


    def color_code(self,party:int,district:int)->Tuple[str,str]:
        """
        A method to get the ANSI color code given a party and a district

        Args:
            party: what party
            district: what district

        Returns:
            A tuple of strings one with the ANSI setting the foreground and background based on the 
            party and district, and a ANSI reset code

        """

        ## while the % makes it possible to have more parties or districts it is too confusing
        return (f"\033[51;{self.party_colors[-1 if party==-1 else party%7]};{self.district_colors[-1 if district==-1 else district%7]}m","\033[0m")

    def color_node(self,node:int) -> Tuple[str,str]:
        """
        A method to get the ANSI color code given a node

        Args:
            node: which node

        Returns:
            A tuple of strings one with the ANSI setting the foreground and background based on the 
            party and district of the node, and a ANSI reset code

        """

        return self.color_code(self.parties[node],self.node_to_district[node])
    
    def set_district(self,node:int,district:int)->bool:
        """
        Sets a node to be a member of a district

        Args:
            node: the node being set
            district: the district the node is added to
        
        Returns:
            If the operation was successful
        """

        if(district>=self.k or district<-1):
            print(f"Illegal operation district {district} out of range")
            # time.sleep(1)
            return False
        if(node<0 or node>= self.g.nodes):
            print(f"Illegal operation node {node} not in graph")
            # time.sleep(1)
            return False
        if(self.node_to_district[node]!=-1):
            self.districts[self.node_to_district[node]].remove(node)
        if(district!=-1):
            self.districts[district].add(node)
        self.node_to_district[node] = district
        return True
    
    def check_win(self) -> bool:
        """
        Checks if the puzzle is complete.

        A Complete puzzle has:
         - k equal sized contiguous subgraphs which cover the graph
         - a clear plurality victor in each district
         - a clear plurality victory for the minority party

        Returns:
            If the puzzle is completed.
        """

        # The districts must be equal contiguous subgraphs
        if(not self.g.isValidEqualContiguousSubgraphs(self.districts)):
            return False
        district_counts : list[int] = [0 for i in range(self.p)]
        for district in self.districts:
            voter_counts = [0 for i in range(self.p)]
            for node in district:
                voter_counts[self.parties[node]] += 1
            m = max(voter_counts)
            if(voter_counts.count(m)!=1):
                # each district must have a strict winner
                return False
            district_counts[voter_counts.index(max(voter_counts))]+=1
        m = max(district_counts)
        if (district_counts.count(m)!=1):
            # there must be a clear plurality winner
            return False
        # if the plurality winner is our minority party that is a win
        return self.minority == district_counts.index(max(district_counts))
    
    def play(self):
        """
        The main gameplay loop of the puzzle done in the terminal.
        """

        while(not self.check_win()):
            # time.sleep(1)
            print("\033[H\033[2J", end="")
            print("\033[H\033[3J", end="")
            print("\033c", end="")
            print(str(self))
            user_input = input()
            #TODO: implement undo and redo
            if(user_input in {"flash","f","show"}):
                # Because the ANSI forground color may not display when the bg is set
                # Allows the user to flash the unaltered board state on screen
                # A bit of a hacky implementation but it works temporarily reset the node to district map
                temp = self.node_to_district
                self.node_to_district = [-1 for i in range(self.g.nodes)]
                print("\033[H\033[2J", end="")
                print("\033[H\033[3J", end="")
                print("\033c", end="")
                print(str(self))
                user_input = input()
                self.node_to_district = temp
                continue
            if(user_input in {"reset","restart"}):
                # Reset the state of the puzzle by clearing the districts and the mapping
                self.node_to_district = [-1 for i in range(self.g.nodes)]
                self.districts = [set() for i in range(self.k)] 
                continue
            if(user_input=="debug"):
                # This is just here to put a breakpoint if something is weird while testing
                continue
            if(user_input in {"forfeit","quit", "give up"}):
                #Allow the user to give up and just display the generated solution
                print("\033[H\033[2J", end="")
                print("\033[H\033[3J", end="")
                print("\033c", end="")
                print(str(self.g))
                for i in range(self.k):
                    for node in self.solution[i]:
                        self.node_to_district[node] = i
                print(str(self))
                print("You forfeit this round. You'll get it next time :)")
                return
            try:
                # The user may input many different node indexes and one district index to place those
                # nodes into a district
                strs = user_input.strip().split()
                for i in strs[:-1]:
                    if(not self.set_district(int(i,16),int(strs[-1],16))):
                        time.sleep(1)
                        break
            except Exception:
                print("Invalid Input")
                time.sleep(1)
        print("\033[H\033[2J", end="")
        print("\033[H\033[3J", end="")
        print("\033c", end="")
        print(str(self))
        print("Congratulations! You gerrymandered the map!")
        
Puzzle(Rectangle(16,16),8,8).play()