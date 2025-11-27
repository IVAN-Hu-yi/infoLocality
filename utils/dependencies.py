"""
This module contains code for reading dependency trees from conll files 
and for shuffling word order in various ways while keeping track of the dependencies
"""
from math import abs
from random import shuffle

#DATA REPRESENTATION
class DependencyTree:

    def __init__(self,tokens=None, edges=None):
        self.edges  = [] if edges is None else edges                      #couples (gov_idx,dep_idx)
        self.tokens = [('$ROOT$','$ROOT$')] if tokens is None else tokens #couples (wordform,postag)
    

    def copy(self):
        """
        Clones the tree and allocates memory for the clone
        """
        return DependencyTree(self.tokens[:],self.edges[:])


    def __str__(self):
        gdict = dict([(d,g) for (g,d) in self.edges])
        return '\n'.join(['\t'.join([str(idx+1),tok[0],tok[1],str(gdict[idx+1])]) for idx,tok in enumerate(self.tokens[1:])])
                     

    def dependency_length(self):
        """
        Returns the sum of all dependencies lengths found in this tree
        """
        return sum( abs(dep_idx-gov_idx) for (gov_idx,dep_idx) in self.edges )

    @property    
    def sentence(self):
        """
        Returns a list of true tokens (without the ROOT) of this tree
        """
        return self.tokens[1:]


    @staticmethod
    def read_trees(istream):
        """
        Yields a tree iterable from a conllu input stream
        @param istream: the stream where to read from
        @return: a DependencyTree instance 
        """
        exclude_chars ='.-'  #excludes empty nodes and compounds from reading

        deptree = DependencyTree()
        for bfr in istream:
            bfr = bfr.strip()
            if (bfr.isspace() or bfr == '' or bfr.startswith('#')):
                if deptree.N() > 1:
                    yield deptree
                    deptree = DependencyTree()
            else:
                idx, word, lemma,tagA,tagB,feats, governor_idx, *other_args  =  bfr.split()    
                
                if not any(c in governor_idx for c in exclude_chars) and not any(c in idx for c in exclude_chars) :
                    deptree.tokens.append((word,tagA))
                    deptree.edges.append((int(governor_idx),int(idx)))
                   
        return deptree

    def shuffle_projective(self):
        """
        Shuffles the words of the tree randomly. It generates projective dependency trees.
        Like python shuffle the method is destructive. perform a copy of the dependency tree to keep the original.
        """
        #create a convenient dependency encoding for the shuffle
        children = { }
        for gov,dep in self.edges:
            deps = children.get(gov,[])
            deps.append(dep)
            children[gov] = deps

        #performs the stratified shuffling of the children
        for key in children:
            if key == 0:
                shuffle(children[key])
                children[key] = [0] + children[key]
            else:
                children[key].append(key)
                shuffle(children[key])
        
        #concatenates all the results
        stack    = set([])
        neworder = [0]
        while len(neworder) < len(self.tokens):
            for idx,elt in enumerate(neworder):
                if elt not in stack:
                    stack.add(elt)
                    left,right = neworder[:idx],neworder[idx+1:]
                    neworder = left+children.get(elt,[elt])+right
                    #print('stack',stack,'order',neworder)
                    break
        assert (len(neworder) == len(self.tokens))
        self.shuffle_tree(neworder)


    def shuffle_tree(self,neworder=None):
        """
        Shuffles the words of the tree randomly without any constraint and keeps the dependencies.
        Like python shuffle the method is destructive. perform a copy of the dependency tree to keep the original.
        
        KwArgs:
            neworder (list of int) a list with a new ordering of the tokens
        """
        src_idx = list(range(len(self.tokens)))
        
        if neworder:
            tgt_idx = neworder
            src_idx,tgt_idx = tgt_idx,src_idx
        else:
            tgt_idx = src_idx[1:] 
            shuffle(tgt_idx)
            tgt_idx = [0] + tgt_idx

        dmap = dict(zip(src_idx,tgt_idx))

        self.edges = [ (dmap[gov],dmap[dep]) for gov,dep in self.edges]
            
        newtokens  = [None]*len(self.tokens)
        newtokens[0] = self.tokens[0]

        for src_idx in range(len(self.tokens)):
            newtokens[dmap[src_idx]]= self.tokens[src_idx]
        self.tokens = newtokens


    def accurracy(self,other):
        """
        Compares this dep tree with another by computing their UAS.
        @param other: other dep tree
        @return : the UAS as a float
        """
        assert(len(self.edges) == len(other.edges))
        S1 = set(self.edges)
        S2 = set(other.edges)
        return len(S1.intersection(S2)) / len(S1)
    
    def N(self):
        """
        Returns the length of the input
        """
        return len(self.tokens)
    
    def __getitem__(self,idx):
        """
        Returns the token at index idx
        """
        return self.tokens[idx]



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
                    prog='dependency word order shuffler',
                    description='Tool for loading and randomizing word order in UD treebanks')

    parser.add_argument('filename')  
    parser.add_argument('--shuffle',default=False,action='store_true')
    parser.add_argument('--projective',default=False,action='store_true')

    args = parser.parse_args()
    istream = open(args.filename)
    for deptree in DependencyTree.read_trees(istream):

        if args.shuffle:
            deptree.shuffle_tree()
        if args.projective:
            deptree.shuffle_projective()
    
        print(deptree.sentence)
        print()
    istream.close()
