import pandas as pd

import os
os.environ['PATH'] += os.pathsep + 'graphviz-2.38/bin'

'''
Branching for decision tree using recursion
    @:param df: the pandas dataframe of the data_set
    @:return root: Node, the root node of decision tree
'''
def TreeGenerate(df):
    # generate a new root node
    new_node = Node(None, None, {})
    label_arr = df[df.columns[-1]]

    label_count = NodeLabel(label_arr)
    # assert the label_count isn't empty
    if label_count:
        new_node.label = max(label_count, key=label_count.get)
        # return when only 1 class in current node or  array is empty
        if len(label_count) == 1 or len(label_arr) == 0:
            return new_node

        # get the optimal attr for a new branching
        new_node.attr, div_value = OptAttr(df)

    # recursion
    if div_value == 0: # categoric variable
        value_count = ValueCount(df[new_node.attr])
        for value in value_count:
            df_v = df[df[new_node.attr].isin([value])] # get sub set
            # delete current attribution
            df_v = df_v.drop(new_node.attr, 1)
            new_node.attr_down[value] = TreeGenerate(df_v)

    else: # left and right child
        value_l = "<=%.3f" % div_value
        value_r = ">%.3f" % div_value
        df_v_l = df[df[new_node.attr] <= div_value]
        df_v_r = df[df[new_node.attr] > div_value]

        new_node.attr_down[value_l] = TreeGenerate(df_v_l)
        new_node.attr_down[value_r] = TreeGenerate(df_v_r)

    return new_node


class Node(object):
    def __init__(self, attr=None, label=None, attr_down={}):
        self.attr = attr
        self.label = label
        self.attr_down = attr_down


def NodeLabel(label_arr):
    # store count of label
    label_count = {}

    for label in label_arr:
        if label in label_count:
            label_count[label] += 1
        else:
            label_count[label] = 1

    return label_count


def OptAttr(df):
    # find the optimal attributes of current data_set
    info_gain = 0

    for attr_id in df.columns[1:-1]:
        info_gain_tmp, div_value_tmp = InfoGain(df, attr_id)
        if info_gain_tmp > info_gain:
            info_gain = info_gain_tmp
            opt_attr = attr_id
            div_value = div_value_tmp

    return opt_attr, div_value


def InfoGain(df, index):
    info_gain = InfoEnt(df.values[:,-1]) # info_gain for the whole label
    div_value = 0 # div_value for continous attribute

    n = len(df[index]) # the number of sample

    # 1. for continuous variable using method of bisection
    if df[index].dtype == float:
        sub_info_ent = {} # store the div_value (div) and it's subset entropy

        df = df.sort_values([index], ascending=1)  # sorting via column
        df = df.reset_index(drop=True)

        data_arr = df[index]
        label_arr = df[df.columns[-1]]

        for i in range(n - 1):
            div = (data_arr[i] + data_arr[i+1]) / 2
            sub_info_ent[div] = ((i+1)*InfoEnt(label_arr[0:i+1]) / n) + ((n - i - 1)*InfoEnt(label_arr[i + 1: -1]) / n)

        # to get the min subset entropy sum and it's divide value
        div_value, sub_info_ent_max = min(sub_info_ent.items(), key=lambda  x: x[1])
        info_gain -= sub_info_ent_max

    # 2. for discrete variable (categoric variable)
    else:
        data_arr = df[index]
        label_arr = df[df.columns[-1]]
        value_count = ValueCount(data_arr)

        for key in value_count:
            key_label_arr = label_arr[data_arr == key]
            info_gain -= value_count[key] * InfoEnt(key_label_arr) / n

    return info_gain, div_value

'''
 calculate the information entropy of an attribution
'''
def InfoEnt(label_arr):
    try:
        from math import log2
    except ImportError:
        print("module math.log2 not found")

    ent = 0
    n = len(label_arr)
    label_count = NodeLabel(label_arr)

    for key in label_count:
        ent -= (label_count[key] / n) * log2(label_count[key] / n)

    return ent


def ValueCount(data_arr):
    value_count ={}

    for label in data_arr:
        if label in value_count:
            value_count[label] += 1
        else:
            value_count[label] = 1

    return value_count


'''
build a graph from root on
@:param i: node number in this tree
@:param g: pydotplus.graphviz.Dot() object
@:param root: the root node

@:return i: node number after modified
@:return g_node: the current root node in graphviz
'''
def TreeToGraph(i, g, root):
    try:
        from pydotplus import graphviz
    except ImportError:
        print("module pydotplus.graphviz not found")

    if root.attr == None:
        g_node_label = "Node: %d\n好瓜: %s" % (i, root.label)
    else:
        g_node_label = "Node: %d\n好瓜: %s\n属性: %s" % (i, root.label, root.attr)

    g_node = i
    g.add_node(graphviz.Node(g_node, label=g_node_label, fontname="FangSong"))

    for value in list(root.attr_down):
        i, g_child = TreeToGraph(i +1, g, root.attr_down[value])
        g.add_edge(graphviz.Edge(g_node, g_child, label=value, fontname="FangSong"))

    return i, g_node


'''
visualization of decision tree from root
@:param root: Node, the root node of tree
@:param out_file: str, name and path of output file
'''
def DrawPNG(root, out_file):
    try:
        from pydotplus import graphviz
    except ImportError:
        print("module pydotplus.graphviz not found")

    g = graphviz.Dot() # generate a new dot

    TreeToGraph(0, g, root)
    g2 = graphviz.graph_from_dot_data(g.to_string())

    g2.write_jpg(out_file)


if __name__ == '__main__':
    '''
    绘制的图为课本中的图4.8  基于信息增益生成的决策树
    '''
    df = pd.read_csv('data/watermelon_3.csv')
    root = TreeGenerate(df)

    DrawPNG(root, 'decision_tree_ID3_4.3.png')

