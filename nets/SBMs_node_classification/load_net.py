"""
    Utility file to select GraphNN model as
    selected by the user
"""

from nets.SBMs_node_classification.graph_transformer_net import GraphTransformerNet

def GraphTransformer(net_params):
    return GraphTransformerNet(net_params)  # net_params: {'L': 10, 'n_heads': 8,...}

def gnn_model(MODEL_NAME, net_params):
    models = {
        'GraphTransformer': GraphTransformer
    }
        
    return models[MODEL_NAME](net_params)