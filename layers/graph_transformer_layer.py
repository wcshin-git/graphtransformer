import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
import numpy as np

"""
    Graph Transformer Layer
    
"""

"""
    Util functions
"""
def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field]).sum(-1, keepdim=True)}  # edges.src[src_field], edges.dst[dst_field]: torch.Size([num_nodes_in_batch, num_heads, dim_head]). *가 element-wise곱셈을 하고 뒤에 sum을 해서 torch.Size([num_nodes_in_batch, num_heads, 1])로 만듦. 즉 dot product를 한 것. 
    return func

def scaled_exp(field, scale_constant):
    def func(edges):
        # clamp for softmax numerical stability
        return {field: torch.exp((edges.data[field] / scale_constant).clamp(-5, 5))}

    return func


"""
    Single Attention Head
"""

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, use_bias):
        super().__init__()
        
        self.out_dim = out_dim    # out_dim//num_heads
        self.num_heads = num_heads
        
        if use_bias:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        else:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        
    
    def propagate_attention(self, g):
        # Compute attention score
        g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score')) #, edges)  # g.apply_edges: Apply the function on the edges to update their features. g.edata['score']에 torch.Size([tot_num_edges, num_heads, 1])가 저장됨. edge의 시작점을 query, 도착점을 key로 하여 dot product 계산한 값임
        g.apply_edges(scaled_exp('score', np.sqrt(self.out_dim)))

        # Send weighted values to target nodes    dgl.DGLGraph.send_and_recv(edges, message_func, reduce_func)
        eids = g.edges() # tuple  (tensor([   0,    0,    0,  ..., 2938, 2938, 2938]), tensor([   2,    5,    6,  ..., 2930, 2936, 2937]))
        g.send_and_recv(eids, fn.src_mul_edge('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))  # src_mul_edge(src, edge, out)  sum(msg, out)    # src(trg이 아님!)의 V_h를 score와 곱한다! 
        g.send_and_recv(eids, fn.copy_edge('score', 'score'), fn.sum('score', 'z')) # copy_edge
    
    def forward(self, g, h):
        
        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)
        
        # Reshaping into [num_nodes_in_batch, num_heads, feat_dim] to 
        # get projections for multi-head attention
        g.ndata['Q_h'] = Q_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['K_h'] = K_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['V_h'] = V_h.view(-1, self.num_heads, self.out_dim)
        
        self.propagate_attention(g)
        
        head_out = g.ndata['wV']/g.ndata['z']  # g.ndata['wV'].shape: torch.Size([num_nodes_in_batch, num_head, dim_head])   g.ndata['z'].shape: torch.Size([2939, 8, 1])
        
        return head_out # head_out: torch.Size([num_nodes_in_batch, num_head, dim_head])
    

class GraphTransformerLayer(nn.Module):
    """
        Param: 
    """
    def __init__(self, in_dim, out_dim, num_heads, dropout=0.0, layer_norm=False, batch_norm=True, residual=True, use_bias=False):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm        
        self.batch_norm = batch_norm
        
        self.attention = MultiHeadAttentionLayer(in_dim, out_dim//num_heads, num_heads, use_bias)
        
        self.O = nn.Linear(out_dim, out_dim)

        if self.layer_norm:
            self.layer_norm1 = nn.LayerNorm(out_dim)
            
        if self.batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(out_dim)
        
        # FFN
        self.FFN_layer1 = nn.Linear(out_dim, out_dim*2)  # 2배로 불리는군
        self.FFN_layer2 = nn.Linear(out_dim*2, out_dim)

        if self.layer_norm:
            self.layer_norm2 = nn.LayerNorm(out_dim)
            
        if self.batch_norm:
            self.batch_norm2 = nn.BatchNorm1d(out_dim)
        
    def forward(self, g, h):
        h_in1 = h # for first residual connection
        
        # multi-head attention out
        attn_out = self.attention(g, h) # attn_out: torch.Size([num_nodes_in_batch, num_head, dim_head])
        h = attn_out.view(-1, self.out_channels)
        
        h = F.dropout(h, self.dropout, training=self.training)
        
        h = self.O(h)
        
        if self.residual:
            h = h_in1 + h # residual connection
        
        if self.layer_norm:
            h = self.layer_norm1(h)
            
        if self.batch_norm:
            h = self.batch_norm1(h)
        
        h_in2 = h # for second residual connection
        
        # FFN
        h = self.FFN_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_layer2(h)

        if self.residual:
            h = h_in2 + h # residual connection
        
        if self.layer_norm:
            h = self.layer_norm2(h)
            
        if self.batch_norm:
            h = self.batch_norm2(h)       

        return h
        
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.num_heads, self.residual)