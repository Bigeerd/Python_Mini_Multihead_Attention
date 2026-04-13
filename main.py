import torch,math
from torch import nn

#搞到设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备：{device}")

class MiniMultiHeadAttention(nn.Module):

    #初始化的对象
    #d_model是向量的总维数
    #n_heads拆成几个头
    #d_queries每个头里QK的维度
    #d_values每个头里V的维度
    #d_model = n_heads * d_queries
    #d_model = n_heads * d_values
    def __init__(self,d_model=128,n_heads=2,d_queries=64,d_values=64,dropout=0.1):

        #豆包说这是固定的咒语
        super().__init__()

        #基本参数
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_queries = d_queries
        self.d_keys = d_queries
        self.d_values = d_values

        #QKV
        #nn.linear是输入和输出维数，bias是偏置值，默认True
        #n_heads*n_queries是多头注意力必须这么写
        self.w_q = nn.Linear(d_model,n_heads*d_queries,bias=True)
        self.w_k = nn.Linear(d_model,n_heads*d_queries)
        self.w_v = nn.Linear(d_model,n_heads*d_values)
        self.w_o = nn.Linear(n_heads*d_values,d_model)

        #dim在哪一层做归一化，-1表示最后一层
        self.softmax = nn.Softmax(dim=-1)

        #p随机关掉神经元的概率，inplace是省内存的
        self.dropout = nn.Dropout(p=dropout,inplace=False)

    #q想查的东西，kv被查的东西
    def forward(self,q,kv):
        #固定参数
        #q.size(0)几句话
        #q.size(1)一句话有几个词
        #q.size(2)一个词有多少维
        batch_size = q.size(0)
        q_len = q.size(1)
        kv_len = kv.size(1)

        print(f"batch_size:{batch_size},q_len:{q_len},kv_len:{kv_len}\n")
        print(f"q.size(0):{q.size(0)},q.size(1):{q.size(1)},q.size(2):{q.size(2)},\n"
              f"kv.size(0):{kv.size(0)},kv.size(1):{kv.size(1)},kv.size(2):{kv.size(2)}\n")

        #算QKV，固定
        Q = self.w_q(q)
        print("投影后 Q shape:", Q.shape)
        print(Q)
        print("\n")
        K = self.w_k(kv)
        print("投影后 K shape:", K.shape)
        print(K)
        print("\n")
        V = self.w_v(kv)
        print("投影后 V shape:", V.shape)
        print(V)
        print("\n")


        #拆多头，固定
        #Q.view(batch_size, q_len, n_heads, d_queries)重排拆成多头
        #transpose(dim1, dim2)dim互换的维度编号
        Q = Q.view(batch_size,q_len,self.n_heads,self.d_queries).transpose(1,2)

        print("view+transpose后 Q shape:", Q.shape)
        print(Q)
        print("\n")

        K = K.view(batch_size,kv_len,self.n_heads,self.d_queries).transpose(1,2)

        print("view+transpose后 K shape:", K.shape)
        print(K)
        print("\n")

        V = V.view(batch_size,kv_len,self.n_heads,self.d_values).transpose(1,2)

        print("view+transpose后 V shape:", V.shape)
        print(V)
        print("\n")

        #分数，固定
        #attn_score分数矩阵
        attn_score = torch.matmul(Q,K.transpose(-2,-1)) / math.sqrt(self.d_queries)

        print("attn_score 分数矩阵 shape:", attn_score.shape)
        print(attn_score)
        print("\n")

        #权重，固定
        attn_weight = self.softmax(attn_score)#做归一化，输出同形状、所有值 0~1、每一行加起来 = 1 的概率权重矩阵
        print("Softmax后 attn_weight shape:", attn_weight.shape)
        print(attn_weight)
        print("\n")
        attn_weight = self.dropout(attn_weight)#在注意力权重矩阵里，随机按概率关掉一部分权重值，临时置 0
        print("Dropout后 attn_weight shape:", attn_weight.shape)
        print(attn_weight)
        print("\n")
        #固定
        out = torch.matmul(attn_weight,V)
        print("加权V后 out shape:", out.shape)
        print(out)
        print("\n")

        #.contiguous() = 整理内存，让数据排整齐
        out = out.transpose(1,2).contiguous().view(batch_size,q_len,-1)
        print("合并多头view后 out shape:", out.shape)
        print(out)
        print("\n")

        out = self.w_o(out)
        print("最终输出 out shape:", out.shape)
        print(out)
        print("\n")

        return out,attn_weight


        pass

torch.set_printoptions(precision=3,
                       threshold=None,
                       edgeitems=10,
                       linewidth=200,
                       sci_mode=False)

model = MiniMultiHeadAttention(d_model = 128,n_heads = 2)

#模拟输入数据测试模型
test_q = torch.randn(2,5,128)
test_k = torch.randn(2,5,128)
print(test_k)
print("\n")
print(test_q)
print("\n")

final_out,final_atten = model(test_q,test_k)