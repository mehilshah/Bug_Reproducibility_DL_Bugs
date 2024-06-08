    Q = self.fc_q(query)
    K = self.fc_k(key)
    V = self.fc_v(value)
    
    #Q = [batch size, query len, hid dim]
    #K = [batch size, key len, hid dim]
    #V = [batch size, value len, hid dim]
            
    # Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
    # K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
    # V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

    Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).view(-1, 1024)
    K = K.view(batch_size, -1, self.n_heads, self.head_dim).view(-1, 1024)
    V = V.view(batch_size, -1, self.n_heads, self.head_dim).view(-1, 1024)
    energy = torch.matmul(Q, K.transpose(1,0)) / self.scale