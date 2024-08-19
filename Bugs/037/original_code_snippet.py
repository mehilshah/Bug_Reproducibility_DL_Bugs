tensor([[ 1., -5.],
        [ 2., -4.],
        [ 3.,  2.],
        [ 4.,  1.],
        [ 5.,  2.]])

tensor([[-1.,  1.],
        [ 1., -1.]], requires_grad=True)

apply_i = lambda x: torch.matmul(x, i)
final = pytorch.tensor([apply_i(a) for a in x])