import torch
s_index, e_index = 4, 9

data = torch.tensor([
[3,4,7,8,9,2,0,0,0,0], 
[1,5,3,4,7,2,8,9,10,0],
[3,4,7,8,10,0,0,0,0,0], # does not contain end value
[3,7,5,9,2,0,0,0,0,0], # does not contain start value
])

def _get_part_from_tokens(
    self,
    data: torch.Tensor,
    s_id: int,
    e_id: int,
) -> list[str]:
    input_ids = []
    for row in data:
        try:
            s_index = (row == s_id).nonzero(as_tuple=True)[0][0]
            e_index = (row == e_id).nonzero(as_tuple=True)[0][0]
        except IndexError:
            input_ids.append(torch.tensor([]))
            continue
        if s_index is None or e_index is None or s_index > e_index:
            input_ids.append(torch.tensor([]))
            continue
        ind = torch.arange(s_id + 1, e_id)
        input_ids.append(row.index_select(0, ind))
    return input_ids

print(_get_part_from_tokens(None, data, s_index, e_index))