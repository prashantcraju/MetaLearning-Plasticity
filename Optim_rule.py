import torch


def my_optimizer(params, loss, logits, activation, Beta, lr, dr):
    """
        One step update of the inner-loop.
    :param params:
    :param loss: loss value
    :param logits: unnormalized prediction values
    :param activation: vector of activations
    :param Beta: smoothness coefficient for non-linearity
    :param lr: learning rate variable
    :param dr: damping rate variable
    :return:
    """
    # -- error
    e = [torch.autograd.grad(loss, logits, create_graph=True)[0]]
    feedback = dict({k: v for k, v in params.items() if 'fk' in k})  # todo: add bias later
    for y, i in zip(reversed(activation), reversed(list(feedback))):
        e.insert(0, torch.matmul(e[0], feedback[i]) * (1 - torch.exp(-Beta * y)))  # note: g'(z) = 1 - e^(-Beta*y)

    # -- weight update
    i = 0
    for k, p in params.items():
        if p.adapt:
            if k[4:] == 'weight':
                p.update = - lr * torch.matmul(e[i+1].T, activation[i])
                params[k] = (1 - dr) * p + p.update
                params[k].adapt = p.adapt
            elif k[4:] == 'bias':
                p.update = - lr * e[i+1].squeeze(0)
                params[k] = (1 - dr) * p + p.update
                params[k].adapt = p.adapt

                i += 1

    return params
