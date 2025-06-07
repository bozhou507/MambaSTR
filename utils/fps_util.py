from .copyutil import copy_internal_nodes


def get_fps(model, inputs, kwargs: dict={}, device='cuda'):
    import time

    model.eval()
    model.to(device)

    for _ in range(10):
        _ = model(*copy_internal_nodes(inputs))

    start_time = time.time()

    iterations = 100
    for _ in range(iterations):
        _ = model(*copy_internal_nodes(inputs), **kwargs)

    end_time = time.time()

    total_time = end_time - start_time
    fps = iterations / total_time
    return fps