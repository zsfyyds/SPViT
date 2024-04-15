import torch.utils.benchmark as benchmark_time

def benchmark(name,model,images,epcho):
    t = benchmark_time.Timer(
        stmt=f'{name}(model,images)',
        setup=f'from __main__ import {name}',
        globals={'model':model,'images':images},
        label='Multithreaded batch dot',
        sub_label='Implemented using mul ')

    print(t.timeit(epcho))