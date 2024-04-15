import numpy as np
from ARIMA import ARIMA

def adaptive_vit_inference_offloading(partitions, total_minibatches, device_count, head_count, neuron_count, eta , tr, te, ts, lambda_value):
    partitions_prime = partitions

    # Initialize lists to store sequence FLOPS and BW for each device
    seq_flops = [[] for _ in range(device_count)]
    seq_bw = [[] for _ in range(device_count)]

    ARIMA_model=ARIMA(p=5, d=1, q=0)

    for n in range(total_minibatches):
        # Compute DL for each device
        DL = []
        for d in range(device_count):
            Params = ...  # Calculate Params_d
            BW = Params  / (tr-ts)  # Calculate BW_d
            FLOPs_a = calculate_flops_MSA(partitions.height, partitions.width, partitions.channels, partitions.num_heads)  # Calculate FLOPs_a
            FLOPS = (FLOPs_a+FLOPs_f) / te-ts  # Calculate FLOPS_d
            FLOPs_f = calculate_flops_FC(partitions.height, partitions.width, partitions.num_output_neurons, partitions.num_input_neurons)  # Calculate FLOPs_f
            DL_d = compute_DL(Params, BW, FLOPs_a, FLOPs_f, FLOPS)
            DL.append(DL_d)

            # Append FLOPS_d and BW_d to corresponding sequences
            seq_flops[d].append(FLOPS[d])
            seq_bw[d].append(BW[d])

            # Perform ARIMA prediction for FLOPS_d and BW_d
            pre_flops_d = ARIMA_model.fit(seq_flops)
            pre_bw_d = ARIMA_model.fit(seq_bw)

            eta=compute_eta(DL, lambda_value)

        # Adjust DL based on eta
        if eta > 0:
            for h in range(head_count):
                d = np.argmin(DL)
                FLOPs_h = calculate_flops_MSA(partitions.height, partitions.width, partitions.channels,1)  # Calculate FLOPs_h
                DL[d] += FLOPs_h / pre_flops_d

            for n in range(neuron_count):
                d = np.argmin(DL)
                FLOPs_n = calculate_flops_FC(partitions.height, partitions.width, partitions.num_output_neurons,1)  # Calculate FLOPs_n
                DL[d] += FLOPs_n / pre_flops_d

        # Update partitions_prime
        partitions_prime = partitions

    return partitions_prime


def calculate_flops_MSA(height, width, channels, num_heads):
    FLOPs_MSA = 4 * num_heads * height * width * channels**2 + 2 * num_heads * (height * width)**2
    return FLOPs_MSA


def calculate_flops_FC(height, width, num_output_neurons, num_input_neurons):
    FLOPs_FC = height * width * (num_output_neurons + 1) * num_input_neurons
    return FLOPs_FC


# 计算 DL^{n+1}
def compute_DL(Params, BW, FLOPs_a, FLOPs_f, FLOPS_d):
    DL = [Params[d] / BW[d] + FLOPs_a / FLOPS_d[d] + FLOPs_f / FLOPS_d[d] for d in range(len(Params))]
    return DL

# 计算 eta
def compute_eta(DL, lambda_value):
    # 计算 eta
    max_DL = max(DL)
    min_DL = min(DL)
    eta = abs(max_DL - min_DL) - lambda_value
    return eta

