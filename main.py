print('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(__file__,__name__,str(__package__)))
import tensorflow as tf
import pkgs.models.Homogeneous_Poisson_NN as HPNN 

if __name__ == '__main__':
    input_norm = {'rhs_max_magnitude':True}
    output_scaling = None#{'max_domain_size_squared':True}

    pbcc = {
        "filters": [4,16,32],
        "kernel_sizes": [19,17,15],
        "padding_mode": "symmetric",
        "activation": tf.nn.leaky_relu,
        "use_bias": False,
        "bias_initializer":"zeros"
	}
    bcc = {
        "downsampling_factors": [1,2,3,4,8,16,32,64],
        "upsampling_factors": [1,2,3,4,8,16,32,64],
        "filters": 32,
        "conv_kernel_sizes": [13,13,13,13,13,13,13,13],
        "n_convs": [2,2,2,2,2,2,2,2],
        "padding_mode": "CONSTANT",
        "constant_padding_value": 4.0,
        "conv_activation": tf.nn.leaky_relu,
        "conv_use_bias": False,
        "use_resnet": True,
        "conv_downsampling_kernel_sizes": [3,2,3,4,8,16,32,64],
        "conv_initializer_constraint_regularizer_options":{"kernel_regularizer":tf.keras.regularizers.l2()},
        "downsampling_method": "conv"
	}
    fcc = {
        "filters": [16,12,8,4,2,1],
        "kernel_sizes": [11,7,5,5,3,3],
        "padding_mode": "CONSTANT",
        "constant_padding_value": 2.0,
        "activation": tf.nn.tanh,
        "use_bias": False,
        "bias_initializer":"zeros"
        }
    
    #mod = HPNN(2, use_batchnorm = False, input_normalization = input_norm, output_scaling = output_scaling, pre_bottleneck_convolutions_config = pbcc, bottleneck_config = bcc, final_convolutions_config = fcc, bottleneck_upsampling = 'multilinear')
    #convinp = 2*tf.random.uniform((1,1,3000,3000))-1
    #denseinp = tf.random.uniform((1,2))

    #print(mod([convinp,denseinp]).shape)
    #mod.summary()
