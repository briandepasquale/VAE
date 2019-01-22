module VAE

using TensorFlow, Distributions, Pkg, Printf
#include(Pkg.dir("TensorFlow", "examples", "mnist_loader.jl"))
joinpath(dirname(pathof(TensorFlow)), "..", "examples", "mnist_loader.jl")

mutable struct VariationalAutoEncoder
    sess
    x
    z
    x_hat_μ
    z_μ
    z_log_σ2
    Loss
    optimizer    
    loader
end

function xavier_init(fan_in, fan_out; constant=1) 
    
    low,high = -constant*sqrt(6. /(fan_in + fan_out)), constant*sqrt(6. /(fan_in + fan_out))
    
    return map(Float32, rand(Uniform(low, high), fan_in, fan_out))
    
end

function initialize_weights(n_hidden_recog_1, n_hidden_recog_2, 
                        n_hidden_gener_1,  n_hidden_gener_2, 
                        n_input, n_z)
    
    all_weights = Dict()
    
    all_weights["weights_recog"] = Dict("h1" => Variable(xavier_init(n_input, n_hidden_recog_1)),
        "h2" => Variable(xavier_init(n_hidden_recog_1, n_hidden_recog_2)),
        "out_mean" => Variable(xavier_init(n_hidden_recog_2, n_z)),
        "out_log_sigma" => Variable(xavier_init(n_hidden_recog_2, n_z)))
    
    all_weights["biases_recog"] = Dict("b1" => Variable(zeros(Float32, n_hidden_recog_1)),
        "b2" => Variable(zeros(Float32, n_hidden_recog_2)),
        "out_mean" => Variable(zeros(Float32, n_z)),
        "out_log_sigma" => Variable(zeros(Float32, n_z)))
    
    all_weights["weights_gener"] = Dict("h1" => Variable(xavier_init(n_z, n_hidden_gener_1)),
        "h2" => Variable(xavier_init(n_hidden_gener_1, n_hidden_gener_2)),
        "out_mean" => Variable(xavier_init(n_hidden_gener_2, n_input)),
        "out_log_sigma" => Variable(xavier_init(n_hidden_gener_2, n_input)))
    
    all_weights["biases_gener"] = Dict("b1" => Variable(zeros(Float32, n_hidden_gener_1)),
        "b2" => Variable(zeros(Float32, n_hidden_gener_2)),
        "out_mean" => Variable(zeros(Float32, n_input)),
        "out_log_sigma" => Variable(zeros(Float32, n_input)))
    
    return all_weights
    
end

function create_network(network_architecture, x, batch_size)
    
    # Initialize autoencode network weights and biases
    weights = initialize_weights(values(network_architecture)...)

    # Use recognition network to determine mean and 
    # (log) variance of Gaussian distribution in latent
    # space
    z_μ, z_log_σ2 = recognition_network(x, weights["weights_recog"], weights["biases_recog"])

    ϵ = map(Float32, rand(Normal(0, 1), batch_size, network_architecture["n_z"]))
    #ϵ = random_normal([batch_size, network_architecture["n_z"]], mean=0, stddev=1, dtype=Float32)
    # z = mu + sigma*epsilon
    z = z_μ + sqrt(exp(z_log_σ2)) .* ϵ

    # Use generator to determine mean of
    # Bernoulli distribution of reconstructed input
    x_hat_μ = generator_network(z, weights["weights_gener"], weights["biases_gener"])
    
    return x_hat_μ, z_μ, z_log_σ2, z
    
end

function recognition_network(x, weights, biases; fct=nn.softplus)

    layer_1 = fct(x * weights["h1"] + biases["b1"]) 
    layer_2 = fct(layer_1 * weights["h2"] + biases["b2"])
    z_μ = layer_2 * weights["out_mean"] + biases["out_mean"]
    z_log_σ2 = layer_2 * weights["out_log_sigma"] + biases["out_log_sigma"]
    
    return z_μ, z_log_σ2
    
end

function generator_network(z, weights, biases; fct=nn.softplus)

    layer_1 = fct(z * weights["h1"] + biases["b1"])
    layer_2 = fct(layer_1 * weights["h2"] + biases["b2"]) 
    x_hat_μ = nn.sigmoid(layer_2 * weights["out_mean"] + biases["out_mean"])
        
end

function create_loss_optimizer(x,x_hat_μ,z_μ,z_log_σ2;learning_rate=0.001)
    
    # The loss is composed of two terms:
    # 1.) The reconstruction loss (the negative log probability
    #     of the input under the reconstructed Bernoulli distribution 
    #     induced by the decoder in the data space).
    #     This can be interpreted as the number of "nats" required
    #     for reconstructing the input when the activation in latent
    #     is given.
    # Adding 1e-10 to avoid evaluation of log(0.0)
    reconstr_loss = -sum(x .* log(1e-10 + x_hat_μ) + (1-x) .* log(1e-10 + 1 - x_hat_μ), 2)
    
    # 2.) The latent loss, which is defined as the Kullback Leibler divergence 
    ##    between the distribution in latent space induced by the encoder on 
    #     the data and some prior. This acts as a kind of regularizer.
    #     This can be interpreted as the number of "nats" required
    #     for transmitting the the latent space distribution given
    #     the prior.
    latent_loss = -0.5 * sum(1 + z_log_σ2 - z_μ.^2 - exp(z_log_σ2), 2)
    
    #Loss = reduce_mean(reconstr_loss + latent_loss, axis=[1])   # average over batch
    Loss = mean(reconstr_loss + latent_loss, 1)   # average over batch
    optimizer = train.minimize(train.AdamOptimizer(learning_rate), Loss) # Use ADAM optimizer
        
    return Loss, optimizer
    
end

function trainVAE(network_architecture; learning_rate=0.001,
          batch_size=100, training_epochs=10, display_step=1, n_samples = 6e4)
        
    sess = Session(Graph())   
    x = placeholder(Float32, shape=[nothing, network_architecture["n_input"]])
    x_hat_μ, z_μ, z_log_σ2, z = create_network(network_architecture, x, batch_size)
    Loss, optimizer = create_loss_optimizer(x, x_hat_μ, z_μ, z_log_σ2)
    
    run(sess, global_variables_initializer())
    
    loader = DataLoader()  
    
    # Training cycle
    for epoch in 1:training_epochs

        avg_cost = 0.
        total_batch = Int(n_samples / batch_size)

        # Loop over all batches
        for i in 1:total_batch

            batch_xs = next_batch(loader, batch_size)
            batch_xs = broadcast(/,batch_xs[1],maximum(batch_xs[1],2));

            # Fit training using batch data
            cur_loss, = run(sess, (Loss, optimizer), Dict(x => batch_xs))
            # Compute average loss
            avg_cost += cur_loss / n_samples * batch_size

        end

        println(@sprintf("Epoch %.0f, current loss is %.9f.", epoch, avg_cost))
    
    end
    
    vae = VariationalAutoEncoder(sess, x, z, x_hat_μ, z_μ, z_log_σ2, Loss, optimizer, loader)
        
    return vae
                
end

"""
    transform(vae,X)

Transform data by mapping it into the latent space
"""
transform(vae, X) = run(vae.sess, vae.z_μ, Dict(vae.x => X))
    
#Generate data by sampling from latent space, drawn from prior in latent space.        
generate(vae, z_mu) = run(vae.sess, vae.x_hat_μ, Dict(vae.z => z_mu))

#Use VAE to reconstruct given data.
reconstruct(vae, X) = run(vae.sess, vae.x_hat_μ, Dict(vae.x => X))

end
